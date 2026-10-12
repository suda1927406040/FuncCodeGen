"""Server-only offline evaluation/diagnosis. Never calls a model or downloads."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import uuid


def inject(container, path, content):
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode='w') as tar:
        member = tarfile.TarInfo(Path(path).name)
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    container.put_archive(str(Path(path).parent), blob.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--mode', choices=['evaluate', 'trace'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'PYTHONNOUSERSITE', 'PIP_NO_INDEX']:
        os.environ[key] = '1'
    spec = importlib.util.spec_from_file_location('frozen_offline', args.bundle / 'offline_harness.py')
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    offline.check_assets(offline.manifest())
    offline.imports()
    sys.path.insert(0, str(args.bundle))
    from evaluate_utf8_errors import replace_output_decoder
    from swebench.harness import docker_utils, run_evaluation
    replace_output_decoder(docker_utils)
    run_evaluation.exec_run_with_timeout = docker_utils.exec_run_with_timeout
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    destination = args.workspace / 'results' / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    rows, specs = offline.assets()
    manifest = offline.manifest()
    if args.mode == 'evaluate':
        command = argparse.Namespace(predictions_path=str(args.predictions), instance_ids=None,
                                     run_id=args.run_id, max_workers=args.workers, timeout=args.timeout)
        offline.evaluate(command)
        run_dir = args.bundle / 'runs' / args.run_id
        for prediction in predictions:
            iid = prediction['instance_id']
            result = destination / iid
            result.mkdir()
            log = run_dir / 'logs' / args.run_id / prediction['model_name_or_path'].replace('/', '__') / iid
            # Harness versions may omit the extra run-id directory.
            if not log.is_dir():
                log = run_dir / 'logs' / prediction['model_name_or_path'].replace('/', '__') / iid
            report = log / 'report.json'
            value = json.loads(report.read_text()).get(iid, {}) if report.is_file() else {}
            (result / 'report.json').write_text(json.dumps(value, indent=2))
            for name in ['test_output.txt', 'run_instance.log']:
                path = log / name
                if path.is_file():
                    # Assertions can follow a multi-MB source dump. Preserve the
                    # full artifact and select bounded evidence on the controller.
                    shutil.copyfile(path, result / name)
        for report in run_dir.glob('*.json'):
            if report.name not in {'predictions.json', 'source_predictions.json'}:
                (destination / report.name).write_bytes(report.read_bytes())
    else:
        source = Path(__file__).resolve().parent.parent
        def diagnose(prediction):
            iid = prediction['instance_id']
            out = destination / iid
            out.mkdir()
            client = offline.docker_client()
            record = manifest['images'][iid]
            c = offline.create_container(client, record['image_id'], 'fea-state-' + uuid.uuid4().hex[:16],
                                         purpose='state-feedback', spec=specs[iid])
            metadata = {'instance_id': iid, 'network_mode': 'none', 'image_id': record['image_id'],
                        'base_commit': record['base_commit'], 'fresh_initial_container': True,
                        'prediction_sha256': hashlib.sha256(prediction['model_patch'].encode()).hexdigest()}
            try:
                c.start()
                offline.inspect_baseline(c, record['base_commit'], record.get('initial_diff_sha256', offline.EMPTY_DIFF))
                metadata['baseline_verified'] = True
                offline.exec_checked(c, 'mkdir -p /tmp/fea-state-probe /tmp/fea-state-output')
                inject(c, '/tmp/fea-state-probe/runtime_probe.py', (source / 'generator/runtime_probe.py').read_bytes())
                inject(c, '/tmp/fea-state-probe/conan_probe.py', (source / 'generator/state_profiles/conan_probe.py').read_bytes())
                inject(c, '/tmp/fea-state-probe/sitecustomize.py', b'from runtime_probe import install\ninstall()\n')
                inject(c, '/tmp/fea-state-config.json', (args.workspace / 'watch' / (iid + '.json')).read_bytes())
                inject(c, '/tmp/fea-current.patch', prediction['model_patch'].encode())
                offline.exec_checked(c, 'git -C /testbed apply --check /tmp/fea-current.patch')
                offline.exec_checked(c, 'git -C /testbed apply /tmp/fea-current.patch')
                # Preserve the frozen test command, test patch, setup and environment.
                env = ('export PYTHONPATH=/tmp/fea-state-probe:${PYTHONPATH:-}; '
                       'export PYTEST_PLUGINS=runtime_probe${PYTEST_PLUGINS:+,$PYTEST_PLUGINS}; '
                       'export FEA_STATE_CONFIG=/tmp/fea-state-config.json; '
                       'export FEA_STATE_OUTPUT=/tmp/fea-state-output; '
                       'export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1;\n')
                script = env + specs[iid].eval_script
                inject(c, '/tmp/fea-state-eval.sh', script.encode())
                output, timed_out, seconds = docker_utils.exec_run_with_timeout(c, '/bin/bash /tmp/fea-state-eval.sh', args.timeout)
                (out / 'diagnostic_output.txt').write_text(output)
                metadata.update(timed_out=timed_out, seconds=seconds)
                stream, _ = c.get_archive('/tmp/fea-state-output')
                # Bounds are in the observer. Allow expanded producer/witness traces.
                count = size = 0
                summaries = []
                with tarfile.open(fileobj=io.BytesIO(b''.join(stream))) as tar, (out / 'state_trace.jsonl').open('wb') as trace:
                    for item in tar.getmembers():
                        if not item.isfile():
                            continue
                        if item.name.endswith('.jsonl'):
                            data = tar.extractfile(item).read(min(item.size, max(0, 60_000_000-size)))
                            # Only emit complete JSONL records when the byte cap is reached.
                            if data and not data.endswith(b'\n'):
                                data = data.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in data else b''
                            trace.write(data)
                            count += data.count(b'\n')
                            size += len(data)
                        # Per-process observer summaries carry the truncation flags
                        # that decide whether an unobserved call is "not called" or unknown.
                        elif Path(item.name).name.startswith('summary-') and item.name.endswith('.json'):
                            if len(summaries) < 64:
                                try:
                                    summaries.append(json.loads(tar.extractfile(item).read(1_000_000)))
                                except (ValueError, OSError, AttributeError):
                                    pass
                metadata.update(trace_events=count, extraction_bytes=size,
                                trace_status='observed' if count else 'no_events',
                                extraction_truncated=size >= 60_000_000,
                                observer_summaries=summaries)
            except Exception as error:
                metadata.update(trace_status='diagnostic_error', error_type=type(error).__name__, error=str(error)[:1000])
            finally:
                c.remove(force=True)
                client.close()
                (out / 'trace_metadata.json').write_text(json.dumps(metadata, indent=2))
            return metadata
        with offline.block_external_sockets(), concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for metadata in pool.map(diagnose, predictions):
                print(json.dumps({'event': 'diagnostic_complete', **metadata}), flush=True)
    offline.check_assets(offline.manifest())
    archive = args.workspace / (args.run_id + '.tar.gz')
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    with tarfile.open(temporary, 'w:gz') as tar:
        tar.add(destination, arcname=args.run_id)
    temporary.replace(archive)
    print(json.dumps({'event': 'remote_complete', 'archive': str(archive)}), flush=True)


if __name__ == '__main__':
    main()
