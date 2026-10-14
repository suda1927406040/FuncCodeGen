"""Server-native subprocess execution; no SSH or SCP needed in same-host mode."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PROXY_KEYS = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']


def offline_process_env(bundle, original=None):
    environment = dict(os.environ if original is None else original)
    for name in PROXY_KEYS + ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_BASE_URL']:
        environment.pop(name, None)
    environment['NO_PROXY'] = '*'
    environment['no_proxy'] = '*'
    environment['PYTHONPATH'] = str(Path(bundle) / 'harness')
    for name in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'PYTHONNOUSERSITE', 'PIP_NO_INDEX']:
        environment[name] = '1'
    return environment


def process_alive(pid, run_id):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    proc = Path('/proc') / str(pid)
    if (proc / 'cmdline').is_file():
        command = (proc / 'cmdline').read_bytes()
        if run_id.encode() not in command:
            return False
        stat = (proc / 'stat').read_text()
        if stat.rsplit(')', 1)[-1].strip().startswith('Z'):
            return False
    return True


def same_host_stage(args, source, mode, number, ids, write_predictions, write_json, poll_seconds=20):
    name = args.run_root.name + '_' + mode + '_' + str(number)
    workspace = Path(args.bundle) / 'state-feedback' / args.run_root.name
    stage = args.run_root / 'rounds' / str(number) / mode
    workspace.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    archive = workspace / (name + '.tar.gz')
    marker = stage / 'remote_pid.json'

    def launch():
        predictions = workspace / (name + '.jsonl')
        write_predictions(predictions, ids)
        if mode == 'trace':
            watch = workspace / 'watch'
            watch.mkdir(exist_ok=True)
            for iid in ids:
                shutil.copy2(args.run_root / 'rounds' / str(number) / 'watch' / (iid + '.json'),
                             watch / (iid + '.json'))
        command = [str(Path(args.bundle) / 'runtime/bin/python'),
                   str(source / 'scripts/state_feedback_remote.py'),
                   '--bundle', str(args.bundle), '--workspace', str(workspace), '--run-id', name,
                   '--predictions', str(predictions), '--mode', mode, '--workers', str(args.workers)]
        with (workspace / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       env=offline_process_env(args.bundle), start_new_session=True)
        write_json(marker, {'pid': process.pid, 'run_id': name, 'execution': 'same-host'})
        return process.pid

    if not archive.is_file():
        # A stale marker from a crashed attempt holds a dead pid; relaunch
        # instead of failing the whole run on resume.
        pid = json.loads(marker.read_text())['pid'] if marker.is_file() else launch()
        relaunches = 0
        while not archive.is_file():
            if not process_alive(pid, name):
                if relaunches >= 3:
                    raise RuntimeError('Offline evaluation/diagnosis exited before completion: ' + name)
                relaunches += 1
                pid = launch()
            time.sleep(poll_seconds)
    result = workspace / 'results' / name
    if not result.is_dir():
        raise RuntimeError('Completed stage has no results: ' + name)
    return result
