"""Full generation -> offline grading -> bounded state repairs.

Model calls run on the controller host. Each evaluation/diagnosis starts a fresh
frozen container. Both remote-server and native same-host execution are supported.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

SOURCE = Path(__file__).resolve().parent.parent
MODEL_LABEL = 'change_1007-state-feedback'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def reuse_initial(source, destination, rows):
    """Copy only generation artifacts; never import old repairs, grades or gold."""
    source, destination = Path(source), Path(destination)
    originals = {r['instance_id']: r for r in json.loads((source/'generation_inputs.json').read_text())}
    selected = ['generated.patch', 'result.json', 'plan.json', 'recall.json', 'generation_recall.json', 'llm_usage.json', 'worker_complete.json']
    for row in rows:
        iid = row['instance_id']
        if originals.get(iid) != row:
            raise ValueError('Reused initial input/base mismatch: ' + iid)
        old, new = source/'initial'/'outputs'/iid, destination/'initial'/'outputs'/iid
        if not (old/'worker_complete.json').is_file() or not (old/'result.json').is_file():
            raise ValueError('Initial generation is incomplete: ' + iid)
        new.mkdir(parents=True, exist_ok=True)
        hashes = {}
        # Completion marker is copied last, allowing interruption recovery.
        for name in selected:
            if (old/name).is_file():
                hashes[name] = hashlib.sha256((old/name).read_bytes()).hexdigest()
                if (new/name).exists() and hashlib.sha256((new/name).read_bytes()).hexdigest() != hashes[name]:
                    raise ValueError('Reused initial artifact changed: ' + iid + '/' + name)
                shutil.copyfile(old/name, new/name)
        write_json(new/'reuse_provenance.json', {'source': str(old.resolve()), 'sha256': hashes})


def credentials(path):
    config = json.loads(path.read_text())
    if config['base_url'].rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('Credential destination mismatch')
    os.environ.update(OPENAI_API_KEY=config['api_key'], OPENAI_BASE_URL=config['base_url'],
                      LLM_PROVIDER='openai', LLM_MODEL_GENERATE='deepseek-flash',
                      LLM_THINKING_MODE='disabled', LLM_MAX_TOKENS='16384', LLM_TEMPERATURE='0')


def worker(args):
    credentials(args.credentials_file)
    sys.path.insert(0, str(SOURCE))
    from run_oracle import run_single
    rows = json.loads((args.run_root / 'generation_inputs.json').read_text())
    row = next(r for r in rows if r['instance_id'] == args.worker)
    return run_single(row, args.run_root / 'initial' / 'outputs', repo_dir_override=args.prepared_repos / args.worker,
                      generation_mode='plan-edit', history_mode='patterns', recall_mode='expanded',
                      plan_max_tokens=2048, planner_json_mode='api', max_repair_attempts=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--prepared-repos', type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path, required=True)
    parser.add_argument('--host', default='5022')
    parser.add_argument('--bundle', default='/data2/ymz/feabench-offline-1007')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--repair-rounds', type=int, default=4)
    parser.add_argument('--feedback-mode', choices=['featurespace', 'state', 'basic'], default='featurespace')
    parser.add_argument('--reuse-initial-from', type=Path, help='Reuse frozen initial candidates in a NEW run; do not regenerate')
    parser.add_argument('--instance-id', action='append', help='Evaluate selected instances from the sanitized 200-case input')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--same-host', action='store_true', help='Run the full framework on the evaluation server without SSH/SCP')
    parser.add_argument('--proxy', help='Explicit HTTP proxy for model calls; excluded from offline evaluation processes')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.proxy:
        for name in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
            os.environ[name] = args.proxy
        for name in ['ALL_PROXY','all_proxy']:
            os.environ.pop(name,None)
    if args.worker:
        worker(args)
        return
    if args.workers < 1 or not 0 <= args.repair_rounds <= 4:
        raise ValueError('Invalid worker/repair bounds')
    if not args.resume:
        args.run_root.mkdir(parents=True, exist_ok=False)
        rows = json.loads(args.inputs.read_text())
        allowed = {'instance_id', 'repo', 'base_commit', 'problem_statement', 'new_components'}
        if len(rows) != 200 or len({r['instance_id'] for r in rows}) != 200 or any(set(r)-allowed for r in rows):
            raise ValueError('Expected 200 sanitized generation inputs')
        if args.instance_id:
            wanted = set(args.instance_id)
            if wanted - {r['instance_id'] for r in rows}:
                raise ValueError('Unknown selected instance')
            rows = [r for r in rows if r['instance_id'] in wanted]
        if any(not (args.prepared_repos/r['instance_id']).is_dir() for r in rows):
            raise ValueError('Missing prepared public repository')
        write_json(args.run_root/'generation_inputs.json', rows)
        hashes = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in SOURCE.rglob('*.py') if '__pycache__' not in p.parts}
        write_json(args.run_root/'run_config.json', {
            'source_revision': (SOURCE/'SOURCE_REVISION').read_text().strip() if (SOURCE/'SOURCE_REVISION').exists() else None,
            'source_sha256': hashes, 'instance_count': len(rows), 'workers': args.workers,
            'model': 'deepseek-flash', 'thinking': 'disabled', 'temperature': 0,
            'generation_max_tokens': 16384, 'structural_repair_limit': 2,
            'semantic_repair_rounds': args.repair_rounds, 'feedback_mode': args.feedback_mode,
            'initial_reused_from': str(args.reuse_initial_from.resolve()) if args.reuse_initial_from else None,
            'generation_location': 'server' if args.same_host else 'local',
            'evaluation_location': 'same-host-offline' if args.same_host else args.host,
            'fresh_initial_container_each_evaluation_and_diagnosis': True,
            'reference_patch_or_trace_in_model_inputs': False,
            'test_feedback_experiment': True, 'development_case': 'conan-io__conan-16871',
            'started_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    rows = json.loads((args.run_root/'generation_inputs.json').read_text())
    config = json.loads((args.run_root/'run_config.json').read_text())
    total = len(rows)
    # The frozen run configuration, not new CLI defaults, controls resume.
    args.feedback_mode = config['feedback_mode']
    args.repair_rounds = config['semantic_repair_rounds']
    if config.get('initial_reused_from'):
        reuse_initial(config['initial_reused_from'], args.run_root, rows)
    for name, expected in config['source_sha256'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + name)
    def progress(stage, **facts):
        value = {'stage': stage, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(), **facts}
        write_json(args.run_root/'progress.json', value)
        print(json.dumps(value), flush=True)
    progress('initial_generation', completed=0, total=total)
    def initial(row):
        iid = row['instance_id']
        out = args.run_root/'initial'/'outputs'/iid
        out.mkdir(parents=True, exist_ok=True)
        result = out/'result.json'
        repo = args.prepared_repos/iid
        head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
        clean = subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True).strip()
        if head != row['base_commit'] or clean:
            raise ValueError('Prepared base changed: ' + iid)
        if (out/'worker_complete.json').is_file():
            return json.loads(result.read_text())
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', iid,
                   '--run-root', str(args.run_root), '--prepared-repos', str(args.prepared_repos),
                   '--credentials-file', str(args.credentials_file)]
        if args.proxy:
            command.extend(['--proxy',args.proxy])
        with (out/'execution.log').open('w') as log:
            done = subprocess.run(command, stdout=log, stderr=log, timeout=7200)
        value = json.loads(result.read_text()) if result.is_file() else {'instance_id':iid,'success':False}
        write_json(out/'worker_complete.json', {'exit_code':done.returncode})
        return value
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(initial,row): row for row in rows}
        for future in concurrent.futures.as_completed(pending):
            results.append(future.result())
            write_json(args.run_root/'initial'/'generation_summary.json',results)
            progress('initial_generation', completed=len(results), total=total,
                     nonempty=sum(bool(r.get('success')) for r in results))
    sys.path.insert(0,str(SOURCE))
    from generator.state_feedback import watch_config, extract_packet, repair_once
    from generator.feature_memory import seed_featurespace, save_memory, focused_elements
    from generator.feedback_evidence import read_failure_output, grade_summary, test_delta
    credentials(args.credentials_file)
    current = {row['instance_id']: (args.run_root/'initial'/'outputs'/row['instance_id']/'generated.patch').read_text()
               if (args.run_root/'initial'/'outputs'/row['instance_id']/'generated.patch').is_file() else '' for row in rows}
    def predictions(path, ids):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps({'instance_id':iid,'model_name_or_path':MODEL_LABEL+'-'+args.feedback_mode,'model_patch':current[iid]})+'\n' for iid in ids))
    remote = args.bundle + '/state-feedback/' + args.run_root.name
    def ssh(command, timeout=120):
        for attempt in range(3):
            try:
                return subprocess.check_output(['ssh',args.host,command],text=True,timeout=timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                time.sleep(5)
    if not args.same_host:
        ssh('mkdir -p ' + shlex.quote(remote))
    package = args.run_root/'remote_sources.tar.gz'
    with tarfile.open(package,'w:gz') as tar:
        for name in ['scripts/state_feedback_remote.py','generator/runtime_probe.py','generator/state_profiles/conan_probe.py']:
            tar.add(SOURCE/name,arcname='source/'+name)
    if not args.same_host:
        subprocess.run(['scp',str(package),args.host+':'+remote+'/remote_sources.tar.gz'],check=True)
        ssh('tar -xzf '+shlex.quote(remote+'/remote_sources.tar.gz')+' -C '+shlex.quote(remote))
    def remote_stage(mode, number, ids):
        if args.same_host:
            from state_feedback_transport import same_host_stage
            return same_host_stage(args,SOURCE,mode,number,ids,predictions,write_json)
        name = args.run_root.name + '_' + mode + '_' + str(number)
        stage = args.run_root/'rounds'/str(number)/mode
        archive = stage/(name+'.tar.gz')
        if not archive.is_file():
            stage.mkdir(parents=True,exist_ok=True)
            pred = stage/'predictions.jsonl'
            predictions(pred,ids)
            subprocess.run(['scp',str(pred),args.host+':'+remote+'/'+name+'.jsonl'],check=True)
            if mode == 'trace':
                watches = stage/'watch.tar.gz'
                with tarfile.open(watches,'w:gz') as tar:
                    for iid in ids:
                        tar.add(args.run_root/'rounds'/str(number)/'watch'/(iid+'.json'),arcname='watch/'+iid+'.json')
                subprocess.run(['scp',str(watches),args.host+':'+remote+'/watch.tar.gz'],check=True)
                ssh('tar -xzf '+shlex.quote(remote+'/watch.tar.gz')+' -C '+shlex.quote(remote))
            command = [args.bundle+'/runtime/bin/python',remote+'/source/scripts/state_feedback_remote.py',
                       '--bundle',args.bundle,'--workspace',remote,'--run-id',name,'--predictions',remote+'/'+name+'.jsonl',
                       '--mode',mode,'--workers',str(args.workers)]
            remote_archive = remote+'/'+name+'.tar.gz'
            marker = stage/'remote_pid.json'
            if not marker.is_file():
                launch = 'nohup env PYTHONPATH='+shlex.quote(args.bundle+'/harness')+' '+shlex.join(command)
                pid = ssh(launch+' > '+shlex.quote(remote+'/'+name+'.log')+' 2>&1 < /dev/null & echo $!').strip()
                if not pid.isdigit():
                    raise ValueError('Invalid server job PID')
                write_json(marker,{'pid':int(pid),'run_id':name})
            pid = json.loads(marker.read_text())['pid']
            while True:
                result = ssh('if test -f '+shlex.quote(remote_archive)+'; then echo done; elif kill -0 '+str(pid)+' 2>/dev/null; then echo running; else echo failed; fi').strip()
                if result == 'done':
                    break
                if result == 'failed':
                    raise RuntimeError('Offline server job failed; inspect '+remote+'/'+name+'.log')
                time.sleep(20)
            subprocess.run(['scp',args.host+':'+remote_archive,str(archive)],check=True)
        target = stage/'results'
        target.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                resolved=(target/member.name).resolve()
                resolved.relative_to(target.resolve())
                if member.issym() or member.islnk():
                    raise ValueError('Unexpected link in diagnostic archive')
            tar.extractall(target,filter='data')
        return target/name
    grade = {}
    execution_counts = {'evaluation_cases': 0, 'diagnostic_cases': 0}
    histories = {row['instance_id']:[] for row in rows}
    memories = {}
    for row in rows:
        initial_dir = args.run_root/'initial'/'outputs'/row['instance_id']
        def artifact(name):
            path = initial_dir/name
            return json.loads(path.read_text()) if path.is_file() else {}
        memories[row['instance_id']] = seed_featurespace(row, artifact('plan.json'), artifact('generation_recall.json') or artifact('recall.json'))
    ids = list(current)
    for number in range(args.repair_rounds+1):
        if number and not ids:
            break
        progress('offline_evaluation',round=number,submitted=len(ids),total=total)
        evaluated = remote_stage('evaluate',number,ids)
        execution_counts['evaluation_cases'] += len(ids)
        for iid in ids:
            grade[iid] = json.loads((evaluated/iid/'report.json').read_text())
            if histories[iid]:
                attempt = histories[iid][-1]
                attempt['grade_after'] = grade_summary(grade[iid])
                attempt['test_delta'] = test_delta(attempt.get('grade_before'), attempt['grade_after'])
                attempt['outcome'] = ('rejected' if attempt.get('error') else
                                      'resolved' if grade[iid].get('resolved') else
                                      'harness_error' if not attempt['grade_after']['has_test_results'] else
                                      'regressed' if attempt['test_delta']['new_failures'] else
                                      'improved' if attempt['test_delta']['fixed'] else 'no_test_improvement')
                memories[iid]['attempts'] = histories[iid]
                save_memory(args.run_root/'featurespaces'/(iid+'.json'), memories[iid])
                write_json(args.run_root/'repairs'/iid/str(attempt['round'])/'outcome.json', attempt)
        write_json(args.run_root/'repair_history.json', histories)
        write_json(args.run_root/'latest_case_reports.json',grade)
        resolved = sum(bool(r.get('resolved')) for r in grade.values())
        progress('evaluation_complete',round=number,resolved=resolved,total=total,resolve_rate=resolved/total)
        if number == args.repair_rounds:
            break
        failed = [iid for iid in ids if not grade.get(iid,{}).get('resolved') and current[iid]
                  and grade.get(iid,{}).get('tests_status')]
        # No-test-result harness errors are preserved for diagnosis; do not invent failure states.
        traceable = [iid for iid in failed if grade.get(iid,{}).get('patch_successfully_applied')
                     and args.feedback_mode != 'basic']
        row_map = {row['instance_id']:row for row in rows}
        for iid in failed:
            log = evaluated/iid/'test_output.txt'
            recall_file = args.run_root/'initial'/'outputs'/iid/'recall.json'
            generated_recall = recall_file.with_name('generation_recall.json')
            if generated_recall.is_file():
                recall_file = generated_recall
            recall = json.loads(recall_file.read_text()) if recall_file.is_file() else {}
            watch = watch_config(args.prepared_repos/iid,row_map[iid],current[iid],recall,log.read_text(errors='replace') if log.is_file() else '')
            if args.feedback_mode == 'featurespace':
                # Newly implicated responsibilities are observed in the next
                # fresh container, without changing initial retrieval/generation.
                focused = focused_elements(memories[iid])
                for key in sorted(focused):
                    entry = memories[iid]['elements'][key]
                    file, symbol = entry['file'], entry.get('symbol', '').split('.')[-1]
                    if file.endswith('.py') and symbol and (file in watch['watch'] or len(watch['watch']) < 24):
                        watch['watch'].setdefault(file, [])
                        watch['watch'][file] = list(dict.fromkeys([symbol] + watch['watch'][file]))[:64]
            watch['failed_tests'] = list(dict.fromkeys(t for group in grade[iid].get('tests_status',{}).values() for t in group.get('failure',[])))
            write_json(args.run_root/'rounds'/str(number)/'watch'/(iid+'.json'),watch)
        traced = None
        if traceable:
            progress('offline_state_diagnosis',round=number,instances=len(traceable))
            traced = remote_stage('trace',number,traceable)
            execution_counts['diagnostic_cases'] += len(traceable)
        def repair(iid):
            directory = args.run_root/'repairs'/iid/str(number+1)
            if (directory/'combined.patch').is_file():
                return iid,(directory/'combined.patch').read_text(),None
            if (directory/'error.json').is_file():
                return iid,None,json.loads((directory/'error.json').read_text())['error']
            base = traced/iid if traced and iid in traceable else None
            meta = json.loads((base/'trace_metadata.json').read_text()) if base else {'trace_status':'not_available'}
            events = []
            if base and (base/'state_trace.jsonl').is_file():
                with (base/'state_trace.jsonl').open() as stream:
                    for line in stream:
                        try: events.append(json.loads(line))
                        except ValueError: pass
            watch=json.loads((args.run_root/'rounds'/str(number)/'watch'/(iid+'.json')).read_text())
            packet = extract_packet(events,grade.get(iid,{}),meta,watch)
            directory.parent.mkdir(parents=True,exist_ok=True)
            packet_path=directory.parent/('state_packet_'+str(number+1)+'.json')
            text, output_source = read_failure_output(evaluated/iid, base)
            packet['output_source'] = output_source
            write_json(packet_path,packet)
            previous=histories[iid][-1].get('error') if histories[iid] else None
            try:
                patch=repair_once(row_map[iid],args.prepared_repos/iid,current[iid],packet,text,watch,directory,previous,
                                  memory=memories[iid], history=histories[iid], round_number=number+1, mode=args.feedback_mode)
                return iid,patch,None
            except Exception as error:
                directory.mkdir(exist_ok=True)
                message=type(error).__name__+': '+str(error)[:2000]
                write_json(directory/'error.json',{'error':message})
                return iid,None,message
        progress('feedback_generation',round=number+1,instances=len(failed),completed=0)
        ids=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done,future in enumerate(concurrent.futures.as_completed([pool.submit(repair,iid) for iid in failed]),1):
                iid,patch,error=future.result()
                directory = args.run_root/'repairs'/iid/str(number+1)
                after = directory/'featurespace_after.json'
                if after.is_file():
                    memories[iid] = json.loads(after.read_text())
                attempt = {'round':number+1,'patch_updated':patch is not None,'error':error,
                           'grade_before':grade_summary(grade[iid]), 'outcome':'pending_evaluation'}
                for phase in ['before', 'after']:
                    record = directory/('candidate_'+phase+'.json')
                    if record.is_file():
                        candidate = json.loads(record.read_text())
                        attempt['behavior_'+phase] = candidate['behavior_sha256']
                        attempt['patch_'+phase] = candidate['patch_sha256']
                        if phase == 'after':
                            attempt['changed_files'] = candidate.get('changed_files', [])
                if memories[iid].get('diagnosis'):
                    attempt['diagnosis'] = memories[iid]['diagnosis']
                histories[iid].append(attempt)
                memories[iid]['attempts'] = histories[iid]
                save_memory(args.run_root/'featurespaces'/(iid+'.json'), memories[iid])
                if patch is not None:
                    current[iid]=patch
                # Failed preflight retains the last valid candidate; retries are bounded by rounds.
                ids.append(iid)
                write_json(args.run_root/'repair_history.json',histories)
                progress('feedback_generation',round=number+1,instances=len(failed),completed=done)
        predictions(args.run_root/'rounds'/str(number+1)/'predictions.jsonl',list(current))
    predictions(args.run_root/'predictions_final.jsonl',list(current))
    if total == 200:
        predictions(args.run_root/'predictions_final200.jsonl',list(current))
    usage_files=list((args.run_root/'initial'/'outputs').glob('*/llm_usage.json'))+list((args.run_root/'repairs').glob('*/*/llm_usage.json'))
    calls=attempts=tokens=0
    initial_calls=initial_tokens=initial_attempts=0
    repair_unknown_usage=0
    for path in usage_files:
        value=json.loads(path.read_text())
        calls+=value.get('logical_calls',1)
        attempts+=value.get('provider_attempts',value.get('attempts',1))
        tokens+=value.get('reported_total_tokens',value.get('total_tokens')) or 0
        if 'initial' in path.relative_to(args.run_root).parts:
            initial_calls+=value.get('logical_calls',1)
            initial_tokens+=value.get('reported_total_tokens',value.get('total_tokens')) or 0
            initial_attempts+=value.get('provider_attempts',value.get('attempts',1))
        else:
            repair_unknown_usage+=value.get('usage_unknown_calls',0)
    summary={'total':total,'resolved':sum(bool(r.get('resolved')) for r in grade.values()),
             'empty':sum(not p for p in current.values()),'logical_calls':calls,'provider_attempts':attempts,
             'harness_error_or_missing_report':sum(bool(current[iid]) and not grade.get(iid,{}).get('tests_status') for iid in current),
             'reported_tokens':tokens,'semantic_repair_limit':args.repair_rounds,
             'final_candidates_per_case':1,'test_feedback_experiment':True,
             'feedback_mode':args.feedback_mode, 'initial_reused_from':config.get('initial_reused_from'),
             'new_run_logical_calls':calls-(initial_calls if config.get('initial_reused_from') else 0),
             'new_run_reported_tokens':tokens-(initial_tokens if config.get('initial_reused_from') else 0),
             'new_run_provider_attempts':attempts-(initial_attempts if config.get('initial_reused_from') else 0),
             'repair_calls_with_unknown_usage':repair_unknown_usage, 'execution_counts':execution_counts,
             'development_case_reported_separately':'conan-io__conan-16871'}
    summary['resolve_rate']=summary['resolved']/total
    other = [r for iid,r in grade.items() if iid != 'conan-io__conan-16871']
    summary['other_cases']={'total':len(other),'resolved':sum(bool(r.get('resolved')) for r in other)}
    summary['other_cases']['resolve_rate']=summary['other_cases']['resolved']/len(other) if other else None
    write_json(args.run_root/'summary.json',summary)
    progress('complete',**summary)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        # Provider wrappers redact secrets. Never dump credential configuration.
        args=parse_args()
        if not args.worker:
            write_json(args.run_root/'failure.json',{'error_type':type(error).__name__,'error':str(error)[:2000]})
            write_json(args.run_root/'progress.json',{'stage':'failed','error_type':type(error).__name__})
        raise
