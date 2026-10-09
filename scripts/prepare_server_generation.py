"""Prepare isolated base worktrees and resume a migrated server checkpoint.

Uses existing public Git caches and existing server credentials. No downloads.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess


def credential_key(source):
    tree=ast.parse(source.read_text())
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):constants[target.id]=value
    def literal(node):
        if isinstance(node,ast.Name):return constants.get(node.id)
        try:return ast.literal_eval(node)
        except (ValueError,TypeError):return None
    found={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Subscript) and ast.unparse(target.value)=='os.environ' and literal(target.slice)=='OPENAI_API_KEY':
                    found['key']=literal(node.value)
        elif isinstance(node,ast.Call):
            if ast.unparse(node.func)=='os.environ.setdefault' and len(node.args)>=2 and literal(node.args[0])=='OPENAI_API_KEY':
                found['key']=literal(node.args[1])
            for keyword in node.keywords:
                if keyword.arg=='api_key':found.setdefault('key',literal(keyword.value))
    for name in ['OPENAI_API_KEY','API_KEY','DEEPSEEK_API_KEY']:
        if constants.get(name):found.setdefault('key',constants[name])
    key=found.get('key')
    if not isinstance(key,str) or not key.strip():raise RuntimeError('Server credentials unavailable')
    return key


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prepared-repos',type=Path,required=True)
    parser.add_argument('--credential-source',type=Path,default=Path('/data2/ymz/TOSEM2026/v5/v5.py'))
    parser.add_argument('--credentials-file',type=Path,required=True)
    parser.add_argument('--repo-cache',type=Path,nargs='+',default=[Path('/data2/ymz/funcgen-transfer-change1007/bare'),Path('/data/ymz/FEA-Bench/feabench-data/testbed')])
    args=parser.parse_args()
    rows=json.loads((args.run_root/'generation_inputs.json').read_text())
    if len(rows)!=200 or any(set(row)-{'instance_id','repo','base_commit','problem_statement','new_components'} for row in rows):
        raise RuntimeError('Expected sanitized public 200-task checkpoint')
    args.prepared_repos.mkdir(parents=True,exist_ok=True)
    def prepare(row):
        iid=row['instance_id'];dest=args.prepared_repos/iid
        def git(root,*arguments):
            return subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','protocol.allow=never','-C',str(root),*arguments],
                                  text=True,capture_output=True,timeout=300,check=True).stdout.strip()
        if not dest.is_dir():
            stem=row['repo'].replace('/','__');origin=None
            for cache in args.repo_cache:
                for candidate in [cache/(stem+'.git'),cache/stem]:
                    if not candidate.is_dir():continue
                    try:git(candidate,'cat-file','-e',row['base_commit']+'^{commit}')
                    except subprocess.CalledProcessError:continue
                    origin=candidate;break
                if origin:break
            if origin is None:raise RuntimeError('Public cached base not available: '+iid)
            git(origin,'worktree','add','--detach',str(dest),row['base_commit'])
        if git(dest,'rev-parse','HEAD')!=row['base_commit'] or git(dest,'status','--porcelain'):
            raise RuntimeError('Prepared base changed: '+iid)
        return iid
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for number,iid in enumerate(pool.map(prepare,rows),1):
            print(json.dumps({'event':'server_base_ready','completed':number,'total':200,'instance_id':iid}),flush=True)
    args.credentials_file.parent.mkdir(parents=True,exist_ok=True)
    args.credentials_file.parent.chmod(0o700)
    if not args.credentials_file.is_file():
        key=credential_key(args.credential_source)
        descriptor=os.open(str(args.credentials_file),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:
            json.dump({'api_key':key,'base_url':'https://api.deepseek.com'},stream)
    config_path=args.run_root/'run_config.json';config=json.loads(config_path.read_text())
    before=dict(config)
    if not (args.run_root/'local_run_config_before_migration.json').exists():
        (args.run_root/'local_run_config_before_migration.json').write_text(json.dumps(before,indent=2))
    config.update(source_revision=(args.source/'SOURCE_REVISION').read_text().strip(),
                  source_sha256={str(p.relative_to(args.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in args.source.rglob('*.py') if '__pycache__' not in p.parts},
                  initial_generation_location='local_completed_before_migration',feedback_generation_location='server',
                  generation_location='server',evaluation_location='same-host-offline',
                  initial_generation_source_revision=before.get('initial_generation_source_revision',before.get('source_revision')),
                  execution_transport='same-host',migration_preserved_initial_outputs=True)
    config_path.write_text(json.dumps(config,indent=2))
    complete=list((args.run_root/'initial/outputs').glob('*/worker_complete.json'))
    print(json.dumps({'event':'server_ready','base_repos':200,'initial_completed_preserved':len(complete),
                      'credential_file_ready':args.credentials_file.is_file(),'secrets_printed':False}),flush=True)


if __name__=='__main__':main()
