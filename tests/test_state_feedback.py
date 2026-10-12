import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from generator.feedback_evidence import evidence_bundle
from generator.llm_interface import LLMResponse
from generator.state_feedback import extract_packet, patch_paths, repair_once, watch_config


class StateFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.repo=self.root/'repo'
        self.repo.mkdir()
        (self.repo/'state.py').write_text('def select(node):\n    node.state = "Missing"\n    return node.state\n')
        for args in [('init','-q'),('add','.'),('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','base')]:
            subprocess.run(['git',*args],cwd=self.repo,check=True,capture_output=True)
        self.instance={'instance_id':'case','repo':'test/repo','problem_statement':'Select a valid state',
                       'new_components':[{'file':'state.py','components':[{'name':'select'}]}],
                       'patch':'GOLD_POISON','test_patch':'HIDDEN_TEST_PATCH_POISON'}
        self.patch='diff --git a/state.py b/state.py\n--- a/state.py\n+++ b/state.py\n@@ -1,3 +1,3 @@\n def select(node):\n-    node.state = "Missing"\n+    node.state = "Invalid"\n     return node.state\n'

    def tearDown(self):
        self.temp.cleanup()

    def reply(self,edits):
        return LLMResponse(json.dumps({'edits':edits}),'openai','test','stop',10,20,30)

    def test_repairs_current_patch_exports_combined_base_diff_without_mutating_base(self):
        watch=watch_config(self.repo,self.instance,self.patch)
        response=self.reply([{'file_path':'state.py','before':'    node.state = "Invalid"','after':'    node.state = "Build"'}])
        with patch('generator.llm_interface.call_llm_with_usage',return_value=response) as llm:
            combined=repair_once(self.instance,self.repo,self.patch,{},'Expected Build, got Invalid',watch,self.root/'round1')
        prompt=llm.call_args.args[0]
        self.assertIn('node.state = "Invalid"',prompt)
        self.assertNotIn('GOLD_POISON',prompt)
        self.assertNotIn('HIDDEN_TEST_PATCH_POISON',prompt)
        self.assertLessEqual(len(prompt),80000)
        subprocess.run(['git','apply','--check','-'],cwd=self.repo,input=combined,text=True,check=True,capture_output=True)
        self.assertIn('+    node.state = "Build"',combined)
        self.assertIn('node.state = "Missing"',(self.repo/'state.py').read_text())
        self.assertEqual(subprocess.check_output(['git','status','--porcelain'],cwd=self.repo,text=True),'')

    def test_test_edits_rejected_and_last_patch_preserved(self):
        response=self.reply([{'file_path':'tests/test_state.py','before':'assert state == "Build"','after':'assert True'}])
        with patch('generator.llm_interface.call_llm_with_usage',return_value=response):
            with self.assertRaisesRegex(ValueError,'production edits'):
                repair_once(self.instance,self.repo,self.patch,{},'failed',watch_config(self.repo,self.instance,self.patch),self.root/'bad')
        self.assertFalse((self.root/'bad/combined.patch').exists())
        self.assertIn('Missing',(self.repo/'state.py').read_text())

    def test_watch_includes_generated_function_not_present_in_base(self):
        patch=self.patch+'\n+def new_producer(node):\n+    return node.state\n'
        self.assertIn('new_producer',watch_config(self.repo,self.instance,patch)['watch']['state.py'])

    def test_guard_snapshot_and_later_write_keep_order_and_identity(self):
        events=[{'seq':1,'test':'failed','process':1,'frame':'f','function':'select','event':'line',
                 'file':'state.py','line':1,'statement':'if node.state == "Invalid":','values':{'node':{'receiver':'o1','state':'Missing'}}},
                {'seq':2,'test':'failed','process':1,'frame':'f','function':'select','event':'return',
                 'file':'state.py','line':2,'statement':'return node','values':{'node':{'receiver':'o1','state':'Invalid'}}}]
        packet=extract_packet(events,{'tests_status':{'FAIL_TO_PASS':{'failure':['failed']}}})
        self.assertEqual(packet['line_timing'],'before_execution')
        self.assertEqual(packet['consumer_guards'][0]['values']['node']['state'],'Missing')
        self.assertEqual(packet['state_transitions'][0]['changes']['node']['after']['state'],'Invalid')

    def test_processes_do_not_share_object_or_frame_identity(self):
        events=[{'seq':1,'test':'failed','process':p,'frame':'f','function':'select','event':'return',
                 'values':{'node':state}} for p,state in [(1,'Missing'),(2,'Invalid')]]
        self.assertEqual(extract_packet(events)['state_transitions'],[])

    def test_branch_outcomes_and_watch_coverage_distinguish_known_from_unknown(self):
        events=[
            {'seq':1,'test':'failed','process':1,'frame':'f1','function':'select','event':'call',
             'file':'state.py','line':1,'statement':'def select(node):','values':{}},
            {'seq':2,'test':'failed','process':1,'frame':'f1','function':'select','event':'line',
             'file':'state.py','line':3,'statement':'if node.state == "Invalid":','values':{},
             'branch':{'condition':'node.state == "Invalid"','guard_line':3,'test_end':3,'body_start':4,
                       'body_end':5,'body_calls':['fallback'],'outcome':False,'next_line':7,
                       'basis':'next_executed_line_in_same_frame',
                       'values_before':{'node':{'state':'Missing'}},'state_before':None}},
        ]
        watch={'watch':{'state.py':['select','unreached_helper']}}
        report={'tests_status':{'FAIL_TO_PASS':{'failure':['failed']}}}
        complete={'trace_status':'observed','observer_summaries':[
            {'process':1,'truncated':False,'truncated_tests':[],'observer_errors':0}]}
        packet=extract_packet(events,report,complete,watch)
        self.assertEqual(packet['trace_coverage']['status'],'complete')
        outcome=packet['branch_outcomes'][0]
        self.assertFalse(outcome['entered_body'])
        self.assertEqual(outcome['condition'],'node.state == "Invalid"')
        self.assertEqual(outcome['values_before']['node']['state'],'Missing')
        self.assertEqual(outcome['body_calls'],['fallback'])
        coverage={record['function']:record for record in packet['watched_call_coverage']}
        self.assertEqual(coverage['select']['called_in_failed_tests'],['failed'])
        self.assertEqual(coverage['unreached_helper']['not_called'][0]['status'],'not_called_confirmed')
        partial=dict(complete,observer_summaries=[
            {'process':1,'truncated':True,'truncated_tests':['failed'],'observer_errors':0}])
        packet=extract_packet(events,report,partial,watch)
        self.assertEqual(packet['trace_coverage']['status'],'partial')
        coverage={record['function']:record for record in packet['watched_call_coverage']}
        self.assertEqual(coverage['unreached_helper']['not_called'][0]['status'],'not_observed_unknown')

    def test_branch_and_coverage_categories_reach_the_evidence_bundle(self):
        events=[{'seq':2,'test':'failed','process':1,'frame':'f1','function':'select','event':'line',
                 'file':'state.py','line':3,'statement':'if node.state == "Invalid":','values':{},
                 'branch':{'condition':'node.state == "Invalid"','guard_line':3,'test_end':3,'body_start':4,
                           'body_end':5,'body_calls':['fallback'],'outcome':False,'next_line':7,
                           'basis':'next_executed_line_in_same_frame','values_before':{},'state_before':None}}]
        watch={'watch':{'state.py':['unreached_helper']}}
        packet=extract_packet(events,{'tests_status':{'FAIL_TO_PASS':{'failure':['failed']}}},
                              {'trace_status':'observed'},watch)
        bundle=evidence_bundle(packet,'E   AssertionError: expected state')
        counts=bundle['categories']
        self.assertGreaterEqual(counts['branch_outcomes']['selected'],1)
        self.assertGreaterEqual(counts['trace_coverage']['selected'],1)
        self.assertGreaterEqual(counts['watched_call_coverage']['selected'],1)
        selected={observation['category'] for observation in bundle['observations']}
        self.assertIn('branch_outcomes',selected)
        self.assertTrue(any('not_called_confirmed' in json.dumps(o) for o in bundle['observations']))

    def test_probe_records_real_python_transition_without_altering_result(self):
        config=self.root/'watch.json'
        config.write_text(json.dumps({'watch':{'state.py':['select']},'repo':'test/repo'}))
        output=self.root/'trace';output.mkdir()
        module=Path(__file__).resolve().parent.parent/'generator/runtime_probe.py'
        program='''import importlib.util,os,linecache\nos.environ['FEA_STATE_CONFIG']=CONFIG\nos.environ['FEA_STATE_OUTPUT']=OUTPUT\nspec=importlib.util.spec_from_file_location('observer',MODULE)\np=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)\ncode='def select(node):\\n    node.state = "Missing"\\n    if node.state == "Invalid":\\n        return "bad"\\n    node.state = "Invalid"\\n    return node.state\\n'\nlinecache.cache['/testbed/state.py']=(len(code),None,code.splitlines(True),'/testbed/state.py')\np.install();p._test='failed'\nns={};exec(compile(code,'/testbed/state.py','exec'),ns)\nclass Node: pass\nassert ns['select'](Node())=='Invalid'\np.finish()\n'''
        program='CONFIG='+repr(str(config))+'\nOUTPUT='+repr(str(output))+'\nMODULE='+repr(str(module))+'\n'+program
        subprocess.run(['python3','-c',program],check=True,capture_output=True,text=True)
        events=[json.loads(line) for f in output.glob('*.jsonl') for line in f.read_text().splitlines()]
        guard=next(e for e in events if e['statement'].startswith('if node.state'))
        self.assertEqual(guard['values']['node']['fields']['state'],'Missing')
        self.assertEqual(events[-1]['return_value'],'Invalid')
        self.assertEqual(json.loads(next(output.glob('summary-*.json')).read_text())['observer_errors'],0)

    def test_unsafe_patch_paths_rejected(self):
        with self.assertRaises(ValueError):
            patch_paths('diff --git a/../private.py b/../private.py\n')


if __name__=='__main__':
    unittest.main()
