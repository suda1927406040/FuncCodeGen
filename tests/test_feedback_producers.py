import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from generator.feedback_context import expand_producers
from generator.feedback_candidates import record_candidate, candidate_view, select_candidate
from generator.state_feedback import extract_packet
from generator.state_profiles.conan import extract
from generator.feedback_edits import parse_current_edits


class ProducerFeedbackTests(unittest.TestCase):
    def test_function_replacement_preserves_binding_and_prevents_append(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'mode.py').write_text('class Mode:\n    @staticmethod\n    def allowed(value):\n        return False\n')
            edit={'file_path':'mode.py','symbol':'Mode.allowed','replacement':'def allowed(value):\n    return value'}
            entries=parse_current_edits(json.dumps({'edits':[edit]}),root)
            self.assertTrue(entries[0]['after'].startswith('    @staticmethod\n    def allowed'))
            self.assertEqual(entries[0]['symbol'], 'Mode.allowed')
            edit['replacement']+='\n\ndef allowed(value):\n    return True\n'
            # A same-name duplicate is ambiguous and still rejected...
            with self.assertRaisesRegex(ValueError,'named function exactly once'):
                parse_current_edits(json.dumps({'edits':[edit]}),root)
            # ...but appended DIFFERENT-name siblings are trimmed to the
            # named function instead of wasting the repair round.
            edit['replacement']=edit['replacement'].replace(
                '\n\ndef allowed(value):\n    return True\n', '\n\ndef _helper(value):\n    return True\n')
            entries=parse_current_edits(json.dumps({'edits':[edit]}),root)
            self.assertNotIn('_helper', entries[0]['after'])
            self.assertIn('return value', entries[0]['after'])

    def test_failure_route_uses_last_command_not_an_earlier_call(self):
        events=[{'seq':1,'test':'case','process':1,'invocation':1,'command':'old',
                 'file':'state.py','function':'helper','event':'call'},
                {'seq':2,'test':'case','process':1,'invocation':2,'command':'new',
                 'file':'state.py','function':'select','event':'line','branch':{
                     'condition':'state == Missing','guard_line':5,'outcome':False,
                     'body_calls':['helper'],'next_line':8,'state_before':{'node':{'binary':'Invalid'}}}}]
        meta={'trace_status':'observed','observer_summaries':[{'process':1,'finished':True,
              'truncated':False,'truncated_tests':[],'observer_errors':0}]}
        packet=extract_packet(events,{'tests_status':{'FAIL_TO_PASS':{'failure':['case']}}},meta,
                              {'watch':{'state.py':['helper']},'declared_interfaces':{'state.py':['helper']}})
        route=packet['failure_routes'][0]
        self.assertEqual(route['command'],'new')
        self.assertEqual(route['target_calls'][0]['status'],'not_called_confirmed')
        self.assertEqual(route['blocked_guards'][0]['condition'],'state == Missing')

    def test_retrieval_finds_unedited_producer_and_excludes_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'consumer.py').write_text('from producer import prepare\ndef select(x):\n    if x.blocked:\n        return False\n    return True\n')
            (root/'producer.py').write_text('def prepare(x):\n    x.blocked = check(x.config)\n')
            (root/'tests').mkdir()
            (root/'tests/secret.py').write_text('def gold(x):\n    x.blocked = GOLD_POISON\n')
            result = expand_producers(root, {'consumer.py':['select']}, '')
            self.assertEqual([r['file'] for r in result], ['producer.py'])
            self.assertIn('prepare', result[0]['symbols'])

    def test_complementary_candidates_are_not_reported_as_union_success(self):
        def report(passes):
            return {'resolved':False, 'patch_successfully_applied':True, 'tests_status':{
                'FAIL_TO_PASS':{'success':passes,'failure':sorted(set('abcde')-set(passes))},
                'PASS_TO_PASS':{'success':['old'],'failure':[]}}}
        with tempfile.TemporaryDirectory() as temp:
            records=record_candidate(temp,'case','patch1',report(['a','b','c']),{},1)
            records=record_candidate(temp,'case','patch2',report(['d','e']),{},2)
            view=candidate_view(temp,'case',records,'patch2')
            self.assertEqual(len(view),2)
            self.assertNotIn('earlier_patch_reference',str(view))
            chosen,patch,grade=select_candidate(temp,'case')
            self.assertEqual(patch,'patch1')
            self.assertEqual(chosen['new_passes'],['a','b','c'])
            self.assertFalse(grade['resolved'])

    def test_unobserved_test_or_missing_summary_never_confirms_absence(self):
        event={'seq':1,'test':'other','process':1,'event':'call','file':'state.py','function':'select'}
        watch={'watch':{'state.py':['helper']}}
        grade={'tests_status':{'FAIL_TO_PASS':{'failure':['failed']}}}
        for summaries in ([],[{'process':1,'finished':True,'truncated_tests':[],'truncated':False,'observer_errors':0}]):
            packet=extract_packet([event],grade,{'trace_status':'observed','observer_summaries':summaries},watch)
            self.assertEqual(packet['watched_call_coverage'][0]['not_called'][0]['status'],'not_observed_unknown')

    def test_optional_validation_is_not_borrowed_from_another_invocation(self):
        events=[{'seq':1,'test':'case','invocation':1,'event':'call','function':'Recipe.validate_build',
                 'file':'recipe.py','line':1,'state':{'current_info':{'receiver':'same','cppstd':'14'}}},
                {'seq':2,'test':'case','invocation':2,'event':'return','function':'Compatibility.compatibles',
                 'file':'compat.py','line':2,'state':{'has_build_validator':True,'returned_candidates':[
                     {'package_id':'id','info':{'receiver':'same','cppstd':'14','cant_build':False}}]}}]
        self.assertEqual(len(extract(events)['unvalidated_candidates']),1)

    def test_real_probe_records_true_false_exception_and_budget_coverage(self):
        code='''counter = 0
def predicate(value):
    global counter
    counter += 1
    if value == 'raise':
        raise ValueError('condition failed')
    return value
def select(value):
    if predicate(value):
        return 'yes'
    return 'no'
'''
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=root/'watch.json'
            config.write_text(json.dumps({'watch':{'state.py':['select']},'event_limit_per_test':20}))
            probe=Path(__file__).resolve().parent.parent/'generator/runtime_probe.py'
            program='''import importlib.util,os,linecache
os.environ['FEA_STATE_CONFIG']=config
os.environ['FEA_STATE_OUTPUT']=output
spec=importlib.util.spec_from_file_location('probe',probe)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
linecache.cache['/testbed/state.py']=(len(code),None,code.splitlines(True),'/testbed/state.py')
ns={};exec(compile(code,'/testbed/state.py','exec'),ns)
p.install()
for label,value in [('true',True),('false',False),('exception','raise')]:
    p._test=label
    try: ns['select'](value)
    except ValueError: pass
p._test='limited'
p._config['event_limit_per_test']=1
ns['select'](True)
p.finish()
assert ns['counter']==4
'''
            bindings={'config':str(config),'output':str(root),'probe':str(probe),'code':code}
            program='\n'.join(k+'='+repr(v) for k,v in bindings.items())+'\n'+program
            subprocess.run([sys.executable,'-c',program],check=True,capture_output=True,text=True)
            events=[json.loads(line) for f in root.glob('trace-*.jsonl') for line in f.read_text().splitlines()]
            by_test={e['test']:e['branch']['outcome'] for e in events if e.get('branch')}
            self.assertEqual(by_test,{'true':True,'false':False})
            summary=json.loads(next(root.glob('summary-*.json')).read_text())
            self.assertEqual(summary['truncated_tests'],['limited'])
            self.assertEqual(summary['observer_errors'],0)
            packet=extract_packet(events,{'tests_status':{'FAIL_TO_PASS':{'failure':['true','false']}}},
                                  {'trace_status':'observed','observer_summaries':[summary]})
            self.assertEqual(len(packet['branch_outcomes']),2)


if __name__=='__main__':
    unittest.main()
