import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
import state_feedback_transport as transport


class ServerTransportTests(unittest.TestCase):
    def test_model_proxy_and_credentials_are_excluded_from_offline_process(self):
        environment=transport.offline_process_env('/bundle',{
            'HTTP_PROXY':'http://127.0.0.1:10805','https_proxy':'http://127.0.0.1:10805',
            'OPENAI_API_KEY':'secret','DEEPSEEK_API_KEY':'other','PATH':'/bin'})
        self.assertNotIn('HTTP_PROXY',environment)
        self.assertNotIn('https_proxy',environment)
        self.assertNotIn('OPENAI_API_KEY',environment)
        self.assertNotIn('DEEPSEEK_API_KEY',environment)
        self.assertEqual(environment['NO_PROXY'],'*')
        self.assertEqual(environment['PIP_NO_INDEX'],'1')

    def test_completed_evaluation_is_reused_without_network_or_process_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=SimpleNamespace(run_root=root/'run',bundle=str(root/'bundle'),workers=4)
            name='run_evaluate_0';workspace=root/'bundle/state-feedback/run'
            result=workspace/'results'/name;result.mkdir(parents=True)
            (workspace/(name+'.tar.gz')).write_bytes(b'completed')
            with patch.object(transport.subprocess,'Popen') as launch:
                actual=transport.same_host_stage(args,root/'source','evaluate',0,['case'],None,None)
            self.assertEqual(actual,result)
            launch.assert_not_called()

    def test_in_progress_evaluation_is_adopted_and_not_restarted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=SimpleNamespace(run_root=root/'run',bundle=str(root/'bundle'),workers=4)
            name='run_evaluate_0';workspace=root/'bundle/state-feedback/run'
            result=workspace/'results'/name;result.mkdir(parents=True)
            marker=root/'run/rounds/0/evaluate/remote_pid.json';marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({'pid':1234,'run_id':name}))
            def complete(_): (workspace/(name+'.tar.gz')).write_bytes(b'completed')
            with patch.object(transport,'process_alive',return_value=True),patch.object(transport.time,'sleep',side_effect=complete),patch.object(transport.subprocess,'Popen') as launch:
                actual=transport.same_host_stage(args,root/'source','evaluate',0,['case'],None,None,poll_seconds=0)
            self.assertEqual(actual,result)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
