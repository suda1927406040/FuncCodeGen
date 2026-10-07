import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from generator import llm_interface as api


class LLMUsageTests(unittest.TestCase):
    def test_openai_usage_json_mode_and_sdk_retry_configuration(self):
        recorded = {}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'), finish_reason='stop')],
                                   usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110,
                                          'prompt_cache_hit_tokens': 80}, _request_id='request-1')
        def factory(**kwargs):
            recorded['client'] = kwargs
            def create(**params): recorded['params'] = params; return response
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=factory)}):
            reply = api._call_openai('prompt', 'json', 'model', 200, 0, True)
        self.assertEqual(reply.total_tokens, 110)
        self.assertEqual(reply.usage['prompt_cache_hit_tokens'], 80)
        self.assertEqual(recorded['client']['max_retries'], 0)
        self.assertEqual(recorded['params']['response_format'], {'type': 'json_object'})
        self.assertNotIn('text', reply.metrics())

    def test_anthropic_cache_accounting_and_text_blocks(self):
        response = SimpleNamespace(content=[SimpleNamespace(type='text', text='one'), SimpleNamespace(type='thinking', text='private'),
                                           SimpleNamespace(type='text', text='two')], stop_reason='end_turn',
                                   usage={'input_tokens': 10, 'cache_read_input_tokens': 80, 'cache_creation_input_tokens': 20,
                                          'output_tokens': 5})
        def factory(**kwargs):
            self.assertEqual(kwargs['max_retries'], 0)
            return SimpleNamespace(messages=SimpleNamespace(create=lambda **params: response))
        with patch.dict(sys.modules, {'anthropic': SimpleNamespace(Anthropic=factory)}):
            reply = api._call_anthropic('prompt', 'system', 'model', 100, 0)
        self.assertEqual(reply.text, 'onetwo')
        self.assertEqual(reply.input_tokens, 110)
        self.assertEqual(reply.total_tokens, 115)

    def test_missing_provider_usage_remains_unknown(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='code'), finish_reason='stop')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **params: response)))
        with patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=lambda **kwargs: client)}):
            reply = api._call_openai('p', '', 'model', 100, 0, False)
        self.assertIsNone(reply.total_tokens)

    def test_transport_retry_is_counted_but_authentication_failure_is_not_retried(self):
        transient = type('APIConnectionError', (Exception,), {})('network')
        reply = api.LLMResponse('code', 'openai', 'model')
        with patch.dict('os.environ', {'LLM_PROVIDER': 'openai'}), patch.object(api, '_call_openai', side_effect=[transient, reply]), patch.object(api.time, 'sleep'):
            result = api.call_llm_with_usage('p', model='model')
        self.assertEqual(result.attempts, 2)
        auth = type('AuthenticationError', (Exception,), {})('do not log secret content')
        with patch.dict('os.environ', {'LLM_PROVIDER': 'openai'}), patch.object(api, '_call_openai', side_effect=auth) as request:
            with self.assertRaises(api.LLMCallError) as caught:
                api.call_llm_with_usage('p', model='model')
        self.assertEqual(request.call_count, 1)
        self.assertNotIn('secret content', str(caught.exception))

    def test_legacy_interface_keeps_string_return(self):
        with patch.object(api, 'call_llm_with_usage', return_value=api.LLMResponse('code', 'openai', 'model')):
            self.assertEqual(api.call_llm('p'), 'code')


if __name__ == '__main__':
    unittest.main()
