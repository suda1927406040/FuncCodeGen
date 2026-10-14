"""Provider-aware credential loading shared by the runner and the baseline harness.

A credential file is JSON: {"api_key": ..., "base_url": ...} plus optional
{"model": ..., "thinking": "enabled"|"disabled"|null, "min_output_tokens": int}.
DeepSeek keys stay as-is; GLM (open.bigmodel.cn) can never disable thinking;
Qwen (DashScope compatible-mode) toggles it with enable_thinking. The provider
is derived from the base_url so callers cannot misconfigure the pair.
"""
import json
import os
from pathlib import Path

PROVIDERS = {
    'api.deepseek.com': {'model': 'deepseek-flash', 'thinking': 'disabled'},
    'open.bigmodel.cn': {'model': 'glm-5.3-flash', 'thinking': 'disabled'},  # maps to thinking.type=low
    'dashscope.aliyuncs.com': {'model': 'qwen3-coder-flash', 'thinking': 'disabled'},
}


def provider_for(base_url):
    host = (base_url or '').lower()
    for marker, config in PROVIDERS.items():
        if marker in host:
            return marker, config
    raise ValueError('Unsupported credential base_url: ' + str(base_url))


def load_credentials(path):
    """Return (env_updates, provider_marker); never leaks the key in errors."""
    config = json.loads(Path(path).read_text())
    marker, defaults = provider_for(config.get('base_url'))
    model = config.get('model') or defaults['model']
    thinking = defaults['thinking'] if config.get('thinking') is None else config['thinking']
    env = {'OPENAI_API_KEY': config['api_key'], 'OPENAI_BASE_URL': config['base_url'],
           'LLM_PROVIDER': 'openai', 'LLM_MODEL_GENERATE': model,
           'LLM_TEMPERATURE': '0', 'LLM_MAX_TOKENS': '16384'}
    env['LLM_THINKING_MODE'] = thinking or ''
    if config.get('min_output_tokens'):
        env['LLM_MIN_OUTPUT_TOKENS'] = str(config['min_output_tokens'])
    elif marker == 'open.bigmodel.cn':
        # GLM always thinks and the budget is shared: 16k starves the visible
        # answer to zero (verified: finish_reason=length with empty content).
        env['LLM_MIN_OUTPUT_TOKENS'] = '49152'
    elif marker != 'api.deepseek.com':
        env['LLM_MIN_OUTPUT_TOKENS'] = '16384'
    return env, marker
