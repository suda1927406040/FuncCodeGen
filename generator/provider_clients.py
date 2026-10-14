"""Explicit per-provider chat clients.

Mirrors the official curl semantics as code:

    curl https://open.bigmodel.cn/api/paas/v4/chat/completions \\
      -H "Authorization: Bearer <key>" -d '{"model":"glm-5.3","messages":[
        {"role":"system", ...}, {"role":"user", ...}], "stream": false, "temperature": 1}'

GLMClient  -> open.bigmodel.cn, thinking always on (never sends a thinking
              parameter; recommended temperature 1), Bearer auth via the
              OpenAI-compatible SDK, non-streaming.
DeepSeekClient -> api.deepseek.com, thinking disabled by default.

Both build on the shared OpenAI-compatible transport and LLMResponse
accounting in llm_interface; env configuration (OPENAI_API_KEY /
OPENAI_BASE_URL) stays authoritative so the pipeline's env-driven calls and
these classes cannot drift apart.
"""
import os

from generator.llm_interface import LLMResponse, _provider_thinking_kwargs


class OpenAICompatibleChat:
    """Base: endpoint/key from arguments or the process environment."""

    base_url = None
    default_model = None
    default_temperature = 0.0
    default_thinking = None

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.base_url = base_url or self.base_url or os.environ.get('OPENAI_BASE_URL')
        self.model = model or os.environ.get('LLM_MODEL_GENERATE') or self.default_model

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

    def chat(self, prompt, system=None, model=None, max_tokens=16384,
             temperature=None, json_mode=False, thinking=None):
        """One non-streaming completion; mirrors the curl request shape."""
        thinking = self.default_thinking if thinking is None else thinking
        temperature = self.default_temperature if temperature is None else temperature
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        kwargs = {'model': model or self.model, 'messages': messages,
                  'max_tokens': max_tokens, 'temperature': temperature}
        if json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        kwargs.update(_provider_thinking_kwargs(self.base_url, thinking or ''))
        response = self._client().chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, 'usage', None)
        usage = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage or {})
        return LLMResponse(choice.message.content or '', 'openai', kwargs['model'],
                            getattr(choice, 'finish_reason', None),
                            usage.get('prompt_tokens'), usage.get('completion_tokens'),
                            usage.get('total_tokens'), usage)


class GLMClient(OpenAICompatibleChat):
    """Zhipu bigmodel.cn endpoint. The model always thinks: a thinking
    parameter is never sent (disabled is rejected with error 1210), and the
    documented recommended sampling is temperature 1."""

    base_url = 'https://open.bigmodel.cn/api/paas/v4'
    default_model = 'glm-5.3-flash'
    default_temperature = 1.0
    default_thinking = None


class GLMProClient(GLMClient):
    default_model = 'glm-5.3'


class DeepSeekClient(OpenAICompatibleChat):
    """DeepSeek endpoint with thinking disabled for deterministic repairs."""

    base_url = 'https://api.deepseek.com'
    default_model = 'deepseek-flash'
    default_temperature = 0.0
    default_thinking = 'disabled'
