"""LLM interface - abstract over OpenAI and Anthropic APIs."""
import os
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Dict, Any, List
from config import (
    LLM_PROVIDER, LLM_MODEL_UNDERSTAND, LLM_MODEL_PATTERN,
    LLM_MODEL_GENERATE, LLM_MAX_TOKENS, LLM_TEMPERATURE,
)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    seconds: float = 0.0
    request_id: Optional[str] = None

    def metrics(self):
        return {key: value for key, value in asdict(self).items() if key != "text"}


class LLMCallError(RuntimeError):
    def __init__(self, provider, model, attempts, seconds, cause):
        self.metrics = {"provider": provider, "model": model, "attempts": attempts,
                        "seconds": seconds, "input_tokens": None, "output_tokens": None,
                        "total_tokens": None, "error_type": type(cause).__name__}
        super().__init__("LLM call failed after %d attempt(s): %s" % (attempts, type(cause).__name__))


def call_llm(prompt: str, system: str = "", model: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: Optional[float] = None,
             json_mode: bool = False, retries: int = 2) -> str:
    """Call LLM with retry logic.

    Args:
        prompt: User message
        system: System message
        model: Model to use (defaults to LLM_MODEL_GENERATE)
        max_tokens: Max output tokens
        temperature: Sampling temperature
        json_mode: Whether to request JSON output
        retries: Number of retries on failure

    Returns:
        LLM response text
    """
    return call_llm_with_usage(prompt, system, model, max_tokens, temperature, json_mode, retries).text


def call_llm_with_usage(prompt: str, system: str = "", model: Optional[str] = None,
                        max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                        json_mode: bool = False, retries: int = 2) -> LLMResponse:
    """Return provider usage; retries cover transport/server errors, never bad code."""
    provider = os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
    model = model or LLM_MODEL_GENERATE
    max_tokens = max_tokens or LLM_MAX_TOKENS
    temperature = temperature if temperature is not None else LLM_TEMPERATURE

    started = time.monotonic()
    for attempt in range(retries + 1):
        try:
            if provider == "anthropic":
                reply = _call_anthropic(prompt, system, model, max_tokens, temperature)
            else:
                reply = _call_openai(prompt, system, model, max_tokens, temperature, json_mode)
            reply.attempts = attempt + 1
            reply.seconds = round(time.monotonic() - started, 6)
            return reply
        except Exception as e:
            status = getattr(e, "status_code", None)
            transient = type(e).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
            transient = transient or (isinstance(status, int) and (status in {408, 409, 429} or status >= 500))
            if transient and attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise LLMCallError(provider, model, attempt + 1, round(time.monotonic() - started, 6), e) from e


def _usage_dict(usage):
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return dict(usage) if isinstance(usage, dict) else vars(usage)


def _call_openai(prompt: str, system: str, model: str,
                 max_tokens: int, temperature: float, json_mode: bool) -> LLMResponse:
    """Call OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        max_retries=0,  # All request retries are counted by our wrapper.
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    usage = _usage_dict(getattr(response, "usage", None))
    return LLMResponse(choice.message.content or "", "openai", model,
                       getattr(choice, "finish_reason", None), usage.get("prompt_tokens"),
                       usage.get("completion_tokens"), usage.get("total_tokens"), usage,
                       request_id=getattr(response, "_request_id", None))


def _call_anthropic(prompt: str, system: str, model: str,
                    max_tokens: int, temperature: float) -> LLMResponse:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=0,
    )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    usage = _usage_dict(getattr(response, "usage", None))
    inputs = usage.get("input_tokens")
    if inputs is not None:
        inputs += (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
    outputs = usage.get("output_tokens")
    total = inputs + outputs if inputs is not None and outputs is not None else None
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return LLMResponse(text, "anthropic", model, getattr(response, "stop_reason", None),
                       inputs, outputs, total, usage, request_id=getattr(response, "_request_id", None))
