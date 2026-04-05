"""LLM interface - abstract over OpenAI and Anthropic APIs."""
import os
import json
import time
from typing import Optional, Dict, Any, List
from config import (
    LLM_PROVIDER, LLM_MODEL_UNDERSTAND, LLM_MODEL_PATTERN,
    LLM_MODEL_GENERATE, LLM_MAX_TOKENS, LLM_TEMPERATURE,
)


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
    provider = os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
    model = model or LLM_MODEL_GENERATE
    max_tokens = max_tokens or LLM_MAX_TOKENS
    temperature = temperature if temperature is not None else LLM_TEMPERATURE

    for attempt in range(retries + 1):
        try:
            if provider == "anthropic":
                return _call_anthropic(prompt, system, model, max_tokens, temperature)
            else:
                return _call_openai(prompt, system, model, max_tokens, temperature, json_mode)
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM call failed after {retries+1} attempts: {e}")


def _call_openai(prompt: str, system: str, model: str,
                 max_tokens: int, temperature: float, json_mode: bool) -> str:
    """Call OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
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
    return response.choices[0].message.content


def _call_anthropic(prompt: str, system: str, model: str,
                    max_tokens: int, temperature: float) -> str:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
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
    return response.content[0].text
