"""Thin OpenRouter client: make one chat-completion call and return content, reasoning, logprobs, usage, latency."""
import time
import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENDPOINT = "https://openrouter.ai/api/v1/key"


class ORError(Exception):
    """Non-200 from OpenRouter, or a transport failure."""


def verify_key(api_key, timeout=30):
    """Return key-info dict if the key authenticates, else raise ORError."""
    r = requests.get(KEY_ENDPOINT, headers={"Authorization": f"Bearer {api_key}"},
                     timeout=timeout)
    if r.status_code != 200:
        raise ORError(f"key check HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("data", {})


def verify_model(api_key, model_id, timeout=60):
    """1-token ping to confirm a model id is live. Returns True or raises ORError."""
    call_model(api_key, model_id, [{"role": "user", "content": "ping"}],
               max_tokens=16, want_logprobs=False, timeout=timeout)  # gpt-5.x needs >=16
    return True


def call_model(api_key, model_id, messages, *, temperature=0.7, max_tokens=4096,
               want_logprobs=True, response_format=None, tools=None, tool_choice=None,
               provider_prefs=None, timeout=180):
    """One completion against OpenRouter. `provider_prefs` pins the backend for open models.
    Returns a dict; raises ORError on non-200/transport error."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "cve-selection-study",
    }
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = response_format
    if provider_prefs:
        body["provider"] = provider_prefs
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5

    t0 = time.time()
    try:
        resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise ORError(f"transport: {e}") from e
    latency_ms = int((time.time() - t0) * 1000)

    if resp.status_code != 200:
        raise ORError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if not data.get("choices"):
        raise ORError(f"no choices: {str(data)[:300]}")
    choice0 = data["choices"][0]
    msg = choice0.get("message", {})
    usage = data.get("usage", {}) or {}
    return {
        "content": msg.get("content"),
        "reasoning": msg.get("reasoning"),
        "tool_calls": msg.get("tool_calls"),
        "message": msg,
        "logprobs": choice0.get("logprobs"),
        "finish_reason": choice0.get("finish_reason"),
        "in_tokens": usage.get("prompt_tokens", 0),
        "out_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        "latency_ms": latency_ms,
        "provider": data.get("provider"),
        "request_id": data.get("id"),
    }
