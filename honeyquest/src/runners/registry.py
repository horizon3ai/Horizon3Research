"""Model registry — teammates extend this when they add a new model.

A registry entry maps a short CLI alias (e.g. ``opus-4-7``) to a
constructor callable that returns a runner instance. The constructor
runs only when the user selects that alias, so no API client or model
loading happens at import time.

To add a model, copy an existing entry below and give it a new alias.
"""

from __future__ import annotations

from typing import Callable

# Type for the constructor side of the registry. Each entry is a zero-arg
# callable returning something that satisfies the RunnerProtocol.
RunnerFactory = Callable[[], object]


def _claude(model_id: str, **extra) -> RunnerFactory:
    def _build():
        from runners.anthropic import AnthropicRunner
        return AnthropicRunner(model=model_id, **extra)
    return _build


def _whiterabbit() -> RunnerFactory:
    def _build():
        from runners.local_hf import WhiteRabbitNeoRunner
        return WhiteRabbitNeoRunner()
    return _build


def _qwen_bedrock(model_id: str) -> RunnerFactory:
    def _build():
        from runners.bedrock import BedrockRunner
        return BedrockRunner(model=model_id)
    return _build


def _bedrock(model_id: str, **extra) -> RunnerFactory:
    def _build():
        from runners.deepseek import DeepSeekRunner
        return DeepSeekRunner(model=model_id, **extra)
    return _build


def _gpt5(model_id: str) -> RunnerFactory:
    def _build():
        from runners.openai_responses import OpenAIResponsesRunner
        return OpenAIResponsesRunner(model=model_id)
    return _build


def _ollama(model_id: str) -> RunnerFactory:
    def _build():
        from runners.ollama import OllamaRunner
        return OllamaRunner(model=model_id)
    return _build


REGISTRY: dict[str, RunnerFactory] = {
    # Anthropic Claude models. AnthropicRunner uses the direct API when
    # ANTHROPIC_API_KEY is set, else falls back to AWS Bedrock automatically.
    "haiku-4-5":  _claude("claude-haiku-4-5-20251001"),
    "sonnet-4-6": _claude("claude-sonnet-4-6"),
    "opus-4-6":   _claude("claude-opus-4-6", temperature=None),
    "opus-4-7":   _claude("claude-opus-4-7", temperature=None),
    "opus-4-8":   _claude("claude-opus-4-8", temperature=None),
    "fable-5":    _claude("claude-fable-5", temperature=None, force_tool=False),

    # DeepSeek (Bedrock)
    "deepseek-r1":   _bedrock("us.deepseek.r1-v1:0"),
    "deepseek-v3.2": _bedrock("deepseek.v3.2"),

    # OpenAI open-weight (Bedrock)
    "gpt-oss-120b":           _bedrock("openai.gpt-oss-120b-1:0"),
    "gpt-oss-safeguard-120b": _bedrock("openai.gpt-oss-safeguard-120b"),

    # NVIDIA (Bedrock)
    "nemotron-super-120b": _bedrock("nvidia.nemotron-super-3-120b"),

    # Moonshot AI (Bedrock)
    "kimi-k2-thinking": _bedrock("moonshot.kimi-k2-thinking"),
    "kimi-k2.5": _bedrock("moonshotai.kimi-k2.5"),

    # Qwen (Bedrock)
    "qwen3-coder-480b": _bedrock("qwen.qwen3-coder-480b-a35b-v1:0"),

    # OpenAI frontier (Bedrock Mantle - Responses API)
    "gpt-5.4": _gpt5("openai.gpt-5.4"),
    "gpt-5.5": _gpt5("openai.gpt-5.5"),

    # Mistral (Bedrock)
    "devstral-2-123b": _bedrock("mistral.devstral-2-123b"),

    # Meta (Bedrock)
    "llama4-maverick-17b": _bedrock("us.meta.llama4-maverick-17b-instruct-v1:0", use_converse=True),

    # Local models via Ollama
    "foundation-sec-8b": _ollama("hf.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q8_0-GGUF:latest"),

    # Local open-weights models
    "whiterabbit-neo-8b": _whiterabbit(),

    # Teammate runners (Bedrock)
    "qwen3-32b":       _qwen_bedrock("qwen.qwen3-32b-v1:0"),
    "qwen3-235b-a22b": _qwen_bedrock("qwen.qwen3-235b-a22b-2507-v1:0"),
}


def list_models() -> list[str]:
    return sorted(REGISTRY.keys())


def build(alias: str):
    if alias not in REGISTRY:
        raise KeyError(
            f"unknown model alias '{alias}'. Available: {', '.join(list_models())}"
        )
    return REGISTRY[alias]()
