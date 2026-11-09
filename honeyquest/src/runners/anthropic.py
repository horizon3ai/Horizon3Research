"""Anthropic Claude runner: Haiku, Sonnet, Opus — direct API or Bedrock.

Single-turn structured output via a forced tool call. The tool's input
schema is the answer (``submit_marks(exploit_marks, trap_marks, reasoning)``);
the call itself is recorded as a tool_call entry in the response.

Transport is chosen automatically from the environment:
  - ``ANTHROPIC_API_KEY`` set  → direct Anthropic API (``anthropic.Anthropic``)
  - otherwise                  → AWS Bedrock (``anthropic.AnthropicBedrock``),
    with the registry model id mapped to its US inference-profile id (see
    ``_BEDROCK_PROFILE``). Credentials resolve via botocore's default chain
    (``AWS_PROFILE`` / SSO / env), so nothing is read from the repo.

Both transports expose the identical ``messages.create()`` surface, so the
request/parse code below is shared — only the client and the invoked model
id differ.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from runners.base import (
    AgentResponse,
    ConversationMessage,
    ToolCall,
    clip_marks,
    split_overlap,
)
from prompt import SUBMIT_TOOL, SYSTEM_PROMPT, render_query
from loader import HoneyQuery


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Registry model id → Bedrock US inference-profile id. Bedrock rejects the
# bare foundation id for these models ("needs an inference profile"), so we
# map to the cross-region profile. Add a row here to enable a model on the
# Bedrock transport. (Direct-API users never touch this table.)
_BEDROCK_PROFILE: dict[str, str] = {
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


class AnthropicRunner:
    """Runner for Anthropic-hosted Claude models (Haiku, Sonnet, Opus).

    Usage:
        runner = AnthropicRunner(model="claude-sonnet-4-6")
        resp = runner.run(query)
    """

    name = "anthropic"
    # 1.1.0: added Bedrock transport (AnthropicBedrock) as a fallback when no
    # ANTHROPIC_API_KEY is present. Same prompt/parse path; different client.
    runner_version = "1.1.0"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        force_tool: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `uv pip install "
                "'anthropic[bedrock]'` or `pip install 'anthropic[bedrock]'`."
            ) from e

        # Record the registry model id in the output (AgentResponse.model)
        # regardless of transport, so analysis grouping stays comparable across
        # direct-API and Bedrock runs. ``_invoke_model`` is what we actually send.
        self.model = model
        self.model_version = model      # Anthropic doesn't expose a separate version field

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            self._client = anthropic.Anthropic(api_key=key)
            self._invoke_model = model
        else:
            profile = _BEDROCK_PROFILE.get(model)
            if profile is None:
                raise RuntimeError(
                    f"ANTHROPIC_API_KEY not set and no Bedrock inference profile "
                    f"mapped for model '{model}'. Either export ANTHROPIC_API_KEY "
                    f"(direct API) or add '{model}' to _BEDROCK_PROFILE. "
                    f"Bedrock-ready models: {', '.join(sorted(_BEDROCK_PROFILE))}."
                )
            region = os.environ.get("AWS_REGION") or "us-east-1"
            # Credentials resolve via botocore's default chain (AWS_PROFILE /
            # SSO / env). No keys are read from the repo.
            self._client = anthropic.AnthropicBedrock(aws_region=region)
            self._invoke_model = profile

        self.max_tokens = max_tokens
        self.force_tool = force_tool
        self.temperature = temperature

    def run(self, q: HoneyQuery) -> AgentResponse:
        body = render_query(q)
        started_at = _now_iso()
        t0 = perf_counter()

        try:
            kwargs: dict[str, Any] = dict(
                model=self._invoke_model,
                max_tokens=self.max_tokens,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[SUBMIT_TOOL],
                tool_choice={"type": "tool", "name": "submit_marks"} if self.force_tool else {"type": "auto"},
                messages=[{"role": "user", "content": body}],
            )
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            return AgentResponse(
                model=self.model,
                qid=q.qid,
                runner=self.name,
                runner_version=self.runner_version,
                model_version=self.model_version,
                started_at=started_at,
                finished_at=_now_iso(),
                latency_s=perf_counter() - t0,
                error=f"api_error: {exc}",
                conversation=[
                    ConversationMessage(role="system", content=SYSTEM_PROMPT),
                    ConversationMessage(role="user", content=body),
                ],
            )
        latency = perf_counter() - t0
        finished_at = _now_iso()

        # Extract the tool_use block (our forced submit_marks call)
        exploit: list[int] = []
        trap: list[int] = []
        reasoning = ""
        raw_text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "tool_use" and block.name == "submit_marks":
                args = block.input or {}
                exploit = list(args.get("exploit_marks") or [])
                trap = list(args.get("trap_marks") or [])
                reasoning = args.get("reasoning", "") or ""
                tool_calls.append(ToolCall(
                    name="submit_marks",
                    input=dict(args),
                    output_excerpt="",   # no follow-up — this ends the turn
                    ts=finished_at,
                ))
            elif btype == "text":
                raw_text += block.text

        n = len(q.lines)
        exploit, halu_e = clip_marks(exploit, n)
        trap, halu_t = clip_marks(trap, n)
        exploit, trap = split_overlap(exploit, trap)

        # Provider usage
        usage_obj = getattr(response, "usage", None)
        usage: dict[str, Any] = {}
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
                "cache_read": getattr(usage_obj, "cache_read_input_tokens", None),
                "cache_create": getattr(usage_obj, "cache_creation_input_tokens", None),
            }

        # Reconstruct the conversation for the record. We only sent one user
        # message + the model's tool_use reply; we serialize the tool call as
        # the assistant content for readability.
        assistant_content = json.dumps({
            "tool": "submit_marks",
            "args": tool_calls[0].input if tool_calls else {},
            "text_blocks": raw_text,
        })

        return AgentResponse(
            model=self.model,
            qid=q.qid,
            runner=self.name,
            runner_version=self.runner_version,
            model_version=self.model_version,
            exploit_marks=exploit,
            trap_marks=trap,
            reasoning=reasoning,
            raw_response_text=raw_text,
            tool_calls=tool_calls,
            conversation=[
                ConversationMessage(role="system", content=SYSTEM_PROMPT),
                ConversationMessage(role="user", content=body, ts=started_at),
                ConversationMessage(role="assistant", content=assistant_content, ts=finished_at),
            ],
            started_at=started_at,
            finished_at=finished_at,
            latency_s=latency,
            usage=usage,
            error="",
            hallucinated_lines=sorted(set(halu_e + halu_t)),
        )
