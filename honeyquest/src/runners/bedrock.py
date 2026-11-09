"""AWS Bedrock Converse runner: Qwen3 (and any other Converse-compatible model).

Single-turn structured output via a forced tool call, mirroring
``runners.anthropic.AnthropicRunner``. The tool's input schema is the
answer (``submit_marks(exploit_marks, trap_marks, reasoning)``); the
call itself is recorded as a tool_call entry in the response.

Credentials come from the standard AWS chain (env vars / ``~/.aws/credentials``
/ IAM role). Region defaults to ``$AWS_REGION`` or ``us-west-2``.
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


class BedrockRunner:
    """Runner for AWS Bedrock Converse-API models (e.g. Qwen3).

    Usage:
        runner = BedrockRunner(model="qwen.qwen3-32b-v1:0")
        resp = runner.run(query)
    """

    name = "bedrock"
    runner_version = "1.0.0"

    def __init__(
        self,
        model: str,
        region: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 not installed. Run `uv pip install boto3` or "
                "`pip install boto3` (or `uv pip install '.[bedrock]'`)."
            ) from e
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_REGION", "us-west-2"),
        )
        self.model = model
        self.model_version = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, q: HoneyQuery) -> AgentResponse:
        body = render_query(q)
        started_at = _now_iso()
        t0 = perf_counter()

        # Bedrock Converse wraps the same JSON schema we ship to Anthropic in a
        # toolSpec envelope. We force the model to call submit_marks so its
        # answer is structured, not free-form text.
        tool_spec = {
            "toolSpec": {
                "name": SUBMIT_TOOL["name"],
                "description": SUBMIT_TOOL["description"],
                "inputSchema": {"json": SUBMIT_TOOL["input_schema"]},
            }
        }

        try:
            response = self._client.converse(
                modelId=self.model,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": body}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                toolConfig={
                    "tools": [tool_spec],
                    "toolChoice": {"tool": {"name": SUBMIT_TOOL["name"]}},
                },
            )
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
                error=f"bedrock_error: {exc}",
                conversation=[
                    ConversationMessage(role="system", content=SYSTEM_PROMPT),
                    ConversationMessage(role="user", content=body),
                ],
            )

        latency = perf_counter() - t0
        finished_at = _now_iso()

        # Walk the Converse output blocks: each is {"text": ...} or
        # {"toolUse": {"name", "toolUseId", "input"}}. We expect exactly one
        # submit_marks toolUse since we forced toolChoice.
        exploit: list[int] = []
        trap: list[int] = []
        reasoning = ""
        raw_text = ""
        tool_calls: list[ToolCall] = []
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "toolUse" in block and block["toolUse"].get("name") == SUBMIT_TOOL["name"]:
                args = block["toolUse"].get("input") or {}
                exploit = list(args.get("exploit_marks") or [])
                trap = list(args.get("trap_marks") or [])
                reasoning = args.get("reasoning", "") or ""
                tool_calls.append(ToolCall(
                    name=SUBMIT_TOOL["name"],
                    input=dict(args),
                    output_excerpt="",
                    ts=finished_at,
                ))
            elif "text" in block:
                raw_text += block["text"]

        n = len(q.lines)
        exploit, halu_e = clip_marks(exploit, n)
        trap, halu_t = clip_marks(trap, n)
        exploit, trap = split_overlap(exploit, trap)

        # Bedrock returns inputTokens / outputTokens / cacheReadInputTokens /
        # cacheWriteInputTokens — remap to the v1 schema names so analyses
        # that group across providers see consistent keys.
        u = response.get("usage") or {}
        usage: dict[str, Any] = {
            "input_tokens": u.get("inputTokens"),
            "output_tokens": u.get("outputTokens"),
            "cache_read": u.get("cacheReadInputTokens"),
            "cache_create": u.get("cacheWriteInputTokens"),
        }

        assistant_content = json.dumps({
            "tool": SUBMIT_TOOL["name"],
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
