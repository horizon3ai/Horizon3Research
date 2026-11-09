"""OpenAI Responses API runner for GPT-5.x on Bedrock Mantle.

GPT-5.5/5.4 use the OpenAI Responses API via the bedrock-mantle
endpoint, not invoke_model or chat completions.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any

from runners.base import (
    AgentResponse,
    ConversationMessage,
    ToolCall,
    clip_marks,
    split_overlap,
)
from prompt import SYSTEM_PROMPT, render_query_free_form
from loader import HoneyQuery


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            patched = text.rstrip()
            if not patched.endswith("}"):
                patched += '"}'
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass
    for m in _JSON_OBJECT.finditer(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


class OpenAIResponsesRunner:
    """Runner for GPT-5.x via bedrock-mantle Responses API."""

    name = "openai_responses"
    runner_version = "1.0.0"

    def __init__(
        self,
        model: str = "openai.gpt-5.5",
        api_key: str | None = None,
        region: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. Run `pip install openai`."
            ) from e
        region = region or os.environ.get("AWS_REGION", "us-east-2")
        key = api_key or os.environ.get("BEDROCK_API_KEY")
        if not key:
            key_file = Path(__file__).resolve().parent.parent.parent / ".bedrock-api-key"
            if key_file.exists():
                key = key_file.read_text().strip()
        if not key:
            raise RuntimeError(
                "BEDROCK_API_KEY not set. Generate one in the Bedrock console."
            )
        self._client = OpenAI(
            base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
            api_key=key,
        )
        self.model = model
        self.model_version = model
        self.max_tokens = max_tokens

    def run(self, q: HoneyQuery) -> AgentResponse:
        body = render_query_free_form(q)
        prompt = f"{SYSTEM_PROMPT}\n\n{body}"
        started_at = _now_iso()
        t0 = perf_counter()

        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_tokens,
            )
            generated = response.output_text or ""
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

        usage: dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }

        parsed = _try_parse_json(generated)
        if parsed is None:
            return AgentResponse(
                model=self.model,
                qid=q.qid,
                runner=self.name,
                runner_version=self.runner_version,
                model_version=self.model_version,
                raw_response_text=generated,
                started_at=started_at,
                finished_at=finished_at,
                latency_s=latency,
                usage=usage,
                error="json_parse_failed",
                conversation=[
                    ConversationMessage(role="system", content=SYSTEM_PROMPT),
                    ConversationMessage(role="user", content=body, ts=started_at),
                    ConversationMessage(role="assistant", content=generated, ts=finished_at),
                ],
            )

        n = len(q.lines)
        exploit, halu_e = clip_marks(list(parsed.get("exploit_marks") or []), n)
        trap, halu_t = clip_marks(list(parsed.get("trap_marks") or []), n)
        exploit, trap = split_overlap(exploit, trap)
        reasoning = parsed.get("reasoning", "") or ""

        tool_calls = [ToolCall(
            name="submit_marks",
            input={
                "exploit_marks": exploit,
                "trap_marks": trap,
                "reasoning": reasoning,
            },
            output_excerpt="",
            ts=finished_at,
        )]

        return AgentResponse(
            model=self.model,
            qid=q.qid,
            runner=self.name,
            runner_version=self.runner_version,
            model_version=self.model_version,
            exploit_marks=exploit,
            trap_marks=trap,
            reasoning=reasoning,
            raw_response_text=generated,
            tool_calls=tool_calls,
            conversation=[
                ConversationMessage(role="system", content=SYSTEM_PROMPT),
                ConversationMessage(role="user", content=body, ts=started_at),
                ConversationMessage(role="assistant", content=generated, ts=finished_at),
            ],
            started_at=started_at,
            finished_at=finished_at,
            latency_s=latency,
            usage=usage,
            error="",
            hallucinated_lines=sorted(set(halu_e + halu_t)),
        )
