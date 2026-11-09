"""Generic AWS Bedrock runner for non-Anthropic hosted models.

Despite the file name, this runner handles every OpenAI-compatible Bedrock
model in the registry (DeepSeek, GPT-OSS, Nemotron, Kimi, Qwen3-Coder,
Devstral, Llama-4 Maverick). It calls Bedrock's ``invoke_model`` API by
default, or the Converse API when ``use_converse=True``. The agent answer
comes back as free-form JSON, parsed by ``_try_parse_json``.

Region defaults to ``$AWS_REGION`` or ``us-east-2``. Auth via botocore's
default credential chain (AWS_PROFILE / SSO / env).
"""

from __future__ import annotations

import json
import os
import re
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


class DeepSeekRunner:
    """Bedrock runner for OpenAI-compatible hosted models (DeepSeek-R1 by
    default). Set ``use_converse=True`` for models served via the Converse API."""

    name = "bedrock_invoke"
    runner_version = "1.2.0"

    def __init__(
        self,
        model: str = "us.deepseek.r1-v1:0",
        region: str | None = None,
        max_tokens: int = 4096,
        use_converse: bool = False,
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 not installed. Run `uv pip install boto3` "
                "or `pip install boto3`."
            ) from e
        self._region = region or os.environ.get("AWS_REGION", "us-east-2")
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self._region,
        )
        self.model = model
        self.model_version = model
        self._model_id = model
        self.max_tokens = max_tokens
        self._use_converse = use_converse

    def _call_invoke(self, body: str) -> tuple[str, str, dict[str, Any]]:
        request_body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
            "max_tokens": self.max_tokens,
        }
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(request_body),
        )
        rb = json.loads(response["body"].read())
        generated = ""
        reasoning_content = ""
        if "choices" in rb and rb["choices"]:
            msg = rb["choices"][0].get("message", {})
            generated = msg.get("content", "") or ""
            reasoning_content = msg.get("reasoning_content", "") or ""
        usage: dict[str, Any] = {}
        usage_obj = rb.get("usage", {})
        if usage_obj:
            usage = {
                "input_tokens": usage_obj.get("prompt_tokens"),
                "output_tokens": usage_obj.get("completion_tokens"),
            }
        return generated, reasoning_content, usage

    def _call_converse(self, body: str) -> tuple[str, str, dict[str, Any]]:
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": body}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        generated = ""
        msg = response.get("output", {}).get("message", {})
        for block in msg.get("content", []):
            if "text" in block:
                generated += block["text"]
        usage: dict[str, Any] = {}
        usage_obj = response.get("usage", {})
        if usage_obj:
            usage = {
                "input_tokens": usage_obj.get("inputTokens"),
                "output_tokens": usage_obj.get("outputTokens"),
            }
        return generated, "", usage

    def run(self, q: HoneyQuery) -> AgentResponse:
        body = render_query_free_form(q)
        started_at = _now_iso()
        t0 = perf_counter()

        try:
            if self._use_converse:
                generated, reasoning_content, usage = self._call_converse(body)
            else:
                generated, reasoning_content, usage = self._call_invoke(body)
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

        full_response = generated
        if reasoning_content:
            full_response = f"<think>\n{reasoning_content}\n</think>\n\n{generated}"

        return AgentResponse(
            model=self.model,
            qid=q.qid,
            runner=self.name,
            runner_version=self.runner_version,
            model_version=self.model_version,
            exploit_marks=exploit,
            trap_marks=trap,
            reasoning=reasoning,
            raw_response_text=full_response,
            tool_calls=tool_calls,
            conversation=[
                ConversationMessage(role="system", content=SYSTEM_PROMPT),
                ConversationMessage(role="user", content=body, ts=started_at),
                ConversationMessage(role="assistant", content=full_response, ts=finished_at),
            ],
            started_at=started_at,
            finished_at=finished_at,
            latency_s=latency,
            usage=usage,
            error="",
            hallucinated_lines=sorted(set(halu_e + halu_t)),
        )
