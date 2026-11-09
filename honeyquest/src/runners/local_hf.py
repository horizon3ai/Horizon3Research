"""Local HuggingFace transformers runner.

For pentest-tuned / uncensored open-weights models (WhiteRabbitNeo,
Sec-8B, etc.) that you want to evaluate side-by-side with hosted
frontier models. Greedy decoding for reproducibility. Tool use is
simulated via a JSON-output convention since most open-weights models
don't have native tool APIs.
"""

from __future__ import annotations

import gc
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
_CODE_FENCE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)```",
    re.DOTALL,
)
_EXPLOIT_ARRAY = re.compile(
    r'"exploit_marks"\s*:\s*\[([^\]]*)\]',
    re.DOTALL,
)
_TRAP_ARRAY = re.compile(
    r'"trap_marks"\s*:\s*\[([^\]]*)\]',
    re.DOTALL,
)
_REASONING_FIELD = re.compile(
    r'"reasoning"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)',  # tolerate truncation
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Unwrap ``` ... ``` markdown fences if present. Many open-weights
    models wrap structured output in fences even when told not to."""
    m = _CODE_FENCE.match(text.lstrip())
    if m:
        return m.group(1).strip()
    # Single opening fence with no close (truncation): everything after it
    if text.lstrip().startswith("```"):
        body = text.split("```", 1)[1]
        # strip the language tag on the first line if present
        if "\n" in body:
            first, rest = body.split("\n", 1)
            if first.strip().lower() in ("json", ""):
                return rest.strip()
        return body.strip()
    return text.strip()


def _try_parse_json(text: str) -> dict | None:
    """Standard parse path: strip fences, try the whole text, then try any
    well-formed JSON object inside. Returns None on failure."""
    text = _strip_code_fences(text)
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    for m in _JSON_OBJECT.finditer(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _try_parse_partial(text: str) -> dict | None:
    """Fallback for when the full JSON won't parse: pull the exploit_marks and
    trap_marks arrays out directly. Works even when the model truncated
    mid-reasoning, when reasoning is malformed, when keys are out of order, or
    when the closing brace is missing. The goal is to salvage the marks;
    reasoning is recovered best-effort and may be partial (tagged [TRUNCATED]
    when it looks cut off)."""
    text = _strip_code_fences(text)
    e_match = _EXPLOIT_ARRAY.search(text)
    t_match = _TRAP_ARRAY.search(text)
    if not e_match and not t_match:
        return None

    def _parse_int_array(s: str) -> list[int]:
        out: list[int] = []
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except (TypeError, ValueError):
                continue
        return out

    r_match = _REASONING_FIELD.search(text)
    reasoning = r_match.group(1).strip() if r_match else ""
    if reasoning and len(reasoning) >= 1900:
        # Mark partially recovered reasoning so analysts can filter it.
        reasoning = reasoning + " [TRUNCATED]"
    return {
        "exploit_marks": _parse_int_array(e_match.group(1)) if e_match else [],
        "trap_marks":    _parse_int_array(t_match.group(1)) if t_match else [],
        "reasoning": reasoning,
        "_recovery": "partial",
    }


class LocalHFRunner:
    """Wraps a HuggingFace causal-LM. Subclass and override ``model_id``
    + ``name``; or instantiate directly with explicit args."""

    name: str = "local_hf"
    runner_version: str = "1.1.0"   # bumped: added code-fence stripping + partial-array recovery
    model_id: str = ""
    max_new_tokens: int = 1024      # doubled from 512 — WhiteRabbitNeo is verbose on filesystem listings
    use_fast_tokenizer: bool = True

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "mps",
        dtype_str: str = "float16",
        max_new_tokens: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "torch / transformers not installed. Run `uv pip install "
                "torch transformers`."
            ) from e
        self._torch = torch

        # Allow caller to override via arg or env
        self.model_id = model_id or self.model_id or os.environ.get("HF_MODEL_ID", "")
        if not self.model_id:
            raise RuntimeError("LocalHFRunner needs model_id (or HF_MODEL_ID env var)")
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens

        dtype = (
            torch.float16 if dtype_str == "float16"
            else torch.bfloat16 if dtype_str == "bfloat16"
            else torch.float32
        )
        cache_dir = os.environ.get("HF_HOME")
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=self.use_fast_tokenizer,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._device = device
        # Exposed name for AgentResponse.model — defaults to the HF id
        self.model = self.model_id
        self.model_version = self.model_id

    def _build_prompt(self, q: HoneyQuery) -> str:
        body = render_query_free_form(q)
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": body},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"{SYSTEM_PROMPT}\n\n{body}\n\nJSON response:"

    def run(self, q: HoneyQuery) -> AgentResponse:
        torch = self._torch
        prompt = self._build_prompt(q)
        started_at = _now_iso()
        t0 = perf_counter()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._device)
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
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
                error=f"generation_error: {exc}",
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        finished_at = _now_iso()
        latency = perf_counter() - t0

        gc.collect()
        if self._device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        parsed = _try_parse_json(generated)
        recovery_flag = ""
        if parsed is None:
            # Fall back to partial extraction: get the arrays even if the
            # JSON object is truncated, malformed, or wrapped in markdown.
            parsed = _try_parse_partial(generated)
            if parsed is not None:
                recovery_flag = "partial"
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
                error="json_parse_failed",
                conversation=[
                    ConversationMessage(role="system", content=SYSTEM_PROMPT),
                    ConversationMessage(role="user", content=prompt, ts=started_at),
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
                ConversationMessage(role="user", content=prompt, ts=started_at),
                ConversationMessage(role="assistant", content=generated, ts=finished_at),
            ],
            started_at=started_at,
            finished_at=finished_at,
            latency_s=latency,
            error="",
            hallucinated_lines=sorted(set(halu_e + halu_t)),
            usage={"recovery": recovery_flag} if recovery_flag else {},
        )


# ---------- pre-configured local model classes ----------

class WhiteRabbitNeoRunner(LocalHFRunner):
    """WhiteRabbitNeo 2-8B (Llama-3.1 base, offensive-fine-tuned)."""
    name = "whiterabbit_neo"
    model_id = os.environ.get(
        "WHITERABBIT_MODEL_ID",
        "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B",
    )
    max_new_tokens = 512
    use_fast_tokenizer = True


class Llama31Base8BRunner(LocalHFRunner):
    """Meta Llama 3.1 8B Instruct — the base WhiteRabbitNeo is fine-tuned from.
    Pair with WhiteRabbitNeoRunner to isolate the offensive-fine-tuning effect.

    Default model id is the NousResearch ungated mirror — verbatim re-upload
    of Meta's weights with no HF gating. Override with LLAMA31_MODEL_ID if you
    have a HF_TOKEN with accepted Meta community license and want the
    meta-llama original."""
    name = "llama_3_1_8b_instruct"
    model_id = os.environ.get(
        "LLAMA31_MODEL_ID",
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    use_fast_tokenizer = True


class FoundationSec8BRunner(LocalHFRunner):
    """Cisco Foundation-Sec-8B (Llama-3.1 base, security-specialized).
    Reference: fdtn-ai/Foundation-Sec-8B on HuggingFace.
    Used as a benchmark in Prinos & Brush 2026 sim-to-real paper."""
    name = "foundation_sec_8b"
    model_id = os.environ.get(
        "FOUNDATION_SEC_MODEL_ID",
        "fdtn-ai/Foundation-Sec-8B",
    )
    use_fast_tokenizer = True
