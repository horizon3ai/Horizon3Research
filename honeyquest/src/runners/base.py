"""Shared runner protocol + enriched output schema.

Every runner emits ``AgentResponse`` objects. The schema is intentionally
verbose — we collect everything that might be analytically interesting
once and then never have to re-run.

Schema v1 fields:

  Identity
    model            "claude-sonnet-4-6", "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B"
    model_version    provider-reported version string (when available)
    runner           name of the python class that produced this record
    runner_version   string, bumped when a runner changes behavior
    qid              query id from the upstream Honeyquest dataset

  Marks (the answer)
    exploit_marks    ordered list of 1-indexed line numbers
    trap_marks       unordered list of 1-indexed line numbers

  Reasoning + forensic capture
    reasoning            agent's verbal justification (1-4 sentences)
    raw_response_text    concatenated text blocks from the model output
    tool_calls           list of {name, input, output_excerpt, ts}; empty
                         for single-turn structured-output runs, populated
                         when the runner gives the agent real tools
    conversation         list of {role, content} for multi-turn runs; the
                         single-turn baseline keeps this as [system, user,
                         assistant]

  Timing / cost
    started_at / finished_at    ISO 8601 timestamps
    latency_s
    usage                       {input_tokens, output_tokens,
                                 cache_read, cache_create}

  Error handling
    error            non-empty string if the call failed; mark fields stay []
    hallucinated_lines  any line numbers the model returned that didn't
                         exist in the artifact (filtered out of marks)

  Schema discipline
    schema_version   string, currently "v1"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from loader import HoneyQuery


SCHEMA_VERSION = "v1"


@dataclass
class ToolCall:
    """One tool invocation during agent reasoning (or the structured-output
    'tool call' that ends a single-turn run)."""
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output_excerpt: str = ""        # truncated for log size
    ts: str = ""                    # ISO timestamp


@dataclass
class ConversationMessage:
    """One message in the agent's conversation history."""
    role: str                       # system | user | assistant | tool
    content: str
    ts: str = ""


@dataclass
class AgentResponse:
    """Normalized agent response across all runners."""

    # Identity
    model: str
    qid: str
    runner: str = ""
    model_version: str = ""
    runner_version: str = "1.0.0"

    # Marks
    exploit_marks: list[int] = field(default_factory=list)
    trap_marks: list[int] = field(default_factory=list)

    # Reasoning + forensic capture
    reasoning: str = ""
    raw_response_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    conversation: list[ConversationMessage] = field(default_factory=list)

    # Timing / cost
    started_at: str = ""
    finished_at: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    # Error handling
    error: str = ""
    hallucinated_lines: list[int] = field(default_factory=list)

    # Schema
    schema_version: str = SCHEMA_VERSION


class RunnerProtocol(Protocol):
    """Every runner implements this single method."""

    name: str
    model: str
    model_version: str

    def run(self, q: HoneyQuery) -> AgentResponse:
        """Run the agent against one HoneyQuery, return the response."""
        ...


# ---------- shared helpers ----------

def clip_marks(marks: list[int], n_lines: int) -> tuple[list[int], list[int]]:
    """Drop out-of-range and duplicate line numbers, preserving order.

    Returns (clean_marks, hallucinated_lines) where hallucinated_lines
    is everything we threw out (for forensic analysis of model accuracy).
    """
    seen: set[int] = set()
    clean: list[int] = []
    hallucinated: list[int] = []
    for m in marks:
        if not isinstance(m, int):
            try:
                m = int(m)
            except (TypeError, ValueError):
                hallucinated.append(m if isinstance(m, int) else -1)
                continue
        if 1 <= m <= n_lines and m not in seen:
            seen.add(m)
            clean.append(m)
        elif m not in seen:
            hallucinated.append(m)
            seen.add(m)
    return clean, hallucinated


def split_overlap(exploit: list[int], trap: list[int]) -> tuple[list[int], list[int]]:
    """Per paper rules a line cannot be both exploit and trap.
    If a model puts the same number in both, drop it from trap."""
    trap_clean = [m for m in trap if m not in set(exploit)]
    return exploit, trap_clean
