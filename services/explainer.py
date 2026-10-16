"""Optional plain-English narrative of a setup via the Claude API.

The strategy engine decides everything. Claude only REPHRASES the
deterministic explanation produced by :func:`core.explain.explain_setup`:

* the prompt contains only engine-computed facts;
* the model is told not to add analysis, advice or numbers;
* the reply is rejected if it contains any significant number that does not
  appear in those facts (a guard against invented prices).

Requires ``pip install anthropic`` and an ``ANTHROPIC_API_KEY`` (or another
credential source the SDK resolves). Server-side refusal fallbacks are enabled
by default (``fallbacks="default"``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_MODEL = "claude-opus-5-5"
SYSTEM = (
    "You rewrite a futures break-and-retest setup summary into 3-5 plain-English sentences for an experienced "
    "discretionary trader. Use ONLY the facts provided. Do not add analysis, opinions, predictions, probabilities "
    "or trading advice, and do not introduce any number, price or percentage that is not in the facts. Keep the "
    "statement that all levels are advisory and that no orders are placed."
)
_NUM = re.compile(r"\d[\d,]*\.?\d*")


@dataclass
class Narrative:
    ok: bool
    text: str
    detail: str


def available() -> tuple[bool, str]:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "The 'anthropic' package is not installed (pip install anthropic)."
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False, "No ANTHROPIC_API_KEY set - the deterministic explanation above is complete without it."
    return True, "Claude narrative available."


def _significant_numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        if not n:
            continue
        if "." in n or len(n.split(".")[0]) >= 3:  # prices, scores with decimals, 3+ digit values
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def unsupported_numbers(reply: str, facts: str) -> set[str]:
    """Numbers in the reply that do not occur in the facts."""
    return _significant_numbers(reply) - _significant_numbers(facts)


def narrate(facts: str, model: str | None = None) -> Narrative:  # pragma: no cover - requires network
    ok, why = available()
    if not ok:
        return Narrative(False, "", why)
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=model or os.environ.get("BRC_EXPLAINER_MODEL") or DEFAULT_MODEL,
            max_tokens=4000,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Facts:\n{facts}"}],
        )
    except anthropic.AuthenticationError:
        return Narrative(False, "", "Claude API authentication failed - check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        return Narrative(False, "", "Claude API rate limit reached - try again shortly.")
    except anthropic.APIStatusError as exc:
        return Narrative(False, "", f"Claude API error {exc.status_code}.")
    except anthropic.APIConnectionError:
        return Narrative(False, "", "Could not reach the Claude API (network).")
    if response.stop_reason == "refusal":
        return Narrative(False, "", "The model declined to produce a narrative.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    bad = unsupported_numbers(text, facts)
    if bad:
        return Narrative(
            False,
            "",
            "Narrative discarded: it contained numbers not produced by the engine "
            f"({', '.join(sorted(bad))}).",
        )
    return Narrative(True, text, f"Rephrased by {response.model}; every number comes from the engine.")
