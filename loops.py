"""Pure feedback loop utilities.

Each function is deterministic and side-effect free.
"""

import re
from collections import OrderedDict
from typing import List


def negative_feedback_chain(signals: List[str], steps: int) -> List[str]:
    """Amplify adverse narratives into a compounding chain.

    The chain elements describe how bearish perception worsens fundamentals.
    """
    if not signals:
        return []
    steps = max(0, steps)
    chain: List[str] = []
    for i in range(steps):
        signal = signals[i % len(signals)]
        chain.append(
            f"Step {i + 1}: {signal} → investor flight → funding stress → liquidity spiral"
        )
    return chain


def positive_feedback_chain(signals: List[str], steps: int) -> List[str]:
    """Amplify constructive narratives into a compounding chain.

    The chain elements describe how bullish perception improves fundamentals.
    """
    if not signals:
        return []
    steps = max(0, steps)
    chain: List[str] = []
    for i in range(steps):
        signal = signals[i % len(signals)]
        chain.append(
            f"Step {i + 1}: {signal} → squeeze/relief rally → easier credit → perceived resilience"
        )
    return chain


def _first_clause(s: str, limit: int = 90) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"^(NEG|POS)\s*:\s*", "", s, flags=re.I)
    s = re.split(r"[.;!?]", s, maxsplit=1)[0]
    return (s[:limit] + "…") if len(s) > limit else s


def neutralization_bridge(neg_summary: str, pos_summary: str) -> str:
    """
    Single-sentence collider summary; avoids quoting either side verbatim.
    <~35 words to stay concise.
    """
    return (
        "Bearish stress meets bullish rescue; credit conditions and timing frictions dampen "
        "both loops, yielding a muted, path-dependent outcome."
    )


def neutralization_bridge_compare(neg_summary: str, pos_summary: str) -> str:
    """Optional: comparative style that references first clauses (not used by default)."""
    n = _first_clause(neg_summary)
    p = _first_clause(pos_summary)
    return f"Bearish stress vs bullish rescue; outcomes muted by funding conditions ({n} vs {p})."


# Helpers
def dedupe_keep_order(items: List[str]) -> List[str]:
    """De-duplicate while preserving first-seen order."""
    return list(OrderedDict.fromkeys(i.strip() for i in items if i and i.strip()))


def blend_paths(
    neg_path: List[float],
    pos_path: List[float],
    w_neg: float,
    w_pos: float,
    damp: float = 0.6,
    eps: float = 0.049,
    horizon: int | None = None,
) -> List[float]:
    """
    Confidence-weighted blend, then dampen reflexivity clash.
    All inputs are % moves (e.g., -3.0, +2.0).
    """
    base = [(w_neg * n + w_pos * p) for n, p in zip(neg_path, pos_path)]
    if horizon and horizon > 5:
        step = max(0.35, min(0.9, damp * 5 / horizon))
    else:
        step = damp
    return _clean_zero_list([x * step for x in base], eps=eps)


def _clean_zero_list(vals: List[float], eps: float = 0.049) -> List[float]:
    """Snap near-zeros to 0 and avoid -0.0 after rounding."""
    out: List[float] = []
    for v in vals:
        v = 0.0 if abs(v) <= eps else v
        v = 0.0 if v == 0 else v
        out.append(round(v, 2))
    return out


def geom_compound(seq: List[float]) -> List[float]:
    """Interpret seq as pct moves; return cumulative pct path (geometric)."""
    acc = 1.0
    out: List[float] = []
    for p in seq:
        acc *= (1.0 + p / 100.0)
        out.append(round((acc - 1.0) * 100.0, 2))
    return out


def smooth_blend(seq: List[float], horizon: int) -> List[float]:
    """Linearly smooth between start and end to avoid plateaus."""
    if horizon <= 1 or not seq:
        return seq
    start, end = float(seq[0]), float(seq[-1])
    return [round(start + (end - start) * i / (horizon - 1), 2) for i in range(horizon)]


def independent_decay(start_step: float, horizon: int, kappa: float = 0.15) -> List[float]:
    """
    Non-reflexive path: per-step % move that decays geometrically toward 0 from the initial step.
    Sign is preserved.
    """
    out: List[float] = []
    step = float(start_step)
    for _ in range(horizon):
        out.append(round(step, 2))
        step *= (1.0 - kappa)
    return out


def independent_drift(start_step: float, horizon: int, drift: float = 0.2) -> List[float]:
    """
    Non-reflexive path: simple linear drift outward from the starting value.
    Positive starts get more positive, negatives more negative. Drift is pct increment per step.
    """
    out: List[float] = []
    step = float(start_step)
    sign = 1.0 if step >= 0 else -1.0
    for _ in range(horizon):
        out.append(round(step, 2))
        step += sign * float(drift)
    return out


