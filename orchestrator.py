"""Scenario orchestrator combining NEG and POS agents into outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from agents import Agent, AgentConfig, LLMClient
from loops import (
    neutralization_bridge,
    neutralization_bridge_compare,
    dedupe_keep_order,
    blend_paths,
    geom_compound,
    smooth_blend,
    independent_decay,
    independent_drift,
)

SCHEMA = {
    "type": "object",
    "required": ["topic", "context", "outcomes"],
    "properties": {
        "outcomes": {
            "type": "object",
            "required": ["negative", "neutralized", "positive"],
        }
    },
}


def run_scenario(
    topic: str,
    context: str,
    client: LLMClient,
    weights: str | None = None,
    damp: float | None = None,
    eps: float | None = None,
    neutral_style: str | None = None,
    horizon: int | None = None,
    unit: str | None = None,
    path_mode: str | None = None,
    kappa: float | None = None,
    indep_mode: str | None = None,
    drift: float | None = None,
) -> Dict:
    neg_cfg = AgentConfig(
        name="NEG",
        stance="negative",
        system_preamble=(
            "You are the NEGATIVE feedback agent. Amplify adverse narratives: fraud contagion, "
            "liquidity spirals, covenant breaches, regulatory overhang. Build a self-reinforcing "
            "causal chain. Be precise and non-defamatory; flag uncertainty."
        ),
    )
    pos_cfg = AgentConfig(
        name="POS",
        stance="positive",
        system_preamble=(
            "You are the POSITIVE feedback agent. Amplify constructive narratives: flight-to-quality "
            "illusions, short-squeezes, turnaround catalysts, accounting clean-up. Build a self-"
            "reinforcing causal chain. Be precise and non-defamatory; flag uncertainty."
        ),
    )

    neg_agent = Agent(neg_cfg, client)
    pos_agent = Agent(pos_cfg, client)

    neg = neg_agent.reason(topic, context, loop_style="amplify")
    pos = pos_agent.reason(topic, context, loop_style="amplify")

    # Confidence weights (normalized)
    c_neg = max(0.0, min(1.0, float(neg.get("confidence", 0.5))))
    c_pos = max(0.0, min(1.0, float(pos.get("confidence", 0.5))))
    total = (c_neg + c_pos) or 1.0
    w_neg, w_pos = c_neg / total, c_pos / total
    if weights:
        try:
            w_neg_i, w_pos_i = [float(x) for x in weights.split(",")]
            t = (w_neg_i + w_pos_i) or 1.0
            w_neg, w_pos = w_neg_i / t, w_pos_i / t
        except Exception:
            pass

    # Thesis via bridge
    if (neutral_style or "concise").lower().startswith("compare"):
        neutral_thesis = neutralization_bridge_compare(neg.get("thesis", ""), pos.get("thesis", ""))
    else:
        neutral_thesis = neutralization_bridge(neg.get("thesis", ""), pos.get("thesis", ""))

    # Drivers/risks: prefer substance, then mix 2 from NEG + 2 from POS, cap to 3
    def _prefer_substance(items: List[str]) -> List[str]:
        return sorted(items, key=lambda x: str(x).lower().startswith("signal-"))

    def _mix(a: List[str], b: List[str], k: int = 3) -> List[str]:
        out: List[str] = []
        for x in (a + b):
            if x not in out:
                out.append(x)
            if len(out) == k:
                break
        return out

    neg_drv = _prefer_substance(neg.get("drivers", [])[:3])
    pos_drv = _prefer_substance(pos.get("drivers", [])[:3])
    drivers_neu = dedupe_keep_order(_mix(neg_drv[:2], pos_drv[:2], k=3))

    neg_rsk = _prefer_substance(neg.get("risks", [])[:3])
    pos_rsk = _prefer_substance(pos.get("risks", [])[:3])
    risks_neu = dedupe_keep_order(_mix(neg_rsk[:2], pos_rsk[:2], k=3))
    # prevent overlap & ensure fallback
    risks_neu = [r for r in risks_neu if r not in drivers_neu][:3]
    if not risks_neu:
        risks_neu = ["governance uncertainty", "liquidity squeeze", "model uncertainty"]

    # Numeric paths: ensure lists of 5 floats
    neg_path = [float(x) for x in neg.get("price_path_week", [])][:5]
    pos_path = [float(x) for x in pos.get("price_path_week", [])][:5]
    while len(neg_path) < 5:
        neg_path.append(0.0)
    while len(pos_path) < 5:
        pos_path.append(0.0)

    # Optional horizon resampling
    def _resample_path(path: List[float], points: int) -> List[float]:
        n = len(path)
        if points <= 0:
            return path
        if points == n:
            return path[:]
        if n == 1:
            return [path[0]] * points
        # Linear interpolation from indices [0..n-1] to points
        out: List[float] = []
        for i in range(points):
            posf = i * (n - 1) / (points - 1)
            j = int(posf)
            t = posf - j
            if j >= n - 1:
                out.append(path[-1])
            else:
                out.append(path[j] * (1 - t) + path[j + 1] * t)
        return out

    if isinstance(horizon, int) and horizon > 0 and horizon != 5:
        neg_path = _resample_path(neg_path, horizon)
        pos_path = _resample_path(pos_path, horizon)
    damp_factor = damp if isinstance(damp, float) else 0.55
    eps_value = eps if isinstance(eps, float) else 0.049
    h = max(len(neg_path), len(pos_path))
    blended = blend_paths(neg_path, pos_path, w_neg, w_pos, damp=damp_factor, eps=eps_value, horizon=h)
    if (path_mode or "linear").lower().startswith("geom"):
        blended = geom_compound(blended)
    blended = smooth_blend(blended, horizon=h)

    def _pad3(xs: List[str], pool: List[str]) -> List[str]:
        xs = xs[:3]
        for c in pool:
            if len(xs) == 3:
                break
            if c not in xs:
                xs.append(c)
        return xs

    drivers_neu = _pad3(drivers_neu, ["dampened feedback"])
    risks_neu = _pad3(risks_neu, ["liquidity squeeze", "governance uncertainty", "model uncertainty"])

    outcomes = {
        "negative": {
            "thesis": neg["thesis"],
            "drivers": neg["drivers"][:3],
            "risks": neg["risks"][:3],
            "price_path_week": neg_path,
            "confidence": float(neg["confidence"]),
        },
        "positive": {
            "thesis": pos["thesis"],
            "drivers": pos["drivers"][:3],
            "risks": pos["risks"][:3],
            "price_path_week": pos_path,
            "confidence": float(pos["confidence"]),
        },
        "neutralized": {
            "thesis": neutral_thesis,
            "drivers": drivers_neu,
            "risks": risks_neu,
            "price_path_week": blended,
            "confidence": round((c_neg + c_pos) / 2, 2),
        },
    }

    # Variance band based on spread between POS and NEG
    try:
        import numpy as _np  # type: ignore

        neg_arr = _np.array(neg_path, dtype=float)
        pos_arr = _np.array(pos_path, dtype=float)
        spread = _np.abs(pos_arr - neg_arr)
        band = float(spread.mean() / 4.0)
        outcomes["neutralized"]["band"] = {
            "upper": [round(x + band, 2) for x in blended],
            "lower": [round(x - band, 2) for x in blended],
        }
    except Exception:
        pass

    # Ensure neutral risks have fallback and no overlap
    risks_neu = [r for r in outcomes["neutralized"]["risks"] if r not in outcomes["neutralized"]["drivers"]][:3]
    if not risks_neu:
        risks_neu = ["governance uncertainty", "liquidity squeeze", "model uncertainty"]
    outcomes["neutralized"]["risks"] = risks_neu

    mode = "mock" if client.__class__.__name__ == "MockClient" else "live"
    result = {
        "topic": topic,
        "context": context,
        "outcomes": outcomes,
        "meta": {
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    result.setdefault("meta", {}).update({
        "horizon": len(outcomes["neutralized"]["price_path_week"]),
        "unit": (unit or "days"),
        "path_mode": (path_mode or "linear"),
    })

    # Independent comparators (non-reflexive baselines)
    mode = (indep_mode or "drift").lower()
    if mode.startswith("decay"):
        k = float(kappa) if isinstance(kappa, (int, float)) else 0.15
        neg_indep = independent_decay(
            start_step=(neg_path[0] if neg_path else -1.0), horizon=h, kappa=k
        )
        pos_indep = independent_decay(
            start_step=(pos_path[0] if pos_path else 1.0), horizon=h, kappa=k
        )
    else:
        d = float(drift) if isinstance(drift, (int, float)) else 0.2
        neg_indep = independent_drift(
            start_step=(neg_path[0] if neg_path else -1.0), horizon=h, drift=d
        )
        pos_indep = independent_drift(
            start_step=(pos_path[0] if pos_path else 1.0), horizon=h, drift=d
        )
    if (path_mode or "linear").lower().startswith("geom"):
        neg_indep = geom_compound(neg_indep)
        pos_indep = geom_compound(pos_indep)
    result.setdefault("comparators", {})["neg_independent"] = neg_indep
    result["comparators"]["pos_independent"] = pos_indep

    # Delta metric: area between coupled vs independent curves
    try:
        import numpy as _np  # type: ignore

        neu_arr = _np.array(outcomes["neutralized"]["price_path_week"], dtype=float)
        neg_arr = _np.array(neg_path, dtype=float)
        pos_arr = _np.array(pos_path, dtype=float)
        neg_ind = _np.array(neg_indep, dtype=float)
        pos_ind = _np.array(pos_indep, dtype=float)
        area_neg = float(_np.abs(neg_arr - neg_ind).sum())
        area_pos = float(_np.abs(pos_arr - pos_ind).sum())
        result.setdefault("meta", {})["interaction_area"] = {
            "neg": round(area_neg, 2),
            "pos": round(area_pos, 2),
        }
    except Exception:
        pass

    try:
        import jsonschema  # type: ignore
        jsonschema.validate(instance=result, schema=SCHEMA)  # type: ignore
    except Exception as e:  # pragma: no cover - optional guardrail
        result.setdefault("meta", {})["validation_error"] = str(e)

    return result


