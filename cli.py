"""Tiny CLI for running reflexivity scenarios."""

from __future__ import annotations

import argparse
import json
from typing import Any

from agents import MockClient, OpenAIClient
from orchestrator import run_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reflexivity loop agents")
    parser.add_argument("--topic", required=True, help="Topic under analysis")
    parser.add_argument("--context", required=True, help="Supporting context text")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock mode")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON")
    parser.add_argument("--weights", type=str, default=None, help="Override weights as 'w_neg,w_pos' (e.g., 0.4,0.6).")
    parser.add_argument("--damp", type=float, default=None, help="Override neutral dampening factor (e.g., 0.55).")
    parser.add_argument("--eps", type=float, default=None, help="Snap-to-zero threshold for neutral path (default 0.049).")
    parser.add_argument("--neutral", type=str, default="concise", choices=["concise", "compare"], help="Neutralization style.")
    parser.add_argument("--export", type=str, default=None, help="Write full JSON to path.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for MockClient determinism.")
    parser.add_argument("--horizon", type=int, default=None, help="Number of points in the projected price path (default 5).")
    parser.add_argument("--unit", type=str, choices=("days","weeks"), default="days", help="Label for horizon steps (default: days).")
    parser.add_argument("--path-mode", choices=("linear","geom"), default="linear", help="Neutral blend mode for path shape.")
    parser.add_argument("--csv", type=str, default=None, help="Export three paths to CSV (neg,neutral,pos).")
    parser.add_argument("--show-indep", action="store_true", help="Print independent (non-reflexive) comparator paths.")
    parser.add_argument("--kappa", type=float, default=0.15, help="Decay for independent comparator paths.")
    parser.add_argument("--drift", type=float, default=0.2, help="Outward drift step for comparator paths.")
    parser.add_argument("--indep-mode", choices=("drift","decay"), default="drift", help="Comparator mode (outward drift or inward decay).")

    args = parser.parse_args(argv)

    client = (MockClient(seed=args.seed) if args.mock else OpenAIClient())
    result = run_scenario(topic=args.topic, context=args.context, client=client, weights=args.weights, damp=args.damp, eps=args.eps, neutral_style=args.neutral, horizon=args.horizon, unit=args.unit, path_mode=args.path_mode, kappa=args.kappa, indep_mode=args.indep_mode, drift=args.drift)

    if args.export:
        import json as _json
        import pathlib as _pathlib
        _path = _pathlib.Path(args.export)
        _path.parent.mkdir(parents=True, exist_ok=True)
        with open(_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved JSON → {args.export}")

    if args.csv:
        import csv as _csv
        import pathlib as _pathlib
        _p = _pathlib.Path(args.csv)
        _p.parent.mkdir(parents=True, exist_ok=True)
        oc = result["outcomes"]
        neg = oc["negative"]["price_path_week"]
        neu = oc["neutralized"]["price_path_week"]
        pos = oc["positive"]["price_path_week"]
        h = max(len(neg), len(neu), len(pos))
        neg_ind = result.get("comparators", {}).get("neg_independent", [None] * h)
        pos_ind = result.get("comparators", {}).get("pos_independent", [None] * h)
        with open(_p, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["step", "NEG", "NEU", "POS", "NEG_INDEP", "POS_INDEP"]) 
            w.writeheader()
            for i in range(h):
                w.writerow({
                    "step": i + 1,
                    "NEG": neg[i] if i < len(neg) else None,
                    "NEU": neu[i] if i < len(neu) else None,
                    "POS": pos[i] if i < len(pos) else None,
                    "NEG_INDEP": neg_ind[i] if i < len(neg_ind) else None,
                    "POS_INDEP": pos_ind[i] if i < len(pos_ind) else None,
                })
        print(f"Saved CSV → {args.csv}")

    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0

    # Pretty print minimal table-like output
    def fmt_path(arr):
        s = ", ".join([f"{(0.0 if abs(x) < 1e-9 else x):+.1f}%" for x in arr])
        return s.replace("-0.0%", "+0.0%")

    def fmt_path_label(horizon: int, unit: str) -> str:
        return f"Price path ({horizon} {unit}):"
    outcomes = result["outcomes"]
    def fmt_path(p: list[float]) -> str:
        return ", ".join(f"{x:+.1f}%" for x in p)

    print(f"Topic: {result['topic']}")
    print(f"Context: {result['context']}")
    print(f"Mode: {result['meta']['mode']}")
    print()
    def _fmt_list(xs):
        return ", ".join(xs) if xs else "—"

    def _print_block(tag, out):
        print(f"\n[{tag.upper()}]")
        print(f"Thesis: {out['thesis']}")
        print(f"Drivers: {_fmt_list(out['drivers'])}")
        print(f"Risks: {_fmt_list(out['risks'])}")
        print(f"{fmt_path_label(len(out['price_path_week']), args.unit)} {fmt_path(out['price_path_week'])}")
        print(f"Confidence: {out.get('confidence', 0.5):.2f}")

    for tag in ("negative", "neutralized", "positive"):
        _print_block(tag, outcomes[tag])

    if args.show_indep and "comparators" in result:
        print("\n[COMPARATORS]")
        print(f"NEG (independent): {fmt_path(result['comparators']['neg_independent'])}")
        print(f"POS (independent): {fmt_path(result['comparators']['pos_independent'])}")
        print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


