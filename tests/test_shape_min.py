from __future__ import annotations

from agents import MockClient
from orchestrator import run_scenario


def test_shape_min():
    res = run_scenario(
        "Enron-like patterns",
        "Company X 10-Q anomalies; auditor churn",
        MockClient(),
    )
    o = res["outcomes"]
    for k in ("negative", "neutralized", "positive"):
        assert {"thesis", "drivers", "risks", "price_path_week", "confidence"}.issubset(o[k].keys())
        assert len(o[k]["price_path_week"]) == 5
        c = float(o[k]["confidence"])
        assert 0.0 <= c <= 1.0
        assert set(map(str.lower, o[k]["drivers"])).isdisjoint(set(map(str.lower, o[k]["risks"])))


