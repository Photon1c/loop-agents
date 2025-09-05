from __future__ import annotations

from agents import MockClient
from orchestrator import run_scenario


def test_horizon_and_labels():
    res = run_scenario(
        "Enron-like patterns",
        "Company X 10-Q anomalies; auditor churn",
        MockClient(),
        horizon=9,
        unit="days",
    )
    m = res["meta"]
    assert m["horizon"] == len(res["outcomes"]["neutralized"]["price_path_week"])
    assert m["unit"] in ("days", "weeks")


