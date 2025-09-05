from __future__ import annotations

from agents import MockClient
from orchestrator import run_scenario


def test_outcomes_exist():
    res = run_scenario("Enron-like patterns", "Company X 10-Q anomalies", MockClient())
    oc = res["outcomes"]
    assert "negative" in oc and "neutralized" in oc and "positive" in oc


def test_price_path_lengths():
    res = run_scenario("Enron-like patterns", "Company X 10-Q anomalies", MockClient())
    oc = res["outcomes"]
    assert len(oc["negative"]["price_path_week"]) == 5
    assert len(oc["positive"]["price_path_week"]) == 5
    assert len(oc["neutralized"]["price_path_week"]) == 5


def test_confidence_bounds():
    res = run_scenario("Enron-like patterns", "Company X 10-Q anomalies", MockClient())
    oc = res["outcomes"]
    for key in ("negative", "neutralized", "positive"):
        c = oc[key]["confidence"]
        assert 0.0 <= c <= 1.0


