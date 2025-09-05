from __future__ import annotations

from agents import MockClient
from orchestrator import run_scenario


def test_agent_no_overlap():
    res = run_scenario(
        "Enron-like patterns",
        "Company X 10-Q anomalies; auditor churn",
        MockClient(),
    )
    for side in ("negative", "neutralized", "positive"):
        d = set(x.lower() for x in res["outcomes"][side]["drivers"])
        r = set(x.lower() for x in res["outcomes"][side]["risks"])
        assert d.isdisjoint(r), f"drivers/risks overlap for {side}"


