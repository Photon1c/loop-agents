"""Agent and client abstractions for reflexivity reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import random
from typing import Dict, List, Literal, Protocol

# Optional .env loader if available
try:  # pragma: no cover - optional
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional
    load_dotenv = None  # type: ignore

from loops import negative_feedback_chain, positive_feedback_chain


Stance = Literal["negative", "positive"]


@dataclass(frozen=True)
class AgentConfig:
    name: str
    stance: Stance
    system_preamble: str


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


class OpenAIClient:
    """Placeholder client. Insert provider-specific call in generate().

    Reads API key from env (e.g., OPENAI_API_KEY). Does not hardcode provider.
    """

    def __init__(self) -> None:
        # Load .env if python-dotenv is available; otherwise rely on process env.
        if load_dotenv is not None:
            try:
                load_dotenv()
            except Exception:
                pass
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def generate(self, prompt: str) -> str:  # pragma: no cover - network
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot run live mode.")
        # OpenAI v1 SDK style
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - import error
            raise RuntimeError(
                "openai package not installed. Run 'pip install openai'."
            ) from exc

        client = OpenAI(api_key=self.api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        system_message = (
            "You are a reflexivity reasoning helper. Return concise, structured content."
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0]
        text = getattr(choice.message, "content", None) or ""
        return text.strip()


class MockClient:
    """Deterministic stubbed client for tests and mock mode."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def generate(self, prompt: str) -> str:
        # Produce deterministic pseudo-content based on stable hash.
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big")
        self.random.seed(seed_int)
        base_signal = [
            "auditor turnover",
            "working-capital strain",
            "governance flags",
            "accounting ambiguity",
            "regulatory chatter",
        ]
        self.random.shuffle(base_signal)
        # Compact summary-like output
        return (
            "Signals: "
            + ", ".join(base_signal[:5])
            + "; Thesis: reflexive dynamics shape price; Drivers: A,B,C; Risks: X,Y,Z;"
        )


class Agent:
    def __init__(self, config: AgentConfig, client: LLMClient) -> None:
        self.config = config
        self.client = client

    def _extract_signals(self, topic: str, context: str) -> List[str]:
        prompt = (
            f"{self.config.system_preamble}\n"
            f"Topic: {topic}\nContext: {context}\n"
            "Task: 1) Extract 5 signals. 2) Build 4-step chain. 3) Output thesis, drivers, risks, price path, confidence."
        )
        raw = self.client.generate(prompt)
        # Very light parsing to retrieve up to 5 signals
        parts = raw.split("Signals:")
        if len(parts) < 2:
            return ["signal-1", "signal-2", "signal-3", "signal-4", "signal-5"]
        segment = parts[1].split(";")[0]
        signals = [s.strip() for s in segment.split(",") if s.strip()]
        if len(signals) < 5:
            signals += [f"signal-{i}" for i in range(len(signals) + 1, 6)]
        return signals[:5]

    def _price_path(self, stance: Stance) -> List[float]:
        # Deterministic path shape by stance
        if stance == "negative":
            return [-3.0, -2.0, -1.0, -0.5, -0.2]
        return [2.0, 1.5, 1.0, 0.5, 0.2]

    def reason(self, topic: str, context: str, loop_style: str) -> Dict:
        signals = self._extract_signals(topic, context)
        if self.config.stance == "negative":
            chain = negative_feedback_chain(signals, steps=4)
        else:
            chain = positive_feedback_chain(signals, steps=4)

        thesis = (
            f"{self.config.name}: reflexive {self.config.stance} thesis under '{topic}'."
        )
        # Clean label prefixes if present (avoid "NEG: NEG:" duplication)
        raw = thesis.strip()
        clean = raw
        for tag in ("NEG:", "POS:"):
            clean = clean[len(tag):].lstrip() if clean.startswith(tag) else clean
        # Stance-aware, topic-aware semanticization and overlap guard
        _ENRON_MAP: Dict[str, Dict[str, List[str]]] = {
            "negative": {
                "drivers": [
                    "auditor churn",
                    "off-balance-sheet obligations",
                    "related-party transactions",
                    "mark-to-market opacity",
                    "credit spread widening",
                    "wholesale funding stress",
                ],
                "risks": [
                    "regulatory intervention window",
                    "short-squeeze reflex",
                    "activist balance-sheet clean-up",
                    "acquisition rumor",
                ],
            },
            "positive": {
                "drivers": [
                    "short-squeeze reflex",
                    "regulatory intervention window",
                    "asset sale / deleveraging",
                    "turnaround guidance",
                    "credit line reaffirmation",
                ],
                "risks": [
                    "forensic accounting exposure",
                    "auditor churn",
                    "related-party transactions",
                    "off-balance-sheet obligations",
                ],
            },
        }

        def _topic_is_enron(topic_text: str, context_text: str) -> bool:
            t = (topic_text + " " + context_text).lower()
            return any(k in t for k in ["enron", "10-q", "auditor", "off-balance", "spe", "mark-to-market"])

        def _semanticize_items(topic_text: str, context_text: str, items: List[str], stance: str, field: str) -> List[str]:
            if not _topic_is_enron(topic_text, context_text):
                return items
            pool = _ENRON_MAP.get(stance, {}).get(field, [])
            out: List[str] = []
            i = 0
            for it in items:
                if isinstance(it, str) and it.strip().lower().startswith("signal-"):
                    out.append(pool[i % len(pool)] if pool else it)
                    i += 1
                else:
                    out.append(it)
            return out

        def _disjoint(top: List[str], others: List[str], fill_pool: List[str], k: int) -> List[str]:
            seen = {x.strip().lower() for x in top}
            cleaned = [x for x in others if x.strip().lower() not in seen]
            for cand in fill_pool:
                if len(cleaned) >= k:
                    break
                if cand.strip().lower() not in seen and cand not in cleaned:
                    cleaned.append(cand)
            return cleaned[:k]

        drv_raw = signals[:3]
        rsk_raw = signals[-3:]
        drivers = _semanticize_items(topic, context, drv_raw, self.config.stance, "drivers")
        risks = _semanticize_items(topic, context, rsk_raw, self.config.stance, "risks")
        if _topic_is_enron(topic, context):
            fill_pool = _ENRON_MAP.get(self.config.stance, {}).get("risks", [])
            risks = _disjoint(drivers, risks, fill_pool, k=3)
        price_path = self._price_path(self.config.stance)
        confidence = 0.65 if self.config.stance == "positive" else 0.6

        return {
            "thesis": clean,
            "drivers": drivers,
            "risks": risks,
            "chain": chain,
            "price_path_week": price_path,
            "confidence": confidence,
        }


