# src/agents/scoring/reliability.py
from __future__ import annotations
from ..schema import Bundle

SOURCE_PRIOR = {
    "sec_8k": 1.00,
    "sec_10q": 0.95,
    "sec_10k": 0.95,
    "sec_form4": 0.90,
    "earnings_call": 0.90,
    "reuters": 0.90,
    "bloomberg": 0.88,
    "ft": 0.87,
    "wsj": 0.87,
    "analyst_rating": 0.80,
    "reddit": 0.55,
    "x": 0.55,
}

class ReliabilityScorer:
    def apply(self, bundle: Bundle) -> None:
        for it in bundle.items:
            base = SOURCE_PRIOR.get(it.source, 0.50)
            cred = float(it.meta.get("author_cred", 1.0))
            cred = max(0.5, min(1.2, cred))
            it.scores["reliability"] = max(0.0, min(1.0, base * cred))