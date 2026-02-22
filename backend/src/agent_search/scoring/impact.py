# src/agents/scoring/impact.py
from __future__ import annotations
from ..schema import Bundle

IMPACT_HINTS = {
    "sec_8k": 0.95,
    "sec_form4": 0.75,
    "earnings_call": 0.90,
    "analyst_rating": 0.70,
    "reuters": 0.70,
    "reddit": 0.40,
    "x": 0.45,
}

class ImpactScorer:
    def apply(self, bundle: Bundle) -> None:
        for it in bundle.items:
            base = IMPACT_HINTS.get(it.source, 0.50)
            # boost if it contains event-like keywords
            text = (it.title + " " + it.text).lower()
            boost = 0.15 if any(k in text for k in ["guidance", "downgrade", "upgrade", "sec", "lawsuit", "ban", "recall", "investigation"]) else 0.0 # list can be expanded with more patterns
            it.scores["impact"] = max(0.0, min(1.0, base + boost))