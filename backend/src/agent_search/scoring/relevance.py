from __future__ import annotations
from ..schema import Bundle

class RelevanceScorer:
    def apply(self, bundle: Bundle) -> None:
        ticker = bundle.ticker.lower()
        for it in bundle.items:
            text = (it.title + " " + it.text).lower()
            hit = 1.0 if ticker in text else 0.6 if bundle.ticker in it.meta.get("entities", []) else 0.3
            it.scores["relevance"] = hit