from __future__ import annotations
from ..schema import Bundle

class SentimentScorer:
    def apply(self, bundle: Bundle) -> None:
        for it in bundle.items:
            # place the existing FinBERT outputs here
            # expected: it.scores["sentiment"] in [-1,1] and it.scores["confidence"] in [0,1]
            if "sentiment" not in it.scores:
                it.scores["sentiment"] = 0.0
            if "confidence" not in it.scores:
                it.scores["confidence"] = 0.7