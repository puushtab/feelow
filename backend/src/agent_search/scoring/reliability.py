# src/agents/scoring/reliability.py
from __future__ import annotations
from ..schema import Bundle

SOURCE_PRIOR = {
    "earnings_video": 0.9,
    "reddit_image": 0.6,
    "twitter_image": 0.65,
    "instagram_image": 0.55,
    "google_trends_image": 0.75,
}

class ReliabilityScorer:
    def apply(self, bundle: Bundle) -> None:
        for it in bundle.items:
            base = SOURCE_PRIOR.get(it.source, 0.50)
            cred = float(it.meta.get("author_cred", 1.0))
            cred = max(0.5, min(1.2, cred))
            it.scores["reliability"] = max(0.0, min(1.0, base * cred))