from __future__ import annotations
import hashlib
from ..schema import Bundle

class NoveltyScorer:
    def __init__(self, cache):
        self.cache = cache

    def apply(self, bundle: Bundle) -> None:
        for it in bundle.items:
            h = hashlib.sha256((it.title + it.text[:500]).encode("utf-8")).hexdigest()
            seen = self.cache.get(f"seen:{bundle.ticker}:{h}")
            if seen:
                it.scores["novelty"] = 0.05
            else:
                it.scores["novelty"] = 1.0
                self.cache.set(f"seen:{bundle.ticker}:{h}", True, ttl_seconds=7*24*3600)