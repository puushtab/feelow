from __future__ import annotations
from datetime import datetime, timezone
from typing import List

from .schema import Bundle, Item
from .router import Router
from .synthesis import SynthesisAgent
from .scoring.reliability import ReliabilityScorer
from .scoring.relevance import RelevanceScorer
from .scoring.novelty import NoveltyScorer
from .scoring.sentiment import SentimentScorer
from .scoring.impact import ImpactScorer

class Orchestrator:
    def __init__(self, collectors: List, llm_client, cache):
        self.collectors = collectors
        self.cache = cache

        # scoring agents
        self.rel0 = ReliabilityScorer()
        self.rel = RelevanceScorer()
        self.nov = NoveltyScorer(cache=cache)
        self.sent = SentimentScorer()
        self.impact = ImpactScorer()

        # router + synthesis
        self.router = Router(llm_client=llm_client)
        self.synth = SynthesisAgent(llm_client=llm_client)

    def run(self, ticker: str) -> dict:
        asof = datetime.now(timezone.utc)
        bundle = Bundle(ticker=ticker, asof=asof)

        # 1) Collect
        items: List[Item] = []
        for c in self.collectors:
            items.extend(c.collect(ticker=ticker, asof=asof))
        bundle.items = items

        # 2) Score (fast)
        self.rel0.apply(bundle)
        self.rel.apply(bundle)
        self.nov.apply(bundle)
        self.sent.apply(bundle)
        self.impact.apply(bundle)

        # 3) Route (select)
        bundle.selected = self.router.select_items(bundle, max_items=18, max_tokens=4500)

        # 4) Aggregate metrics (Reliable Source Score etc.)
        bundle.metrics = self._aggregate(bundle)

        # 5) Synthesize narrative + recommendation
        report = self.synth.generate(bundle)

        return {
            "bundle_metrics": bundle.metrics,
            "selected_items": [self._item_view(i) for i in bundle.selected],
            "report": report,
        }

    def _aggregate(self, bundle: Bundle) -> dict:
        # Example reliable score aggregator
        qs = []
        for it in bundle.selected:
            R  = it.scores.get("reliability", 0.0)
            Re = it.scores.get("relevance", 0.0)
            N  = it.scores.get("novelty", 0.0)
            I  = it.scores.get("impact", 0.0)
            C  = it.scores.get("confidence", 1.0)
            q = max(0.0, min(1.0, R*Re*N*I*C))
            it.scores["quality"] = q
            qs.append(q)

        rss = 1.0
        for q in qs:
            rss *= (1.0 - q)
        rss = 1.0 - rss

        # diversity: count by source
        by_source = {}
        for it in bundle.selected:
            by_source[it.source] = by_source.get(it.source, 0) + 1

        return {
            "reliable_source_score": float(rss),
            "selected_count": len(bundle.selected),
            "source_mix": by_source,
        }

    def _item_view(self, it: Item) -> dict:
        return {
            "id": it.id,
            "source": it.source,
            "ts": it.ts.isoformat(),
            "title": it.title,
            "url": it.url,
            "scores": it.scores,
            "summary": it.summary,
            "tags": it.tags,
        }