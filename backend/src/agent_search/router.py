from __future__ import annotations
from typing import List
from .schema import Bundle, Item

class Router:
    def __init__(self, llm_client):
        self.llm = llm_client

    def select_items(self, bundle: Bundle, max_items: int, max_tokens: int) -> List[Item]:
        items = list(bundle.items)

        # compute quality
        for it in items:
            R  = it.scores.get("reliability", 0.0)
            Re = it.scores.get("relevance", 0.0)
            N  = it.scores.get("novelty", 0.0)
            I  = it.scores.get("impact", 0.0)
            C  = it.scores.get("confidence", 1.0)
            it.scores["quality"] = max(0.0, min(1.0, R*Re*N*I*C))

        # Always-include critical items
        always = [it for it in items if it.source in {"sec_8k", "sec_form4", "earnings_call"} and it.scores["quality"] > 0.2]

        # Rank remaining by quality
        remaining = [it for it in items if it not in always]
        remaining.sort(key=lambda x: x.scores.get("quality", 0.0), reverse=True)

        # diversify with caps
        selected = []
        per_source_cap = {
            "reddit": 3,
            "x": 3,
            "reuters": 3,
            "bloomberg": 2,
            "ft": 2,
            "wsj": 2,
        }
        counts = {}

        def can_take(it: Item) -> bool:
            cap = per_source_cap.get(it.source, 4)
            return counts.get(it.source, 0) < cap

        # add always first
        for it in always:
            if len(selected) >= max_items:
                break
            selected.append(it)
            counts[it.source] = counts.get(it.source, 0) + 1

        # fill with best remaining respecting caps
        for it in remaining:
            if len(selected) >= max_items:
                break
            if not can_take(it):
                continue
            selected.append(it)
            counts[it.source] = counts.get(it.source, 0) + 1

        return selected