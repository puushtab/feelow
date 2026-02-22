# src/agents/synthesis.py
from __future__ import annotations
import json
from .schema import Bundle

from .prompts import SYNTHESIS_PROMPT

class SynthesisAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate(self, bundle: Bundle) -> str:
        # Build compact JSON for prompt
        bundle_metrics_json = json.dumps(bundle.metrics, indent=2, default=str)

        items_payload = []
        for it in bundle.selected:
            items_payload.append({
                "id": it.id,
                "source": it.source,
                "ts": it.ts.isoformat(),
                "title": it.title,
                "summary": it.summary or it.text[:280],
                "stance": it.scores.get("stance"),
                "confidence": it.scores.get("confidence", 0.7),
                "horizon": it.scores.get("horizon"),
                "quality": it.scores.get("quality"),
                "url": it.url
            })

        items_json = json.dumps(items_payload, indent=2)

        prompt = SYNTHESIS_PROMPT.format(
            ticker=bundle.ticker,
            bundle_metrics_json=bundle_metrics_json,
            items_json=items_json
        )

        # llm_client is your wrapper (Anthropic/OpenAI). Returns text.
        return self.llm.complete(prompt)