import math
from src.agent_search.prompts import MULTIMODAL_SENTIMENT_PROMPT

def _clip(x, lo, hi):
    return max(lo, min(hi, x))

class MultimodalSentimentScorer:
    """
    Uses Gemini to score sentiment for items that are images/videos
    (or any item where sentiment isn't already set).
    """
    def __init__(self, llm_client, max_items: int = 20):
        self.llm = llm_client
        self.max_items = max_items

    def apply(self, bundle) -> None:
        n = 0
        for it in bundle.items:
            if n >= self.max_items:
                break

            # If something already set sentiment, keep it
            if "sentiment" in it.scores and it.scores["sentiment"] is not None:
                continue

            prompt = MULTIMODAL_SENTIMENT_PROMPT.format(
                ticker=bundle.ticker,
                source=it.source,
                title=it.title,
                text=(it.text or "")[:800],
            )

            media_parts = []
            p = it.meta.get("asset_path")
            if isinstance(p, str) and p:
                # call your existing helper
                from src.agent_search.utils import load_media_part
                try:
                    media_parts.append(load_media_part(p))
                except Exception:
                    media_parts = []

            raw = self.llm.complete(prompt=prompt, media_parts=media_parts or None)

            from src.agent_search.utils import safe_json_loads
            data = safe_json_loads(raw)
            if not isinstance(data, dict):
                it.scores["sentiment"] = 0.0
                it.scores["confidence"] = 0.4
                it.scores["stance"] = "neutral"
                continue

            s = float(data.get("sentiment", 0.0))
            c = float(data.get("confidence", 0.6))
            stance = str(data.get("stance", "neutral")).lower()

            it.scores["sentiment"] = _clip(s, -1.0, 1.0)
            it.scores["confidence"] = _clip(c, 0.0, 1.0)
            it.scores["stance"] = stance

            n += 1