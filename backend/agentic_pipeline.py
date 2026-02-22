"""
agentic_pipeline.py (Gemini + multimodal demo)

Collectors -> Scorers -> (optional) Item summarizer -> (optional) LLM Router -> Synthesis -> report

Changes vs Claude/text-only version:
- Uses Gemini (google-generativeai).
- Supports MULTIMODAL inputs (images + video) for summarization / routing / synthesis.
- Collectors are local demo collectors:
  - Earnings call video (local file)
  - Reddit / Twitter / Instagram / Google Trends images (local files)

Important note:
- Gemini API expects media to be provided as bytes/parts; this file loads local assets
  and passes them as multimodal parts.
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Your existing imports (adjust paths to your project)
from src.agent_search.schema import Bundle, Item
from src.agent_search.prompts import ITEM_SUMMARY_PROMPT, ROUTER_LLM_PROMPT, SYNTHESIS_PROMPT
from src.agent_search.scoring.reliability import ReliabilityScorer
from src.agent_search.scoring.relevance import RelevanceScorer
from src.agent_search.scoring.novelty import NoveltyScorer
from src.agent_search.scoring.sentiment import SentimentScorer
from src.agent_search.scoring.impact import ImpactScorer

from src.agent_search.collectors.local_video import EarningsVideoCollector
from src.agent_search.collectors.local_images import LocalImageCollector


# ---------------------------
# Config
# ---------------------------

@dataclass
class AgenticPipelineConfig:
    # selection budgets
    max_items: int = 14
    per_source_cap_default: int = 4
    per_source_caps: Optional[Dict[str, int]] = None

    # summarization policy
    summarize_long_items: bool = True
    summarize_min_chars: int = 1600
    summarize_max_chars: int = 8000

    # multimodal policy
    attach_media_to_llm: bool = True            # pass image/video to Gemini when available
    max_media_per_call: int = 6                 # keep calls small (demo-friendly)

    # LLM router policy
    use_llm_router: bool = True
    llm_router_min_items: int = 10
    llm_router_max_chars_table: int = 12000

    # synthesis policy
    synthesis_max_items_json_chars: int = 20000

    # cache behavior
    novelty_ttl_seconds: int = 7 * 24 * 3600


# ---------------------------
# Robust JSON extraction
# ---------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

def _extract_json_text(raw: str) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    m = _FIRST_JSON_OBJ_RE.search(raw.strip())
    if m:
        return m.group(1).strip()
    return None

def _safe_json_loads(raw: str) -> Optional[Any]:
    jtxt = _extract_json_text(raw) or raw
    try:
        return json.loads(jtxt)
    except Exception:
        try:
            jtxt2 = re.sub(r",\s*([}\]])", r"\1", jtxt)
            return json.loads(jtxt2)
        except Exception:
            return None


# ---------------------------
# Media helpers (local assets)
# ---------------------------

def _guess_mime(path: Union[str, Path]) -> str:
    p = str(path).lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    if p.endswith(".mp4"):
        return "video/mp4"
    if p.endswith(".mov"):
        return "video/quicktime"
    if p.endswith(".mp3"):
        return "audio/mpeg"
    if p.endswith(".wav"):
        return "audio/wav"
    return "application/octet-stream"


def _load_media_part(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Returns a Gemini "part" dict: {"mime_type": "...", "data": bytes}
    Compatible with google-generativeai content parts.
    """
    path = Path(path)
    mime = _guess_mime(path)
    data = path.read_bytes()
    return {"mime_type": mime, "data": data}


def _item_media_paths(it: Item) -> List[str]:
    """
    For demo collectors we store:
      it.meta["asset_type"] in {"image","video"}
      it.meta["asset_path"] = "...local path..."
    """
    p = it.meta.get("asset_path")
    if isinstance(p, str) and p:
        return [p]
    return []


# ---------------------------
# Simple deterministic router (fallback)
# ---------------------------

def _deterministic_select(
    items: List[Item],
    max_items: int,
    per_source_caps: Dict[str, int],
    default_cap: int,
    always_include_sources: Tuple[str, ...] = ("earnings_video", "google_trends_image"),
) -> List[Item]:
    """
    Select items by quality with diversity caps.
    Always include key "macro sentiment anchors" like earnings_video and google_trends_image if present.
    """
    always = [it for it in items if it.source in always_include_sources and it.scores.get("quality", 0.0) > 0.05]

    remaining = [it for it in items if it not in always]
    remaining.sort(key=lambda x: x.scores.get("quality", 0.0), reverse=True)

    selected: List[Item] = []
    counts: Dict[str, int] = {}

    def cap_for(source: str) -> int:
        return per_source_caps.get(source, default_cap)

    def take(it: Item) -> bool:
        cap = cap_for(it.source)
        if counts.get(it.source, 0) >= cap:
            return False
        selected.append(it)
        counts[it.source] = counts.get(it.source, 0) + 1
        return True

    for it in always:
        if len(selected) >= max_items:
            break
        take(it)

    for it in remaining:
        if len(selected) >= max_items:
            break
        take(it)

    return selected


# ---------------------------
# Quality aggregation
# ---------------------------

def _compute_quality(it: Item) -> float:
    R  = float(it.scores.get("reliability", 0.0))
    Re = float(it.scores.get("relevance", 0.0))
    N  = float(it.scores.get("novelty", 0.0))
    I  = float(it.scores.get("impact", 0.0))
    C  = float(it.scores.get("confidence", 1.0))
    q = R * Re * N * I * C
    return float(max(0.0, min(1.0, q)))


def _aggregate_bundle_metrics(bundle: Bundle) -> Dict[str, Any]:
    qs = []
    by_source: Dict[str, int] = {}

    for it in bundle.selected:
        q = _compute_quality(it)
        it.scores["quality"] = q
        qs.append(q)
        by_source[it.source] = by_source.get(it.source, 0) + 1

    prod = 1.0
    for q in qs:
        prod *= (1.0 - q)
    rss = 1.0 - prod

    wsum = 0.0
    ssum = 0.0
    for it in bundle.selected:
        q = float(it.scores.get("quality", 0.0))
        s = float(it.scores.get("sentiment", 0.0))
        wsum += q
        ssum += q * s
    avg_sent = (ssum / wsum) if wsum > 1e-12 else 0.0

    return {
        "reliable_source_score": float(max(0.0, min(1.0, rss))),
        "selected_count": len(bundle.selected),
        "source_mix": by_source,
        "avg_sentiment_weighted": float(avg_sent),
    }


# ---------------------------
# Gemini client (multimodal)
# ---------------------------

class GeminiLLMClient:
    """
    Minimal Gemini wrapper that supports multimodal parts.

    Requires:
      pip install google-generativeai

    Usage:
      llm = GeminiLLMClient(api_key=os.environ["GEMINI_API_KEY"], model="gemini-1.5-pro")
      text = llm.complete(prompt="...", media_parts=[{"mime_type":"image/png","data":...}, ...])
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", temperature: float = 0.2):
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self._genai = genai
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str, media_parts: Optional[List[Dict[str, Any]]] = None) -> str:
        generation_config = self._genai.types.GenerationConfig(
            temperature=self.temperature,
        )

        contents: List[Any] = []
        if media_parts:
            # Each part: {"mime_type": "...", "data": bytes}
            for p in media_parts:
                contents.append(self._genai.types.Part.from_bytes(p["data"], mime_type=p["mime_type"]))

        contents.append(prompt)

        resp = self._model.generate_content(
            contents,
            generation_config=generation_config,
        )
        # resp.text is usually present; if blocked/empty, return safe string
        return (getattr(resp, "text", None) or "").strip()


# ---------------------------
# Item summarization step (Gemini multimodal)
# ---------------------------

def _summarize_item_if_needed(llm_client: GeminiLLMClient, it: Item, cfg: AgenticPipelineConfig) -> None:
    """
    For multimodal demo:
    - If item has media (image/video), summarize using Gemini with the media attached.
    - If item is text-only and long, summarize as before.
    """
    if it.summary:
        return

    media_paths = _item_media_paths(it)
    has_media = bool(media_paths)

    # If short text + no media, trivial summary
    if not has_media and (not it.text or len(it.text) < cfg.summarize_min_chars):
        it.summary = it.title.strip()
        return

    # Build prompt
    text_for_prompt = (it.text or "")
    if len(text_for_prompt) > cfg.summarize_max_chars:
        text_for_prompt = text_for_prompt[: cfg.summarize_max_chars]

    prompt = ITEM_SUMMARY_PROMPT.format(
        source=it.source,
        title=it.title,
        text=text_for_prompt,
    )

    media_parts: List[Dict[str, Any]] = []
    if cfg.attach_media_to_llm and has_media:
        # keep bounded
        for p in media_paths[: cfg.max_media_per_call]:
            try:
                media_parts.append(_load_media_part(p))
            except Exception:
                pass

    raw = llm_client.complete(prompt=prompt, media_parts=media_parts if media_parts else None)
    data = _safe_json_loads(raw)

    if not isinstance(data, dict):
        # Fallback: store a compact string
        it.summary = it.title.strip()
        if it.text:
            it.summary += " — " + it.text[:200].strip()
        return

    summary = data.get("summary")
    if isinstance(summary, list):
        it.summary = "\n".join([str(x) for x in summary][:3])
    elif isinstance(summary, str):
        it.summary = summary.strip()
    else:
        it.summary = it.title.strip()

    key_facts = data.get("key_facts", [])
    if isinstance(key_facts, list):
        it.tags.extend([str(x) for x in key_facts[:10]])

    stance = data.get("stance")
    if isinstance(stance, str):
        it.scores["stance"] = stance.strip().lower()

    conf = data.get("confidence")
    try:
        if conf is not None:
            it.scores["confidence"] = float(conf)
    except Exception:
        pass

    horizon = data.get("horizon")
    if isinstance(horizon, str):
        it.scores["horizon"] = horizon.strip()

    why = data.get("why_market_moves")
    if isinstance(why, str):
        it.meta["why_market_moves"] = why.strip()


# ---------------------------
# LLM Router step (Gemini)
# ---------------------------

def _llm_route_items(
    llm_client: GeminiLLMClient,
    items: List[Item],
    cfg: AgenticPipelineConfig
) -> Tuple[List[Item], Dict[str, Any]]:
    rows = []
    for it in items:
        rows.append({
            "id": it.id,
            "source": it.source,
            "ts": it.ts.isoformat(),
            "title": (it.title or "")[:140],
            "quality": round(float(it.scores.get("quality", 0.0)), 3),
            "short_summary": ((it.summary or it.text[:220]) if it.text else (it.summary or "")).replace("\n", " ")[:260],
        })

    table = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(table) > cfg.llm_router_max_chars_table:
        table = table[: cfg.llm_router_max_chars_table] + "\n...truncated..."

    prompt = ROUTER_LLM_PROMPT.format(max_items=cfg.max_items, items_table=table)
    raw = llm_client.complete(prompt=prompt)
    decision = _safe_json_loads(raw)

    if not isinstance(decision, dict):
        return [], {"error": "router_json_parse_failed", "raw": raw[:800]}

    selected_ids = decision.get("selected_ids", [])
    if not isinstance(selected_ids, list) or not selected_ids:
        return [], {"error": "router_missing_selected_ids", "decision": decision}

    sel_set = set(str(x) for x in selected_ids)
    selected = [it for it in items if it.id in sel_set]
    return selected, decision


# ---------------------------
# Synthesis step (Gemini multimodal)
# ---------------------------

def _synthesize_report(llm_client: GeminiLLMClient, bundle: Bundle, cfg: AgenticPipelineConfig) -> str:
    bundle_metrics_json = json.dumps(bundle.metrics, indent=2, default=str, ensure_ascii=False)

    payload = []
    media_parts: List[Dict[str, Any]] = []

    # Attach up to max_media_per_call media assets to the final synthesis
    for it in bundle.selected:
        payload.append({
            "id": it.id,
            "source": it.source,
            "ts": it.ts.isoformat(),
            "title": it.title,
            "summary": it.summary or (it.text[:280] if it.text else ""),
            "key_facts": it.tags[:12],
            "stance": it.scores.get("stance"),
            "confidence": it.scores.get("confidence", 0.7),
            "horizon": it.scores.get("horizon"),
            "quality": it.scores.get("quality"),
            "url": it.url,
        })

        if cfg.attach_media_to_llm and len(media_parts) < cfg.max_media_per_call:
            for p in _item_media_paths(it):
                if len(media_parts) >= cfg.max_media_per_call:
                    break
                try:
                    media_parts.append(_load_media_part(p))
                except Exception:
                    pass

    items_json = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    if len(items_json) > cfg.synthesis_max_items_json_chars:
        items_json = items_json[: cfg.synthesis_max_items_json_chars] + "\n...truncated..."

    prompt = SYNTHESIS_PROMPT.format(
        ticker=bundle.ticker,
        bundle_metrics_json=bundle_metrics_json,
        items_json=items_json,
    )
    return llm_client.complete(prompt=prompt, media_parts=media_parts if media_parts else None).strip()


# ---------------------------
# Main entry: run_agentic_pipeline
# ---------------------------

def run_agentic_pipeline(
    ticker: str,
    collectors: List[Any],
    llm_client: GeminiLLMClient,
    cache: Any,
    cfg: AgenticPipelineConfig = AgenticPipelineConfig(),
) -> Dict[str, Any]:
    asof = datetime.now(timezone.utc)
    bundle = Bundle(ticker=ticker, asof=asof)

    # 1) Collect
    items: List[Item] = []
    for c in collectors:
        got = c.collect(ticker=ticker, asof=asof)
        if got:
            items.extend(got)
    bundle.items = items

    # 2) Score
    reliability = ReliabilityScorer()
    relevance = RelevanceScorer()
    novelty = NoveltyScorer(cache=cache)
    sentiment = SentimentScorer()
    impact = ImpactScorer()

    reliability.apply(bundle)
    relevance.apply(bundle)
    novelty.apply(bundle)
    sentiment.apply(bundle)
    impact.apply(bundle)

    # Compute quality
    for it in bundle.items:
        it.scores["quality"] = _compute_quality(it)

    # 3) Summarize (multimodal)
    for it in bundle.items:
        _summarize_item_if_needed(llm_client, it, cfg)

    # 4) Route (update caps for your sources)
    per_source_caps = cfg.per_source_caps or {
        "earnings_video": 1,
        "reddit_image": 3,
        "twitter_image": 3,
        "instagram_image": 2,
        "google_trends_image": 2,
        "social_image": 2,
    }

    router_info: Dict[str, Any] = {"mode": "deterministic"}
    candidates = sorted(bundle.items, key=lambda x: x.scores.get("quality", 0.0), reverse=True)

    selected: List[Item] = []
    if cfg.use_llm_router and len(candidates) >= cfg.llm_router_min_items:
        llm_selected, decision = _llm_route_items(llm_client, candidates, cfg)
        if llm_selected:
            selected = llm_selected
            router_info = {"mode": "llm", "decision": decision}
        else:
            router_info = {"mode": "deterministic_fallback", "llm_router_error": decision}
            selected = _deterministic_select(
                candidates,
                max_items=cfg.max_items,
                per_source_caps=per_source_caps,
                default_cap=cfg.per_source_cap_default,
            )
    else:
        selected = _deterministic_select(
            candidates,
            max_items=cfg.max_items,
            per_source_caps=per_source_caps,
            default_cap=cfg.per_source_cap_default,
        )

    bundle.selected = selected

    # 5) Aggregate bundle metrics
    bundle.metrics = _aggregate_bundle_metrics(bundle)

    # 6) Synthesis report (multimodal)
    report = _synthesize_report(llm_client, bundle, cfg)

    # 7) Output
    selected_items_payload = []
    for it in bundle.selected:
        selected_items_payload.append({
            "id": it.id,
            "source": it.source,
            "ts": it.ts.isoformat(),
            "title": it.title,
            "url": it.url,
            "scores": it.scores,
            "summary": it.summary,
            "tags": it.tags[:12],
            "meta": it.meta,
        })

    return {
        "bundle_metrics": bundle.metrics,
        "selected_items": selected_items_payload,
        "router_info": router_info,
        "report": report,
    }


# ---------------------------
# Minimal cache example
# ---------------------------

class SimpleDictCache:
    def __init__(self):
        self._d = {}

    def get(self, key: str):
        return self._d.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 0):
        self._d[key] = value


"""
Example usage (demo):
"""
collectors = [
    EarningsVideoCollector("demo_assets/nvda/video/earnings_clip.mp4"),
    LocalImageCollector("demo_assets/nvda/images/"),
]

cache = SimpleDictCache()
llm = GeminiLLMClient(api_key="AIzaSyDufl2aS_gN71WiBsxLAvltIUs-l562Wws", model="gemini-1.5-pro")

res = run_agentic_pipeline(
    ticker="NVDA",
    collectors=collectors,
    llm_client=llm,
    cache=cache,
)

print(res["bundle_metrics"])
print(res["report"])
