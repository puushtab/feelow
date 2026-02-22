"""
A single, practical runner:
Collectors -> Scorers -> (optional) Item summarizer -> (optional) LLM Router -> Synthesis -> report

- Uses your prompts.py: ITEM_SUMMARY_PROMPT, ROUTER_LLM_PROMPT, SYNTHESIS_PROMPT
- Includes robust JSON parsing (handles code fences, trailing text, invalid JSON)
- Includes deterministic fallback routing if router JSON fails
- Summarizes ONLY long items (configurable)
- Produces:
  - final markdown report
  - bundle metrics (including Reliable Source Score)
  - selected items payload (auditable)

You only need to provide:
- collectors: list of objects with .collect(ticker, asof) -> List[Item]
- llm_client: object with .complete(prompt: str) -> str
- cache: object with .get(key) and .set(key, value, ttl_seconds)

This file assumes you already have:
- schema.py with Item and Bundle
- prompts.py with three prompts
- scoring agents: ReliabilityScorer, RelevanceScorer, NoveltyScorer, SentimentScorer, ImpactScorer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Your existing imports (adjust paths to your project)
from src.agent_search.schema import Bundle, Item
from src.agent_search.prompts import ITEM_SUMMARY_PROMPT, ROUTER_LLM_PROMPT, SYNTHESIS_PROMPT
from src.agent_search.scoring.reliability import ReliabilityScorer
from src.agent_search.scoring.relevance import RelevanceScorer
from src.agent_search.scoring.novelty import NoveltyScorer
from src.agent_search.scoring.sentiment import SentimentScorer
from src.agent_search.scoring.impact import ImpactScorer


# ---------------------------
# Config
# ---------------------------

@dataclass
class AgenticPipelineConfig:
    # selection budgets
    max_items: int = 14
    per_source_cap_default: int = 4
    per_source_caps: Dict[str, int] = None  # e.g. {"reddit": 3, "x": 3, "reuters": 3}

    # summarization policy
    summarize_long_items: bool = True
    summarize_min_chars: int = 1600       # summarize only if text longer than this
    summarize_max_chars: int = 8000       # truncate input to LLM
    # If you already have summaries from collectors, you can keep them.

    # LLM router policy
    use_llm_router: bool = True
    llm_router_min_items: int = 10        # use router only if enough candidates
    llm_router_max_chars_table: int = 12000  # keep router table bounded

    # synthesis policy
    synthesis_max_items_json_chars: int = 20000  # truncate items JSON if huge

    # cache behavior
    novelty_ttl_seconds: int = 7 * 24 * 3600


# ---------------------------
# Robust JSON extraction
# ---------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

def _extract_json_text(raw: str) -> Optional[str]:
    """
    Tries to extract a JSON object/array from an LLM response.
    Handles code fences and extra commentary.
    """
    if not raw or not isinstance(raw, str):
        return None

    # 1) code-fenced JSON
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()

    # 2) try to find first {...} or [...]
    m = _FIRST_JSON_OBJ_RE.search(raw.strip())
    if m:
        return m.group(1).strip()

    return None


def _safe_json_loads(raw: str) -> Optional[Any]:
    """
    Best-effort JSON parsing.
    Returns parsed object, or None.
    """
    jtxt = _extract_json_text(raw) or raw
    try:
        return json.loads(jtxt)
    except Exception:
        # Try a small cleanup: remove trailing commas (common LLM mistake)
        try:
            jtxt2 = re.sub(r",\s*([}\]])", r"\1", jtxt)
            return json.loads(jtxt2)
        except Exception:
            return None


# ---------------------------
# Simple deterministic router (fallback)
# ---------------------------

def _deterministic_select(
    items: List[Item],
    max_items: int,
    per_source_caps: Dict[str, int],
    default_cap: int,
    always_include_sources: Tuple[str, ...] = ("sec_8k", "sec_form4", "earnings_call"),
) -> List[Item]:
    """
    Select items by quality with diversity caps + always include critical sources if present.
    """
    # Always include critical, if decent quality
    always = [it for it in items if it.source in always_include_sources and it.scores.get("quality", 0.0) > 0.2]

    # Sort remaining by quality
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
    """
    Computes reliable_source_score + source mix.
    RSS = 1 - Π (1 - q_i) over selected items.
    """
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

    # Optional: average sentiment weighted by quality
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
# Item summarization step
# ---------------------------

def _summarize_item_if_needed(llm_client, it: Item, cfg: AgenticPipelineConfig) -> None:
    """
    If item is long and has no summary, summarize into structured JSON and fill fields.
    """
    if not cfg.summarize_long_items:
        return
    if it.summary:  # already summarized
        return
    if not it.text or len(it.text) < cfg.summarize_min_chars:
        # For short items, a minimal "summary" is just the title.
        it.summary = it.title.strip()
        return

    prompt = ITEM_SUMMARY_PROMPT.format(
        source=it.source,
        title=it.title,
        text=it.text[: cfg.summarize_max_chars],
    )

    raw = llm_client.complete(prompt)
    data = _safe_json_loads(raw)

    # Fallback if JSON fails
    if not isinstance(data, dict):
        it.summary = (it.title + " — " + it.text[:280]).strip()
        return

    # Fill
    summary = data.get("summary")
    if isinstance(summary, list):
        it.summary = "\n".join([str(x) for x in summary][:3])
    elif isinstance(summary, str):
        it.summary = summary.strip()
    else:
        it.summary = it.title.strip()

    key_facts = data.get("key_facts", [])
    if isinstance(key_facts, list):
        # store key facts as tags for now
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
# LLM Router step
# ---------------------------

def _llm_route_items(
    llm_client,
    items: List[Item],
    cfg: AgenticPipelineConfig
) -> Tuple[List[Item], Dict[str, Any]]:
    """
    Use ROUTER_LLM_PROMPT to select ids.
    Returns (selected_items, router_decision_dict).
    Falls back to deterministic if parsing fails.
    """
    # Build compact table for router
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
    raw = llm_client.complete(prompt)
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
# Synthesis step
# ---------------------------

def _synthesize_report(llm_client, bundle: Bundle, cfg: AgenticPipelineConfig) -> str:
    bundle_metrics_json = json.dumps(bundle.metrics, indent=2, default=str, ensure_ascii=False)

    payload = []
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

    items_json = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    if len(items_json) > cfg.synthesis_max_items_json_chars:
        items_json = items_json[: cfg.synthesis_max_items_json_chars] + "\n...truncated..."

    prompt = SYNTHESIS_PROMPT.format(
        ticker=bundle.ticker,
        bundle_metrics_json=bundle_metrics_json,
        items_json=items_json,
    )
    return llm_client.complete(prompt).strip()


# ---------------------------
# Main entry: run_agentic_pipeline
# ---------------------------

def run_agentic_pipeline(
    ticker: str,
    collectors: List[Any],
    llm_client: Any,
    cache: Any,
    cfg: AgenticPipelineConfig = AgenticPipelineConfig(),
) -> Dict[str, Any]:
    """
    End-to-end agentic pipeline runner.

    Returns dict:
      {
        "bundle_metrics": {...},
        "selected_items": [ ...auditable payload... ],
        "router_info": {...},
        "report": "markdown string"
      }
    """
    asof = datetime.now(timezone.utc)
    bundle = Bundle(ticker=ticker, asof=asof)

    # 1) Collect
    items: List[Item] = []
    for c in collectors:
        got = c.collect(ticker=ticker, asof=asof)
        if got:
            items.extend(got)
    bundle.items = items

    # 2) Score (fast)
    # Instantiate scorers
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

    # Compute quality now (used by router)
    for it in bundle.items:
        it.scores["quality"] = _compute_quality(it)

    # 3) Summarize long items (optional but recommended)
    for it in bundle.items:
        _summarize_item_if_needed(llm_client, it, cfg)

    # 4) Route
    per_source_caps = cfg.per_source_caps or {
        "reddit": 3,
        "x": 3,
        "reuters": 3,
        "bloomberg": 2,
        "ft": 2,
        "wsj": 2,
        "polymarket": 2,
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
            # fallback to deterministic
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

    # 6) Synthesis report
    report = _synthesize_report(llm_client, bundle, cfg)

    # 7) Output (auditable)
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
        })

    return {
        "bundle_metrics": bundle.metrics,
        "selected_items": selected_items_payload,
        "router_info": router_info,
        "report": report,
    }


# ---------------------------
# Minimal cache + LLM client examples (optional)
# ---------------------------

class SimpleDictCache:
    """
    Tiny in-memory cache (good for local dev). TTL is ignored for simplicity.
    Replace with Redis in production.
    """
    def __init__(self):
        self._d = {}

    def get(self, key: str):
        return self._d.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 0):
        self._d[key] = value


class ClaudeLLMClient:
    """
    Minimal wrapper that matches llm_client.complete(prompt)->str
    Requires `pip install anthropic`
    """
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 1200, temperature: float = 0.2):
        import anthropic  # type: ignore
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


"""
Example usage:

collectors = [
    NewsCollector(...),
    RedditCollector(...),
    SecFilingsCollector(...),
    EarningsTranscriptCollector(...),
    PolymarketCollector(...),
]

cache = SimpleDictCache()
llm = ClaudeLLMClient(api_key="YOUR_KEY")

res = run_agentic_pipeline(
    ticker="NVDA",
    collectors=collectors,
    llm_client=llm,
    cache=cache,
)

print(res["bundle_metrics"])
print(res["report"])
"""