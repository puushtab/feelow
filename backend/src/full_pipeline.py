"""
Full Polymarket Pipeline
========================

Connects the **agent-search** feature (`PolymarketPipeline`) with the
**polymarket-analysis** scoring feature (`process_polymarket_markets`)
into one unified function: `get_polymarket`.

Flow:
  1. PolymarketPipeline (Gemini LLM) searches Polymarket for markets
     related to a company, then scores each market's pertinence (0-100).
  2. The raw results are converted and fed into process_polymarket_markets
     which computes advanced metrics (momentum, volatility, concentration,
     composite signal, etc.), ranks markets, and produces a Claude-ready
     summary block.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Any

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_pipeline")

# ─── Dynamic imports for hyphenated directories ──────────────────────────────
# Python doesn't allow hyphens in package names, so we use importlib.

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))

# The workspace root contains polymarket_api.py which agent-search needs
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

# agent_search directory (contains polymarket_pipeline.py)
_AGENT_SEARCH_DIR = os.path.join(_SRC_DIR, "agent_search")
if _AGENT_SEARCH_DIR not in sys.path:
    sys.path.insert(0, _AGENT_SEARCH_DIR)


def _import_from_path(module_name: str, file_path: str):
    """Import a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import PolymarketPipeline from agent-search/polymarket_pipeline.py
_pipeline_mod = _import_from_path(
    "polymarket_pipeline",
    os.path.join(_AGENT_SEARCH_DIR, "polymarket_pipeline.py"),
)
PolymarketPipeline = _pipeline_mod.PolymarketPipeline

# Import process_polymarket_markets from polymarket-analysis/market_scorer.py (v2)
_score_mod = _import_from_path(
    "market_scorer",
    os.path.join(_SRC_DIR, "polymarket-analysis", "market_scorer.py"),
)
process_polymarket_markets = _score_mod.process_polymarket_markets


# ─── Format bridge ────────────────────────────────────────────────────────────

def _convert_price_history(market_dict: dict) -> dict:
    """
    Bridge between the two data formats:

    agent-search returns ``price_history``:
        [{"timestamp": epoch_sec, "price": float}, ...]

    score_polymarket expects ``history``:
        [{"t": epoch_sec, "p": float}, ...]
    """
    result = dict(market_dict)
    price_history = result.get("price_history", [])
    if isinstance(price_history, list):
        history = []
        skipped = 0
        for pt in price_history:
            if isinstance(pt, dict):
                t = pt.get("timestamp") or pt.get("t")
                p = pt.get("price") or pt.get("p")
                if t is not None and p is not None:
                    history.append({"t": t, "p": p})
                else:
                    skipped += 1
            else:
                skipped += 1
        result["history"] = history
        log.debug(
            "CONVERT    ▸ %r: %d price_history → %d history points (%d skipped)",
            market_dict.get("question", "?")[:50],
            len(price_history),
            len(history),
            skipped,
        )
    else:
        log.debug(
            "CONVERT    ▸ %r: price_history is not a list (%s), skipping",
            market_dict.get("question", "?")[:50],
            type(price_history).__name__,
        )
    return result


# ─── Main entry point ────────────────────────────────────────────────────────

def get_polymarket(
    company: str,
    date: str | None = None,
    gemini_api_key: str | None = None,
    max_queries: int = 1,
    limit_per_query: int = 10,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Full end-to-end Polymarket analysis for a company.

    **Step 1 — Agent Search** (`PolymarketPipeline`):
        Uses Gemini LLM with forced tool-calling to search Polymarket,
        then scores each market's pertinence (0–100).

    **Step 2 — Advanced Scoring** (`process_polymarket_markets`):
        Builds Market objects, computes momentum / volatility / concentration /
        composite signal, ranks by score × engagement, and generates a
        Claude-ready summary block.

    Args:
        company:          Company name (e.g. ``"NVIDIA"``).
        date:             Optional date context (e.g. ``"February 2026"``).
        gemini_api_key:   Gemini API key. Falls back to env vars
                          ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
        max_queries:      Number of varied search queries (1–3).
        limit_per_query:  Max markets returned per query.
        top_k:            Number of top markets in the final summary.

    Returns:
        dict with keys:
          - ``raw_markets``:          list of raw market dicts from agent-search
          - ``top_markets_summary``:  list of scored/analysed market summaries
          - ``corr_top2``:            Pearson correlation between top 2 markets
          - ``global_score``:         weighted global score
          - ``claude_block``:         text block for injecting into Claude prompts
    """
    # ── Resolve API key ──────────────────────────────────────────────────────
    api_key = (
        gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        log.error("API KEY    ▸ No Gemini API key found (param, GEMINI_API_KEY, GOOGLE_API_KEY)")
        raise ValueError(
            "No Gemini API key provided. "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY env var, or pass gemini_api_key."
        )
    log.info("API KEY    ▸ resolved (%s…%s)", api_key[:4], api_key[-4:])

    # ── Step 1: Agent search ─────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("PIPELINE   ▸ company=%r  date=%r  max_queries=%d  limit=%d  top_k=%d",
             company, date, max_queries, limit_per_query, top_k)
    log.info("═" * 60)
    log.info("")
    log.info("╔══ STEP 1: PolymarketPipeline (agent-search) ══╗")
    pipeline = PolymarketPipeline(
        api_key=api_key,
        max_queries=max_queries,
        limit_per_query=limit_per_query,
    )
    raw_markets = pipeline.run(company, date)
    log.info("╚══ STEP 1 DONE: %d raw markets returned ══╝", len(raw_markets))

    if not raw_markets:
        log.warning("PIPELINE   ▸ No markets found for %r — returning empty result.", company)
        return {
            "raw_markets": [],
            "top_markets_summary": [],
            "corr_top2": 0.0,
            "global_score": 0.0,
            "claude_block": "No Polymarket markets found for this query.",
        }

    # ── Convert formats & normalise pertinence ───────────────────────────────
    log.info("")
    log.info("╔══ FORMAT BRIDGE: converting price_history → history ══╗")
    markets_for_scoring: list[dict] = []
    for i, m in enumerate(raw_markets):
        converted = _convert_price_history(m)

        # PolymarketPipeline returns pertinence on 0–100 scale;
        # score_polymarket.Market expects 0–1.
        pert = converted.get("pertinence_score", 50)
        pert_normalised = float(pert) / 100.0 if float(pert) > 1.0 else float(pert)
        converted["pertinence_score"] = pert_normalised
        log.info(
            "  [%d/%d] %s  pertinence: %s → %.4f  history_pts: %d",
            i + 1,
            len(raw_markets),
            m.get("question", "?")[:50],
            pert,
            pert_normalised,
            len(converted.get("history", [])),
        )
        markets_for_scoring.append(converted)
    log.info("╚══ FORMAT BRIDGE DONE: %d markets ready for scoring ══╝", len(markets_for_scoring))

    # ── Step 2: Advanced scoring ─────────────────────────────────────────────
    log.info("")
    log.info("╔══ STEP 2: process_polymarket_markets (score_polymarket) ══╗")
    scored = process_polymarket_markets(markets_for_scoring, top_k=top_k)
    log.info("╚══ STEP 2 DONE ══╝")

    # ── Convert Market objects to summary dicts ──────────────────────────────
    top_markets_summary = []
    for m in scored["top_markets"]:
        top_markets_summary.append({
            "question": m.question,
            "url": m.url,
            "score": m.score,
            "engagement": m.engagement,
            "metrics": m.metrics,
            "advanced": m.advanced,
        })

    # ── Build Claude block ───────────────────────────────────────────────────
    claude_lines = ["## Polymarket correlated markets\n"]
    for i, m in enumerate(scored["top_markets"], 1):
        adv = m.advanced
        p_last = adv.get("p_last")
        entropy = adv.get("entropy_nats")
        slope_recent = adv.get("slope_recent_per_day")
        tv = adv.get("total_variation", 0)
        mj = adv.get("max_jump", 0)
        stale = adv.get("staleness_ratio_0_5", 0)
        tte = adv.get("time_to_event_days")
        comp = adv.get("composite_signal")
        hq = adv.get("history_quality")

        p_pct = f"{(p_last or 0)*100:.2f}%" if p_last is not None else "N/A"
        bias = "strong YES bias" if (p_last or 0) > 0.8 else ("strong NO bias" if (p_last or 0) < 0.2 else "mixed")
        consensus = "very high consensus" if (entropy or 999) < 0.1 else ("high consensus" if (entropy or 999) < 0.3 else "moderate")

        claude_lines.append(f"### Market #{i}")
        claude_lines.append(f"Polymarket market: {m.question}")
        claude_lines.append(f"Implied YES probability (latest): {p_pct} ({bias}).")
        claude_lines.append(f"Consensus: {consensus} (entropy={entropy or 0:.3f} nats).")
        claude_lines.append(
            f"History dynamics: slope_recent={slope_recent or 0:.5f} prob/day | "
            f"total_variation={tv:.4f} | max_jump={mj:.4f} | staleness@0.5={stale or 0:.3f}."
        )
        claude_lines.append(
            f"Reliability proxies: pertinence={m.metrics.get('pertinence', 0):.2f} | "
            f"liquidity(now)={m.liquidity:,.2f} | volume(now)={m.volume:,.2f}."
        )
        if tte is not None:
            claude_lines.append(f"Time to resolution: {tte:.2f} days.")
        claude_lines.append(f"Composite signal: {comp or 0:.3f} ({'weak/neutral' if abs(comp or 0) < 0.05 else 'notable'})")
        claude_lines.append(f"URL: {m.url}\n")

    claude_block = "\n".join(claude_lines)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══ FINAL RESULT ══╗")
    log.info("  raw_markets:          %d", len(raw_markets))
    log.info("  top_markets_summary:  %d", len(top_markets_summary))
    log.info("  corr_top2:            %.4f", scored["corr_top2"])
    log.info("  global_score:         %.4f", scored["global_score"])
    for i, s in enumerate(top_markets_summary, 1):
        log.info("  #%d  score=%.4f  eng=%.4f  %s",
                 i, s["score"], s["engagement"], s["question"][:50])
    log.info("╚══ PIPELINE COMPLETE ══╝")

    return {
        "raw_markets": raw_markets,
        "top_markets_summary": top_markets_summary,
        "corr_top2": scored["corr_top2"],
        "global_score": scored["global_score"],
        "claude_block": claude_block,
    }
