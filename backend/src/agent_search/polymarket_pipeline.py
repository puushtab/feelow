"""
Polymarket Intelligence Pipeline
=================================

End-to-end pipeline that:
  0. Defines a FastMCP server with `search_with_history` tool (typed schema)
  1. Builds a natural query from a company name + optional date
  2. Uses Gemini LLM with FORCED tool calling to search Polymarket
     (up to max_queries varied search angles per run)
  3. Scores each market's pertinence (0–100) via structured LLM output
  4. Returns all markets with: metadata + price_history + pertinence_score

Usage:
    from polymarket_pipeline import PolymarketPipeline

    pipeline = PolymarketPipeline(api_key="...", max_queries=2)
    results  = pipeline.run("NVIDIA", date="February 2026")

    for m in results:
        print(m["question"], m["pertinence_score"], len(m["price_history"]))

CLI:
    GEMINI_API_KEY=... python polymarket_pipeline.py NVIDIA --max-queries 2
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai.types import (
    Content,
    Part,
    GenerateContentConfig,
    FunctionDeclaration,
    Tool as GeminiTool,
)

from polymarket_api import PolymarketAPI

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
#  0. FastMCP Server — Tool Definition (source of truth for schema & typing)
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP("polymarket-search")
_api = PolymarketAPI()


@mcp.tool()
def search_with_history(
    query: Annotated[str, Field(description="Search keywords for Polymarket prediction markets (e.g. 'NVIDIA stock')")],
    limit: Annotated[int, Field(description="Maximum number of markets to return", ge=1, le=50)] = 10,
    interval: Annotated[
        Literal["all", "1d", "1w", "1m"],
        Field(description="Price history time window"),
    ] = "all",
    include_closed: Annotated[bool, Field(description="Include resolved/closed markets")] = False,
) -> list[dict]:
    """
    Search Polymarket prediction markets by keyword and return each result
    with its full price history attached.
    """
    results = _api.search_markets_with_history(
        query,
        limit=limit,
        interval=interval,
        include_closed=include_closed,
    )
    return [m.to_dict() for m in results]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Pydantic Models — Imposed LLM Output Format for Scoring
# ═══════════════════════════════════════════════════════════════════════════════

class MarketScore(BaseModel):
    """Pertinence score for one market."""
    id: str = Field(description="The market ID")
    score: int = Field(ge=0, le=100, description="Pertinence score 0 (irrelevant) to 100 (perfectly relevant)")


class ScoringResult(BaseModel):
    """Batch pertinence scores — returned by the LLM."""
    scores: list[MarketScore] = Field(description="One score entry per market, covering every market provided")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Gemini Tool Declaration (mirrors the FastMCP tool schema above)
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_TOOL_DECL = GeminiTool(
    function_declarations=[
        FunctionDeclaration(
            name="search_with_history",
            description=(
                "Search Polymarket prediction markets by keyword and return "
                "each result with its full price history attached."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "Search keywords (e.g. 'NVIDIA stock price')",
                    },
                    "limit": {
                        "type": "INTEGER",
                        "description": "Max markets to return (1–50)",
                    },
                    "interval": {
                        "type": "STRING",
                        "description": "Price history window",
                        "enum": ["all", "1d", "1w", "1m"],
                    },
                    "include_closed": {
                        "type": "BOOLEAN",
                        "description": "Include resolved/closed markets",
                    },
                },
                "required": ["query"],
            },
        )
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class PolymarketPipeline:
    """
    Full pipeline:
      1. Build natural query from company name
      2. Gemini calls search_with_history N times (forced, varied angles)
      3. Gemini scores pertinence of every collected market (structured output)
      4. Return enriched market list sorted by pertinence
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        scoring_model: str = "gemini-2.5-flash",
        max_queries: int = 1,
        limit_per_query: int = 10,
        use_llm_search: bool = False,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.scoring_model = scoring_model
        self.max_queries = min(max_queries, 3)
        self.limit_per_query = limit_per_query
        self.use_llm_search = use_llm_search

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _execute_tool(name: str, args: dict) -> list[dict]:
        """Execute a tool call locally (same logic as the FastMCP server)."""
        if name != "search_with_history":
            raise ValueError(f"Unknown tool: {name}")
        # Sanitise types (Gemini may send floats for ints)
        clean_args = {
            "query": str(args.get("query", "")),
            "limit": int(args.get("limit", 10)),
            "interval": str(args.get("interval", "all")),
            "include_closed": bool(args.get("include_closed", False)),
        }
        log.info("TOOL CALL  ▸ %s(%s)", name, json.dumps(clean_args))
        t0 = time.time()
        results = search_with_history(**clean_args)
        elapsed = time.time() - t0
        log.info("TOOL RESULT▸ %d markets returned in %.1fs", len(results), elapsed)
        for r in results:
            log.debug("  • [%s] %s  (vol=$%s, yes=%.2f%%)",
                      r.get('id','?')[:8], r.get('question','?')[:60],
                      f"{r.get('volume',0):,.0f}",
                      (r.get('yes_price',0) or 0) * 100)
        return results

    @staticmethod
    def _build_query(company: str, date: str | None = None) -> str:
        q = f"{company} stock"
        log.info("QUERY BUILT▸ %r", q)
        return q

    # ── Step 1+2: LLM-driven search with forced tool calls ───────────────────

    def _search_step(self, company: str, date: str | None) -> list[dict]:
        natural_query = self._build_query(company, date)

        system = (
            f"You are a Polymarket research assistant.\n"
            f"Your goal: find ALL prediction markets that could impact the stock price of «{company}».\n"
            f"You have ONE tool: search_with_history. You MUST call it.\n"
            f"Search broadly: the company name, its stock ticker, earnings, revenue, "
            f"market cap, its sector/industry (e.g. AI, tech, EV, semiconductors), "
            f"key competitors, regulation, tariffs, or any macro event that could move the stock.\n"
            f"Use SHORT, simple keywords (1-3 words). Do NOT add dates or long phrases.\n"
            f"Set limit={self.limit_per_query} and interval=\"all\" for full history."
        )
        log.info("SEARCH STEP▸ system_prompt=%d chars", len(system))

        config = GenerateContentConfig(
            system_instruction=system,
            tools=[SEARCH_TOOL_DECL],
            temperature=0.7,
            max_output_tokens=1024,
            # Force the LLM to call the tool — it cannot reply with text only
            tool_config={"function_calling_config": {"mode": "ANY"}},
        )
        log.info("SEARCH STEP▸ config: model=%s, temperature=%.1f, tool_mode=ANY (forced)",
                 self.model, 0.7)

        all_markets: list[dict] = []
        seen_ids: set[str] = set()
        searched_queries: list[str] = []

        for i in range(self.max_queries):
            log.info("─" * 50)
            log.info("SEARCH ITER▸ %d/%d", i + 1, self.max_queries)

            # Build iteration prompt
            if searched_queries:
                prompt = (
                    f"Already searched: {searched_queries}.\n"
                    f"Now search with DIFFERENT keywords: try the sector, competitors, "
                    f"regulation, tariffs, or macro events related to {company}. "
                    f"Use 1-3 word queries only."
                )
            else:
                prompt = (
                    f"Search for Polymarket prediction markets related to {company}. "
                    f"Use the company name as query."
                )
            log.info("LLM INPUT  ▸ %s", prompt)

            t0 = time.time()
            response = self.client.models.generate_content(
                model=self.model,
                contents=[Content(role="user", parts=[Part.from_text(text=prompt)])],
                config=config,
            )
            llm_elapsed = time.time() - t0
            log.info("LLM OUTPUT ▸ received in %.1fs", llm_elapsed)

            # Extract function call from response
            called = False
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        args = dict(fc.args)
                        query_used = args.get("query", "")
                        searched_queries.append(query_used)
                        log.info("LLM CALL   ▸ %s(%s)", fc.name, json.dumps(args))

                        try:
                            results = self._execute_tool(fc.name, args)
                            new = 0
                            for m in results:
                                mid = m.get("id", "")
                                if mid and mid not in seen_ids:
                                    seen_ids.add(mid)
                                    all_markets.append(m)
                                    new += 1
                            log.info("DEDUP      ▸ %d returned, %d new, %d total unique",
                                     len(results), new, len(all_markets))
                        except Exception as e:
                            log.error("TOOL ERROR ▸ %s", e)

                        called = True
                        break

            if not called:
                log.warning("NO CALL    ▸ LLM did not produce a function_call on iter %d", i + 1)

        log.info("SEARCH DONE▸ %d unique markets from %d queries",
                 len(all_markets), len(searched_queries))
        return all_markets

    # ── Direct search (no LLM) ────────────────────────────────────────────────

    def _search_step_direct(self, company: str) -> list[dict]:
        """
        Search Polymarket directly with two fixed queries:
          1. "{company}"
          2. "{company} stock"
        No LLM involved — faster and cheaper.
        """
        queries = [company, f"{company} stock"]
        all_markets: list[dict] = []
        seen_ids: set[str] = set()

        for q in queries:
            log.info("DIRECT SEARCH▸ query=%r  limit=%d", q, self.limit_per_query)
            t0 = time.time()
            try:
                results = search_with_history(
                    query=q,
                    limit=self.limit_per_query,
                    interval="all",
                    include_closed=False,
                )
            except Exception as e:
                log.error("DIRECT SEARCH ERROR▸ %s", e)
                continue
            elapsed = time.time() - t0
            new = 0
            for m in results:
                mid = m.get("id", "")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_markets.append(m)
                    new += 1
            log.info("DIRECT SEARCH▸ %d returned, %d new, %d total (%.1fs)",
                     len(results), new, len(all_markets), elapsed)

        log.info("DIRECT SEARCH DONE▸ %d unique markets from %d queries",
                 len(all_markets), len(queries))
        return all_markets

    # ── Step 3: Batch pertinence scoring (structured output) ──────────────────

    def _score_step(
        self, markets: list[dict], company: str, date: str | None
    ) -> dict[str, int]:
        """
        Give ALL market titles + IDs to the LLM in one call.
        Returns a {market_id: score} mapping.
        """
        if not markets:
            log.warning("SCORE SKIP ▸ no markets to score")
            return {}

        natural_query = self._build_query(company, date)

        # Build compact list of (id, title) for the prompt
        entries = [
            {"id": m["id"], "title": m["question"]}
            for m in markets
        ]
        log.info("SCORE INPUT▸ %d markets to score against query %r", len(entries), natural_query)
        for e in entries:
            log.info("  • id=%s  title=%s", e["id"][:10], e["title"][:60])

        prompt = (
            f"You are evaluating prediction markets for pertinence.\n\n"
            f"Original query: «{natural_query}»\n\n"
            f"Score each market from 0 to 100:\n"
            f"  100 = directly about the stock price / valuation / financial performance of {company}\n"
            f"  70–99 = strongly related (market cap, revenue, rankings involving {company})\n"
            f"  30–69 = moderately related (industry trends, competitors, sector events)\n"
            f"  1–29 = weakly related (tangentially mentions {company} or its sector)\n"
            f"  0 = completely irrelevant\n\n"
            f"Markets to score:\n{json.dumps(entries, indent=2)}\n\n"
            f"Return a score for EVERY market. Do not skip any."
        )
        log.info("SCORE LLM  ▸ prompt=%d chars, model=%s, temp=0.2, response=JSON (imposed schema)",
                 len(prompt), self.scoring_model)
        log.info("SCORE LLM  ▸ imposed schema: %s", ScoringResult.model_json_schema())

        config = GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ScoringResult.model_json_schema(),
            temperature=0.2,
            max_output_tokens=4096,
        )

        t0 = time.time()
        response = self.client.models.generate_content(
            model=self.scoring_model,
            contents=[Content(role="user", parts=[Part.from_text(text=prompt)])],
            config=config,
        )
        elapsed = time.time() - t0
        log.info("SCORE RAW  ▸ LLM responded in %.1fs: %s", elapsed, response.text[:500])

        try:
            result = ScoringResult.model_validate_json(response.text)
            score_map = {s.id: s.score for s in result.scores}
            log.info("SCORE OUT  ▸ parsed %d scores via Pydantic:", len(score_map))
            for s in result.scores:
                log.info("  • id=%s → score=%d", s.id[:10], s.score)
            return score_map
        except Exception as e:
            log.error("SCORE PARSE▸ Pydantic validation failed: %s", e)
            # Fallback: try raw JSON
            try:
                raw = json.loads(response.text)
                score_map = {s["id"]: s["score"] for s in raw.get("scores", [])}
                log.warning("SCORE FALL ▸ used raw JSON fallback, got %d scores", len(score_map))
                return score_map
            except Exception:
                log.error("SCORE FAIL ▸ could not parse scores at all")
                return {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, company: str, date: str | None = None) -> list[dict]:
        """
        Full pipeline: search → score → return enriched markets.

        Args:
            company: Company name (e.g. "NVIDIA")
            date:    Optional date context (e.g. "February 2026")

        Returns:
            List of market dicts, each containing:
              - All standard fields (question, volume, liquidity, outcomes…)
              - price_history: list of {timestamp, price}
              - pertinence_score: int 0–100
            Sorted by pertinence_score descending.
        """
        t_start = time.time()
        log.info("═" * 60)
        log.info("PIPELINE   ▸ company=%r  date=%r", company, date)
        log.info("PIPELINE   ▸ model=%s  scoring_model=%s  max_queries=%d  limit=%d",
                 self.model, self.scoring_model, self.max_queries, self.limit_per_query)
        log.info("═" * 60)

        # ── Step 1+2: Search ──────────────────────────────────────────────────
        log.info("")
        if self.use_llm_search:
            log.info("╔══ STEP 1+2: SEARCH (LLM → tool calls → Polymarket API) ══╗")
            t0 = time.time()
            markets = self._search_step(company, date)
        else:
            log.info("╔══ STEP 1+2: SEARCH (DIRECT — no LLM) ══╗")
            t0 = time.time()
            markets = self._search_step_direct(company)
        log.info("╚══ SEARCH COMPLETE: %d unique markets in %.1fs ══╝",
                 len(markets), time.time() - t0)

        if not markets:
            log.warning("PIPELINE   ▸ No markets found. Aborting.")
            return []

        # ── Step 3: Score ─────────────────────────────────────────────────────
        log.info("")
        log.info("╔══ STEP 3: SCORING (LLM structured output → pertinence) ══╗")
        t0 = time.time()
        score_map = self._score_step(markets, company, date)
        log.info("╚══ SCORING COMPLETE: %d/%d scored in %.1fs ══╝",
                 len(score_map), len(markets), time.time() - t0)

        # ── Step 4: Merge & sort ──────────────────────────────────────────────
        log.info("")
        log.info("╔══ STEP 4: MERGE + SORT ══╗")
        for m in markets:
            m["pertinence_score"] = score_map.get(m["id"], 0)

        markets.sort(key=lambda m: m["pertinence_score"], reverse=True)
        log.info("MERGE      ▸ attached scores and sorted by pertinence (desc)")

        # ── Summary ───────────────────────────────────────────────────────────
        log.info("")
        log.info("╔══ FINAL OUTPUT ══╗")
        for m in markets:
            score = m["pertinence_score"]
            ph = m.get("price_history", [])
            ph_len = len(ph) if isinstance(ph, list) else "dict"
            bar = "█" * (score // 5) + "░" * (20 - score // 5)
            log.info("  [%3d] %s  %s", score, bar, m['question'][:55])
            log.info("        history=%s pts  vol=$%s", ph_len, f"{m.get('volume', 0):,.0f}")

        total = time.time() - t_start
        log.info("╚══ PIPELINE DONE in %.1fs — %d markets returned ══╝", total, len(markets))
        return markets


# ═══════════════════════════════════════════════════════════════════════════════
#  4. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Polymarket Intelligence Pipeline")
    p.add_argument("company", help="Company name (e.g. NVIDIA)")
    p.add_argument("--date", default=None, help="Date context (e.g. 'February 2026')")
    p.add_argument("--max-queries", type=int, default=1, help="Search queries 1–3")
    p.add_argument("--limit", type=int, default=10, help="Results per query")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model for search")
    p.add_argument("--scoring-model", default="gemini-2.5-flash", help="Gemini model for scoring")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        exit(1)

    pipeline = PolymarketPipeline(
        api_key=api_key,
        model=args.model,
        scoring_model=args.scoring_model,
        max_queries=args.max_queries,
        limit_per_query=args.limit,
    )

    results = pipeline.run(args.company, args.date)

    if args.json:
        # Strip price_history for compact JSON output
        for r in results:
            ph = r.get("price_history", [])
            r["price_history_length"] = len(ph) if isinstance(ph, list) else "dict"
        print(json.dumps(results, indent=2, default=str))
