ITEM_SUMMARY_PROMPT = """\
You are a financial research assistant.
Summarize the item for trading relevance.

Return JSON with:
- summary: <= 3 bullet points
- key_facts: list of short factual statements
- stance: one of ["bullish","bearish","mixed","neutral"]
- confidence: 0..1
- horizon: one of ["intraday","1-3d","1-4w","long_term"]
- why_market_moves: 1 sentence explaining mechanism

Item:
SOURCE: {source}
TITLE: {title}
TEXT: {text}
"""

ROUTER_LLM_PROMPT = """\
You are routing evidence for a stock report under a strict context budget.

Goal: select the most decision-relevant, non-duplicative items to explain likely price movement.

Return JSON:
{
  "selected_ids": [...],
  "dropped_as_duplicates": [...],
  "notes": "short"
}

Rules:
- Prefer higher reliability sources (SEC filings, earnings call, Reuters/FT/WSJ/Bloomberg).
- Prefer forward-looking, specific, market-moving information.
- Ensure diversity across sources (not all social posts).
- Keep at most {max_items} items.

Items (each has id, source, ts, title, quality, short_summary):
{items_table}
"""

SYNTHESIS_PROMPT = """\
You are an institutional-grade equity analyst.

You must:
1) Produce a clear directional view (bullish/bearish/neutral) for {ticker}
2) Provide an intensity score 0-100
3) Provide a confidence score 0-100 that reflects evidence quality
4) Explain top drivers and key risks
5) Explicitly mention contradictions/disagreements in sources
6) Provide a "Reliable Source Score" 0-100 based on evidence quality and diversity
7) Output MUST be grounded only in the provided items and metrics. Do not invent facts.

Bundle metrics:
{bundle_metrics_json}

Selected evidence items:
For each item:
- id, source, ts, title, summary, key_facts, stance, confidence, horizon, url

{items_json}

Output format (markdown):
## Signal
- Direction:
- Intensity (0-100):
- Confidence (0-100):
- Reliable Source Score (0-100):

## Key Drivers (Top 3)
1) ...
2) ...
3) ...

## Risks / What could invalidate this
- ...

## Evidence Map
- Bullish evidence: [id, id, ...]
- Bearish evidence: [id, id, ...]
- Mixed/neutral: [id, ...]

## Actionable view (not financial advice)
- If bullish: what to watch + key levels / events
- If bearish: what to watch + key levels / events

## Sources used
- List each item with id + url
"""