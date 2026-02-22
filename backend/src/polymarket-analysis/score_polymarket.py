import math
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple


# ----------------------------
# HISTORY HELPERS
# ----------------------------

def _clip01(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def _history_arrays(history: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert history list [{'t': epoch_seconds, 'p': prob_yes}, ...]
    into (t_seconds, p) arrays sorted by time.
    """
    if not isinstance(history, list) or len(history) == 0:
        return np.array([]), np.array([])

    t_list, p_list = [], []
    for row in history:
        if not isinstance(row, dict):
            continue
        t = row.get("t")
        p = row.get("p")
        try:
            tf = float(t)
            pf = float(p)
        except (TypeError, ValueError):
            continue
        t_list.append(tf)
        p_list.append(_clip01(pf))

    if not t_list:
        return np.array([]), np.array([])

    t_arr = np.array(t_list, dtype=float)
    p_arr = np.array(p_list, dtype=float)
    idx = np.argsort(t_arr)
    return t_arr[idx], p_arr[idx]


def _ols_slope_per_day(t: np.ndarray, p: np.ndarray) -> Optional[float]:
    """
    Least squares slope of p vs time (time in days), returns slope per day.
    """
    if len(t) < 3:
        return None
    x = (t - t.mean()) / 86400.0  # seconds -> days, centered
    y = p - p.mean()
    denom = float((x * x).sum())
    if denom <= 1e-12:
        return None
    return float((x * y).sum() / denom)


def _count_changes(p: np.ndarray, eps: float = 1e-12) -> int:
    if len(p) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(p)) > eps))


def _staleness_ratio(p: np.ndarray, value: float = 0.5, tol: float = 1e-6) -> Optional[float]:
    if len(p) == 0:
        return None
    return float(np.mean(np.abs(p - value) <= tol))


def _time_since_last_change(t: np.ndarray, p: np.ndarray, threshold: float = 0.005) -> Optional[float]:
    """
    Seconds since last |Δp| >= threshold.
    If never changed that much, returns total span.
    """
    if len(t) < 2:
        return None
    dp = np.abs(np.diff(p))
    idx = np.where(dp >= threshold)[0]
    if len(idx) == 0:
        return float(t[-1] - t[0])
    last_i = int(idx[-1] + 1)
    return float(t[-1] - t[last_i])


def _parse_iso_utc(dt_str: str) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _binary_entropy(p: float, eps: float = 1e-12) -> float:
    p = _clip01(p, eps=eps)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ----------------------------
# MARKET CLASS
# ----------------------------
class Market:
    """
    Represents a Polymarket event/market with outcomes, historical prices, and other features.
    """

    def __init__(self, data: Dict[str, Any]):
        self.question = data["question"]
        self.event_title = data.get("event_title", "")
        self.active = data.get("active", True)
        self.closed = data.get("closed", False)
        self.end_date = data.get("end_date", None)

        self.volume = data.get("volume", 0.0)          # snapshot volume (no history)
        self.liquidity = data.get("liquidity", 0.0)    # snapshot liquidity
        self.outcomes = data.get("outcomes", [])
        self.outcome_prices = data.get("outcome_prices", [])
        self.url = data.get("url", "")
        self.history = data.get("history", [])
        self.pertinence_score = data.get("pertinence_score", 0.5)

        self.metrics: Dict[str, float] = {}
        self.advanced: Dict[str, Any] = {}  # advanced metrics (history quality, slope, etc.)
        self.score = 0.0
        self.engagement = 0.0

    # ----------------------------
    # SIMPLE METRICS (existing)
    # ----------------------------
    def compute_metrics(self) -> Dict[str, float]:
        """
        Computes all relevant metrics for the market, including:
        - momentum: measures recent trend in price history
        - volatility: range of price changes in history
        - concentration: how skewed the probabilities are (entropy inverse)
        - volume: normalized trade volume
        - liquidity: normalized available liquidity
        - pertinence: external relevance score (0-1)
        Returns a dictionary of normalized metrics (0-1).
        """
        momentum = self.compute_momentum()
        volatility = self.compute_volatility()
        concentration = self.compute_concentration()

        # Normalize volume and liquidity to 0-1 range using log
        volume_score = math.log(float(self.volume) + 1.0) / 12.0
        liquidity_score = math.log(float(self.liquidity) + 1.0) / 10.0

        # Include external pertinence score if available
        pertinence = float(self.pertinence_score)

        self.metrics = {
            "momentum": float(max(0, min(momentum, 1))),
            "volatility": float(max(0, min(volatility, 1))),
            "concentration": float(max(0, min(concentration, 1))),
            "volume": float(max(0, min(volume_score, 1))),
            "liquidity": float(max(0, min(liquidity_score, 1))),
            "pertinence": float(max(0, min(pertinence, 1))),
        }
        return self.metrics

    def compute_momentum(self) -> float:
        """
        Computes a momentum metric based on the historical prices.
        Momentum measures how far the last price has moved from the first.
        If no history exists, fallback to a simple estimate using outcome_prices.
        Returns a normalized value between 0 and 1.
        """
        if not self.history or len(self.history) < 2:
            # fallback using snapshot price (YES vs 0.5)
            if len(self.outcome_prices) >= 1:
                p_yes = float(self.outcome_prices[0])
                # your previous fallback returned ~"closeness to 0.5"; keep behavior
                return 1 - abs(p_yes - 0.5) * 2
            return 0.5

        _, p = _history_arrays(self.history)
        if len(p) < 2:
            return 0.5
        return float(max(0.0, min(abs(p[-1] - p[0]), 1.0)))

    def compute_volatility(self) -> float:
        """
        Computes volatility based on historical prices.
        Volatility is defined as the difference between max and min price observed in history.
        If no history exists, fallback to max-min of outcome_prices.
        Returns a normalized value between 0 and 1.
        """
        if not self.history:
            if self.outcome_prices:
                return float(max(self.outcome_prices) - min(self.outcome_prices))
            return 0.0
        _, p = _history_arrays(self.history)
        if len(p) == 0:
            return 0.0
        return float(p.max() - p.min())

    def compute_concentration(self) -> float:
        """
        Computes concentration as the inverse of entropy of outcome probabilities.
        High concentration → most of the probability mass on a single outcome.
        Low concentration → probability evenly spread across outcomes.
        Returns a normalized value between 0 and 1.
        """
        prices = self.outcome_prices or []
        if len(prices) <= 1:
            return 0.0
        entropy = 0.0
        for p in prices:
            if p and p > 0:
                entropy -= float(p) * math.log(float(p))
        max_entropy = math.log(len(prices))
        return float(1.0 - (entropy / max_entropy))

    # ----------------------------
    # ADVANCED METRICS (NEW)
    # ----------------------------
    def compute_advanced_metrics(
        self,
        recent_points: int = 20,
        jump_threshold: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Computes more sophisticated history-based metrics without needing volume history.

        Produces:
          - p_first, p_last, net_change
          - slope_per_day, slope_recent_per_day
          - total_variation, max_jump, vol_dp
          - staleness_ratio_0_5, change_count, time_since_last_reprice_sec
          - history_points, history_span_hours
          - entropy_nats, time_to_event_days
          - history_quality (0..1)
          - composite_signal (-1..1): direction+strength scaled by pertinence, liquidity, history quality, staleness
        """
        t, p = _history_arrays(self.history)

        n = int(len(p))
        span_sec = float(t[-1] - t[0]) if n >= 2 else None
        span_hours = (span_sec / 3600.0) if span_sec is not None else None

        p_first = float(p[0]) if n else None
        p_last = float(p[-1]) if n else (float(self.outcome_prices[0]) if self.outcome_prices else None)

        dp = np.diff(p) if n >= 2 else np.array([])
        abs_dp = np.abs(dp) if dp.size else np.array([])

        net_change = (p_last - p_first) if (p_last is not None and p_first is not None) else None
        total_variation = float(abs_dp.sum()) if abs_dp.size else 0.0
        max_jump = float(abs_dp.max()) if abs_dp.size else 0.0
        vol_dp = float(dp.std(ddof=0)) if dp.size else 0.0

        slope_all = _ols_slope_per_day(t, p) if n >= 3 else None
        if n >= recent_points:
            slope_recent = _ols_slope_per_day(t[-recent_points:], p[-recent_points:])
        else:
            slope_recent = slope_all

        change_count = _count_changes(p)
        stale = _staleness_ratio(p)  # fraction at ~0.5
        time_since_reprice = _time_since_last_change(t, p, threshold=jump_threshold)

        entropy_nats = _binary_entropy(p_last) if p_last is not None else None

        # time to event
        tte_days = None
        if self.end_date:
            end_dt = _parse_iso_utc(self.end_date)
            if end_dt is not None:
                tte_days = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0

        # history quality proxy
        history_quality = None
        if n > 0:
            q_points = math.tanh(n / 50.0)                         # saturates
            q_span = math.tanh((span_hours or 0.0) / 24.0)         # saturates after ~1 day
            q_stale = 1.0 - (stale or 0.0)                         # penalize stuck @0.5
            history_quality = float(max(0.0, min(1.0, 0.45*q_points + 0.35*q_span + 0.20*q_stale)))

        # composite signal in [-1,1]
        composite_signal = None
        if slope_recent is not None:
            # reliability: pertinence * history_quality * liquidity_factor * (1 - staleness)
            pertinence = float(max(0.0, min(1.0, self.pertinence_score)))
            liq = max(float(self.liquidity or 0.0), 0.0)
            liquidity_factor = math.tanh(math.sqrt(liq) / 150.0)   # tweak constant as needed
            stale_factor = 1.0 - (stale or 0.0)

            rel = pertinence * (history_quality if history_quality is not None else 1.0) * liquidity_factor * stale_factor

            # intensity uses both slope magnitude and how much repricing happened
            intensity = math.tanh(20.0 * abs(slope_recent)) * math.tanh(5.0 * total_variation)
            composite_signal = float(rel * math.copysign(intensity, slope_recent))

        self.advanced = {
            "history_points": n,
            "history_span_hours": span_hours,
            "p_first": p_first,
            "p_last": p_last,
            "net_change": net_change,
            "total_variation": total_variation,
            "max_jump": max_jump,
            "vol_dp": vol_dp,
            "change_count": change_count,
            "staleness_ratio_0_5": stale,
            "time_since_last_reprice_sec": time_since_reprice,
            "slope_per_day": slope_all,
            "slope_recent_per_day": slope_recent,
            "entropy_nats": entropy_nats,
            "time_to_event_days": tte_days,
            "history_quality": history_quality,
            "composite_signal": composite_signal,
        }
        return self.advanced

    # ----------------------------
    # FINAL SCORE (unchanged but you can add advanced term)
    # ----------------------------
    def compute_score(self) -> float:
        """
        Computes a final score for the market using weighted combination of metrics.
        Weights can be adjusted according to importance of each metric.
        Returns a normalized score between 0 and 1.
        """
        if not self.metrics:
            self.compute_metrics()

        weights = {
            "momentum": 0.25,
            "volatility": 0.15,
            "concentration": 0.2,
            "volume": 0.1,
            "liquidity": 0.1,
            "pertinence": 0.2,
        }
        self.score = sum(self.metrics[k] * weights[k] for k in weights)
        self.score = float(max(0.0, min(self.score, 1.0)))
        return self.score

    def compute_engagement(self) -> float:
        """
        Computes engagement score to reflect market activity/interest.
        Weighted combination of volume, concentration, and momentum.
        Returns a normalized score between 0 and 1.
        """
        if not self.metrics:
            self.compute_metrics()
        self.engagement = float(
            0.4 * self.metrics["volume"] +
            0.3 * self.metrics["concentration"] +
            0.3 * self.metrics["momentum"]
        )
        return self.engagement

    # ----------------------------
    # NEW: SUMMARY TEXT FOR CLAUDE
    # ----------------------------
    def polymarket_summary_text(self) -> str:
        """
        Returns a concise text block describing the market snapshot + history dynamics.
        Good for injecting into your Claude report prompt.
        """
        if not self.advanced:
            self.compute_advanced_metrics()

        p_last = self.advanced.get("p_last", None)
        entropy = self.advanced.get("entropy_nats", None)
        slope = self.advanced.get("slope_recent_per_day", None)
        tv = self.advanced.get("total_variation", None)
        mj = self.advanced.get("max_jump", None)
        stale = self.advanced.get("staleness_ratio_0_5", None)
        tte = self.advanced.get("time_to_event_days", None)
        comp = self.advanced.get("composite_signal", None)

        def pct(x):
            return "N/A" if x is None else f"{100.0*float(x):.2f}%"

        def num(x, nd=4):
            return "N/A" if x is None else f"{float(x):.{nd}f}"

        def money(x):
            if x is None:
                return "N/A"
            x = float(x)
            if x >= 1_000_000:
                return f"{x/1_000_000:.2f}M"
            if x >= 1_000:
                return f"{x/1_000:.2f}k"
            return f"{x:.2f}"

        def consensus_label(h):
            if h is None:
                return "unknown"
            if h < 0.20:
                return "very high consensus"
            if h < 0.45:
                return "moderate consensus"
            if h < 0.62:
                return "low consensus"
            return "high uncertainty"

        def direction_label(p):
            if p is None:
                return "N/A"
            if p >= 0.65:
                return "strong YES bias"
            if p >= 0.55:
                return "slight YES bias"
            if p <= 0.35:
                return "strong NO bias"
            if p <= 0.45:
                return "slight NO bias"
            return "near 50/50"

        comp_line = "Composite signal: N/A"
        if isinstance(comp, (int, float)):
            if comp > 0.15:
                comp_line = f"Composite signal: +{num(comp, 3)} (bullish pressure proxy)"
            elif comp < -0.15:
                comp_line = f"Composite signal: {num(comp, 3)} (bearish pressure proxy)"
            else:
                comp_line = f"Composite signal: {num(comp, 3)} (weak/neutral)"

        lines = [
            f"Polymarket market: {self.question}",
            f"Implied YES probability (latest): {pct(p_last)} ({direction_label(p_last)}).",
            f"Consensus: {consensus_label(entropy)} (entropy={num(entropy, 3)} nats).",
            f"History dynamics: slope_recent={num(slope, 5)} prob/day | total_variation={num(tv, 4)} | max_jump={num(mj, 4)} | staleness@0.5={num(stale, 3)}.",
            f"Reliability proxies: pertinence={num(self.pertinence_score, 2)} | liquidity(now)={money(self.liquidity)} | volume(now)={money(self.volume)}.",
            f"Time to resolution: {num(tte, 2)} days." if tte is not None else "Time to resolution: N/A.",
            comp_line,
            f"URL: {self.url}" if self.url else "",
        ]
        return "\n".join([ln for ln in lines if ln.strip()])

    def summary(self) -> Dict[str, Any]:
        if not self.metrics:
            self.compute_metrics()
        if not self.advanced:
            self.compute_advanced_metrics()
        return {
            "question": self.question,
            "url": self.url,
            "score": round(self.score, 4),
            "engagement": round(self.engagement, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "advanced": {
                k: (round(v, 6) if isinstance(v, (int, float)) and v is not None else v)
                for k, v in self.advanced.items()
            },
        }


# ----------------------------
# CORRELATION BETWEEN TWO MARKETS
# ----------------------------
def market_correlation(market_a: Market, market_b: Market) -> float:
    """
    Computes the Pearson correlation between two markets based on outcome_prices.
    High correlation → markets likely to move together (similar outcomes)
    Low correlation → markets independent
    Returns a value between -1 and 1.
    """
    ta, pa = _history_arrays(market_a.history)
    tb, pb = _history_arrays(market_b.history)

    if len(pa) >= 3 and len(pb) >= 3:
        # Align by index length (simple); for more rigor align by time buckets
        n = min(len(pa), len(pb))
        return float(np.corrcoef(pa[-n:], pb[-n:])[0, 1])

    prices_a = market_a.outcome_prices
    prices_b = market_b.outcome_prices
    n = min(len(prices_a), len(prices_b))
    if n == 0:
        return 0.0
    return float(np.corrcoef(prices_a[:n], prices_b[:n])[0, 1])


# ----------------------------
# NEW: OVERALL PIPELINE FUNCTION
# ----------------------------
def process_polymarket_markets(
    markets_data: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    End-to-end helper for your app:
      - builds Market objects
      - computes simple + advanced metrics, score, engagement
      - ranks markets by (score * engagement) but also exposes composite_signal ranking
      - computes correlation between top 2
      - produces a Claude-ready summary block for top_k markets

    Returns dict:
      {
        "markets_sorted": List[Market],
        "top_markets_summary": List[dict],
        "corr_top2": float,
        "global_score": float,
        "claude_block": str
      }
    """
    markets = [Market(d) for d in markets_data]

    for m in markets:
        m.compute_metrics()
        m.compute_advanced_metrics()
        m.compute_score()
        m.compute_engagement()

    # Primary ranking: your original intent
    markets_sorted = sorted(markets, key=lambda x: x.score * x.engagement, reverse=True)

    # Correlation between top 2
    corr_top2 = market_correlation(markets_sorted[0], markets_sorted[1]) if len(markets_sorted) >= 2 else 0.0

    # Global score (fixing your weighting bug: previously you multiplied by weight twice)
    weights = np.array([m.score * m.engagement for m in markets_sorted], dtype=float)
    denom = float(weights.sum()) if float(weights.sum()) > 1e-12 else 1.0
    global_score = float(np.sum(weights * weights) / denom)  # keeps your spirit: emphasizes strong markets

    # Claude block
    top = markets_sorted[:max(1, top_k)]
    claude_block = "## Polymarket correlated markets\n\n" + "\n\n".join(
        f"### Market #{i}\n{m.polymarket_summary_text()}"
        for i, m in enumerate(top, 1)
    )

    return {
        "markets_sorted": markets_sorted,
        "top_markets_summary": [m.summary() for m in top],
        "corr_top2": float(corr_top2),
        "global_score": global_score,
        "claude_block": claude_block,
    }
    


if __name__ == "__main__":
    # ----------------------------
    # EXAMPLE USAGE (advanced + pipeline)
    # ----------------------------

    market1_data = {
        "question": "Will NVIDIA be the third-largest company in the world by market cap on February 28?",
        "event_title": "3rd largest company end of February?",
        "active": True,
        "closed": False,
        "end_date": "2026-02-28T00:00:00Z",
        "volume": 67907.85,
        "liquidity": 13568.55,
        "outcomes": ["Yes", "No"],
        "outcome_prices": [0.0045, 0.9955],
        "url": "https://polymarket.com/event/will-nvidia-be-the-third-largest-company-in-the-world-by-market-cap-on-february-28",
        "history": [
            {"t": 1771536017, "p": 0.5},
            {"t": 1771537219, "p": 0.5},
            {"t": 1771538422, "p": 0.5},
            {"t": 1771677618, "p": 0.495},
            {"t": 1771678825, "p": 0.495},
            {"t": 1771701619, "p": 0.49},
            {"t": 1771702818, "p": 0.495},
            {"t": 1771704024, "p": 0.49},
            {"t": 1771704258, "p": 0.5},
        ],
        "pertinence_score": 0.8,
    }

    market2_data = {
        "question": "Will NVIDIA be the second-largest company in the world by market cap on February 28?",
        "event_title": "2nd largest company end of February?",
        "active": True,
        "closed": False,
        "end_date": "2026-02-28T00:00:00Z",
        "volume": 55000,
        "liquidity": 12000,
        "outcomes": ["Yes", "No"],
        "outcome_prices": [0.01, 0.99],
        "url": "https://polymarket.com/event/will-nvidia-be-the-second-largest-company-in-the-world-by-market-cap-on-february-28",
        "history": [
            {"t": 1771536017, "p": 0.5},
            {"t": 1771537219, "p": 0.5},
            {"t": 1771538422, "p": 0.5},
            {"t": 1771677618, "p": 0.50},
            {"t": 1771678825, "p": 0.505},
            {"t": 1771701619, "p": 0.50},
            {"t": 1771702818, "p": 0.505},
            {"t": 1771704024, "p": 0.50},
            {"t": 1771704258, "p": 0.495},
        ],
        "pertinence_score": 0.7,
    }

    # 1) Manual usage (simple + advanced)
    markets = [Market(market1_data), Market(market2_data)]

    for m in markets:
        m.compute_metrics()
        m.compute_advanced_metrics(recent_points=20, jump_threshold=0.005)
        m.compute_score()
        m.compute_engagement()

    markets_sorted = sorted(markets, key=lambda x: x.score * x.engagement, reverse=True)

    corr = market_correlation(markets_sorted[0], markets_sorted[1]) if len(markets_sorted) >= 2 else 0.0

    weights = np.array([m.score * m.engagement for m in markets_sorted], dtype=float)
    denom = float(weights.sum()) if float(weights.sum()) > 1e-12 else 1.0
    global_score = float(np.sum(weights * weights) / denom)

    print("\n===== MANUAL (simple + advanced) =====")
    for i, m in enumerate(markets_sorted, 1):
        print(f"\n#{i} Market: {m.question}")
        print(f"URL: {m.url}")
        print(f"Score individuel: {m.score:.4f}")
        print(f"Engagement: {m.engagement:.4f}")
        print(f"Breakdown metrics: { {k: round(v,4) for k,v in m.metrics.items()} }")
        print(f"Advanced: { {k: (round(v,6) if isinstance(v,(int,float)) and v is not None else v) for k,v in m.advanced.items()} }")
        print("\n--- Claude-ready summary block ---")
        print(m.polymarket_summary_text())

    print("\n--- Corrélation entre marchés ---")
    print("Corrélation:", round(corr, 4))
    print("\n--- Score global pondéré ---")
    print("Global score:", round(global_score, 4))

    # 2) Pipeline usage (recommended for your app)
    print("\n\n===== PIPELINE (process_polymarket_markets) =====")
    pipeline_res = process_polymarket_markets([market1_data, market2_data], top_k=2)

    print("\nTop markets summary objects:")
    for s in pipeline_res["top_markets_summary"]:
        print(s)

    print("\nCorrelation top2:", round(pipeline_res["corr_top2"], 4))
    print("Global score:", round(pipeline_res["global_score"], 4))

    print("\n--- Claude block to inject into your report prompt ---")
    print(pipeline_res["claude_block"])