import math
import numpy as np
from dataclasses import dataclass
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
    """Least squares slope of p vs time (time in days), returns slope per day."""
    if len(t) < 3:
        return None
    x = (t - t.mean()) / 86400.0
    y = p - p.mean()
    denom = float((x * x).sum())
    if denom <= 1e-12:
        return None
    return float((x * y).sum() / denom)


def _count_changes(p: np.ndarray, threshold: float = 0.005) -> int:
    """Count only meaningful changes: |Δp| >= threshold."""
    if len(p) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(p)) >= float(threshold)))


def _staleness_ratio_at_value(p: np.ndarray, value: float = 0.5, tol: float = 1e-6) -> Optional[float]:
    """Fraction of points close to a given value (default 0.5)."""
    if len(p) == 0:
        return None
    return float(np.mean(np.abs(p - value) <= tol))


def _staleness_flat_ratio(p: np.ndarray, dp_tol: float = 0.001) -> Optional[float]:
    """Fraction of steps where price barely moves, regardless of level."""
    if len(p) < 2:
        return None
    dp = np.abs(np.diff(p))
    return float(np.mean(dp <= float(dp_tol)))


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
    Represents a Polymarket market with snapshot + history-derived features.
    """

    def __init__(self, data: Dict[str, Any]):
        self.question = data["question"]
        self.event_title = data.get("event_title", "")
        self.active = data.get("active", True)
        self.closed = data.get("closed", False)
        self.end_date = data.get("end_date", None)

        self.volume = data.get("volume", 0.0)
        self.liquidity = data.get("liquidity", 0.0)
        self.outcomes = data.get("outcomes", [])
        self.outcome_prices = data.get("outcome_prices", [])
        self.url = data.get("url", "")
        self.history = data.get("history", [])
        self.pertinence_score = data.get("pertinence_score", 0.5)

        self.metrics: Dict[str, float] = {}
        self.advanced: Dict[str, Any] = {}
        self.score = 0.0
        self.engagement = 0.0

    # ----------------------------
    # SIMPLE METRICS
    # ----------------------------
    def compute_metrics(self) -> Dict[str, float]:
        momentum = self.compute_momentum()
        volatility = self.compute_volatility()
        concentration = self.compute_concentration()

        volume_score = math.log(float(self.volume) + 1.0) / 12.0
        liquidity_score = math.log(float(self.liquidity) + 1.0) / 10.0
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
        If no history, momentum is unknown -> 0.0 (avoid rewarding 50/50 by accident).
        Otherwise: |p_last - p_first|.
        """
        if not self.history or len(self.history) < 2:
            return 0.0
        _, p = _history_arrays(self.history)
        if len(p) < 2:
            return 0.0
        return float(max(0.0, min(abs(p[-1] - p[0]), 1.0)))

    def compute_volatility(self) -> float:
        if not self.history:
            if self.outcome_prices:
                return float(max(self.outcome_prices) - min(self.outcome_prices))
            return 0.0
        _, p = _history_arrays(self.history)
        if len(p) == 0:
            return 0.0
        return float(p.max() - p.min())

    def compute_concentration(self) -> float:
        prices = [float(x) for x in (self.outcome_prices or []) if x is not None]
        if len(prices) <= 1:
            return 0.0
        s = sum(prices)
        if s > 0:
            prices = [p / s for p in prices]
        entropy = 0.0
        for p in prices:
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(len(prices))
        return float(1.0 - (entropy / max_entropy))

    # ----------------------------
    # ADVANCED METRICS (reviewed)
    # ----------------------------
    def compute_advanced_metrics(self, recent_points: int = 20, jump_threshold: float = 0.005) -> Dict[str, Any]:
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
        if n >= max(3, recent_points):
            slope_recent = _ols_slope_per_day(t[-recent_points:], p[-recent_points:])
        else:
            slope_recent = slope_all

        change_count = _count_changes(p, threshold=jump_threshold)

        stale_05 = _staleness_ratio_at_value(p, value=0.5, tol=1e-6)
        stale_flat = _staleness_flat_ratio(p, dp_tol=jump_threshold / 2)

        time_since_reprice = _time_since_last_change(t, p, threshold=jump_threshold)
        entropy_nats = _binary_entropy(p_last) if p_last is not None else None

        tte_days = None
        if self.end_date:
            end_dt = _parse_iso_utc(self.end_date)
            if end_dt is not None:
                tte_days = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0

        history_quality = None
        if n > 0:
            q_points = math.tanh(n / 50.0)
            q_span = math.tanh((span_hours or 0.0) / 24.0)
            q_flat = 1.0 - (stale_flat or 0.0)
            history_quality = float(max(0.0, min(1.0, 0.45*q_points + 0.35*q_span + 0.20*q_flat)))

        whale_dom = (max_jump / total_variation) if total_variation > 1e-12 else None
        jump_mean = (total_variation / max(1, change_count)) if total_variation > 0 else 0.0

        composite_signal = None
        if slope_recent is not None:
            pertinence = float(max(0.0, min(1.0, self.pertinence_score)))
            liq = max(float(self.liquidity or 0.0), 0.0)
            liquidity_factor = math.tanh(math.sqrt(liq) / 150.0)

            stale_factor = 1.0 - max((stale_05 or 0.0), (stale_flat or 0.0))

            whale_factor = 1.0
            if whale_dom is not None:
                whale_factor = float(max(0.0, min(1.0, 1.0 - whale_dom)))

            rel = pertinence * (history_quality if history_quality is not None else 1.0) * liquidity_factor * stale_factor * whale_factor

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
            "jump_mean": jump_mean,
            "whale_dom": whale_dom,

            "staleness_ratio_0_5": stale_05,
            "staleness_flat_ratio": stale_flat,
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
    # SCORE / ENGAGEMENT (keep)
    # ----------------------------
    def compute_score(self) -> float:
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
        if not self.metrics:
            self.compute_metrics()
        self.engagement = float(
            0.4 * self.metrics["volume"] +
            0.3 * self.metrics["concentration"] +
            0.3 * self.metrics["momentum"]
        )
        return self.engagement


# ----------------------------
# FILTER CONFIG + TESTS
# ----------------------------
@dataclass
class FilterConfig:
    min_volume: float = 10_000.0
    min_liquidity: float = 3_000.0
    min_yes_prob: float = 0.60
    min_composite: float = 0.10
    max_whale_dom: float = 0.60
    max_jump_abs: float = 0.03
    min_change_count: int = 5
    min_time_to_event_days: Optional[float] = None  # ex: 7.0 if you want


def tradable_reasons(m: Market, cfg: FilterConfig) -> List[str]:
    if not m.advanced:
        m.compute_advanced_metrics(jump_threshold=0.005)

    reasons = []
    vol = float(m.volume or 0.0)
    liq = float(m.liquidity or 0.0)

    p_yes = m.advanced.get("p_last")
    comp = m.advanced.get("composite_signal")
    whale_dom = m.advanced.get("whale_dom")
    max_jump = float(m.advanced.get("max_jump") or 0.0)
    change_count = int(m.advanced.get("change_count") or 0)
    tte = m.advanced.get("time_to_event_days")

    if vol < cfg.min_volume:
        reasons.append(f"volume<{cfg.min_volume} (vol={vol:.0f})")
    if liq < cfg.min_liquidity:
        reasons.append(f"liquidity<{cfg.min_liquidity} (liq={liq:.0f})")
    if p_yes is None or float(p_yes) < cfg.min_yes_prob:
        reasons.append(f"p_yes<{cfg.min_yes_prob} (p_yes={None if p_yes is None else float(p_yes):.4f})")
    if comp is None or float(comp) < cfg.min_composite:
        reasons.append(f"composite<{cfg.min_composite} (comp={None if comp is None else float(comp):.6f})")
    if whale_dom is None or float(whale_dom) > cfg.max_whale_dom:
        reasons.append(f"whale_dom>{cfg.max_whale_dom} (whale_dom={whale_dom})")
    if max_jump >= cfg.max_jump_abs:
        reasons.append(f"max_jump>={cfg.max_jump_abs} (max_jump={max_jump:.4f})")
    if change_count < cfg.min_change_count:
        reasons.append(f"change_count<{cfg.min_change_count} (change_count={change_count})")
    if cfg.min_time_to_event_days is not None and tte is not None and float(tte) < cfg.min_time_to_event_days:
        reasons.append(f"tte<{cfg.min_time_to_event_days} (tte={float(tte):.2f})")

    return reasons


def is_tradable(m: Market, cfg: FilterConfig) -> bool:
    return len(tradable_reasons(m, cfg)) == 0


def sentiment_rank(m: Market) -> float:
    if not m.advanced:
        m.compute_advanced_metrics()
    comp = float(m.advanced.get("composite_signal") or 0.0)
    vol_factor = math.tanh(float(m.volume or 0.0) / 50_000.0)
    liq_factor = math.tanh(float(m.liquidity or 0.0) / 10_000.0)
    return comp * (0.5 + 0.5 * vol_factor) * (0.5 + 0.5 * liq_factor)


def test_markets(markets: List[Market], cfg: FilterConfig) -> None:
    print("=== TEST MARKETS ===")
    print("Config:", cfg)
    for i, m in enumerate(markets, 1):
        if not m.metrics:
            m.compute_metrics()
        if not m.advanced:
            m.compute_advanced_metrics(jump_threshold=0.005)
        m.compute_score()
        m.compute_engagement()

        p_yes = m.advanced.get("p_last")
        comp = m.advanced.get("composite_signal")
        whale_dom = m.advanced.get("whale_dom")
        cc = m.advanced.get("change_count")
        mj = m.advanced.get("max_jump")
        tv = m.advanced.get("total_variation")
        stale05 = m.advanced.get("staleness_ratio_0_5")
        stalef = m.advanced.get("staleness_flat_ratio")
        tte = m.advanced.get("time_to_event_days")

        reasons = tradable_reasons(m, cfg)
        ok = (len(reasons) == 0)

        print(f"\n[{i}] {m.question}")
        print(f"  volume={float(m.volume or 0):.0f}  liquidity={float(m.liquidity or 0):.0f}  tte_days={None if tte is None else float(tte):.2f}")
        print(f"  p_yes={None if p_yes is None else float(p_yes):.4f}  composite={None if comp is None else float(comp):.6f}")
        print(f"  whale_dom={whale_dom}  max_jump={None if mj is None else float(mj):.4f}  total_var={None if tv is None else float(tv):.4f}  change_count={cc}")
        print(f"  stale@0.5={stale05}  stale_flat={stalef}")
        print(f"  score={m.score:.4f}  engagement={m.engagement:.4f}  rank={sentiment_rank(m):.4f}")
        print("  tradable:", ok)
        if not ok:
            print("  reasons:", "; ".join(reasons))


# ----------------------------
# PIPELINE (filter + rank + correlation)
# ----------------------------
def market_correlation(market_a: Market, market_b: Market) -> float:
    ta, pa = _history_arrays(market_a.history)
    tb, pb = _history_arrays(market_b.history)
    if len(pa) >= 3 and len(pb) >= 3:
        n = min(len(pa), len(pb))
        return float(np.corrcoef(pa[-n:], pb[-n:])[0, 1])
    prices_a = market_a.outcome_prices
    prices_b = market_b.outcome_prices
    n = min(len(prices_a), len(prices_b))
    if n == 0:
        return 0.0
    return float(np.corrcoef(prices_a[:n], prices_b[:n])[0, 1])


def process_polymarket_markets(
    markets_data: List[Dict[str, Any]],
    top_k: int = 5,
    cfg: Optional[FilterConfig] = None,
    apply_filter: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or FilterConfig()
    markets = [Market(d) for d in markets_data]

    for m in markets:
        m.compute_metrics()
        m.compute_advanced_metrics(jump_threshold=0.005)
        m.compute_score()
        m.compute_engagement()

    filtered = markets
    if apply_filter:
        filtered = [m for m in markets if is_tradable(m, cfg)]
        if not filtered:
            filtered = markets  # fallback

    markets_sorted = sorted(filtered, key=sentiment_rank, reverse=True)

    corr_top2 = market_correlation(markets_sorted[0], markets_sorted[1]) if len(markets_sorted) >= 2 else 0.0

    weights = np.array([m.score * m.engagement for m in markets_sorted], dtype=float)
    denom = float(weights.sum()) if float(weights.sum()) > 1e-12 else 1.0
    global_score = float(np.sum(weights * weights) / denom)

    top = markets_sorted[:max(1, top_k)]
    return {
        "filtering": {"total_input": len(markets), "after_filter": len(filtered), "filter_applied": bool(apply_filter)},
        "markets_sorted": markets_sorted,
        "top_markets": top,
        "corr_top2": float(corr_top2),
        "global_score": float(global_score),
    }


# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
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

    markets = [Market(market1_data), Market(market2_data)]

    cfg = FilterConfig(
        min_volume=10_000,
        min_liquidity=3_000,
        min_yes_prob=0.60,
        min_composite=0.10,
        max_whale_dom=0.60,
        max_jump_abs=0.03,
        min_change_count=5,
        min_time_to_event_days=None,  # set 7.0 if you want to exclude near-resolution
    )

    # Manual tests
    for m in markets:
        m.compute_metrics()
        m.compute_advanced_metrics(jump_threshold=0.005)
        m.compute_score()
        m.compute_engagement()

    test_markets(markets, cfg)

    # Pipeline
    res = process_polymarket_markets([market1_data, market2_data], top_k=2, cfg=cfg, apply_filter=True)
    print("\n=== PIPELINE ===")
    print("Filtering:", res["filtering"])
    print("Correlation top2:", round(res["corr_top2"], 4))
    print("Global score:", round(res["global_score"], 4))
    print("\nTop markets by sentiment rank:")
    for i, m in enumerate(res["top_markets"], 1):
        print(f"\n#{i} {m.question}")
        print(f"  rank={sentiment_rank(m):.4f}  p_yes={m.advanced.get('p_last')}  comp={m.advanced.get('composite_signal')}")
        print(f"  whale_dom={m.advanced.get('whale_dom')}  change_count={m.advanced.get('change_count')}")
        print(f"  url={m.url}")