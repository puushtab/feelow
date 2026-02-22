from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class Item:
    id: str
    source: str                  # "reuters", "reddit", "sec_8k", "earnings_call", "x", ...
    ticker: str
    ts: datetime
    title: str
    text: str
    url: Optional[str] = None

    # raw metadata from the source (karma, author, filing type, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    # scores added by scoring agents
    scores: Dict[str, float] = field(default_factory=dict)

    # derived tags / extracted facts
    tags: List[str] = field(default_factory=list)

    # short extract the router can use without stuffing full text
    summary: Optional[str] = None


@dataclass
class Bundle:
    ticker: str
    asof: datetime
    items: List[Item] = field(default_factory=list)

    # bundle-level aggregated metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # what router selected
    selected: List[Item] = field(default_factory=list)