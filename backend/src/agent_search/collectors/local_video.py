from datetime import datetime
from pathlib import Path
from typing import List
import hashlib

from src.agent_search.schema import Item


class EarningsVideoCollector:
    """
    Loads a pre-downloaded earnings call video for demo.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def collect(self, ticker: str, asof: datetime) -> List[Item]:
        if not self.video_path.exists():
            return []

        uid = hashlib.sha256(str(self.video_path).encode()).hexdigest()

        return [
            Item(
                id=uid,
                source="earnings_video",
                ticker=ticker,
                ts=asof,
                title=f"{ticker} Earnings Call Video",
                text="",  # no text, Gemini processes video
                url=str(self.video_path),
                meta={
                    "asset_type": "video",
                    "asset_path": str(self.video_path),
                }
            )
        ]