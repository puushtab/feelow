from datetime import datetime
from pathlib import Path
from typing import List
import hashlib

from src.agent_search.schema import Item


class LocalImageCollector:
    """
    Collects local images for demo (reddit, twitter, instagram, trends).
    """

    def __init__(self, base_folder: str):
        self.base_folder = Path(base_folder)

    def collect(self, ticker: str, asof: datetime) -> List[Item]:
        items = []

        if not self.base_folder.exists():
            return items

        for img_path in self.base_folder.glob("*.png"):
            source_type = self._infer_source(img_path.name)

            uid = hashlib.sha256(str(img_path).encode()).hexdigest()

            items.append(
                Item(
                    id=uid,
                    source=source_type,
                    ticker=ticker,
                    ts=asof,
                    title=f"{source_type} image",
                    text="",
                    url=str(img_path),
                    meta={
                        "asset_type": "image",
                        "asset_path": str(img_path),
                    }
                )
            )

        return items

    def _infer_source(self, filename: str) -> str:
        name = filename.lower()

        if "reddit" in name:
            return "reddit_image"
        if "twitter" in name or "x_" in name:
            return "twitter_image"
        if "instagram" in name:
            return "instagram_image"
        if "trend" in name:
            return "google_trends_image"

        return "social_image"