import mimetypes
from pathlib import Path

def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def load_media_part(path: str) -> dict:
    """
    Returns dict with mime_type and raw bytes for Gemini.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with open(p, "rb") as f:
        data = f.read()

    return {
        "mime_type": _guess_mime(path),
        "data": data,
    }
    
import json
import re
from typing import Optional, Any


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.DOTALL | re.IGNORECASE,
)

_FIRST_JSON_OBJ_RE = re.compile(
    r"(\{.*\}|\[.*\])",
    re.DOTALL,
)


def extract_json_text(raw: str) -> Optional[str]:
    """
    Extracts JSON object/array from LLM response.
    Handles code fences and trailing commentary.
    """
    if not raw or not isinstance(raw, str):
        return None

    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()

    m = _FIRST_JSON_OBJ_RE.search(raw.strip())
    if m:
        return m.group(1).strip()

    return None


def safe_json_loads(raw: str) -> Optional[Any]:
    """
    Best-effort JSON parsing with small cleanup.
    """
    jtxt = extract_json_text(raw) or raw
    try:
        return json.loads(jtxt)
    except Exception:
        try:
            jtxt2 = re.sub(r",\s*([}\]])", r"\1", jtxt)
            return json.loads(jtxt2)
        except Exception:
            return None