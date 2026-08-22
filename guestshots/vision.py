"""Optional final ranking of candidate shots with a vision LLM via OpenRouter."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.4-mini"


class ShotRating(BaseModel):
    index: int = Field(description="0-based index of the image in the order given")
    eyes: str = Field(description="open | squint | closed")
    flattering: int = Field(ge=1, le=10)
    active: int = Field(ge=1, le=10)
    usable: bool | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.eyes.strip().lower() == "open" and (self.usable is not False)


class Ratings(BaseModel):
    ratings: list[ShotRating]


def available() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _client() -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={"HTTP-Referer": "https://alphalist.com", "X-Title": "guestshots"},
    )


PROMPT = (
    "These are candidate screenshots of a podcast guest taken from a video interview{hint}. "
    "We want to pick stills for social media / episode artwork: the guest should look good "
    "(flattering expression, open eyes, natural smile or focused look, good angle and light) "
    "AND look active — engaged, animated, gesturing or mid-speech rather than passive or frozen. "
    "Rate every image. FIRST look closely at the eyes: are both eyes clearly open with visible iris "
    "(\"open\"), narrowed in a laugh/squint (\"squint\"), or shut/blinking (\"closed\")? Anything but "
    "\"open\" is unusable, no matter how nice the smile. Also unusable: awkward mid-word mouth shapes, "
    "rolled-up eyes, motion blur, unflattering angles.\n\n"
    "Respond with JSON only, no prose, exactly this shape:\n"
    '{{"ratings": [{{"index": 0, "eyes": "open|squint|closed", "flattering": 1-10, "active": 1-10, '
    '"usable": true|false, "note": "one short sentence"}}, ...]}}'
)


def rate(images: list[Path], guest_hint: str = "", model: str = DEFAULT_MODEL) -> list[ShotRating]:
    content: list[dict] = []
    for i, p in enumerate(images):
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.standard_b64encode(p.read_bytes()).decode()}})
    content.append({"type": "text", "text": PROMPT.format(hint=f" ({guest_hint})" if guest_hint else "")})

    resp = _client().chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": content}],
        extra_body={"usage": {"include": True}},
    )
    u = resp.usage
    cost = getattr(u, "cost", None) or (getattr(u, "model_extra", {}) or {}).get("cost")
    print(f"    LLM usage: {u.prompt_tokens} in / {u.completion_tokens} out"
          + (f" → ${cost:.4f}" if cost is not None else ""))
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(f"{model} returned no JSON: {text[:300]}")
    try:
        return Ratings.model_validate(json.loads(m.group(0))).ratings
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"{model} returned unparsable ratings: {e}\n{text[:300]}") from e
