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
    gaze: str = "unknown"          # camera | away | unknown
    hand_near_face: bool = False   # any hand/finger touching or overlapping the face
    flattering: int = Field(ge=1, le=10)
    active: int = Field(ge=1, le=10)
    usable: bool | None = None
    note: str = ""

    def ok(self, require_gaze_camera: bool = False) -> bool:
        """Hard gates: eyes open, no hand at the face, the model's own verdict, optionally gaze at camera."""
        if self.eyes.strip().lower() != "open" or self.hand_near_face or self.usable is False:
            return False
        return not require_gaze_camera or self.gaze.strip().lower() == "camera"


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
    "rolled-up eyes, motion blur, unflattering angles.\n"
    "Also report for every image: gaze — is the guest looking into the camera/at the viewer (\"camera\") "
    "or elsewhere (\"away\")? hand_near_face — is any hand or finger touching or overlapping the face "
    "(true/false)? A hand at the face is unusable: on a still it looks like nose-picking or hiding.\n"
    "{criteria}"
    "Respond with JSON only, no prose, exactly this shape:\n"
    '{{"ratings": [{{"index": 0, "eyes": "open|squint|closed", "gaze": "camera|away", "hand_near_face": true|false, '
    '"flattering": 1-10, "active": 1-10, "usable": true|false, "note": "one short sentence"}}, ...]}}'
)

MAX_CRITERIA = 500


def _criteria_block(criteria: str) -> str:
    criteria = " ".join(criteria.split())[:MAX_CRITERIA]
    if not criteria:
        return "\n"
    return ("\nAdditional requirements from the person ordering these stills — treat a violation as "
            f"unusable and say why in the note: {criteria}\n\n")


def rate(images: list[Path], guest_hint: str = "", model: str = DEFAULT_MODEL,
         criteria: str = "") -> tuple[list[ShotRating], dict]:
    """Rate candidate crops. `criteria` = free-text extra requirements appended to the prompt.
    Returns (ratings, usage) with usage = {prompt_tokens, completion_tokens, cost}."""
    content: list[dict] = []
    for i, p in enumerate(images):
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.standard_b64encode(p.read_bytes()).decode()}})
    content.append({"type": "text", "text": PROMPT.format(hint=f" ({guest_hint})" if guest_hint else "",
                                                          criteria=_criteria_block(criteria))})

    resp = _client().chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": content}],
        extra_body={"usage": {"include": True}},
    )
    u = resp.usage
    cost = getattr(u, "cost", None) or (getattr(u, "model_extra", {}) or {}).get("cost")
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "cost": cost}
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(f"{model} returned no JSON: {text[:300]}")
    try:
        return Ratings.model_validate(json.loads(m.group(0))).ratings, usage
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"{model} returned unparsable ratings: {e}\n{text[:300]}") from e
