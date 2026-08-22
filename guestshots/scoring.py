"""Heuristic 'flattering + active' scoring of guest frames and diverse top-N selection.

Thresholds are calibrated per video from the guest's own feature distribution, so a
squinty face or a slightly off-axis camera doesn't break absolute cut-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faces import FaceObs


def _bell(x: float, center: float, width: float) -> float:
    return float(np.exp(-((x - center) / width) ** 2))


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))


@dataclass
class Calib:
    eye_lo: float      # p15 of eye aspect — blinks / looking down
    eye_hi: float      # p70 — clearly open
    mouth_closed: float  # p10 of mouth aspect
    mouth_wide: float    # p95
    smile_base: float    # p50 of mouth width / inter-ocular
    smile_hi: float      # p95
    sharp_ref: float     # p80 laplacian variance
    motion_hi: float     # p85 of face-region motion

    @classmethod
    def from_obs(cls, obs: list[FaceObs], motions: list[float]) -> "Calib":
        p = lambda attr, q: float(np.percentile([getattr(o, attr) for o in obs], q))  # noqa: E731
        return cls(
            eye_lo=p("eye_open", 15), eye_hi=p("eye_open", 70),
            mouth_closed=p("mouth_open", 10), mouth_wide=p("mouth_open", 95),
            smile_base=p("smile", 50), smile_hi=p("smile", 95),
            sharp_ref=p("sharpness", 80),
            motion_hi=float(np.percentile(motions, 85)) if motions else 1.0,
        )


@dataclass
class Scored:
    obs: FaceObs
    solo: bool  # guest is the only (relevant) face in frame
    activity: float  # raw motion in the face region
    score: float
    parts: dict


WEIGHTS = {"frontal": 2.0, "eyes": 3.0, "expression": 2.0, "motion": 0.8, "sharp": 1.2,
           "light": 0.5, "size": 1.0, "solo": 2.0, "det": 0.3}


def score_frame(o: FaceObs, solo: bool, activity: float, c: Calib) -> Scored:
    speaking = _ramp(o.mouth_open, c.mouth_closed, c.mouth_wide)   # 0 closed … 1 gaping
    smile = _ramp(o.smile, c.smile_base, c.smile_hi)
    parts = {
        # looking roughly at camera, not mid-turn
        "frontal": 0.6 * _bell(o.yaw, 0, 30) + 0.4 * _bell(o.pitch, 0, 22),
        # eyes clearly open: both landmark models must agree (kills blinks, squints, looking down)
        "eyes": min(_ramp(o.eye_open, c.eye_lo, c.eye_hi),
                    _ramp(o.ear_mp, 0.14, 0.24) if o.ear_mp >= 0 else 0.5),
        # flattering + active: a smile, or a moderately open "talking" mouth — not a gape
        "expression": max(smile, 0.8 * _bell(speaking, 0.45, 0.25)),
        "smile": smile,
        "speaking": speaking,
        # gesturing / moving — but normalized so one wild frame doesn't dominate
        "motion": _ramp(activity, 0, c.motion_hi),
        # crisp, not motion-blurred
        "sharp": _ramp(o.sharpness, 0.4 * c.sharp_ref, c.sharp_ref),
        "light": _bell(o.brightness, 125, 55),
        "size": _ramp(o.area_frac, 0.005, 0.04),
        "solo": 1.0 if solo else 0.35,
        "det": _ramp(o.det_score, 0.5, 0.9),
    }
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    # hard gates: never "flattering"
    if parts["eyes"] < 0.3 or abs(o.yaw) > 45:
        score *= 0.2
    if speaking > 0.95:  # mid-word gape
        score *= 0.6
    return Scored(obs=o, solo=solo, activity=activity, score=score, parts=parts)


def select_diverse(cands: list[Scored], n: int, min_gap: float) -> list[Scored]:
    """Greedy top-N with a minimum time gap between picks, so we don't return 5 near-identical frames."""
    picked: list[Scored] = []
    for c in sorted(cands, key=lambda s: s.score, reverse=True):
        if all(abs(c.obs.t - p.obs.t) >= min_gap for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    return sorted(picked, key=lambda s: s.obs.t)
