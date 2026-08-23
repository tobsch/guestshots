"""The guestshots pipeline as a library: analyze a video, pick the guest, produce shots.

Used by both the CLI (`cli.py`) and the web service (`server.py`).
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import faces as F
from . import video as V
from .scoring import Calib, Scored, score_frame, select_diverse

Progress = Callable[[str, int, int, str], None]  # stage, done, total, message


def _noop(stage: str, done: int = 0, total: int = 0, msg: str = "") -> None:
    pass


PROFILES: dict[str, dict] = {
    # Stills meant as background for quote graphics: guest alone, looking at the camera, no hand at the
    # face, face in the upper part of the frame so there is room for text below the chin.
    "portrait": {"solo_only": True, "require_gaze_camera": True, "max_face_bottom": 0.55, "llm_pool": 6},
}


@dataclass
class Options:
    n: int = 5
    fps: float = 1.0
    min_gap: float = 20.0
    host_sim: float = 0.45
    guest_id: int | None = None
    solo_only: bool = False
    llm: bool = True
    llm_model: str = "openai/gpt-5.4-mini"
    llm_pool: int = 4
    criteria: str = ""                    # free-text extra requirements for the LLM stage
    require_gaze_camera: bool = False     # LLM gate: guest must look into the camera
    max_face_bottom: float | None = None  # drop shots whose face box bottom is below this fraction of frame height
    profile: str | None = None            # name from PROFILES; explicit values win over the profile

    @classmethod
    def build(cls, profile: str | None = None, **explicit) -> "Options":
        """Profile defaults, overridden by explicitly passed values (None = not passed)."""
        if profile and profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
        values = dict(PROFILES.get(profile or "", {}))
        values.update({k: v for k, v in explicit.items() if v is not None})
        return cls(profile=profile, **values)


@dataclass
class Analysis:
    vid: V.Video
    frames: list[tuple[float, Path]]
    obs: list[F.FaceObs]
    idents: list[F.Identity]
    host_ids: set[int] = field(default_factory=set)


@dataclass
class Result:
    out_dir: Path
    shots: list[Path]
    contact_sheet: Path
    report: dict


class PipelineError(Exception):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _motion(frames: list[tuple[float, Path]]) -> dict[Path, np.ndarray]:
    """Cheap per-frame motion map (abs diff to previous frame, downscaled)."""
    out, prev = {}, None
    for _, p in frames:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4))
        out[p] = np.zeros_like(g, dtype=np.float32) if prev is None else cv2.absdiff(g, prev).astype(np.float32)
        prev = g
    return out


def host_embeddings(paths: list[Path]) -> list[np.ndarray]:
    embs = []
    for p in paths:
        e = F.embed_image(p)
        if e is not None:
            embs.append(e)
    return embs


def analyze(url: str, cache_dir: Path, fps: float = 1.0, progress: Progress = _noop) -> Analysis:
    """Download, extract frames, detect + cluster faces. Everything is cached under cache_dir/<video id>."""
    progress("downloading", 0, 0, url)
    vid = V.download(url, cache=cache_dir)
    progress("extracting", 0, 0, f"{vid.title} ({vid.duration / 60:.1f} min, {vid.width}x{vid.height})")
    frames = V.extract_frames(vid, fps)

    obs_cache = vid.path.parent / f"faces_{fps:g}fps_v2.pkl"
    if obs_cache.exists():
        obs: list[F.FaceObs] = pickle.loads(obs_cache.read_bytes())
        frames_dir = frames[0][1].parent if frames else None
        for o in obs:  # cache may have been written under a different root
            o.frame = frames_dir / o.frame.name
    else:
        obs = []
        for i, (t, p) in enumerate(frames):
            if i % 25 == 0:
                progress("detecting", i, len(frames), "")
            obs.extend(F.detect(t, p))
        obs_cache.write_bytes(pickle.dumps(obs))
    progress("detecting", len(frames), len(frames), "")
    idents = F.cluster(obs)
    return Analysis(vid=vid, frames=frames, obs=obs, idents=idents)


def mark_hosts(analyses: list[Analysis], host_embs: list[np.ndarray], host_sim: float) -> None:
    """Label host identities: by reference photos, or (no refs, ≥2 videos) by recurrence across videos."""
    if host_embs:
        for a in analyses:
            for i in a.idents:
                if max(float(i.centroid @ h) for h in host_embs) >= host_sim:
                    a.host_ids.add(i.id)
    elif len(analyses) > 1:
        cents = [(vi, i) for vi, a in enumerate(analyses) for i in a.idents]
        for vi, a in enumerate(analyses):
            for i in a.idents:
                if any(vj != vi and float(i.centroid @ j.centroid) >= host_sim for vj, j in cents):
                    a.host_ids.add(i.id)


def produce(a: Analysis, opts: Options, out: Path, progress: Progress = _noop) -> Result:
    """Pick the guest, score frames, verify full-res grabs, optionally LLM-rank, write shots."""
    from . import vision

    vid, obs, idents, host_ids = a.vid, a.obs, a.idents, a.host_ids
    out.mkdir(parents=True, exist_ok=True)
    idd = out / "identities"
    idd.mkdir(exist_ok=True)
    for i in idents:
        if i.sample:
            F.save_crop(i.sample, idd / f"id{i.id}_{'host' if i.id in host_ids else 'cand'}_x{i.count}.jpg")

    non_host = [i for i in idents if i.id not in host_ids]
    if opts.guest_id is not None:
        guest = next((i for i in idents if i.id == opts.guest_id), None)
    elif non_host:
        guest = max(non_host, key=lambda i: i.count)
    else:
        guest = None
    if guest is None:
        raise PipelineError("could not determine the guest — check identities/ and pass guest_id, or add a host photo")
    if not host_ids:
        progress("scoring", 0, 0, f"no host identified — assuming guest = most frequent face (#{guest.id})")
    progress("scoring", 0, 0, f"guest = identity #{guest.id} ({guest.count} frames), host = {sorted(host_ids) or '-'}")

    by_frame: dict[Path, list[F.FaceObs]] = defaultdict(list)
    for o in obs:
        by_frame[o.frame].append(o)
    motion = _motion(a.frames)
    mean_motion = float(np.mean([m.mean() for m in motion.values()]))
    if mean_motion < 0.5:
        progress("scoring", 0, 0, f"video is (almost) static (motion {mean_motion:.2f}) — cover art / audio-only?")

    def face_motion(o: F.FaceObs) -> float:
        x1, y1, x2, y2 = [v // 4 for v in o.bbox]
        m = motion[o.frame][y1:y2, x1:x2]
        return float(m.mean()) if m.size else 0.0

    guest_obs = [o for o in obs if o.identity == guest.id]
    if opts.max_face_bottom is not None and guest_obs:
        ah = cv2.imread(str(guest_obs[0].frame)).shape[0]  # analysis frame height; same aspect as full-res
        kept = [o for o in guest_obs if o.bbox[3] / ah <= opts.max_face_bottom]
        progress("scoring", 0, 0, f"face-bottom filter ≤{opts.max_face_bottom:g}: {len(kept)}/{len(guest_obs)} frames kept")
        guest_obs = kept
    if not guest_obs:
        raise PipelineError("no guest frames left after filtering")
    calib = Calib.from_obs(guest_obs, [face_motion(o) for o in guest_obs])
    median_area = float(np.median([o.area_frac for o in guest_obs]))
    scored: list[Scored] = []
    for o in guest_obs:
        others = [x for x in by_frame[o.frame] if x is not o]
        solo = not any(x.identity in host_ids or x.area_frac > 0.5 * o.area_frac for x in others)
        solo = solo and o.area_frac >= 0.5 * median_area  # wide shot → host in frame even if undetected
        if opts.solo_only and not solo:
            continue
        scored.append(score_frame(o, solo, face_motion(o), calib))
    if not scored:
        raise PipelineError("no usable guest frames")

    use_llm = opts.llm and vision.available()
    if opts.llm and not use_llm:
        progress("scoring", 0, 0, "LLM re-ranking skipped: OPENROUTER_API_KEY not set")
    pool = select_diverse(scored, opts.n * opts.llm_pool if use_llm else opts.n,
                          min_gap=opts.min_gap / (2 if use_llm else 1))

    # Full-res grabs, verified against the guest embedding (analysis frame may be 0.5s off a cut).
    cand_dir = out / "candidates"
    cand_dir.mkdir(exist_ok=True)
    scale = vid.width / 960
    verified: list[Scored] = []
    for k, c in enumerate(pool):
        progress("verifying", k, len(pool), "")
        full = cand_dir / f"full_t{int(c.obs.t):05d}.jpg"
        hit, img = None, None
        for dt in (0.0, 0.4, -0.4, 0.8):
            V.grab_fullres(vid, max(0.0, c.obs.t + dt), full)
            img = cv2.imread(str(full))
            small = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
            for o in F.detect_img(small, c.obs.t + dt):
                if float(o.embedding @ guest.centroid) >= 0.6 and _iou(o.bbox, c.obs.bbox) > 0.3:
                    hit = o
                    break
            if hit:
                break
        if not hit:
            full.unlink(missing_ok=True)
            continue
        c.obs.bbox = hit.bbox
        fh, fw = img.shape[:2]
        bx1, by1, bx2, by2 = [v * scale for v in hit.bbox]
        c.parts["face"] = {"x": round(bx1 / fw, 4), "y": round(by1 / fh, 4),
                           "w": round((bx2 - bx1) / fw, 4), "h": round((by2 - by1) / fh, 4)}
        if opts.max_face_bottom is not None and by2 / fh > opts.max_face_bottom:
            full.unlink(missing_ok=True)
            continue
        F.save_crop(c.obs, cand_dir / f"t{int(c.obs.t):05d}.jpg", pad=0.8, img=img, scale=scale)
        verified.append(c)
    pool = verified
    if not pool:
        raise PipelineError("no verifiable guest frames")

    picks = pool
    usage = None
    gates: dict[str, int] = {}
    if use_llm:
        progress("ranking", 0, len(pool), f"asking {opts.llm_model} to rank {len(pool)} candidates")
        imgs = [cand_dir / f"t{int(c.obs.t):05d}.jpg" for c in pool]
        ratings, usage = vision.rate(imgs, guest_hint=vid.title, model=opts.llm_model, criteria=opts.criteria)
        passed: list[Scored] = []
        gated: dict[str, int] = {"eyes": 0, "hand_near_face": 0, "gaze": 0, "unusable": 0, "unrated": 0}
        rated = {r.index: r for r in ratings if 0 <= r.index < len(pool)}
        for i, c in enumerate(pool):
            r = rated.get(i)
            if r is None:
                gated["unrated"] += 1
                continue
            c.parts.update(llm_flattering=r.flattering, llm_active=r.active, llm_eyes=r.eyes,
                           llm_gaze=r.gaze, llm_hand_near_face=r.hand_near_face, llm_note=r.note)
            if r.eyes.strip().lower() != "open":
                gated["eyes"] += 1
            elif r.hand_near_face:
                gated["hand_near_face"] += 1
            elif opts.require_gaze_camera and r.gaze.strip().lower() != "camera":
                gated["gaze"] += 1
            elif r.usable is False:
                gated["unusable"] += 1
            else:
                c.score = 0.35 * c.score + 0.65 * (0.6 * r.flattering + 0.4 * r.active) / 10
                passed.append(c)
        gates = {k: v for k, v in gated.items() if v}
        progress("ranking", len(pool), len(pool), f"{len(passed)}/{len(pool)} candidates passed the LLM gates"
                 + (f" (rejected: {gates})" if gates else ""))
        picks = select_diverse(passed, opts.n, min_gap=opts.min_gap)
        if len(picks) < opts.n:
            progress("ranking", len(pool), len(pool),
                     f"only {len(picks)} of {opts.n} requested shots passed — relax the gates or raise llm_pool")

    shots_dir = out / "shots"
    shots_dir.mkdir(exist_ok=True)
    for f in shots_dir.glob("*.jpg"):
        f.unlink()
    report, shots = [], []
    for k, c in enumerate(picks, 1):
        t = c.obs.t
        fn = shots_dir / f"{k:02d}_{int(t) // 60:02d}m{int(t) % 60:02d}s_score{c.score:.2f}.jpg"
        (cand_dir / f"full_t{int(t):05d}.jpg").rename(fn)
        shots.append(fn)
        parts = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in c.parts.items()}
        report.append({"rank": k, "t": round(t, 2), "file": fn.name, "score": round(c.score, 3),
                       "solo": c.solo, "face": parts.pop("face", None), "bbox_analysis": c.obs.bbox,
                       "parts": parts})
    for f in cand_dir.glob("full_*.jpg"):
        f.unlink()
    sheet = [cv2.resize(cv2.imread(str(p)), (640, 360)) for p in shots]
    sheet += [np.zeros((360, 640, 3), np.uint8)] * (-len(sheet) % 3)
    sheet_path = out / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), np.vstack([np.hstack(sheet[i:i + 3]) for i in range(0, len(sheet), 3)]))
    rep = {"url": f"https://youtu.be/{vid.id}", "title": vid.title, "guest_identity": guest.id,
           "options": asdict(opts), "requested": opts.n, "llm_rejected": gates,
           "host_identities": sorted(host_ids), "llm": opts.llm_model if use_llm else None,
           "llm_usage": usage, "shots": report}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    progress("done", len(shots), len(shots), f"{len(shots)} shots")
    return Result(out_dir=out, shots=shots, contact_sheet=sheet_path, report=rep)
