"""guestshots — pick flattering, active guest screenshots from podcast YouTube videos."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import track

from . import faces as F
from . import video as V
import pickle

from .scoring import Calib, Scored, score_frame, select_diverse

app = typer.Typer(add_completion=False, help=__doc__)
con = Console()
ROOT = Path(__file__).resolve().parent.parent
HOSTS_DIR = ROOT / "hosts"
OUT_DIR = ROOT / "output"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _host_embeddings(host_dir: Path) -> list[np.ndarray]:
    embs = []
    for p in sorted(host_dir.glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            e = F.embed_image(p)
            if e is not None:
                embs.append(e)
    return embs


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


def analyze_video(url: str, fps: float, host_embs: list[np.ndarray], host_sim: float):
    vid = V.download(url)
    con.print(f"[bold]{vid.title}[/] ({vid.duration/60:.1f} min, {vid.width}x{vid.height})")
    frames = V.extract_frames(vid, fps)
    con.print(f"  {len(frames)} frames @ {fps:g} fps")

    obs_cache = vid.path.parent / f"faces_{fps:g}fps_v2.pkl"
    if obs_cache.exists():
        obs: list[F.FaceObs] = pickle.loads(obs_cache.read_bytes())
    else:
        obs = []
        for t, p in track(frames, description="  detecting faces", console=con):
            obs.extend(F.detect(t, p))
        obs_cache.write_bytes(pickle.dumps(obs))
    idents = F.cluster(obs)
    con.print(f"  {len(obs)} faces → {len(idents)} identities: "
              + ", ".join(f"#{i.id}×{i.count}" for i in idents))

    # label host identities via reference photos
    host_ids = set()
    for i in idents:
        if host_embs and max(float(i.centroid @ h) for h in host_embs) >= host_sim:
            host_ids.add(i.id)
    return vid, frames, obs, idents, host_ids


@app.command()
def main(
    urls: list[str] = typer.Argument(..., help="YouTube URLs"),
    n: int = typer.Option(5, "-n", "--shots", help="Screenshots per video"),
    fps: float = typer.Option(1.0, help="Analysis frames per second"),
    min_gap: float = typer.Option(20.0, help="Min seconds between picked shots"),
    host_dir: Path = typer.Option(HOSTS_DIR, help="Folder with reference photos of the host"),
    host_sim: float = typer.Option(0.45, help="Cosine similarity to call an identity 'host'"),
    guest_id: int | None = typer.Option(None, help="Force identity id as guest (see identities/ crops)"),
    solo_only: bool = typer.Option(False, help="Only frames where the guest is alone in the picture"),
    llm: bool = typer.Option(True, help="Re-rank top candidates with a vision LLM via OpenRouter (needs OPENROUTER_API_KEY)"),
    llm_model: str = typer.Option("openai/gpt-5.4-mini", help="OpenRouter model id for re-ranking"),
    llm_pool: int = typer.Option(4, help="Candidates per final shot sent to Claude (n × llm_pool)"),
):
    """Download each video, find the guest (not the host), and save N flattering + active screenshots."""
    host_embs = _host_embeddings(host_dir)
    if host_embs:
        con.print(f"host refs: {len(host_embs)} photo(s) from {host_dir}")
    else:
        con.print(f"[yellow]no host reference photos in {host_dir}[/] — "
                  "will infer host as the identity shared across videos (needs ≥2 URLs), else ask you.")

    results = [analyze_video(u, fps, host_embs, host_sim) for u in urls]

    # no host refs: host = identity recurring across videos
    if not host_embs and len(results) > 1:
        cents = [(vi, i) for vi, (_, _, _, idents, _) in enumerate(results) for i in idents]
        for vi, (_, _, _, idents, host_ids) in enumerate(results):
            for i in idents:
                hits = sum(1 for vj, j in cents if vj != vi and float(i.centroid @ j.centroid) >= host_sim)
                if hits >= 1:
                    host_ids.add(i.id)

    from . import vision
    use_llm = llm and vision.available()
    if llm and not use_llm:
        con.print("[yellow]LLM re-ranking skipped: OPENROUTER_API_KEY not set (env or .env).[/]")

    for vid, frames, obs, idents, host_ids in results:
        out = OUT_DIR / f"{vid.id}_{_slug(vid.title)}"
        out.mkdir(parents=True, exist_ok=True)
        idd = out / "identities"
        idd.mkdir(exist_ok=True)
        for i in idents:
            if i.sample:
                tag = "host" if i.id in host_ids else "cand"
                F.save_crop(i.sample, idd / f"id{i.id}_{tag}_x{i.count}.jpg")

        # guest = forced id, else biggest non-host identity by face-time
        non_host = [i for i in idents if i.id not in host_ids]
        if guest_id is not None:
            guest = next((i for i in idents if i.id == guest_id), None)
        elif non_host:
            guest = max(non_host, key=lambda i: i.count)
        else:
            guest = None
        if guest is None:
            con.print(f"[red]{vid.title}: could not determine the guest.[/] "
                      f"Check crops in {idd} and re-run with --guest-id N "
                      f"(or drop a host photo into {host_dir}).")
            continue
        if not host_ids:
            con.print(f"[yellow]  no host identified — assuming guest = most frequent face (#{guest.id}). "
                      f"Verify in {idd}.[/]")
        con.print(f"  guest = identity #{guest.id} ({guest.count} frames), host = {sorted(host_ids) or '—'}")

        # per-frame context
        by_frame: dict[Path, list[F.FaceObs]] = defaultdict(list)
        for o in obs:
            by_frame[o.frame].append(o)
        motion = _motion(frames)
        mean_motion = float(np.mean([m.mean() for m in motion.values()]))
        if mean_motion < 0.5:
            con.print(f"[yellow]  video is (almost) static (motion {mean_motion:.2f}) — looks like cover art / audio-only. "
                      "Screenshots will all be the same picture.[/]")
        def face_motion(o: F.FaceObs) -> float:
            x1, y1, x2, y2 = [v // 4 for v in o.bbox]
            m = motion[o.frame][y1:y2, x1:x2]
            return float(m.mean()) if m.size else 0.0

        guest_obs = [o for o in obs if o.identity == guest.id]
        calib = Calib.from_obs(guest_obs, [face_motion(o) for o in guest_obs])

        median_area = float(np.median([o.area_frac for o in guest_obs]))
        scored: list[Scored] = []
        for o in guest_obs:
            others = [x for x in by_frame[o.frame] if x is not o]
            solo = not any(x.identity in host_ids or x.area_frac > 0.5 * o.area_frac for x in others)
            # wide shots: guest much smaller than usual → host is in frame even if undetected
            solo = solo and o.area_frac >= 0.5 * median_area
            if solo_only and not solo:
                continue
            scored.append(score_frame(o, solo, face_motion(o), calib))

        if not scored:
            con.print("[red]  no usable guest frames[/]")
            continue

        pool = select_diverse(scored, n * llm_pool if use_llm else n, min_gap=min_gap / (2 if use_llm else 1))
        cand_dir = out / "candidates"
        cand_dir.mkdir(exist_ok=True)
        # Grab full-res frames and VERIFY the guest is really in them (the 1fps analysis frame
        # can sit up to 0.5s away from the exact timestamp — fatal at a camera cut).
        scale = vid.width / 960
        verified: list[Scored] = []
        for c in pool:
            full = cand_dir / f"full_t{int(c.obs.t):05d}.jpg"
            hit = None
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
                con.print(f"    [dim]dropped t={c.obs.t:.0f}s: guest not found in full-res frame (camera cut?)[/]")
                full.unlink(missing_ok=True)
                continue
            c.obs.bbox = hit.bbox  # refine to the verified frame
            F.save_crop(c.obs, cand_dir / f"t{int(c.obs.t):05d}.jpg", pad=0.8, img=img, scale=scale)
            verified.append(c)
        pool = verified
        if not pool:
            con.print("[red]  no verifiable guest frames[/]")
            continue

        picks = pool
        ratings = None
        if use_llm:
            con.print(f"  asking {llm_model} to rank {len(pool)} candidates …")
            imgs = [cand_dir / f"t{int(c.obs.t):05d}.jpg" for c in pool]
            ratings = vision.rate(imgs, guest_hint=vid.title, model=llm_model)
            for r in ratings:
                if 0 <= r.index < len(pool):
                    c = pool[r.index]
                    c.parts["llm_flattering"] = r.flattering
                    c.parts["llm_active"] = r.active
                    c.parts["llm_eyes"] = r.eyes
                    c.parts["llm_note"] = r.note
                    llm_score = 0 if not r.ok else (0.6 * r.flattering + 0.4 * r.active) / 10
                    c.score = 0.35 * c.score + 0.65 * llm_score
            picks = select_diverse(pool, n, min_gap=min_gap)

        shots_dir = out / "shots"
        shots_dir.mkdir(exist_ok=True)
        for f in shots_dir.glob("*.jpg"):
            f.unlink()
        report = []
        for k, c in enumerate(picks, 1):
            t = c.obs.t
            fn = shots_dir / f"{k:02d}_{int(t)//60:02d}m{int(t)%60:02d}s_score{c.score:.2f}.jpg"
            (cand_dir / f"full_t{int(t):05d}.jpg").rename(fn)
            report.append({"rank": k, "t": round(t, 2), "file": fn.name, "score": round(c.score, 3),
                           "solo": c.solo, "bbox_analysis": c.obs.bbox,
                           "parts": {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in c.parts.items()}})
        sheet = [cv2.resize(cv2.imread(str(shots_dir / r["file"])), (640, 360)) for r in report]
        sheet += [np.zeros((360, 640, 3), np.uint8)] * (-len(sheet) % 3)
        cv2.imwrite(str(out / "contact_sheet.jpg"),
                    np.vstack([np.hstack(sheet[i:i + 3]) for i in range(0, len(sheet), 3)]))
        for f in cand_dir.glob("full_*.jpg"):
            f.unlink()
        (out / "report.json").write_text(json.dumps({
            "url": f"https://youtu.be/{vid.id}", "title": vid.title, "guest_identity": guest.id,
            "host_identities": sorted(host_ids), "llm": llm_model if use_llm else None, "shots": report}, indent=2))
        con.print(f"  [green]✓ {len(picks)} shots → {shots_dir}[/]")
        for r in report:
            p = r["parts"]
            note = (f"[{p['llm_eyes']}] " + p["llm_note"]) if p.get("llm_note") else f"eyes {p['eyes']:.2f} smile {p['smile']:.2f} speak {p['speaking']:.2f} motion {p['motion']:.2f}"
            con.print(f"    {r['file']}  {'solo' if r['solo'] else 'two-shot'}  {note}")


if __name__ == "__main__":
    app()
