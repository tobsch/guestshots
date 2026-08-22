"""Download YouTube videos and extract frames with yt-dlp + ffmpeg."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

CACHE = Path(__file__).resolve().parent.parent / "cache"


@dataclass
class Video:
    id: str
    title: str
    path: Path
    duration: float  # seconds
    width: int
    height: int


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Cannot parse YouTube id from {url}")
    return m.group(1)


def download(url: str, max_height: int = 1080) -> Video:
    """Download (cached) and return video metadata."""
    vid = video_id(url)
    vdir = CACHE / vid
    vdir.mkdir(parents=True, exist_ok=True)
    meta_file = vdir / "meta.json"

    existing = list(vdir.glob("video.*"))
    if existing and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        return Video(vid, meta["title"], existing[0], meta["duration"], meta["width"], meta["height"])

    opts = {
        "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "outtmpl": str(vdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    path = next(vdir.glob("video.*"))
    w, h, dur = probe(path)
    meta = {"title": info.get("title", vid), "duration": dur, "width": w, "height": h}
    meta_file.write_text(json.dumps(meta))
    return Video(vid, meta["title"], path, dur, w, h)


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def extract_frames(video: Video, fps: float, width: int = 960) -> list[tuple[float, Path]]:
    """Extract analysis frames at `fps`; returns [(timestamp_sec, path)]."""
    fdir = video.path.parent / f"frames_{fps:g}fps_{width}"
    marker = fdir / ".done"
    if not marker.exists():
        fdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video.path),
             "-vf", f"fps={fps},scale={width}:-2", "-q:v", "3",
             str(fdir / "f%06d.jpg")],
            check=True,
        )
        marker.touch()
    frames = sorted(fdir.glob("f*.jpg"))
    # ffmpeg fps filter: frame n (1-based) sits at (n-1)/fps
    return [((i) / fps, p) for i, p in enumerate(frames)]


def grab_fullres(video: Video, t: float, out: Path) -> None:
    """Grab a single full-resolution frame at timestamp t."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video.path),
         "-frames:v", "1", "-q:v", "2", str(out)],
        check=True,
    )
