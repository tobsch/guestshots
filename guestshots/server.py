"""guestshots web service: REST API + single-page web app.

State = directories on disk (no database):
  $GUESTSHOTS_DATA/jobs/<id>/job.json, hosts/, out/{shots,contact_sheet.jpg,report.json,...}
  $GUESTSHOTS_DATA/cache/<video id>/  (downloaded video + frames + detections, TTL-cleaned)
One worker thread processes jobs sequentially (the pipeline is CPU-bound).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from . import pipeline as P

DATA = Path(os.environ.get("GUESTSHOTS_DATA", "/data"))
JOBS = DATA / "jobs"
CACHE = DATA / "cache"
# Keys: GUESTSHOTS_API_KEYS="tobi:abc123,iskender:def456" (name:key pairs) and/or legacy GUESTSHOTS_API_KEY=abc123.
API_KEYS: dict[str, str] = {}  # key -> owner name
for _pair in os.environ.get("GUESTSHOTS_API_KEYS", "").split(","):
    if ":" in _pair:
        _name, _key = _pair.split(":", 1)
        API_KEYS[_key.strip()] = _name.strip()
if os.environ.get("GUESTSHOTS_API_KEY"):
    API_KEYS[os.environ["GUESTSHOTS_API_KEY"]] = "default"
JOB_TTL = float(os.environ.get("GUESTSHOTS_JOB_TTL_DAYS", "7")) * 86400
CACHE_TTL = float(os.environ.get("GUESTSHOTS_CACHE_TTL_HOURS", "24")) * 3600
MAX_HOSTS = 10
WEB = Path(__file__).parent / "web" / "index.html"

app = FastAPI(title="guestshots", docs_url="/api/docs", openapi_url="/api/openapi.json")


# ---------- auth ----------

def require_key(request: Request) -> str:
    """Returns the owner name of the presented key. Each owner only sees their own jobs."""
    if not API_KEYS:
        raise HTTPException(500, "GUESTSHOTS_API_KEYS not configured")
    key = request.headers.get("x-api-key") or request.query_params.get("key") or ""
    if key not in API_KEYS:
        raise HTTPException(401, "invalid api key")
    return API_KEYS[key]


# ---------- jobs on disk ----------

@dataclass
class Job:
    id: str
    url: str
    opts: dict
    created: float
    status: str = "queued"       # queued | running | done | failed
    stage: str = "queued"
    done: int = 0
    total: int = 0
    title: str = ""
    log: list[str] = field(default_factory=list)
    error: str | None = None
    shots: list[str] = field(default_factory=list)
    finished: float | None = None
    n_hosts: int = 0
    owner: str = "default"

    @property
    def dir(self) -> Path:
        return JOBS / self.id

    def save(self) -> None:
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(self.dir / "job.json")

    @classmethod
    def load(cls, d: Path) -> "Job":
        return cls(**json.loads((d / "job.json").read_text()))


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_queue: Queue[str] = Queue()


def _public(j: Job) -> dict:
    d = asdict(j)
    d["queue_position"] = (
        [x for x in sorted(_jobs.values(), key=lambda x: x.created) if x.status == "queued"].index(j) + 1
        if j.status == "queued" else None)
    return d


def _worker() -> None:
    while True:
        jid = _queue.get()
        j = _jobs.get(jid)
        if not j or j.status != "queued":
            continue

        def progress(stage: str, done: int = 0, total: int = 0, msg: str = "") -> None:
            with _lock:
                j.stage, j.done, j.total = stage, done, total
                if msg:
                    j.log.append(msg)
                    j.log = j.log[-50:]
                if stage == "extracting" and msg:
                    j.title = msg.split(" (")[0]
                j.save()

        with _lock:
            j.status = "running"
            j.save()
        try:
            refs = sorted((j.dir / "hosts").glob("*"))
            host_embs = P.host_embeddings(refs)
            opts = P.Options(**j.opts)
            a = P.analyze(j.url, CACHE, opts.fps, progress)
            P.mark_hosts([a], host_embs, opts.host_sim)
            res = P.produce(a, opts, j.dir / "out", progress)
            with _lock:
                j.status, j.stage = "done", "done"
                j.title = a.vid.title
                j.shots = [p.name for p in res.shots]
        except Exception as e:  # noqa: BLE001 — whatever failed, the job failed
            with _lock:
                j.status, j.stage, j.error = "failed", "failed", f"{type(e).__name__}: {e}"
        finally:
            shutil.rmtree(j.dir / "hosts", ignore_errors=True)  # stateless: host photos never outlive the job
            shutil.rmtree(j.dir / "out" / "candidates", ignore_errors=True)
            with _lock:
                j.finished = time.time()
                j.save()


def _janitor() -> None:
    while True:
        now = time.time()
        with _lock:
            for j in list(_jobs.values()):
                if j.status in ("done", "failed") and now - (j.finished or j.created) > JOB_TTL:
                    shutil.rmtree(j.dir, ignore_errors=True)
                    _jobs.pop(j.id, None)
        active = {P_video_id(j.url) for j in _jobs.values() if j.status in ("queued", "running")}
        for d in CACHE.glob("*"):
            if d.is_dir() and d.name not in active and now - d.stat().st_mtime > CACHE_TTL:
                shutil.rmtree(d, ignore_errors=True)
        time.sleep(600)


def P_video_id(url: str) -> str:
    try:
        return P.V.video_id(url)
    except ValueError:
        return ""


@app.on_event("startup")
def _startup() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for d in sorted(JOBS.glob("*")):
        if (d / "job.json").exists():
            j = Job.load(d)
            if j.status == "running":
                j.status, j.stage = "queued", "queued"  # server restarted mid-job → redo
                j.log.append("server restarted, job re-queued")
                j.save()
            _jobs[j.id] = j
    for j in sorted(_jobs.values(), key=lambda x: x.created):
        if j.status == "queued":
            _queue.put(j.id)
    threading.Thread(target=_worker, daemon=True, name="worker").start()
    threading.Thread(target=_janitor, daemon=True, name="janitor").start()
    # warm the models so the first job doesn't pay for it
    threading.Thread(target=lambda: (P.F.get_app(), P.F.get_mesh()), daemon=True).start()


# ---------- API ----------

async def _create_job(owner: str, url: str, hosts: list[UploadFile], form: dict) -> Job:
    if not P_video_id(url):
        raise HTTPException(400, "not a YouTube URL")
    if len(hosts) > MAX_HOSTS:
        raise HTTPException(400, f"max {MAX_HOSTS} host photos")
    try:
        opts = P.Options.build(form.pop("profile"), **form)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    opts.n = max(1, min(opts.n, 30))
    opts.llm_pool = max(1, min(opts.llm_pool, 8))
    opts.criteria = opts.criteria[:500]
    j = Job(id=uuid.uuid4().hex[:12], url=url, created=time.time(), opts=asdict(opts),
            n_hosts=len(hosts), owner=owner)
    (j.dir / "hosts").mkdir(parents=True)
    for i, h in enumerate(hosts):
        ext = Path(h.filename or "x.jpg").suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(400, f"unsupported host photo type {ext}")
        (j.dir / "hosts" / f"host{i}{ext}").write_bytes(await h.read())
    with _lock:
        j.save()
        _jobs[j.id] = j
    _queue.put(j.id)
    return j


@app.post("/api/jobs", status_code=202)
async def create_job(
    owner: str = Depends(require_key),
    url: str = Form(...), hosts: list[UploadFile] = File(default=[]),
    n: int | None = Form(None), solo_only: bool | None = Form(None), min_gap: float | None = Form(None),
    llm: bool | None = Form(None), llm_model: str | None = Form(None), llm_pool: int | None = Form(None),
    guest_id: int | None = Form(None), host_sim: float | None = Form(None),
    profile: str | None = Form(None), criteria: str | None = Form(None),
    max_face_bottom: float | None = Form(None), require_gaze_camera: bool | None = Form(None),
):
    j = await _create_job(owner, url, hosts, dict(n=n, solo_only=solo_only, min_gap=min_gap, llm=llm, llm_model=llm_model,
                                                  llm_pool=llm_pool, guest_id=guest_id, host_sim=host_sim, profile=profile,
                                                  criteria=criteria, max_face_bottom=max_face_bottom,
                                                  require_gaze_camera=require_gaze_camera))
    return _public(j)


@app.get("/api/jobs")
def list_jobs(owner: str = Depends(require_key)):
    with _lock:
        return [_public(j) for j in sorted(_jobs.values(), key=lambda x: x.created, reverse=True) if j.owner == owner]


def _get(jid: str, owner: str) -> Job:
    j = _jobs.get(jid)
    if not j or j.owner != owner:
        raise HTTPException(404, "no such job")
    return j


@app.get("/api/jobs/{jid}")
def get_job(jid: str, owner: str = Depends(require_key)):
    with _lock:
        return _public(_get(jid, owner))


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str, owner: str = Depends(require_key)):
    with _lock:
        j = _get(jid, owner)
        if j.status == "running":
            raise HTTPException(409, "job is running")
        shutil.rmtree(j.dir, ignore_errors=True)
        _jobs.pop(jid, None)
    return {"deleted": jid}


@app.get("/api/jobs/{jid}/events")
async def job_events(jid: str, owner: str = Depends(require_key)):
    _get(jid, owner)

    async def gen():
        last = None
        while True:
            with _lock:
                j = _jobs.get(jid)
                if not j:
                    break
                snap = json.dumps(_public(j))
            if snap != last:
                yield f"data: {snap}\n\n"
                last = snap
            if j.status in ("done", "failed"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _file(j: Job, rel: str) -> FileResponse:
    p = (j.dir / "out" / rel).resolve()
    if not p.is_relative_to(j.dir.resolve()) or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/api/jobs/{jid}/shots/{name}")
def get_shot(jid: str, name: str, owner: str = Depends(require_key)):
    return _file(_get(jid, owner), f"shots/{name}")


@app.get("/api/jobs/{jid}/contact_sheet.jpg")
def get_sheet(jid: str, owner: str = Depends(require_key)):
    return _file(_get(jid, owner), "contact_sheet.jpg")


@app.get("/api/jobs/{jid}/report.json")
def get_report(jid: str, owner: str = Depends(require_key)):
    return _file(_get(jid, owner), "report.json")


@app.get("/api/jobs/{jid}/identities/{name}")
def get_identity(jid: str, name: str, owner: str = Depends(require_key)):
    return _file(_get(jid, owner), f"identities/{name}")


@app.get("/api/jobs/{jid}/identities")
def list_identities(jid: str, owner: str = Depends(require_key)):
    j = _get(jid, owner)
    return sorted(p.name for p in (j.dir / "out" / "identities").glob("*.jpg"))


def _zip(j: Job) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name in j.shots:
            z.write(j.dir / "out" / "shots" / name, name)
        for extra in ("contact_sheet.jpg", "report.json"):
            if (j.dir / "out" / extra).exists():
                z.write(j.dir / "out" / extra, extra)
    return buf.getvalue()


@app.get("/api/jobs/{jid}/shots.zip")
def get_zip(jid: str, owner: str = Depends(require_key)):
    j = _get(jid, owner)
    if j.status != "done":
        raise HTTPException(409, f"job is {j.status}")
    return Response(_zip(j), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="guestshots_{P_video_id(j.url)}.zip"'})


@app.post("/api/shots")
async def sync_shots(
    owner: str = Depends(require_key),
    url: str = Form(...), hosts: list[UploadFile] = File(default=[]),
    n: int | None = Form(None), solo_only: bool | None = Form(None), min_gap: float | None = Form(None),
    llm: bool | None = Form(None), llm_model: str | None = Form(None), llm_pool: int | None = Form(None),
    guest_id: int | None = Form(None), host_sim: float | None = Form(None),
    profile: str | None = Form(None), criteria: str | None = Form(None),
    max_face_bottom: float | None = Form(None), require_gaze_camera: bool | None = Form(None), timeout: int = Form(2700),
):
    """Synchronous variant: blocks until the job is done and returns the ZIP (for scripts / agents)."""
    j = await _create_job(owner, url, hosts, dict(n=n, solo_only=solo_only, min_gap=min_gap, llm=llm, llm_model=llm_model,
                                                  llm_pool=llm_pool, guest_id=guest_id, host_sim=host_sim, profile=profile,
                                                  criteria=criteria, max_face_bottom=max_face_bottom,
                                                  require_gaze_camera=require_gaze_camera))
    t0 = time.time()
    while j.status not in ("done", "failed"):
        if time.time() - t0 > timeout:
            raise HTTPException(504, f"job {j.id} still {j.status}; poll /api/jobs/{j.id}")
        await asyncio.sleep(2)
    if j.status == "failed":
        raise HTTPException(422, j.error or "failed")
    return Response(_zip(j), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="guestshots_{P_video_id(j.url)}.zip"',
                             "X-Job-Id": j.id})


@app.get("/api/profiles")
def profiles():
    return P.PROFILES


@app.get("/api/health")
def health():
    with _lock:
        return {"ok": True, "jobs": len(_jobs),
                "running": sum(1 for j in _jobs.values() if j.status == "running"),
                "queued": sum(1 for j in _jobs.values() if j.status == "queued"),
                "llm": bool(os.environ.get("OPENROUTER_API_KEY")), "keys": len(API_KEYS)}


@app.get("/", response_class=HTMLResponse)
def index():
    return WEB.read_text()


@app.exception_handler(HTTPException)
async def _http_exc(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8765")), log_level="info")
