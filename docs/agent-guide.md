# guestshots — guide for AI agents (Claude, Cowork, scripts)

guestshots turns a YouTube podcast episode into a handful of flattering, *active* screenshots of the **guest** — eyes open, smiling or gesturing, sharp, not the host. You give it a YouTube URL and one or more photos of the host (so it knows whom to exclude); it returns JPEGs at the video's full resolution.

This page is written so an agent can use the HTTP API end to end without reading the source.

## 0. What you need

| | |
|---|---|
| Base URL | `https://guestshots.mhw.wtf` (replace if self-hosted) |
| API key | given to you by the operator — send it as header `X-Api-Key: <key>` (or `?key=<key>` on GET links) |
| Host photo(s) | 1–3 clear frontal JPG/PNG of the **host**. Without them the tool assumes the most frequent face is the guest — usually wrong in a two-person interview. |
| Time | an 80-minute episode takes ~8–10 minutes end to end. Jobs run one at a time; yours may queue. |
| Cost | ~1 cent per job (vision LLM re-ranking), paid by the operator. Nothing else. |

Every API key sees **only its own jobs**. Jobs and their images are deleted after 7 days.

## 1. The fast path: one synchronous call

If your runtime can hold an HTTP connection open for ~10–15 minutes, this is all you need:

```bash
curl -sS -X POST https://guestshots.mhw.wtf/api/shots \
  -H "X-Api-Key: $GUESTSHOTS_KEY" \
  -F url='https://www.youtube.com/watch?v=XXXXXXXXXXX' \
  -F hosts=@host1.jpg -F hosts=@host2.jpg \
  -F n=6 -F solo_only=true \
  -o shots.zip

# quote-graphic backgrounds with an extra rule:
curl -sS -X POST https://guestshots.mhw.wtf/api/shots -H "X-Api-Key: $GUESTSHOTS_KEY" \
  -F url='https://www.youtube.com/watch?v=XXXXXXXXXXX' -F hosts=@host1.jpg \
  -F n=6 -F profile=portrait -F criteria='no coffee cup or microphone covering the mouth' -o shots.zip
```

Returns `200` with a ZIP containing `01_…jpg … 06_…jpg`, `contact_sheet.jpg` and `report.json`. The response header `X-Job-Id` lets you fetch the same files individually later (section 3).

Errors: `401` bad key · `400` not a YouTube URL / bad photo type · `422` the pipeline could not find a guest (see `report`/`error`) · `504` timeout (default 45 min; the job keeps running — poll it via `/api/jobs/{id}`).

## 2. The robust path: async job + polling

Use this when your tool calls time out after a minute or two (most agent sandboxes).

### 2.1 Create the job

```
POST /api/jobs        (multipart/form-data)
```

| field | type | default | meaning |
|---|---|---|---|
| `url` | string | required | YouTube URL (`watch?v=`, `youtu.be/`, `live/`, `shorts/`) |
| `hosts` | file, repeatable | — | host reference photos (jpg/png/webp, max 10) |
| `n` | int | 5 | how many screenshots you want (1–30) |
| `solo_only` | bool | false | only frames where the guest is alone in the picture — **recommended `true`** for social media |
| `min_gap` | float | 20 | minimum seconds between two screenshots (variety) |
| `llm` | bool | true | vision-LLM re-ranking (keeps closed eyes / mid-word mouths out). Leave on. |
| `llm_model` | string | `openai/gpt-5.4-mini` | any OpenRouter vision model id |
| `llm_pool` | int | 4 | candidates per final shot sent to the LLM (n × pool images) |
| `guest_id` | int | — | force a specific detected identity as the guest (see 3.1) |
| `profile` | string | — | option preset. `portrait` = stills meant as backgrounds for quote graphics: `solo_only=true`, `require_gaze_camera=true`, `max_face_bottom=0.55`, `llm_pool=6`. Explicit fields override the profile. |
| `criteria` | string ≤500 chars | — | free-text extra requirements for the vision LLM, e.g. `"no hand anywhere in the picture, no strong reflections in the glasses"`. A violation makes the image unusable; the reason lands in `parts.llm_note`. |
| `require_gaze_camera` | bool | false | LLM gate: only frames where the guest looks into the camera. Strict — in two-person studio recordings the guest usually looks at the host, expect few or zero survivors. |
| `max_face_bottom` | float 0–1 | — | drop frames whose face box bottom is below this fraction of the frame height (leaves room for text under the chin) |

Response `202`:

```json
{"id":"58ce1db49a2d","status":"queued","stage":"queued","queue_position":1,"url":"…","created":1787462546.2, …}
```

Keep `id`.

### 2.2 Poll until done

```
GET /api/jobs/{id}
```

```json
{"id":"58ce1db49a2d","status":"running","stage":"detecting","done":2025,"total":5053,
 "title":"Dominik Schwarz, Partner Verdane – …","log":["guest = identity #1 (2469 frames), host = [0, 12]"], …}
```

`status` is one of `queued → running → done | failed`. `stage` walks through `downloading → extracting → detecting (done/total) → scoring → verifying → ranking → done`. Poll every 15–30 s; a realistic budget is 40 polls. On `failed`, `error` holds the reason (most common: *could not determine the guest* → add host photos or use `guest_id`).

If you can consume server-sent events instead of polling: `GET /api/jobs/{id}/events` streams the same JSON whenever it changes and closes on `done`/`failed`.

### 2.3 Fetch the results

When `status == "done"`, `shots` lists the file names in rank order:

```json
"shots":["01_27m13s_score0.74.jpg","02_51m27s_score0.75.jpg", …]
```

| | |
|---|---|
| `GET /api/jobs/{id}/shots/{file}` | one screenshot, `image/jpeg`, full video resolution (e.g. 1920×1080). Name encodes rank and timestamp. |
| `GET /api/jobs/{id}/shots.zip` | everything in one ZIP |
| `GET /api/jobs/{id}/contact_sheet.jpg` | all shots tiled — good for showing a human a quick overview |
| `GET /api/jobs/{id}/report.json` | per shot: timestamp `t` (seconds), `score`, `solo`, **`face`** = `{x, y, w, h}` of the guest's face box normalised to 0..1 of the delivered image (use it to compute crops and text placement), `parts.llm_note` (one-sentence reason), `parts.llm_eyes`, `parts.llm_gaze` (`camera|away`), `parts.llm_hand_near_face`. Top level: `requested` (the `n` you asked for), `llm_rejected` = `{eyes: 2, gaze: 21, hand_near_face: 1}` — why candidates were thrown out. |

Image GETs also accept `?key=…` so you can hand the URL to something that cannot set headers (an `<img>` tag, a chat attachment fetcher).

**You may get fewer than `n` shots.** The LLM gates (eyes open, no hand at the face, optionally gaze at camera, your `criteria`) drop candidates instead of returning bad images. Check `len(shots)` against `report.requested`; `report.llm_rejected` tells you which gate bit. Remedies: raise `llm_pool` (up to 8), drop `require_gaze_camera`, loosen `criteria`, or re-run with a larger `n`.

### 2.4 Clean up (optional)

`DELETE /api/jobs/{id}` removes the job and its images immediately. Otherwise they expire after 7 days.

## 3. Things worth knowing

### 3.1 How the host is excluded
Faces are clustered into identities; any identity matching your host photos (cosine similarity ≥ 0.45) is removed, and the most frequent remaining identity is the guest. `GET /api/jobs/{id}/identities` lists crops like `id0_host_x808.jpg`, `id1_cand_x2402.jpg` (`x…` = frames in which it appears); `GET /api/jobs/{id}/identities/{file}` returns the crop. If the wrong person was picked, re-run with `guest_id=<N>`.

### 3.2 Hard gates vs. soft ranking
Gates (an image is dropped): eyes not `open`, `hand_near_face`, `gaze != camera` when `require_gaze_camera` is set, `usable=false` (the model's verdict, including violations of your `criteria`), face box below `max_face_bottom`. Everything else is soft: `flattering` and `active` (1–10) are blended with the local heuristic score for the final ranking.

### 3.3 What "good" means here
Locally each guest frame is scored on: eyes open (two independent landmark models must agree), facing the camera, a smile or a moderately open "talking" mouth (not a gape), motion in the face region (gesturing = active), sharpness, lighting, size in frame, alone in frame. The top `n × llm_pool` candidates are then cropped at full resolution and sent to a vision LLM that classifies eyes `open | squint | closed` and rates *flattering* and *active* 1–10. Only `open` survives. This LLM gate is what actually keeps laugh-squints and blinks out — landmark models miss those.

### 3.4 Limits
- YouTube only (anything yt-dlp can download works, but only YouTube ids are validated).
- Audio-only uploads with a static cover image produce identical "screenshots"; the log will say *video is (almost) static*.
- Max 10 host photos, `n` ≤ 30, one job at a time server-wide.
- Videos are cached for 24 h; re-running the same URL with different options is fast (~1 min).

### 3.5 Idempotency / retries
Creating the same job twice creates two jobs (and pays the LLM twice). Keep the `id` and poll instead of re-posting. `GET /api/jobs` lists all your jobs, newest first — check there before re-submitting after a crash.

## 4. Minimal Python client

```python
import time, requests

BASE, KEY = "https://guestshots.mhw.wtf", "…"
H = {"X-Api-Key": KEY}

def guestshots(url, host_photos, n=6, solo_only=True):
    files = [("hosts", open(p, "rb")) for p in host_photos]
    job = requests.post(f"{BASE}/api/jobs", headers=H, files=files,
                        data={"url": url, "n": n, "solo_only": solo_only}).json()
    while job["status"] not in ("done", "failed"):
        time.sleep(20)
        job = requests.get(f"{BASE}/api/jobs/{job['id']}", headers=H).json()
    if job["status"] == "failed":
        raise RuntimeError(job["error"])
    return [(s, requests.get(f"{BASE}/api/jobs/{job['id']}/shots/{s}", headers=H).content)
            for s in job["shots"]]

for name, jpeg in guestshots("https://youtu.be/XXXXXXXXXXX", ["host.jpg"]):
    open(name, "wb").write(jpeg)
```

## 5. Endpoint summary

| method | path | auth | purpose |
|---|---|---|---|
| POST | `/api/shots` | key | synchronous: returns ZIP |
| POST | `/api/jobs` | key | create job → `202 {id}` |
| GET | `/api/jobs` | key | your jobs, newest first |
| GET | `/api/jobs/{id}` | key | status / progress |
| GET | `/api/jobs/{id}/events` | key | SSE stream of status |
| GET | `/api/jobs/{id}/shots/{file}` | key | one JPEG |
| GET | `/api/jobs/{id}/shots.zip` | key | ZIP of all |
| GET | `/api/jobs/{id}/contact_sheet.jpg` | key | overview image |
| GET | `/api/jobs/{id}/report.json` | key | scores + LLM notes |
| GET | `/api/jobs/{id}/identities` · `/identities/{file}` | key | who was detected |
| DELETE | `/api/jobs/{id}` | key | delete |
| GET | `/api/health` | — | liveness (`{"ok":true,…}`) |
| GET | `/api/docs` | — | interactive OpenAPI UI |

A human-friendly web app lives at `/` (same API key, host photos are kept in the browser).
