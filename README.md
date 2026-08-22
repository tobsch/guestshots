# guestshots

Give it YouTube links of a podcast episode and it returns N screenshots in which **the guest** (not the host) looks good and active — eyes open, smiling or gesturing, sharp, facing the camera.

```
output/<id>_<title>/shots/01_18m28s_score0.72.jpg
                          02_27m13s_score0.77.jpg
                          ...
```

## Setup on a fresh Mac (Apple Silicon or Intel)

```bash
# 1. Homebrew (if missing): https://brew.sh
# 2. System tools
brew install ffmpeg uv git

# 3. Clone + build the Python environment (uv downloads Python 3.12 itself, nothing else needed)
git clone https://github.com/tobsch/guestshots.git
cd guestshots
uv sync

# 4. OpenRouter key for the LLM stage (optional but recommended — ~1 cent per video)
echo 'OPENROUTER_API_KEY=<your-key>' > .env

# 5. Drop host photo(s) in so the tool knows who is NOT the guest
cp ~/Pictures/me.jpg hosts/
```

On first run InsightFace downloads its models once (~300 MB) to `~/.insightface/`.

That's it. No Docker, no GPU, no CUDA — runs on CPU (Apple Silicon uses CoreML automatically).

## Usage

```bash
uv run guestshots 'https://www.youtube.com/watch?v=XXXX'
uv run guestshots 'https://youtu.be/XXXX' -n 8 --solo-only
uv run guestshots URL1 URL2 URL3            # several episodes in one go
```

**Always quote the URL** — zsh chokes on `?` and `&` otherwise.

| Option | Default | Meaning |
|---|---|---|
| `-n` | 5 | Screenshots per video |
| `--solo-only` | off | Only frames where the guest is alone (and large) in the picture |
| `--min-gap` | 20 | Minimum seconds between two shots |
| `--fps` | 1 | Analysis frames per second (higher = more precise, slower) |
| `--host-dir` | `hosts/` | Folder with reference photos of the host |
| `--guest-id N` | — | Force a guest identity (see the `identities/` crops) |
| `--llm / --no-llm` | on | LLM re-ranking via OpenRouter |
| `--llm-model` | `openai/gpt-5.4-mini` | Any OpenRouter vision model, e.g. `anthropic/claude-sonnet-5` |
| `--llm-pool` | 4 | Candidates per final shot sent to the LLM (n × pool images) |

Runtime: ~10 min for an 80-minute episode on the first run (download + face detection), seconds afterwards — video and detections are cached in `cache/`.

## Output

```
output/<id>_<title>/
  shots/01_12m34s_score0.81.jpg   ← the screenshots, full resolution
  contact_sheet.jpg               ← all shots at a glance
  candidates/                     ← face crops the LLM saw
  identities/id0_host_x1423.jpg   ← who was recognised as whom (host / cand)
  report.json                     ← timestamps, scores, LLM notes
```

## How it works

1. `yt-dlp` downloads the video (≤1080p), `ffmpeg` extracts 1 frame/s.
2. InsightFace detects faces, embeddings and landmarks; faces are clustered into identities.
3. **Host vs. guest**
   - Reference photos in `hosts/` → identity with cosine similarity ≥ 0.45 is the host.
   - No reference but ≥2 URLs → the person appearing in several videos is the host.
   - Otherwise: most frequent person = guest (with a warning). Fix with `--guest-id N`.
4. **Per-frame scoring of the guest**, calibrated to the distribution in that video: eyes open (InsightFace **and** MediaPipe must agree), facing the camera, smile or moderately open "talking" mouth, motion in the face region (gesturing), sharpness, light, size, alone in frame. Blinks, laugh-squints, wide-open mouths and turned-away heads are dropped.
5. Candidates are grabbed at full resolution and **verified** (the face in the frame must match the guest embedding) — the 1-fps analysis frame can sit up to 0.5 s off the exact timestamp, which at a camera cut would otherwise hand you the host.
6. **LLM stage** (optional): the best `n × pool` crops go through OpenRouter to the vision model, which first classifies the eyes (`open|squint|closed` — only `open` is usable) and then rates "flattering" and "active" 1–10. Landmark models are unreliable on laugh-squints; a vision LLM with a full-res crop isn't.
7. Top N with minimum spacing → `shots/`.

## Cost

Only the LLM stage costs money: **~1 cent per video** (24 images, `openai/gpt-5.4-mini`; Sonnet would be ~3.5 cents and is no better at the eye check). Everything else runs locally. The usage line is printed on every run.

## Notes / troubleshooting

- **Audio episodes with a static cover** (e.g. older alphalist uploads, 720×720 artwork) → warning "video is (almost) static". Only works with real interview video.
- **"could not determine the guest"** → check `output/.../identities/` to see who was detected, then `--guest-id N` or put a host photo in `hosts/`.
- **The host shows up in the shots** → add a host photo to `hosts/` (easiest: copy the host crop from `identities/`) or use `--solo-only`.
- `mediapipe` is deliberately pinned to 0.10.14 — newer versions crash on macOS (Metal delegate). Don't bump it.
- `.env`, `cache/`, `output/` and `hosts/*` are gitignored — your key and photos never end up in the repo.
