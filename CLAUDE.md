# guestshots — notes for Claude

CLI tool: YouTube podcast URL → N screenshots where the guest (not the host) looks flattering and active. See README.md for usage and pipeline.

- Python 3.12 via `uv`; run with `uv run guestshots ...`. Put dependency pins in `pyproject.toml` — ad-hoc `uv pip install` gets reverted by the next `uv run`.
- `mediapipe` is pinned to 0.10.14 on purpose (1.x crashes on macOS/Metal). Don't bump.
- LLM stage goes through **OpenRouter** (`guestshots/vision.py`, OpenAI SDK). Key in `.env` as `OPENROUTER_API_KEY` (gitignored). Default model `openai/gpt-5.4-mini` — benchmarked equal to claude-sonnet-5 on the eye-open check at ~1/3 the cost.
- Iterate on scoring using the cached detections (`cache/<id>/faces_1fps_v2.pkl`); only bump the cache version suffix when `FaceObs` fields change.
- Landmark EARs cannot detect laugh-squints reliably; the LLM `eyes: open|squint|closed` gate is what actually keeps closed eyes out.
- Full-res grabs are verified against the guest embedding because the 1-fps analysis frame can be 0.5 s off (camera cuts).
- Web service: `guestshots/server.py` (FastAPI, jobs as dirs under `$GUESTSHOTS_DATA`, one worker thread, SSE, `X-Api-Key`), SPA in `guestshots/web/index.html` (host photos in IndexedDB). Image built by `.github/workflows/docker.yml` → `ghcr.io/tobsch/guestshots`. Deployed on Tobi's s1max (LXC 200 dockge, `https://guestshots.mhw.wtf`, stack dir `/home/tobias/stacks/guestshots`).
- **onnxruntime in docker with a CPU quota**: without `intra_op_num_threads` = quota and `session.intra_op.allow_spinning=0`, ORT spawns one spinning thread per *host* core and throughput collapses 13x (0.9 vs 11.8 fps measured). `faces.get_app()` reads the cgroup quota / `GUESTSHOTS_THREADS`. Keep `cpus`, `OMP_NUM_THREADS`, `GUESTSHOTS_THREADS` in compose.yaml equal.
- LLM stage = hard gates + soft ranking (`pipeline.produce`): eyes not open, hand_near_face, gaze≠camera (only with `require_gaze_camera`), `usable=false` (incl. free-text `criteria`) DROP the candidate; fewer than `n` shots may come back, `report.llm_rejected` says why. Presets live in `pipeline.PROFILES` (`portrait` = Iskender's quote-graphic use case). Don't "fill up" to `n` with rejected frames — that was the original bug.
- Everything in the repo is English. `hosts/`, `cache/`, `output/`, `.env` are gitignored.
