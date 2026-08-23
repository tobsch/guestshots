# guestshots — notes for Claude

CLI tool: YouTube podcast URL → N screenshots where the guest (not the host) looks flattering and active. See README.md for usage and pipeline.

- Python 3.12 via `uv`; run with `uv run guestshots ...`. Put dependency pins in `pyproject.toml` — ad-hoc `uv pip install` gets reverted by the next `uv run`.
- `mediapipe` is pinned to 0.10.14 on purpose (1.x crashes on macOS/Metal). Don't bump.
- LLM stage goes through **OpenRouter** (`guestshots/vision.py`, OpenAI SDK). Key in `.env` as `OPENROUTER_API_KEY` (gitignored). Default model `openai/gpt-5.4-mini` — benchmarked equal to claude-sonnet-5 on the eye-open check at ~1/3 the cost.
- Iterate on scoring using the cached detections (`cache/<id>/faces_1fps_v2.pkl`); only bump the cache version suffix when `FaceObs` fields change.
- Landmark EARs cannot detect laugh-squints reliably; the LLM `eyes: open|squint|closed` gate is what actually keeps closed eyes out.
- Full-res grabs are verified against the guest embedding because the 1-fps analysis frame can be 0.5 s off (camera cuts).
- Everything in the repo is English. `hosts/`, `cache/`, `output/`, `.env` are gitignored.
