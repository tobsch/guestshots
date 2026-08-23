FROM python:3.12-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 curl unzip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# InsightFace models baked into the image (else every fresh container downloads ~300 MB)
RUN mkdir -p /root/.insightface/models/buffalo_l && cd /root/.insightface/models/buffalo_l \
    && curl -fsSL -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && unzip -q buffalo_l.zip && rm buffalo_l.zip

COPY guestshots ./guestshots
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" GUESTSHOTS_DATA=/data PORT=8765 OMP_NUM_THREADS=6
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s CMD curl -fs localhost:8765/api/health || exit 1
# keep yt-dlp fresh — YouTube changes break old versions within weeks
CMD ["sh", "-c", "uv pip install -q -U yt-dlp 2>/dev/null; nice -n 10 guestshots-server"]
