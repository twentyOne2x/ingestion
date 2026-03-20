# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# NOTE: `docker compose build` can build multiple images in parallel. If multiple Dockerfiles
# share the same apt cache mount id, apt can fail with a lock contention error.
RUN --mount=type=cache,id=icmfyi-ingestion-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=icmfyi-ingestion-apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc curl ca-certificates unzip && \
    # yt-dlp's YouTube extractor increasingly relies on running small JS snippets (EJS).
    # Installing deno keeps this container self-contained and avoids host-level deps.
    curl -fsSL https://deno.land/install.sh | sh && \
    mv /root/.deno/bin/deno /usr/local/bin/deno && \
    rm -rf /root/.deno && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements-base.txt
COPY services/diarization_indexer/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "services.diarization_indexer.app:app", "--host", "0.0.0.0", "--port", "8080"]
