# syntax=docker/dockerfile:1.7

# Pin immutable multi-architecture indexes. The release build selects linux/amd64;
# the index pins keep local arm64 contract builds possible without mutable tags.
ARG PYTHON_BUILD_IMAGE=python:3.12.13-bookworm@sha256:3cd9086bdb30f7c9bc08a3fa621d9842e0d3f6f9291aeb4677e0547817c10b12
ARG PYTHON_RUNTIME_IMAGE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ARG DENO_IMAGE=denoland/deno:2.5.6@sha256:3ea71953ff50e3ff15c377ead1a8521f624e2f43d27713675a8bed7b33f166aa
ARG FFMPEG_IMAGE=mwader/static-ffmpeg:7.1.1@sha256:11a44711684c0b9f754c047dcd64235b8b52deab251bd0e0a86f22faa160749c

FROM ${DENO_IMAGE} AS deno
FROM ${FFMPEG_IMAGE} AS ffmpeg

# The pinned full Python image contains the compiler toolchain needed by the
# few source distributions in requirements.txt, so no mutable apt install is
# needed. Only built wheels cross the stage boundary.
FROM ${PYTHON_BUILD_IMAGE} AS wheels

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY requirements.txt ./requirements-base.txt
COPY services/diarization_indexer/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir=/wheels -r requirements.txt

FROM ${PYTHON_RUNTIME_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DENO_DIR=/data/deno-cache \
    HOME=/home/icmfyi

WORKDIR /app

# Deno is copied from an immutable official image instead of executing a
# network installer. yt-dlp uses it for YouTube EJS challenge snippets.
COPY --from=deno /usr/bin/deno /usr/local/bin/deno
COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe
COPY --from=wheels /wheels /wheels
COPY requirements.txt ./requirements-base.txt
COPY services/diarization_indexer/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels && \
    groupadd --gid 10001 icmfyi && \
    useradd --uid 10001 --gid 10001 --create-home --home-dir /home/icmfyi \
      --shell /usr/sbin/nologin icmfyi && \
    mkdir -p /data/deno-cache /data/pipeline /data/exports /data/hot-media /data/probes /models && \
    chown -R 10001:10001 /data /models /home/icmfyi

COPY --chown=0:0 . /app
RUN chmod -R a-w /app

USER 10001:10001

CMD ["uvicorn", "services.diarization_indexer.app:app", "--host", "0.0.0.0", "--port", "8080"]
