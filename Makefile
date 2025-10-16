# Makefile
SHELL = /bin/bash

PROJECT ?= just-skyline-474622-e1
REGION ?= us-central1
ARTIFACT_REPO ?= ingestion
TAG ?= latest
PUBSUB_TOKEN_AUDIENCE ?= https://diarization-indexer-406386298457.us-central1.run.app
PUBSUB_SERVICE_ACCOUNT ?= diarization-indexer@$(PROJECT).iam.gserviceaccount.com
PIPELINE_NOTIFICATIONS_TOPIC ?= ingestion-diarization-ready
GLOBAL_THREAD_LIMIT ?= 6

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .
	pyupgrade

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*

# ---------------------------------------------------------------------------
# Diarization indexer Cloud Run service
# ---------------------------------------------------------------------------
IMAGE_DIARIZATION_INDEXER = $(REGION)-docker.pkg.dev/$(PROJECT)/$(ARTIFACT_REPO)/diarization-indexer:$(TAG)

.PHONY: build-diarization-indexer
build-diarization-indexer:
	docker build -t $(IMAGE_DIARIZATION_INDEXER) -f services/diarization_indexer/Dockerfile .

.PHONY: push-diarization-indexer
push-diarization-indexer:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	docker push $(IMAGE_DIARIZATION_INDEXER)

.PHONY: deploy-diarization-indexer
deploy-diarization-indexer:
	gcloud run deploy diarization-indexer \
		--project $(PROJECT) \
		--region $(REGION) \
		--image $(IMAGE_DIARIZATION_INDEXER) \
		--service-account $(PUBSUB_SERVICE_ACCOUNT) \
		--platform managed \
		--set-env-vars YT_NAMESPACE_CONFIG=/app/src/ingest_v2/configs/namespaces.json,PUBSUB_TOKEN_AUDIENCE=$(PUBSUB_TOKEN_AUDIENCE),PUBSUB_TOKEN_ISSUER=https://accounts.google.com,PUBSUB_SERVICE_ACCOUNT=$(PUBSUB_SERVICE_ACCOUNT),PIPELINE_STORAGE_ROOT=/tmp/pipeline_storage_v2,PIPELINE_NOTIFICATIONS_TOPIC=$(PIPELINE_NOTIFICATIONS_TOPIC),GLOBAL_THREAD_LIMIT=$(GLOBAL_THREAD_LIMIT),GLOBAL_THREAD_STATE_DIR=/tmp/global_thread_state,GLOBAL_THREAD_SCOPE=diarization-indexer \
		--set-secrets YOUTUBE_API_KEY=youtube-api-key:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest,PINECONE_API_ENVIRONMENT=PINECONE_API_ENVIRONMENT:latest,PINECONE_INDEX_NAME=PINECONE_INDEX_NAME:latest,PINECONE_NAMESPACE_VIDEOS=PINECONE_NAMESPACE_VIDEOS:latest,PINECONE_NAMESPACE_STREAMS=PINECONE_NAMESPACE_STREAMS:latest,EMBED_MODEL=EMBED_MODEL:latest,EMBED_DIM=EMBED_DIM:latest,EMBED_PROVIDER=EMBED_PROVIDER:latest \
		--no-allow-unauthenticated
