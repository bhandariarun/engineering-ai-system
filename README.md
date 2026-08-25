# Engineering AI Assistant

A production-shaped AI assistant that demonstrates LLM integration, structured output, tool calling, RAG, reliability controls, and containerized deployment. It runs offline by default and can connect to OpenAI, Azure OpenAI, or a vLLM OpenAI-compatible endpoint.

## Features

- FastAPI backend with Pydantic request and response validation
- Streamlit chat UI
- Markdown/text ingestion with overlapping chunks and local TF-IDF retrieval
- OpenAI-compatible provider with configurable temperature, top-p, retries, JSON output, and function calling
- Allow-listed calculator and UTC time tools
- In-process response cache and sliding-window rate limiting
- Graceful retrieval-grounded fallback when a model provider is unavailable
- Dockerfile and Docker Compose configuration
- Architecture diagram in [docs/architecture.md](docs/architecture.md)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
API_URL=http://localhost:8000 streamlit run app/ui.py
```

Open `http://localhost:8501`. The API is documented at `http://localhost:8000/docs`.

Without `OPENAI_API_KEY`, the assistant uses the local retrieval-grounded fallback. To use OpenAI or vLLM, set `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`, and `CHAT_MODEL` in `.env`.

## Docker Compose

```bash
docker compose up --build
```

The UI is at `http://localhost:8501`; the API is at `http://localhost:8000`.

## Local model with vLLM

Start vLLM separately with its OpenAI-compatible server, then set for example:

```env
OPENAI_API_KEY=token
OPENAI_BASE_URL=http://host.docker.internal:8001/v1
CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

The application does not convert the model to ONNX because the target providers are decoder-only generative models served through an OpenAI-compatible API. vLLM's continuous batching and optimized kernels are the selected inference optimization for local serving.

## Tests

```bash
pytest -q
```

## Project layout

- `app/main.py`: API, health, ingestion, and rate limiting
- `app/service.py`: orchestration, caching, fallback, citations
- `app/rag.py`: ingestion, chunking, and retrieval
- `app/providers.py`: structured LLM calls and retry policy
- `app/tools.py`: allow-listed function tools
- `app/ui.py`: Streamlit client