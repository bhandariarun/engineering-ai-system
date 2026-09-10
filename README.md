# Engineering AI Assistant

A production-shaped AI assistant that demonstrates LLM integration, structured output, tool calling, RAG, reliability controls, and containerized deployment. It runs offline by default and can connect to OpenAI, Azure OpenAI, or a vLLM OpenAI-compatible endpoint.

## Features

- FastAPI backend with Pydantic request and response validation
- Streamlit chat UI
- Markdown/text ingestion with overlapping chunks and local TF-IDF retrieval
- OpenAI-compatible provider with configurable temperature, top-p, retries, JSON output, and function calling
- Allow-listed calculator and UTC time tools
- Bounded agentic research loop that can search again, use a tool, ask for clarification, or finalize
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

## W16 Agentic Extension

### Why a Fixed Pipeline Is Not Enough

A fixed pipeline cannot handle cross-source verification because it must decide how many searches to run before it has seen whether the first evidence is complete or contradictory.

### Context Engineering Technique

The agent applies context capping and progressive history pruning inside `AssistantService.ask`. Each decision receives at most `agent_context_chars` of evidence and the last four compact history records, while the full retrieval result is not repeatedly copied into the model prompt. This addresses context growth during repeated searches and keeps the model focused on the latest evidence and action outcomes. Retrieved chunks are retained separately for citations.

### Agentic Pattern

This is a single-agent loop. One model owns the decision sequence because the task requires shared evidence and lightweight branching, not independent specialist outputs. A multi-agent design would add coordination tokens and a sequential bottleneck without providing useful context isolation; the bounded loop already limits context saturation. The model chooses `search`, `tool`, `clarify`, or `final` after each result, with `agent_max_steps` as the stopping condition.

### Evaluation Harness

`scripts/evaluate_agent.py` is a from-scratch harness using scripted model decisions against the real `AssistantService` loop. It measures task completion, tool-call correctness, trajectory length, and token usage. The checked-in report is in [evaluation/results.md](evaluation/results.md). A provider timeout is injected on the second decision; the service records `agent_failure` and returns an explicit degraded response instead of inventing a final answer. A never-ending scripted trajectory is also tested as a soft failure at the step cap; an invalid intermediate result that causes a later wrong decision is the cascading soft-failure category.

### Skill Versus Agent

This capability could be described as a research Skill, but a Skill alone would not be sufficient because the model must choose the next action from intermediate results; the bounded decision loop is therefore implemented as an agent.

### Tool Versus Agent Boundary

The calculator and UTC clock are modeled as bounded tool calls, not agent-to-agent interactions. They are stateless, allow-listed functions with one request and one result; the research agent decides when to call them and receives their result on the next iteration. The provider is a model endpoint, while orchestration, state, limits, and failure handling remain in the application.

## Project layout

- `app/main.py`: API, health, ingestion, and rate limiting
- `app/service.py`: orchestration, caching, fallback, citations
- `app/rag.py`: ingestion, chunking, and retrieval
- `app/providers.py`: structured LLM calls and retry policy
- `app/tools.py`: allow-listed function tools
- `app/ui.py`: Streamlit client