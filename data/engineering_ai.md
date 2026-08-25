# Engineering AI Assistant

This assistant answers questions about the reference architecture, reliability controls, and deployment model in this repository.

## Architecture

The system has a Streamlit client, a FastAPI service, a retrieval layer, and an OpenAI-compatible model provider. Retrieval uses local TF-IDF vectors by default so the demo remains runnable offline. In production, the provider can point to OpenAI, Azure OpenAI, vLLM, or another compatible endpoint.

## Reliability

The API applies an in-process rate limit, retries transient provider errors, caches successful responses, and returns a retrieval-grounded degraded response when no model provider is configured or the provider fails.

## Operations

Run the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000` and the UI with `streamlit run app/ui.py`. Docker Compose starts both services.
