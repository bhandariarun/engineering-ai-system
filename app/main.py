from collections import defaultdict, deque
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import ChatRequest, HealthResponse, IngestResponse
from .service import AssistantService

settings = get_settings()
service = AssistantService(settings)
service.ingest()
requests_by_client: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    recent = requests_by_client[client]
    while recent and recent[0] <= now - 60:
        recent.popleft()
    if len(recent) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    recent.append(now)
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", provider=service.provider.name, indexed_chunks=len(service.retriever.chunks))


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    documents, chunks = service.ingest()
    return IngestResponse(documents=documents, chunks=chunks)


@app.post("/chat")
def chat(request: ChatRequest):
    return service.ask(request.question)
