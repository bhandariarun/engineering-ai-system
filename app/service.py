import hashlib
import json
import time
from typing import Any

from .config import Settings
from .providers import ModelProvider
from .rag import Retriever
from .schemas import AssistantResponse, Citation
from .tools import TOOLS


class AssistantService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = Retriever()
        self.provider = ModelProvider(settings)
        self.cache: dict[str, tuple[float, AssistantResponse]] = {}

    def ingest(self) -> tuple[int, int]:
        return self.retriever.ingest_directory("data")

    def ask(self, question: str) -> AssistantResponse:
        cache_key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached and time.time() - cached[0] < self.settings.cache_ttl_seconds:
            return cached[1].model_copy(update={"cached": True})

        matches = self.retriever.search(question, self.settings.retrieval_k)
        context = "\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _ in matches)
        degraded = False
        try:
            answer, requested_tools = self.provider.answer(question, context)
            tool_calls = []
            for request in requested_tools:
                function = TOOLS.get(request["name"])
                if function:
                    result = function(**request.get("arguments", {}))
                    tool_calls.append({**request, "result": result})
        except Exception:
            degraded = True
            answer = "The model provider is temporarily unavailable. " + (context or "No matching context was found.")
            tool_calls = []

        response = AssistantResponse(
            answer=answer,
            citations=[Citation(source=chunk.source, chunk_id=chunk.chunk_id, score=round(score, 4)) for chunk, score in matches],
            tool_calls=tool_calls,
            provider=self.provider.name,
            degraded=degraded,
        )
        self.cache[cache_key] = (time.time(), response)
        return response
