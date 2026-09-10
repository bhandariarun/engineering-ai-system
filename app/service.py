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

        degraded = False
        matches = []
        cited_matches = {}
        evidence = ""
        history: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        answer = ""
        total_tokens = 0
        iterations = 0
        stop_reason = "max_steps"
        try:
            for _ in range(self.settings.agent_max_steps):
                iterations += 1
                decision, tokens = self.provider.agent_step(question, evidence[-self.settings.agent_context_chars:], history[-4:])
                total_tokens += tokens
                action = decision.get("action")
                if action == "search":
                    query = str(decision.get("query") or question)
                    matches = self.retriever.search(query, self.settings.retrieval_k)
                    cited_matches.update({chunk.chunk_id: (chunk, score) for chunk, score in matches})
                    evidence = "\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _ in matches)
                    history.append({"action": "search", "query": query, "result_count": len(matches)})
                elif action == "tool":
                    requested_tools = decision.get("tool_calls", [])
                    results = []
                    for request in requested_tools:
                        function = TOOLS.get(request.get("name"))
                        if not function:
                            results.append({"name": request.get("name"), "error": "tool is unavailable"})
                            continue
                        try:
                            result = function(**request.get("arguments", {}))
                            call = {**request, "result": result}
                        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                            call = {**request, "error": str(exc)}
                        tool_calls.append(call)
                        results.append(call)
                    evidence = f"{evidence}\nTool results: {json.dumps(results)}"[-self.settings.agent_context_chars:]
                    history.append({"action": "tool", "result_count": len(results)})
                elif action == "clarify":
                    answer = str(decision.get("answer") or "Could you clarify your request?")
                    stop_reason = "clarification"
                    break
                elif action == "final":
                    answer = str(decision.get("answer") or "I could not produce an answer.")
                    stop_reason = "completed"
                    break
                else:
                    raise ValueError(f"unsupported agent action: {action}")
                history.append({"action": action, "evidence_chars": len(evidence)})
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            degraded = True
            answer = "The agent could not complete its research loop safely. " + str(exc or evidence or "No matching context was found.")
            stop_reason = "agent_failure"
        if not answer:
            answer = "The agent reached its step limit before it could verify a final answer."
            degraded = True

        response = AssistantResponse(
            answer=answer,
            citations=[Citation(source=chunk.source, chunk_id=chunk.chunk_id, score=round(score, 4)) for chunk, score in cited_matches.values()],
            tool_calls=tool_calls,
            provider=self.provider.name,
            degraded=degraded,
            iterations=iterations,
            token_usage=total_tokens,
            stop_reason=stop_reason,
        )
        self.cache[cache_key] = (time.time(), response)
        return response
