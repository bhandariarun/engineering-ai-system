import json
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .config import Settings


class ModelProvider:
    name = "offline"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url) if settings.openai_api_key else None
        if self.client:
            self.name = "openai-compatible"

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=0.2, max=2))
    def answer(self, question: str, context: str) -> tuple[str, list[dict[str, Any]]]:
        if not self.client:
            return self._offline_answer(question, context), []
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_tokens=self.settings.max_tokens,
            response_format={"type": "json_object"},
            tools=[
                {"type": "function", "function": {"name": "calculator", "description": "Evaluate basic arithmetic", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
                {"type": "function", "function": {"name": "current_time", "description": "Get the current UTC time", "parameters": {"type": "object", "properties": {}}}},
            ],
            messages=[
                {"role": "system", "content": "Return only JSON with keys answer and tool_calls. Use the supplied context and cite source names in the answer."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        message = response.choices[0].message
        tool_calls = [{"name": call.function.name, "arguments": json.loads(call.function.arguments)} for call in (message.tool_calls or [])]
        payload = json.loads(message.content or "{}")
        return str(payload.get("answer", "I could not produce an answer.")), tool_calls

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=0.2, max=2))
    def agent_step(self, question: str, evidence: str, history: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        """Choose the next action from current evidence instead of following a fixed workflow."""
        if not self.client:
            return self._offline_agent_step(question, evidence, history), 0
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_tokens=self.settings.max_tokens,
            response_format={"type": "json_object"},
            tools=[
                {"type": "function", "function": {"name": "calculator", "description": "Evaluate basic arithmetic", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
                {"type": "function", "function": {"name": "current_time", "description": "Get the current UTC time", "parameters": {"type": "object", "properties": {}}}},
            ],
            messages=[
                {"role": "system", "content": "You are a bounded research agent. Return JSON with action, query, answer, and tool_calls. action must be one of search, tool, clarify, final. Start with search when evidence is empty. After a search, assess whether evidence is sufficient; search again with a narrower or alternate query when needed. Use tool only when it adds evidence. Use clarify when the request is ambiguous. Use final only when supported by evidence or when explaining that evidence is unavailable."},
                {"role": "user", "content": json.dumps({"question": question, "evidence": evidence, "history": history})},
            ],
        )
        message = response.choices[0].message
        tool_calls = [{"name": call.function.name, "arguments": json.loads(call.function.arguments)} for call in (message.tool_calls or [])]
        payload = json.loads(message.content or "{}")
        payload["tool_calls"] = tool_calls or payload.get("tool_calls", [])
        usage = getattr(response, "usage", None)
        return payload, int(getattr(usage, "total_tokens", 0) or 0)

    @staticmethod
    def _offline_answer(question: str, context: str) -> str:
        normalized = question.strip().lower().rstrip(".!?")
        if normalized in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
            return "Hi! I can help with the assistant architecture, reliability controls, deployment, and indexed project notes."
        if context:
            source, _, text = context.partition("] ")
            sentences = [sentence.strip() for sentence in text.replace("##", ".").split(".") if sentence.strip()]
            keywords = [word for word in normalized.split() if len(word) > 3]
            relevant = next((sentence for sentence in sentences if any(word in sentence.lower() for word in keywords)), sentences[0])
            return f"Based on the indexed project notes: {relevant}."
        return f"I can answer once relevant documents are indexed. Your question was: {question}"

    @staticmethod
    def _offline_agent_step(question: str, evidence: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = question.strip().lower()
        if normalized in {"hi", "hello", "hey"}:
            return {"action": "final", "answer": "Hi! I can help with the assistant architecture, reliability controls, deployment, and indexed project notes."}
        if not evidence:
            return {"action": "search", "query": question}
        if any(word in normalized for word in ("compare", "versus", " vs ", "recommend")) and len(history) < 3:
            return {"action": "search", "query": f"alternative evidence for {question}"}
        return {"action": "final", "answer": ModelProvider._offline_answer(question, evidence)}
