import json
import re
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
        if ModelProvider._is_conversational(normalized):
            if "know me" in normalized:
                return "I do not know personal details about you, but I can help with this project's indexed notes and architecture."
            return "Hi! I can help with the assistant architecture, reliability controls, deployment, and indexed project notes."
        if context:
            parts = re.findall(r"\[[^\]]+\]\s*(.*?)(?=(?:\n\[[^\]]+\])|$)", context, flags=re.S)
            text = " ".join(part.strip() for part in parts if part.strip())
            if not text:
                return f"I can answer once relevant documents are indexed. Your question was: {question}"

            text = text.replace("## ", "\n## ")
            blocks = [block.strip() for block in re.split(r"\n##\s*", text) if block.strip()]
            if blocks and len(blocks) > 1:
                question_terms = ModelProvider._meaningful_terms(normalized)
                run_terms = {"run", "local", "start", "docker", "compose", "streamlit", "uvicorn", "api", "ui"}
                scored_blocks = []
                for block in blocks:
                    block_lower = block.lower()
                    score = sum(term in ModelProvider._meaningful_terms(block) for term in question_terms)
                    if any(term in block_lower for term in run_terms):
                        score += 5
                    scored_blocks.append((score, block))
                best_blocks = [block for _, block in sorted(scored_blocks, key=lambda item: item[0], reverse=True)[:2]]
                if any(block for block in best_blocks if block):
                    return f"Based on the indexed project notes: {' '.join(best_blocks)}"

            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.replace("`", "")) if sentence.strip()]
            keywords = ModelProvider._meaningful_terms(normalized)
            if not keywords:
                return f"Based on the indexed project notes: {sentences[0] if sentences else text}."
            scored = []
            for sentence in sentences:
                sentence_terms = ModelProvider._meaningful_terms(sentence)
                score = sum(term in sentence_terms for term in keywords)
                scored.append((score, sentence))
            if scored:
                picked = [sentence for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]
                if picked:
                    return f"Based on the indexed project notes: {' '.join(picked)}"
            return f"Based on the indexed project notes: {text[:800].rstrip()}."
        return f"I can answer once relevant documents are indexed. Your question was: {question}"

    @staticmethod
    def _offline_agent_step(question: str, evidence: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = question.strip().lower()
        if ModelProvider._is_conversational(normalized):
            return {"action": "final", "answer": ModelProvider._offline_answer(question, "")}

        expression = ModelProvider._calculator_expression(question)
        if expression:
            if not any(item.get("action") == "tool" for item in history):
                return {"action": "tool", "tool_calls": [{"name": "calculator", "arguments": {"expression": expression}}]}
            try:
                results = json.loads(evidence.split("Tool results:", 1)[1])
                result = results[0].get("result")
                if result is not None:
                    return {"action": "final", "answer": f"The result is {result}."}
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if ModelProvider._is_time_question(normalized):
            if not any(item.get("action") == "tool" for item in history):
                return {"action": "tool", "tool_calls": [{"name": "current_time", "arguments": {}}]}
            try:
                results = json.loads(evidence.split("Tool results:", 1)[1])
                result = results[0].get("result")
                if result is not None:
                    return {"action": "final", "answer": f"The current UTC time is {result}."}
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if not evidence:
            return {"action": "search", "query": question}
        if any(word in normalized for word in ("compare", "versus", " vs ", "recommend")) and len(history) < 3:
            return {"action": "search", "query": f"alternative evidence for {question}"}
        return {"action": "final", "answer": ModelProvider._offline_answer(question, evidence)}

    @staticmethod
    def _is_conversational(normalized: str) -> bool:
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        return normalized in greetings or "know me" in normalized

    @staticmethod
    def _is_time_question(normalized: str) -> bool:
        time_keywords = ("time", "date", "utc", "timezone", "what time", "what date")
        return any(keyword in normalized for keyword in time_keywords)

    @staticmethod
    def _calculator_expression(question: str) -> str | None:
        expression = question.strip().rstrip(".!?")
        expression = re.sub(r"^(calculate|compute|what is)\s+", "", expression, flags=re.IGNORECASE)
        if re.fullmatch(r"[0-9\s.+*/()\-]+", expression) and any(operator in expression for operator in "+-*/"):
            return expression
        return None

    @staticmethod
    def _meaningful_terms(text: str) -> set[str]:
        stop_words = {"a", "an", "and", "does", "for", "handle", "how", "is", "system", "the", "this", "what"}
        terms = set(re.findall(r"[a-z0-9]+", text.lower())) - stop_words
        normalized = set(terms)
        for term in terms:
            if term.endswith("ures"):
                normalized.add(term[:-4])
            elif term.endswith("ies"):
                normalized.add(term[:-3] + "y")
            elif term.endswith("s"):
                normalized.add(term[:-1])
        return normalized
