import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .config import Settings


def load_prompt_templates() -> dict[str, str]:
    """Load versioned prompt templates used for agent tracing and regression evaluation."""
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    templates: dict[str, str] = {}
    if prompt_dir.exists():
        for path in sorted(prompt_dir.glob("*.txt")):
            key = path.stem
            templates[key] = path.read_text(encoding="utf-8").strip()
    if templates:
        return templates

    return {
        "prompt_v1": """
You are a careful research assistant. Search before finalizing. Use the smallest query that can answer the user question, prefer a final answer only when the evidence is sufficient, and clearly explain when information is missing.
""".strip(),
        "prompt_v2": """
You are a bounded research agent. Return JSON with action, query, answer, and tool_calls. Use search when evidence is empty. After each search, decide whether more evidence is needed; if yes, search again with a narrower or alternative query. Use tool only for deterministic calculations or time checks. Use final only when the answer is supported by retrieved evidence or when the evidence is unavailable.
""".strip(),
        "prompt_v3": """
You are a production-grade research agent. Verify claims with evidence before finalizing. Prefer precise search terms, keep the evidence brief but sufficient, and stop only after a verified answer or an explicit explanation that the task cannot be completed. Rely on tool calls for calculations and time lookups, and record why each step was chosen so that failures are diagnosable.
""".strip(),
    }


class ModelProvider:
    name = "offline"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url) if settings.openai_api_key else None
        if self.client:
            self.name = "openai-compatible"
        self.prompt_version = settings.prompt_version

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
                {"role": "system", "content": self._system_prompt_for_version(self.prompt_version)},
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
    def _system_prompt_for_version(version: str) -> str:
        templates = load_prompt_templates()
        return templates.get(version, templates.get("prompt_v2", "Return JSON with action, query, answer, and tool_calls."))

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
