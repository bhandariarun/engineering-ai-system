"""Small evaluation harness for the agentic feature; intentionally no evaluation framework."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.service import AssistantService


class ScriptedProvider:
    name = "evaluation-scripted"

    def __init__(self, decisions, fail_at=None):
        self.decisions = iter(decisions)
        self.calls = []
        self.fail_at = fail_at

    def agent_step(self, question, evidence, history):
        self.calls.append({"question": question, "evidence": evidence, "history": history})
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected provider timeout")
        return next(self.decisions), 17


@dataclass
class Case:
    name: str
    question: str
    decisions: list[dict]
    expected_iterations: int
    expected_tool: str | None = None
    fail_at: int | None = None


def run_case(case: Case) -> dict:
    service = AssistantService(Settings(cache_ttl_seconds=0, agent_max_steps=5))
    service.ingest()
    provider = ScriptedProvider(case.decisions, case.fail_at)
    service.provider = provider
    response = service.ask(case.question)
    tool_ok = case.expected_tool is None or any(
        call.get("name") == case.expected_tool and "result" in call for call in response.tool_calls
    )
    completed = response.stop_reason == "completed" and not response.degraded and bool(response.answer)
    if case.fail_at is not None:
        failure_type = "hard failure" if response.stop_reason == "agent_failure" else "cascading soft failure"
    elif not completed:
        failure_type = "soft failure" if response.stop_reason == "max_steps" else "hard failure"
    else:
        failure_type = "none"
    return {
        "name": case.name,
        "completed": completed,
        "tool_correct": tool_ok,
        "iterations": response.iterations,
        "expected_iterations": case.expected_iterations,
        "tokens": response.token_usage,
        "stop_reason": response.stop_reason,
        "failure_type": failure_type,
        "provider_calls": len(provider.calls),
    }


def main() -> None:
    cases = [
        Case("cross-source verification", "Compare reliability and fallback", [
            {"action": "search", "query": "reliability"},
            {"action": "search", "query": "failure fallback"},
            {"action": "final", "answer": "Verified"},
        ], 3),
        Case("tool-assisted calculation", "Calculate 2 + 2", [
            {"action": "tool", "tool_calls": [{"name": "calculator", "arguments": {"expression": "2 + 2"}}]},
            {"action": "final", "answer": "The result is 4."},
        ], 2, expected_tool="calculator"),
        Case("failure injection", "Handle a provider timeout", [
            {"action": "search", "query": "reliability"},
            {"action": "final", "answer": "Should not be reached"},
        ], 2, fail_at=2),
    ]
    results = [run_case(case) for case in cases]
    completed = sum(result["completed"] for result in results)
    tool_cases = [result for result in results if result["name"] == "tool-assisted calculation"]
    tool_correct = sum(result["tool_correct"] for result in tool_cases)
    print(json.dumps({
        "task_completion_rate": f"{completed}/{len(results)}",
        "tool_call_correctness": f"{tool_correct}/{len(tool_cases)}",
        "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()