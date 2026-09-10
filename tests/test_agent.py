from app.config import Settings
from app.service import AssistantService


class ScriptedProvider:
    name = "scripted"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def agent_step(self, question, evidence, history):
        self.calls.append({"question": question, "evidence": evidence, "history": history})
        return next(self.decisions), 17


def make_service(provider):
    service = AssistantService(Settings(cache_ttl_seconds=0, agent_max_steps=5))
    service.ingest()
    service.provider = provider
    return service


def test_agent_can_search_twice_before_finalizing():
    provider = ScriptedProvider([
        {"action": "search", "query": "reliability"},
        {"action": "search", "query": "failure fallback"},
        {"action": "final", "answer": "Verified answer"},
    ])
    response = make_service(provider).ask("Compare reliability and fallback behavior")

    assert response.answer == "Verified answer"
    assert response.iterations == 3
    assert response.token_usage == 51
    assert len(provider.calls[1]["evidence"]) > 0
    assert response.citations


def test_agent_executes_allowlisted_tool_and_feeds_result_back():
    provider = ScriptedProvider([
        {"action": "tool", "tool_calls": [{"name": "calculator", "arguments": {"expression": "2 + 2"}}]},
        {"action": "final", "answer": "The result is 4."},
    ])
    response = make_service(provider).ask("Calculate 2 + 2")

    assert response.tool_calls[0]["result"] == "4"
    assert response.iterations == 2
    assert "Tool results" in provider.calls[1]["evidence"]


def test_agent_failure_is_explicit_and_bounded():
    provider = ScriptedProvider([{"action": "search", "query": "missing"}] * 5)
    response = make_service(provider).ask("Will this stop?")

    assert response.degraded is True
    assert response.stop_reason == "max_steps"
    assert response.iterations == 5
    assert "step limit" in response.answer