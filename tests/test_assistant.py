from app.config import Settings
from app.service import AssistantService
from app.tools import calculator


def test_calculator_rejects_code():
    try:
        calculator("__import__('os').getcwd()")
    except ValueError:
        return
    raise AssertionError("unsafe expression was accepted")


def test_offline_question_returns_citation():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("How does this system handle provider failures?")
    assert response.answer
    assert response.citations
    assert response.degraded is False
    assert "retries transient provider errors" in response.answer


def test_offline_greeting_is_conversational():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("Hi")
    assert response.answer.startswith("Hi!")
    assert response.citations == []


def test_offline_personal_question_is_conversational():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("Hello, do you know me?")

    assert "do not know personal details" in response.answer
    assert response.citations == []
    assert response.stop_reason == "completed"
    assert response.iterations == 1


def test_offline_calculator_uses_tool():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("Calculate 25 * 4.")

    assert response.answer == "The result is 100."
    assert response.tool_calls[0]["name"] == "calculator"
    assert response.tool_calls[0]["result"] == "100"
    assert response.stop_reason == "completed"


def test_offline_time_question_uses_current_time_tool():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("What time is it?")

    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "current_time"
    assert "current utc time" in response.answer.lower()
    assert response.stop_reason == "completed"


def test_offline_run_instructions_are_not_truncated():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    response = service.ask("How do I run locally?")

    assert "uvicorn app.main:app" in response.answer
    assert "streamlit run app/ui.py" in response.answer
    assert "docker compose" in response.answer.lower()


def test_cache_marks_repeated_question():
    service = AssistantService(Settings(openai_api_key=None))
    service.ingest()
    service.ask("What is the architecture?")
    response = service.ask("What is the architecture?")
    assert response.cached is True
