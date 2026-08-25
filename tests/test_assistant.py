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
    service = AssistantService(Settings())
    service.ingest()
    response = service.ask("How does the system handle failures?")
    assert response.answer
    assert response.citations
    assert response.degraded is False


def test_offline_greeting_is_conversational():
    service = AssistantService(Settings())
    service.ingest()
    response = service.ask("Hi")
    assert response.answer.startswith("Hi!")
    assert response.citations == []


def test_cache_marks_repeated_question():
    service = AssistantService(Settings())
    service.ingest()
    service.ask("What is the architecture?")
    response = service.ask("What is the architecture?")
    assert response.cached is True
