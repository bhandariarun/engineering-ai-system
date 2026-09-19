"""Minimal MLflow agent experiment runner for prompt-version comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from app.config import Settings
from app.service import AssistantService


def run_case(prompt_version: str, question: str):
    settings = Settings(prompt_version=prompt_version)
    service = AssistantService(settings)
    service.ingest()
    response = service.ask(question)
    payload = {
        "question": question,
        "answer": response.answer,
        "provider": response.provider,
        "degraded": response.degraded,
        "iterations": response.iterations,
        "token_usage": response.token_usage,
        "stop_reason": response.stop_reason,
        "tool_calls": response.tool_calls,
        "citations": [citation.model_dump() for citation in response.citations],
    }
    return payload


def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    questions = [
        "How do I run locally?",
        "Calculate 25 * 4.",
        "How does this system handle provider failures?",
    ]
    for version in ["prompt_v1", "prompt_v2", "prompt_v3"]:
        with mlflow.start_run(run_name=f"agent-{version}") as run:
            mlflow.log_param("prompt_version", version)
            mlflow.log_param("agent_max_steps", 5)
            payloads = [run_case(version, question) for question in questions]
            mlflow.log_metric("avg_iterations", sum(item["iterations"] for item in payloads) / max(len(payloads), 1))
            mlflow.log_metric("avg_token_usage", sum(item["token_usage"] for item in payloads) / max(len(payloads), 1))
            mlflow.log_metric("degraded_runs", sum(1 for item in payloads if item["degraded"]))
            artifact_path = Path("artifacts") / f"{version}-trace.json"
            artifact_path.parent.mkdir(exist_ok=True)
            artifact_path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact_path))
            mlflow.set_tag("prompt_version", version)
            print(f"Logged run: {run.info.run_id} with {version}")


if __name__ == "__main__":
    main()
