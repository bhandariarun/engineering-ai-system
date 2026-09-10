# Agent Evaluation Results

Generated with `/home/arun/Office/engineering-ai-system/venv/bin/python scripts/evaluate_agent.py` on the offline scripted provider.

| Case | Completed | Tool correct | Iterations | Tokens | Failure classification |
| --- | --- | --- | ---: | ---: | --- |
| Cross-source verification | yes | n/a | 3 | 51 | none |
| Tool-assisted calculation | yes | yes | 2 | 34 | none |
| Failure injection: provider timeout | no | n/a | 2 | 17 | hard failure |

Task completion rate: **2/3 (66.7%)**. Tool-call correctness: **1/1 (100%)**. The injected timeout is surfaced as `agent_failure`; the service does not produce a confident final answer. The scripted provider uses 17 tokens per decision so the harness records the total consumed per trajectory. The step cap is separately tested with a soft failure when the agent never chooses `final`.