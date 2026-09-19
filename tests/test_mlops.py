from app.providers import load_prompt_templates


def test_prompt_versions_are_available():
    prompts = load_prompt_templates()
    assert {"prompt_v1", "prompt_v2", "prompt_v3"}.issubset(prompts)
    assert "search" in prompts["prompt_v2"].lower()
    assert "final" in prompts["prompt_v3"].lower()
