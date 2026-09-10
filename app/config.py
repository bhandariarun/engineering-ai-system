from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Engineering AI Assistant"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 700
    retrieval_k: int = 4
    agent_max_steps: int = 5
    agent_context_chars: int = 6000
    rate_limit_per_minute: int = 30
    cache_ttl_seconds: int = 300
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
