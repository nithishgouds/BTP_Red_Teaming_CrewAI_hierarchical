from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "btp-usecase"
    app_version: str = "0.1.0"
    environment: str = "development"
    storage_directory: str = "./data/documents"
    max_upload_size_mb: int = 10
    allowed_file_types: str = "pdf,txt,docx"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def allowed_extensions(self) -> set[str]:
        return {item.strip().lower() for item in self.allowed_file_types.split(",") if item.strip()}

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_directory).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
