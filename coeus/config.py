from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    tavily_api_key: SecretStr = SecretStr("")
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    db_path: Path = Field(Path("data/coeus.sqlite3"), validation_alias="COEUS_DB_PATH")
    workers: int = Field(3, ge=1, le=8, validation_alias="COEUS_WORKERS")
    max_tasks: int = Field(4, ge=1, le=8, validation_alias="COEUS_MAX_TASKS")
    results_per_task: int = Field(3, ge=1, le=5, validation_alias="COEUS_RESULTS_PER_TASK")
    request_timeout: float = Field(45, ge=1, le=120, validation_alias="COEUS_REQUEST_TIMEOUT")
    run_timeout: float = Field(600, ge=1, le=1800, validation_alias="COEUS_RUN_TIMEOUT")

    extract_chunk_chars: int = Field(
        4000, ge=1000, le=8000, validation_alias="COEUS_EXTRACT_CHUNK_CHARS"
    )
    evidence_budget_bytes: int = Field(
        8000, ge=4000, le=24000, validation_alias="COEUS_EVIDENCE_BUDGET_BYTES"
    )
    llm_max_output_tokens: int = Field(
        2048, ge=512, le=4096, validation_alias="COEUS_LLM_MAX_OUTPUT_TOKENS"
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:3b"
    ollama_timeout: float = Field(180, ge=10, le=600)

    def validate_offline(self) -> None:
        url = urlsplit(self.ollama_base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must be a local HTTP address, such as http://127.0.0.1:11434."
            )
        if "cloud" in self.ollama_model.lower():
            raise ValueError(
                "Offline mode requires a locally installed model, not an Ollama cloud model."
            )

    def validate_live(self) -> None:
        if not self.tavily_api_key.get_secret_value() or not self.llm_api_key.get_secret_value():
            raise ValueError("Set TAVILY_API_KEY and LLM_API_KEY in .env for live research.")
        url = urlsplit(self.llm_base_url)
        if (
            not url.hostname
            or url.username
            or url.password
            or not (
                url.scheme == "https"
                or (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"})
            )
        ):
            raise ValueError("LLM_BASE_URL must use HTTPS (or a local HTTP server).")
