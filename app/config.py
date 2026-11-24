"""Zentrale Konfiguration. Alle Werte kommen aus Umgebungsvariablen bzw. der .env-Datei."""
from functools import lru_cache
from pathlib import Path
import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    app_username: str = "admin"
    app_password: str = ""
    secret_key: str | None = None

    data_dir: Path = Path("data")

    llm_provider: str = "ollama"
    llm_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen-paperless"
    llm_api_key: str | None = None
    llm_timeout: float = 1200
    llm_num_ctx: int | None = None
    llm_max_input_chars: int = 6000

    ocr_lang: str = "deu+eng"
    sum_tolerance: float = 0.05

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "receipts.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.upload_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def get_secret_key(self) -> str:
        """Nimmt SECRET_KEY aus der .env oder erzeugt einmalig einen und speichert ihn in ./data."""
        if self.secret_key:
            return self.secret_key
        path = self.data_dir / "secret_key"
        if path.exists():
            return path.read_text().strip()
        key = secrets.token_urlsafe(48)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()
