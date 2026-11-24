import os
import pytest


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_USERNAME", "test")
    monkeypatch.setenv("APP_PASSWORD", "geheim")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
