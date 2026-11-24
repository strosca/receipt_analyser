"""Einfache Anmeldung mit Benutzername/Passwort aus der .env und einem Session-Cookie."""
import secrets

from fastapi import Request

from .config import get_settings


class NotAuthenticated(Exception):
    pass


def check_credentials(username: str, password: str) -> bool:
    s = get_settings()
    if not s.app_password:
        return False
    ok_user = secrets.compare_digest(username.encode(), s.app_username.encode())
    ok_pass = secrets.compare_digest(password.encode(), s.app_password.encode())
    return ok_user and ok_pass


def require_login(request: Request) -> str:
    """FastAPI-Abhängigkeit: Wirft NotAuthenticated, wenn niemand angemeldet ist."""
    user = request.session.get("user")
    if not user:
        raise NotAuthenticated()
    return user
