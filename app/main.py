"""Einstiegspunkt der Web-App."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .auth import NotAuthenticated, check_credentials, require_login
from .config import get_settings
from .logging_setup import setup_logging
from .templating import templates

log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings.log_dir)
    if not settings.app_password:
        log.error("APP_PASSWORD ist nicht gesetzt – Anmeldung ist nicht möglich. Bitte in .env eintragen.")
    log.info("Receipt Analyser gestartet (KI: %s / %s)", settings.llm_provider, settings.llm_model)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    app = FastAPI(title="Kassenbon-Auswertung", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.get_secret_key(),
        session_cookie="ra_session",
        max_age=60 * 60 * 24 * 30,
        same_site="lax",
        https_only=False,
    )
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.exception_handler(NotAuthenticated)
    async def _not_authenticated(request: Request, exc: NotAuthenticated):
        if request.headers.get("HX-Request"):
            # HTMX-Anfragen: komplette Seite auf Login umleiten
            return HTMLResponse("", status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request):
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        if check_credentials(username, password):
            request.session["user"] = username
            return RedirectResponse("/", status_code=303)
        log.warning("Fehlgeschlagene Anmeldung für Benutzer %r", username)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Benutzername oder Passwort falsch."}, status_code=401
        )

    @app.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, user: str = Depends(require_login)):
        return templates.TemplateResponse(request, "index.html", {})

    return app


app = create_app()
