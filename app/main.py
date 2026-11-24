"""Einstiegspunkt der Web-App."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .auth import NotAuthenticated, check_credentials
from .config import get_settings
from .db import init_db
from .logging_setup import setup_logging
from .pipeline import process_receipt
from .templating import templates
from .web import router as web_router
from .worker import Worker

log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings.log_dir)
    if not settings.app_password:
        log.error("APP_PASSWORD ist nicht gesetzt – Anmeldung ist nicht möglich. Bitte in .env eintragen.")
    init_db()
    worker = Worker(process_receipt)
    app.state.worker = worker
    if app.state.start_worker:
        worker.start()
    log.info("Receipt Analyser gestartet (Ollama: %s / %s)", settings.llm_base_url, settings.llm_model)
    yield
    await worker.stop()


def create_app(start_worker: bool = True) -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    app = FastAPI(title="Kassenbon-Auswertung", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.start_worker = start_worker
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

    app.include_router(web_router)
    return app


app = create_app()
