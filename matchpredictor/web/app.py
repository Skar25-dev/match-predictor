"""Aplicación web (FastAPI + Jinja2) sobre ``PredictorService``.

Páginas: ``/`` (selector), ``/team/{nombre}`` (ficha) y ``/predict`` (predicción de un
partido). La API JSON equivalente cuelga de ``/api``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from matchpredictor.analysis import UnknownTeamError
from matchpredictor.names import display_name, division_name
from matchpredictor.service import PredictorService

TEMPLATES = Path(__file__).parent / "templates"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + " %"


def _signed(value: float, places: int = 1) -> str:
    """Número con signo explícito y coma decimal (``+1,2`` / ``−0,4``)."""
    rounded = round(value, places)
    sign = "+" if rounded >= 0 else "−"
    return sign + f"{abs(rounded):.{places}f}".replace(".", ",")


def _decimal(value: float, places: int = 2) -> str:
    """Número con coma decimal y un número fijo de decimales (``2,90``)."""
    return f"{value:.{places}f}".replace(".", ",")


def create_app(service: PredictorService) -> FastAPI:
    app = FastAPI(title="Match Predictor", docs_url="/api/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters.update(
        {
            "team_name": display_name,
            "division_name": division_name,
            "pct": _percent,
            "dec": _decimal,
            "signed": _signed,
        }
    )

    def page(request: Request, name: str, context: dict, status_code: int = 200) -> HTMLResponse:
        base = {"teams": service.teams_by_division(), "latest_season": service.latest_season}
        return templates.TemplateResponse(
            request, name, {**base, **context}, status_code=status_code
        )

    # --- páginas ---------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page(request, "index.html", {})

    @app.get("/team", response_class=HTMLResponse)
    def team_redirect(name: str = ""):
        if not name:
            return RedirectResponse("/", status_code=303)
        return RedirectResponse(f"/team/{quote(name, safe='')}", status_code=303)

    @app.get("/team/{name}", response_class=HTMLResponse)
    def team(request: Request, name: str):
        try:
            data = service.overview(name)
        except UnknownTeamError as error:
            return page(request, "error.html", {"message": str(error)}, status_code=404)
        return page(request, "team.html", {"data": data})

    @app.get("/predict", response_class=HTMLResponse)
    def predict(request: Request, home: str = "", away: str = "", division: str = ""):
        context = {"home": home, "away": away, "division": division, "result": None}
        if not home and not away:
            return page(request, "predict.html", context)
        try:
            context["result"] = service.predict(home, away, division or None)
        except (UnknownTeamError, ValueError) as error:
            return page(request, "predict.html", {**context, "error": str(error)}, 400)
        return page(request, "predict.html", context)

    # --- API JSON --------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "latest_season": service.latest_season}

    @app.get("/api/teams")
    def api_teams():
        return service.teams_by_division()

    @app.get("/api/team/{name}")
    def api_team(name: str):
        try:
            return service.overview(name)
        except UnknownTeamError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/predict")
    def api_predict(home: str, away: str, division: str | None = None):
        try:
            return service.predict(home, away, division)
        except UnknownTeamError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app
