"""Aplicación web (FastAPI + Jinja2) sobre ``PredictorService``.

Páginas: ``/`` (selector), ``/team/{nombre}`` (ficha) y ``/predict`` (predicción de un
partido). La API JSON equivalente cuelga de ``/api``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
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


_WEEKDAYS = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")


def _short_date(iso: str) -> str:
    """``2026-11-20`` -> ``vie 20/11``."""
    day = date.fromisoformat(iso)
    return f"{_WEEKDAYS[day.weekday()]} {day.day:02d}/{day.month:02d}"


def _triple(probabilities: dict) -> str:
    """``58 / 24 / 18`` (local / empate / visitante) en porcentajes sin decimales."""
    return " / ".join(f"{probabilities[key] * 100:.0f}" for key in ("home", "draw", "away"))


def _filter_fixtures(entries: list[dict], division: str, team: str) -> list[dict]:
    chosen = entries
    if division in ("SP1", "SP2"):
        chosen = [e for e in entries if e["division"] == division]
    if team:
        chosen = [e for e in chosen if team in (e["home"], e["away"])]
    return chosen


def _decimal(value: float, places: int = 2) -> str:
    """Número con coma decimal y un número fijo de decimales (``2,90``)."""
    return f"{value:.{places}f}".replace(".", ",")


def create_app(
    service: PredictorService,
    fixtures: pd.DataFrame | None = None,
    today: date | None = None,
) -> FastAPI:
    """``fixtures``: calendario de ``load_fixtures`` (opcional); ``today`` solo para pruebas."""
    app = FastAPI(title="Match Predictor", docs_url="/api/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters.update(
        {
            "team_name": display_name,
            "division_name": division_name,
            "pct": _percent,
            "dec": _decimal,
            "signed": _signed,
            "short_date": _short_date,
            "triple": _triple,
        }
    )
    entries = service.predict_fixtures(fixtures) if fixtures is not None else None
    reference_day = today or date.today()
    outcome_names = {"home": "Local", "draw": "Empate", "away": "Visitante"}

    def fixture_context(division: str, team: str) -> dict:
        shown = _filter_fixtures(entries or [], division, team)
        latest = max((e["date"] for e in entries), default=None) if entries else None
        return {
            "loaded": entries is not None,
            "entries": shown,
            "all_entries": bool(entries),
            "latest": latest,
            "stale": latest is not None and date.fromisoformat(latest) < reference_day,
            "today": reference_day.isoformat(),
            "division": division,
            "team": team,
            "fixture_teams": sorted({t for e in entries or [] for t in (e["home"], e["away"])}),
            "outcome_names": outcome_names,
        }

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

    @app.get("/fixtures", response_class=HTMLResponse)
    def fixtures_page(request: Request, division: str = "", team: str = ""):
        return page(request, "fixtures.html", fixture_context(division, team))

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

    @app.get("/api/fixtures")
    def api_fixtures(division: str = "", team: str = ""):
        context = fixture_context(division, team)
        return {
            "fixtures": context["entries"],
            "latest_date": context["latest"],
            "stale": context["stale"],
        }

    return app
