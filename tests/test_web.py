import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from matchpredictor.service import PredictorService
from matchpredictor.web.app import _signed, create_app
from tests.synthetic import add_shots, synthetic_raw_two_divisions, write_as_csvs


@pytest.fixture(scope="module")
def client():
    raw = add_shots(synthetic_raw_two_divisions())
    for column in ("home_team", "away_team"):
        raw[column] = raw[column].replace({"T00": "Ath Madrid"})  # nombre con espacio
    return TestClient(create_app(PredictorService(raw)))


@pytest.fixture(scope="module")
def client_without_shots():
    return TestClient(create_app(PredictorService(synthetic_raw_two_divisions())))


def test_index_lists_teams_grouped_by_division(client):
    response = client.get("/")

    assert response.status_code == 200
    assert 'optgroup label="La Liga"' in response.text
    assert 'optgroup label="Segunda División"' in response.text
    assert "Atlético de Madrid" in response.text  # nombre para mostrar
    assert 'value="Ath Madrid"' in response.text  # valor interno del dato


def test_team_page_shows_form_and_history(client):
    response = client.get("/team/T03")

    assert response.status_code == 200
    for text in ("Forma reciente", "Últimos partidos", "Temporada a temporada", "Elo"):
        assert text in response.text


def test_team_selector_redirects_even_with_spaces_in_the_name(client):
    redirect = client.get("/team", params={"name": "Ath Madrid"}, follow_redirects=False)

    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/team/Ath%20Madrid"
    assert "Atlético de Madrid" in client.get(redirect.headers["location"]).text


def test_team_selector_encodes_slashes_and_question_marks(client):
    redirect = client.get("/team", params={"name": "A/B?c"}, follow_redirects=False)

    assert redirect.headers["location"] == "/team/A%2FB%3Fc"


def test_team_selector_without_a_name_goes_home(client):
    redirect = client.get("/team", follow_redirects=False)

    assert (redirect.status_code, redirect.headers["location"]) == (303, "/")


def test_unknown_team_is_a_404_and_user_text_is_escaped(client):
    response = client.get("/team/%3Cscript%3Ealert(1)")

    assert response.status_code == 404
    assert "<script>alert" not in response.text
    assert "&lt;script&gt;" in response.text


def test_predict_page_without_params_shows_only_the_form(client):
    response = client.get("/predict")

    assert response.status_code == 200
    assert "Probabilidades de victoria local" in response.text
    assert "cuota justa" not in response.text


def test_predict_page_shows_probabilities(client):
    response = client.get("/predict", params={"home": "T01", "away": "T02"})

    assert response.status_code == 200
    assert response.text.count("%</b>") == 3
    assert "cuota justa" in response.text
    assert "hipotético" not in response.text


def test_numbers_use_spanish_decimal_comma(client):
    predict = client.get("/predict", params={"home": "T01", "away": "T02"}).text
    team = client.get("/team/T03").text

    assert len(re.findall(r"cuota justa \d+,\d{2}<", predict.replace("<br>", "<"))) == 3
    assert re.search(r"Elo \d+ ·", team)  # el Elo se muestra sin decimales
    assert re.search(r"\d,\d{2}</b><span>pts/partido", team)
    assert not re.search(r"\d\.\d{2}</b><span>pts/partido", team)


def test_predict_across_divisions_warns(client):
    response = client.get("/predict", params={"home": "T01", "away": "U02"})

    assert "Partido hipotético" in response.text


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"home": "T01", "away": "T01"}, "distintos"),
        ({"home": "T01", "away": "Atlantis"}, "desconocido"),
        ({"home": "T01", "away": ""}, "desconocido"),
        ({"home": "T01", "away": "T02", "division": "SP9"}, "División no válida"),
    ],
)
def test_predict_page_reports_errors_with_status_400(client, params, message):
    response = client.get("/predict", params=params)

    assert response.status_code == 400
    assert message in response.text


def test_api_predict_returns_json(client):
    response = client.get("/api/predict", params={"home": "T01", "away": "T02"})

    body = response.json()
    assert response.status_code == 200
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=2e-4)
    assert body["home"] == "T01"


def test_api_errors_use_proper_status_codes(client):
    assert client.get("/api/predict", params={"home": "T01", "away": "Atlantis"}).status_code == 404
    assert client.get("/api/predict", params={"home": "T01", "away": "T01"}).status_code == 400
    assert client.get("/api/predict", params={"home": "T01"}).status_code == 422
    assert client.get("/api/team/Atlantis").status_code == 404


def test_api_teams_team_and_health(client):
    assert set(client.get("/api/teams").json()) == {"SP1", "SP2"}
    assert client.get("/api/team/T03").json()["team"] == "T03"
    assert client.get("/health").json()["status"] == "ok"


def test_server_entrypoint_builds_the_app_and_starts_uvicorn(tmp_path, monkeypatch):
    import uvicorn

    from matchpredictor.web.__main__ import main

    write_as_csvs(synthetic_raw_two_divisions(), tmp_path)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))

    assert main(["--data", str(tmp_path), "--host", "0.0.0.0", "--port", "9123"]) == 0

    assert (started["host"], started["port"]) == ("0.0.0.0", 9123)
    assert TestClient(started["app"]).get("/health").status_code == 200


def test_entrypoint_passes_the_division_offset_and_the_page_mentions_it(tmp_path, monkeypatch):
    import uvicorn

    from matchpredictor.web.__main__ import main

    write_as_csvs(synthetic_raw_two_divisions(), tmp_path)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))

    main(["--data", str(tmp_path), "--division-offset", "-50"])

    web = TestClient(started["app"])
    assert web.get("/api/predict", params={"home": "T01", "away": "U02"}).json()[
        "division_offset"
    ] == pytest.approx(-50.0)
    page = " ".join(web.get("/predict", params={"home": "T01", "away": "U02"}).text.split())
    assert "ajuste de -50 puntos Elo" in page
    same_division = web.get("/predict", params={"home": "T01", "away": "T02"}).text
    assert "ajuste de" not in same_division


def test_shot_form_is_shown_when_data_exists(client):
    team = client.get("/team/T03").text
    predict = " ".join(client.get("/predict", params={"home": "T01", "away": "T02"}).text.split())

    assert re.search(r"Tiros a puerta \(últimos 10 partidos\): <b>[+−]\d+,\d</b>", team)
    assert re.search(r"Tiros a puerta \(últimos 10\): T01 [+−]\d+,\d · T02 [+−]\d+,\d", predict)


def test_shot_form_is_hidden_without_shot_data(client_without_shots):
    team = client_without_shots.get("/team/T03").text
    predict = client_without_shots.get("/predict", params={"home": "T01", "away": "T02"}).text

    assert "Tiros a puerta" not in team and "Tiros a puerta" not in predict


@pytest.mark.parametrize(
    ("value", "text"),
    [(1.26, "+1,3"), (-0.44, "−0,4"), (0.0, "+0,0"), (-0.04, "+0,0"), (12.0, "+12,0")],
)
def test_signed_formatting(value, text):
    assert _signed(value) == text


def test_cross_division_page_explains_that_only_elo_is_used(client):
    page = " ".join(client.get("/predict", params={"home": "T01", "away": "U02"}).text.split())

    assert "solo se usa el Elo" in page
    assert "Tiros a puerta (últimos 10):" not in page


def test_same_division_page_shows_shot_form_and_no_elo_only_note(client):
    page = " ".join(client.get("/predict", params={"home": "T01", "away": "T02"}).text.split())

    assert "Tiros a puerta (últimos 10):" in page
    assert "solo se usa el Elo" not in page


# --- Página de la jornada ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixtures_setup():
    from tests.synthetic import synthetic_fixtures

    raw = add_shots(synthetic_raw_two_divisions())
    return raw, synthetic_fixtures(raw)


def _fixtures_client(setup, today):
    raw, fixtures = setup
    return TestClient(create_app(PredictorService(raw), fixtures, today=today))


def test_fixtures_page_lists_matches_with_model_and_market(fixtures_setup):
    raw, fixtures = fixtures_setup
    web = _fixtures_client(fixtures_setup, raw["date"].max().date())

    page = " ".join(web.get("/fixtures").text.split())

    for text in ("Próxima jornada", "T01", "T02", "T03", "13:00", "Modelo", "Mercado"):
        assert text in page
    assert re.search(r"\d+ / \d+ / \d+", page)
    assert "Sin predicción" in page and "Nuevo FC" in page
    assert "ya ha pasado" not in page  # calendario vigente
    assert "pasado</span>" not in page


def test_stale_calendar_is_flagged(fixtures_setup):
    raw, fixtures = fixtures_setup
    future = fixtures["date"].max().date() + pd.Timedelta(days=3)
    web = _fixtures_client(fixtures_setup, future)

    page = " ".join(web.get("/fixtures").text.split())

    assert "ya ha pasado" in page
    assert page.count("pasado</span>") == 4  # los 4 partidos del calendario ya han pasado
    assert web.get("/api/fixtures").json()["stale"] is True


def test_filters_by_division_and_team(fixtures_setup):
    raw, _ = fixtures_setup
    web = _fixtures_client(fixtures_setup, raw["date"].max().date())

    sp2 = web.get("/api/fixtures", params={"division": "SP2"}).json()["fixtures"]
    team = web.get("/api/fixtures", params={"team": "T01"}).json()["fixtures"]
    everything = web.get("/api/fixtures", params={"division": "XX"}).json()["fixtures"]
    nothing = web.get("/api/fixtures", params={"division": "SP2", "team": "T01"}).json()["fixtures"]

    assert [(e["home"], e["away"]) for e in sp2] == [("U01", "U02")]
    assert [(e["home"], e["away"]) for e in team] == [("T01", "T02")]
    assert len(everything) == 4 and nothing == []
    assert (
        "No hay partidos que coincidan"
        in web.get("/fixtures", params={"division": "SP2", "team": "T01"}).text
    )


def test_without_a_calendar_the_page_explains_how_to_get_one():
    raw = synthetic_raw_two_divisions()
    web = TestClient(create_app(PredictorService(raw)))

    page = web.get("/fixtures").text

    assert "No hay calendario cargado" in page and "download --fixtures" in page
    assert web.get("/api/fixtures").json() == {"fixtures": [], "latest_date": None, "stale": False}


def test_fixtures_page_escapes_team_names_from_the_file(fixtures_setup):
    raw, fixtures = fixtures_setup
    evil = fixtures.copy()
    evil.loc[0, "away_team"] = "<script>alert(1)</script>"
    web = TestClient(create_app(PredictorService(raw), evil, today=raw["date"].max().date()))

    page = web.get("/fixtures").text

    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_navigation_links_to_the_fixtures_page(client):
    assert 'href="/fixtures"' in client.get("/").text


def test_entrypoint_loads_the_calendar_file(tmp_path, monkeypatch):
    import uvicorn

    from matchpredictor.web.__main__ import main

    write_as_csvs(synthetic_raw_two_divisions(), tmp_path)
    calendar = tmp_path.parent / f"{tmp_path.name}_calendar.csv"
    calendar.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nSP1,01/01/2030,20:00,T01,T02,2,3.4,3.8\n"
    )
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))

    main(["--data", str(tmp_path), "--fixtures", str(calendar)])

    page = TestClient(started["app"]).get("/fixtures").text
    assert "T01" in page and "01/01" in page


def test_entrypoint_without_a_calendar_file_still_starts(tmp_path, monkeypatch, capsys):
    import uvicorn

    from matchpredictor.web.__main__ import main

    write_as_csvs(synthetic_raw_two_divisions(), tmp_path)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))

    main(["--data", str(tmp_path), "--fixtures", str(tmp_path / "no_existe.csv")])

    assert "Sin calendario" in capsys.readouterr().out
    assert "No hay calendario cargado" in TestClient(started["app"]).get("/fixtures").text
