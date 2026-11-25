from datetime import date

import pytest
import requests

from matchpredictor.data import download
from matchpredictor.data.download import (
    build_url,
    current_season_start,
    download_season,
    file_name,
    season_code,
)

CSV = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,15/08/2025,Alpha,Beta,1,0\n"


class FakeResponse:
    def __init__(self, content: bytes = b"", status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Sustituye a requests: devuelve respuestas predefinidas y apunta las llamadas."""

    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.get(url, FakeResponse(status=404))


# --- Funciones puras ---------------------------------------------------------------------------
@pytest.mark.parametrize(("year", "code"), [(2025, "2526"), (2016, "1617"), (1999, "9900")])
def test_season_code(year, code):
    assert season_code(year) == code


def test_build_url_and_file_name():
    assert build_url(2025, "SP1") == "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
    assert file_name(2024, "SP2") == "SP2_2024-25.csv"


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 11, 17), 2026),
        (date(2026, 8, 1), 2026),
        (date(2026, 7, 31), 2025),
        (date(2027, 1, 5), 2026),
    ],
)
def test_current_season_start(today, expected):
    assert current_season_start(today) == expected


# --- download_season ---------------------------------------------------------------------------
def test_download_writes_the_file_and_identifies_itself(tmp_path):
    session = FakeSession({build_url(2025, "SP1"): FakeResponse(CSV)})

    path, downloaded = download_season(2025, "SP1", tmp_path, session=session)

    assert downloaded is True
    assert path == tmp_path / "SP1_2025-26.csv"
    assert path.read_bytes() == CSV
    headers = session.calls[0][1]["headers"]
    assert headers["User-Agent"].startswith("match-predictor")
    assert "@" not in headers["User-Agent"]  # nunca se envía un email


def test_existing_file_is_skipped_unless_forced(tmp_path):
    session = FakeSession({build_url(2025, "SP1"): FakeResponse(CSV)})
    existing = tmp_path / "SP1_2025-26.csv"
    existing.write_bytes(b"viejo")

    _, downloaded = download_season(2025, "SP1", tmp_path, session=session)
    assert downloaded is False
    assert session.calls == []
    assert existing.read_bytes() == b"viejo"

    _, downloaded = download_season(2025, "SP1", tmp_path, force=True, session=session)
    assert downloaded is True
    assert existing.read_bytes() == CSV


def test_html_page_with_status_200_is_rejected(tmp_path):
    session = FakeSession({build_url(2025, "SP1"): FakeResponse(b"<html>No encontrado</html>")})

    with pytest.raises(ValueError, match="CSV"):
        download_season(2025, "SP1", tmp_path, session=session)

    assert list(tmp_path.iterdir()) == []


def test_http_error_raises_and_leaves_no_file(tmp_path):
    session = FakeSession({})  # todo da 404

    with pytest.raises(requests.HTTPError):
        download_season(2025, "SP1", tmp_path, session=session)

    assert list(tmp_path.iterdir()) == []


# --- main --------------------------------------------------------------------------------------
def test_main_keeps_going_when_one_season_fails(tmp_path, capsys):
    session = FakeSession({build_url(2024, "SP1"): FakeResponse(CSV)})  # 2025 dará 404

    code = download.main(
        ["--from-season", "2024", "--to-season", "2025", "--divisions", "SP1",
         "--out", str(tmp_path), "--pause", "0"],
        session=session,
    )  # fmt: skip

    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "SP1_2024-25.csv").exists()
    assert not (tmp_path / "SP1_2025-26.csv").exists()
    assert "Descargados: 1" in captured.out
    assert "SP1 2025-26" in captured.err


def test_main_returns_error_when_everything_fails(tmp_path):
    code = download.main(
        ["--from-season", "2025", "--to-season", "2025", "--out", str(tmp_path), "--pause", "0"],
        session=FakeSession({}),
    )

    assert code == 1


# --- Calendario de próximos partidos ------------------------------------------------------
FIXTURES_CSV = b"Div,Date,Time,HomeTeam,AwayTeam\nSP1,20/11/2026,20:00,Alpha,Beta\n"


def test_download_fixtures_writes_and_overwrites(tmp_path):
    from matchpredictor.data.download import FIXTURES_URL, download_fixtures

    target = tmp_path / "data" / "fixtures.csv"
    target.parent.mkdir()
    target.write_bytes(b"old content")
    session = FakeSession({FIXTURES_URL: FakeResponse(FIXTURES_CSV)})

    path = download_fixtures(target, session=session)

    assert path == target and target.read_bytes() == FIXTURES_CSV
    assert session.calls[0][1]["headers"]["User-Agent"].startswith("match-predictor")


def test_download_fixtures_creates_the_folder(tmp_path):
    from matchpredictor.data.download import FIXTURES_URL, download_fixtures

    target = tmp_path / "nueva" / "fixtures.csv"

    download_fixtures(target, session=FakeSession({FIXTURES_URL: FakeResponse(FIXTURES_CSV)}))

    assert target.exists()


@pytest.mark.parametrize(
    "response",
    [FakeResponse(b"<html>no encontrado</html>"), FakeResponse(status=500)],
    ids=["html", "http-error"],
)
def test_download_fixtures_rejects_bad_responses_and_keeps_the_old_file(tmp_path, response):
    from matchpredictor.data.download import FIXTURES_URL, download_fixtures

    target = tmp_path / "fixtures.csv"
    target.write_bytes(b"old")

    with pytest.raises((ValueError, requests.HTTPError)):
        download_fixtures(target, session=FakeSession({FIXTURES_URL: response}))

    assert target.read_bytes() == b"old"


def test_cli_fixtures_flag_downloads_only_the_calendar(tmp_path, capsys):
    from matchpredictor.data.download import FIXTURES_URL

    session = FakeSession({FIXTURES_URL: FakeResponse(FIXTURES_CSV)})
    out = tmp_path / "fixtures.csv"

    code = download.main(["--fixtures", "--fixtures-out", str(out)], session=session)

    assert code == 0 and out.read_bytes() == FIXTURES_CSV
    assert [url for url, _ in session.calls] == [FIXTURES_URL]  # ninguna temporada
    assert "calendario" in capsys.readouterr().out


def test_cli_fixtures_flag_reports_failures(tmp_path, capsys):
    session = FakeSession({})  # 404

    code = download.main(["--fixtures", "--fixtures-out", str(tmp_path / "f.csv")], session=session)

    assert code == 1 and "calendario" in capsys.readouterr().err
