from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"


@pytest.fixture
def write_csv(tmp_path):
    """Devuelve una función que escribe un CSV mínimo de football-data.co.uk en tmp_path."""

    def _write(body: str, name: str = "season.csv", header: str = HEADER, encoding: str = "utf-8"):
        path = tmp_path / name
        path.write_bytes((header + "\n" + body).encode(encoding))
        return path

    return _write
