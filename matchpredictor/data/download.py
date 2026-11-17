"""Descarga los CSV de football-data.co.uk.

Uso:
    python -m matchpredictor.data.download                       # últimas 10 temporadas
    python -m matchpredictor.data.download --from-season 2018 --divisions SP1 SP2
    python -m matchpredictor.data.download --force               # volver a descargar todo

Los archivos se guardan en ``data/raw/`` como ``SP1_2025-26.csv``. Los que ya existen se
saltan (salvo con ``--force``), así que se puede relanzar sin problema.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DIVISIONS = ("SP1", "SP2")  # La Liga y Segunda División
USER_AGENT = "match-predictor/0.1 (proyecto educativo)"
DEFAULT_OUT_DIR = Path("data/raw")


def season_code(start_year: int) -> str:
    """Código de temporada de la URL: 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def build_url(start_year: int, division: str) -> str:
    return f"{BASE_URL}/{season_code(start_year)}/{division}.csv"


def file_name(start_year: int, division: str) -> str:
    return f"{division}_{start_year}-{(start_year + 1) % 100:02d}.csv"


def current_season_start(today: date | None = None) -> int:
    """Año en que empezó la temporada en curso (la temporada arranca en agosto)."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def download_season(
    start_year: int,
    division: str,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[Path, bool]:
    """Descarga una temporada. Devuelve (ruta, descargado); descargado=False si ya existía."""
    target = Path(out_dir) / file_name(start_year, division)
    if target.exists() and not force:
        return target, False

    http = session or requests
    response = http.get(
        build_url(start_year, division),
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    # La web puede devolver una página HTML con código 200 si el archivo no existe.
    if b"HomeTeam" not in response.content[:5000]:
        raise ValueError(f"{build_url(start_year, division)} no devolvió un CSV de resultados")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target, True


def main(argv: list[str] | None = None, *, session: requests.Session | None = None) -> int:
    current = current_season_start()
    parser = argparse.ArgumentParser(description="Descarga los CSV de football-data.co.uk")
    parser.add_argument("--from-season", type=int, default=current - 9, help="año de inicio")
    parser.add_argument("--to-season", type=int, default=current, help="año de inicio")
    parser.add_argument("--divisions", nargs="+", default=list(DIVISIONS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause", type=float, default=0.5, help="segundos entre descargas")
    args = parser.parse_args(argv)

    downloaded = skipped = 0
    failed: list[str] = []
    for start_year in range(args.from_season, args.to_season + 1):
        for division in args.divisions:
            label = f"{division} {start_year}-{(start_year + 1) % 100:02d}"
            try:
                path, was_downloaded = download_season(
                    start_year, division, args.out, force=args.force, session=session
                )
            except (requests.RequestException, ValueError) as error:
                failed.append(label)
                print(f"  ✗ {label}: {error}", file=sys.stderr)
                continue
            if was_downloaded:
                downloaded += 1
                print(f"  ↓ {label} -> {path}")
                time.sleep(args.pause)
            else:
                skipped += 1
                print(f"  = {label} ya existe ({path})")

    print(f"\nDescargados: {downloaded} | ya existían: {skipped} | fallidos: {len(failed)}")
    if failed:
        print("Fallidos: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed and not (downloaded or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
