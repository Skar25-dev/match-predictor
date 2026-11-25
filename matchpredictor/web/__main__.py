"""Arranca la web:  python -m matchpredictor.web --data data/raw"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from pathlib import Path

    import uvicorn

    from matchpredictor.data import load_all, load_fixtures
    from matchpredictor.service import PredictorService
    from matchpredictor.web.app import create_app

    parser = argparse.ArgumentParser(description="Servidor web de Match Predictor")
    parser.add_argument("--data", default="data/raw", help="directorio con los CSV")
    parser.add_argument("--host", default="127.0.0.1", help="usa 0.0.0.0 dentro de Docker")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--fixtures",
        default="data/fixtures.csv",
        help="calendario de próximos partidos (python -m matchpredictor.data.download --fixtures)",
    )
    parser.add_argument(
        "--division-offset",
        type=float,
        default=0.0,
        help="un rating de Segunda equivale a (rating + d) en La Liga (ver evaluation.divisions)",
    )
    args = parser.parse_args(argv)

    print("Cargando datos y entrenando el modelo...")
    service = PredictorService(load_all(args.data), division_offset=args.division_offset)
    fixtures = None
    if Path(args.fixtures).exists():
        fixtures = load_fixtures(args.fixtures)
        print(f"Calendario: {len(fixtures)} partidos de {args.fixtures}")
    else:
        print(f"Sin calendario ({args.fixtures} no existe): la página Jornada quedará vacía")
    print(f"Listo: temporada más reciente {service.latest_season}. http://{args.host}:{args.port}")
    uvicorn.run(create_app(service, fixtures), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
