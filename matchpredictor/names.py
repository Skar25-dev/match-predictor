"""Nombres para mostrar. football-data.co.uk abrevia y quita tildes (``Ath Madrid``, ``Cadiz``)."""

DISPLAY_NAMES = {
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético de Madrid",
    "Alaves": "Alavés",
    "Almeria": "Almería",
    "Cadiz": "Cádiz",
    "Cordoba": "Córdoba",
    "Espanol": "Espanyol",
    "La Coruna": "Deportivo de La Coruña",
    "Leganes": "Leganés",
    "Malaga": "Málaga",
    "Mirandes": "Mirandés",
    "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting de Gijón",
    "Vallecano": "Rayo Vallecano",
    "Castellon": "CD Castellón",
}

DIVISION_NAMES = {"SP1": "La Liga", "SP2": "Segunda División"}


def display_name(team: str) -> str:
    """Nombre legible del equipo; si no hay uno propio se deja el original."""
    return DISPLAY_NAMES.get(team, team)


def division_name(division: str) -> str:
    return DIVISION_NAMES.get(division, division)
