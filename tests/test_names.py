from matchpredictor.names import display_name, division_name


def test_display_name_restores_accents_and_full_names():
    assert display_name("Cadiz") == "Cádiz"
    assert display_name("Ath Madrid") == "Atlético de Madrid"
    assert display_name("Sociedad") == "Real Sociedad"


def test_display_name_leaves_unknown_names_untouched():
    assert display_name("Real Madrid") == "Real Madrid"
    assert display_name("Equipo Raro") == "Equipo Raro"


def test_division_names():
    assert division_name("SP1") == "La Liga"
    assert division_name("SP2") == "Segunda División"
    assert division_name("XX") == "XX"
