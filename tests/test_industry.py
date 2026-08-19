import numpy as np

from osrisk.factors.industry import sic_to_industry


def test_known_sic_mappings():
    assert sic_to_industry(7372) == "BusEq"   # prepackaged software
    assert sic_to_industry(2834) == "Hlth"    # pharmaceutical preparations
    assert sic_to_industry(6022) == "Money"   # state commercial banks
    assert sic_to_industry(1311) == "Enrgy"   # crude petroleum & natural gas
    assert sic_to_industry(4911) == "Utils"   # electric services
    assert sic_to_industry(5411) == "Shops"   # grocery stores
    assert sic_to_industry(3711) == "Durbl"   # motor vehicles


def test_unknown_and_missing_map_to_other():
    assert sic_to_industry(9999) == "Other"
    assert sic_to_industry(None) == "Other"
    assert sic_to_industry(np.nan) == "Other"
