import numpy as np

from riskprism.factors.industry import INDUSTRIES, sic_to_industry


def test_known_sic_mappings():
    # FF30 (v0.9), per Ken French's published Siccodes30 definitions
    assert sic_to_industry(7372) == "Servs"   # computer programming & data processing
    assert sic_to_industry(3571) == "BusEq"   # electronic computers
    assert sic_to_industry(2834) == "Hlth"    # pharmaceutical preparations
    assert sic_to_industry(6022) == "Fin"     # state commercial banks
    assert sic_to_industry(1311) == "Oil"     # crude petroleum & natural gas
    assert sic_to_industry(4911) == "Util"    # electric services
    assert sic_to_industry(5411) == "Rtail"   # grocery stores
    assert sic_to_industry(3711) == "Autos"   # motor vehicles
    assert sic_to_industry(2082) == "Beer"    # malt beverages
    assert sic_to_industry(2111) == "Smoke"   # cigarettes
    assert sic_to_industry(5812) == "Meals"   # eating places
    assert sic_to_industry(1220) == "Coal"    # bituminous coal


def test_thirty_industries_other_last():
    assert len(INDUSTRIES) == 30
    assert INDUSTRIES[-1] == "Other"
    assert len(set(INDUSTRIES)) == 30


def test_unknown_and_missing_map_to_other():
    assert sic_to_industry(9100) == "Other"   # government — not in any FF30 range
    assert sic_to_industry(None) == "Other"
    assert sic_to_industry(np.nan) == "Other"
