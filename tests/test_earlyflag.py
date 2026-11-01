import numpy as np
import pandas as pd
import pytest

from earlyflag.audit import capacity_curve, flag_top, group_audit
from earlyflag.causal import e_value, ipw_risk_difference
from earlyflag.data import snake


def test_snake_handles_raw_headers():
    assert snake("Daytime/evening attendance\t") == "daytime_evening_attendance"
    assert snake("Mother's qualification") == "mothers_qualification"
    assert snake("Curricular units 1st sem (approved)") == "curricular_units_1st_sem_approved"


def test_flag_top_and_capacity():
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    y = np.array([1, 0, 0, 0, 1])
    assert flag_top(p, 0.4).tolist() == [True, False, True, False, False]
    row = capacity_curve(y, p, [0.6]).iloc[0]
    assert row.flagged == 3 and row.dropouts_reached == 2
    assert row.recall == 1.0 and row.precision == pytest.approx(2 / 3)


def test_group_audit_rates():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.7, 0.6]
    f = [True, False, True, False, True, False]
    g = ["a", "a", "a", "a", "b", "b"]
    a = group_audit(y, p, f, pd.Series(g)).set_index("group")
    assert a.loc["a", "recall"] == 0.5
    assert a.loc["a", "false_positive_rate"] == 0.5
    assert a.loc["b", "recall"] == 1.0 and a.loc["b", "false_positive_rate"] == 0.0


def test_e_value_reference():
    assert e_value(2.0) == pytest.approx(3.414, abs=1e-3)
    assert e_value(0.5) == pytest.approx(3.414, abs=1e-3)


def test_ipw_removes_confounding():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-1.5 * x)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 1.2 * x - 0.0 * t))))
    df = pd.DataFrame({"x": x, "t": t, "y": y})
    r = ipw_risk_difference(df, "t", "y", ["x"], n_boot=50)
    assert r["naive_rd"] > 0.1
    assert abs(r["ipw_rd"]) < 0.02
