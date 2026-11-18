"""Tests determinísticos de las fórmulas estadísticas (Fases 1 y 2)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from tpi import stats as S


@pytest.fixture
def datos_ols():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = 1 + 2 * X.a - X.b + rng.normal(scale=1 + np.abs(X.c), size=200)  # heterocedástico
    return X, y


def test_ols_coincide_con_sklearn(datos_ols):
    X, y = datos_ols
    r = S.ols(X, y)
    lr = LinearRegression().fit(X, y)
    np.testing.assert_allclose(r.beta[1:], lr.coef_, rtol=1e-10)
    np.testing.assert_allclose(r.beta[0], lr.intercept_, rtol=1e-10)
    assert abs(r.r2 - lr.score(X, y)) < 1e-10


def test_hc3_formula_explicita(datos_ols):
    X, y = datos_ols
    r = S.ols(X, y)
    Xm = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    h = np.einsum("ij,jk,ik->i", Xm, XtX_inv, Xm)
    omega = np.diag(r.resid**2 / (1 - h) ** 2)
    se = np.sqrt(np.diag(XtX_inv @ Xm.T @ omega @ Xm @ XtX_inv))
    np.testing.assert_allclose(r.se_hc3, se, rtol=1e-8)
    np.testing.assert_allclose(r.hat, h, rtol=1e-8)


def test_cook_y_residuo_externo_por_leave_one_out(datos_ols):
    X, y = datos_ols
    r = S.ols(X, y)
    inf = S.influence(r)
    for i in [0, 17, 123]:
        mask = np.arange(len(X)) != i
        ri = S.ols(X[mask], y[mask])
        Xm = np.column_stack([np.ones(len(X)), X])
        cook_loo = np.sum((Xm @ r.beta - Xm @ ri.beta) ** 2) / (r.p * r.sigma2)
        assert abs(inf.cook[i] - cook_loo) < 1e-8
        t_ext = r.resid[i] / np.sqrt(ri.sigma2 * (1 - r.hat[i]))
        assert abs(inf.resid_stud_ext[i] - t_ext) < 1e-8


def test_breusch_pagan_detecta_heterocedasticidad(datos_ols):
    X, y = datos_ols
    r = S.ols(X, y)
    assert S.breusch_pagan(r, X[["c"]].abs().to_numpy())["p_value"] < 0.05


def test_durbin_watson_ruido_blanco_cerca_de_2():
    e = np.random.default_rng(1).normal(size=5000)
    assert 1.9 < S.durbin_watson(e) < 2.1


def test_vif_independientes_y_colinealidad_perfecta():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    assert (S.vif(X) < 1.1).all()
    X["d"] = X.a + X.b
    with np.errstate(divide="ignore", invalid="ignore"):
        assert S.vif(X).max() > 1e6


def test_welch_anova_igual_a_scipy():
    rng = np.random.default_rng(3)
    g = [rng.normal(0, 1, 50), rng.normal(0.5, 3, 80), rng.normal(1, 0.5, 40)]
    ours = S.welch_anova(*g)
    ref = stats.f_oneway(*g, equal_var=False)
    assert abs(ours["F"] - ref.statistic) < 1e-8
    assert abs(ours["p_value"] - ref.pvalue) < 1e-10


def test_two_proportion_z2_igual_chi2():
    r = S.two_proportion_test(90, 1000, 120, 1000)
    chi2 = stats.chi2_contingency([[90, 910], [120, 880]], correction=False)[0]
    assert abs(r["z"] ** 2 - chi2) < 1e-8
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_potencia_en_el_mde_es_80():
    mde = S.mde_two_proportions(0.09, 12000, 57000)
    assert abs(S.power_two_proportions(0.09, 0.09 - mde, 12000, 57000) - 0.80) < 1e-3


def test_mann_whitney_rank_biserial():
    a, b = np.arange(10, 20), np.arange(0, 10)
    r = S.mann_whitney(a, b)
    assert r["rank_biserial"] == pytest.approx(1.0)
    assert r["rb_ci_low"] <= r["rank_biserial"]


def test_games_howell_sin_diferencias_no_rechaza():
    rng = np.random.default_rng(4)
    g = {k: rng.normal(0, s, 300) for k, s in zip("abc", [1, 2, 3])}
    assert (S.games_howell(g)["p_ajustado"] > 0.01).all()
