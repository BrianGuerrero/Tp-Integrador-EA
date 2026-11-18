"""Tests del DAG, del criterio backdoor y de los estimadores causales con efecto conocido."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tpi import causal as C
from tpi.config import Z_CAT, Z_NUM

Z = set(Z_NUM + Z_CAT)


@pytest.fixture(scope="module")
def G():
    return C.build_dag()


def test_dag_aciclico(G):
    assert nx.is_directed_acyclic_graph(G)


def test_backdoor_valido_con_z(G):
    assert C.backdoor_check(G, Z)["valido"]


@pytest.mark.parametrize("extra", [C.M, C.C, C.LOS])
def test_ajustar_descendientes_invalida(G, extra):
    r = C.backdoor_check(G, Z | {extra})
    assert not r["valido"] and extra in r["descendientes_incluidos"]


def test_sin_ajuste_quedan_caminos_abiertos(G):
    assert not C.backdoor_check(G, set())["bloquea_backdoor"]


def _sintetico(n=20000, tau=-0.05, seed=0):
    """Datos con confusión: Z -> T y Z -> Y; efecto verdadero tau."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    e = 1 / (1 + np.exp(-(-1 + 1.2 * z)))
    t = rng.binomial(1, e)
    y = rng.binomial(1, np.clip(0.15 + 0.08 * z + tau * t, 0.01, 0.99))
    return y.astype(float), t.astype(float), e, z


def test_ipw_recupera_efecto_conocido():
    y, t, e, _ = _sintetico()
    naive = y[t == 1].mean() - y[t == 0].mean()
    ate = C.ipw_ate(y, t, e)
    assert abs(ate - (-0.05)) < 0.015
    assert abs(naive - (-0.05)) > 0.03  # el naive está sesgado por confusión


def test_e_value():
    assert C.e_value(1.0) == pytest.approx(1.0)
    assert C.e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert C.e_value(0.5) == C.e_value(2.0)


def test_cate_predict_ic_contiene_estimacion():
    coef = np.array([0.0, 0.01, -0.02])
    cov = np.diag([1e-4, 1e-4, 1e-4])
    tau, lo, hi = C.cate_predict(coef, cov, np.array([[1.0, 0.0], [0.0, 2.0]]))
    assert np.all(lo < tau) and np.all(tau < hi)
    np.testing.assert_allclose(tau, [0.01, -0.04])


def test_policy_table_medir_a_todos():
    cate = np.array([-0.02, -0.01, 0.0, 0.01])
    tab = C.policy_table(cate, cate + 0.005, np.array([1, 0, 0, 0]))
    assert tab.loc[tab["Política"] == "Medir a todos", "Reinternaciones evitadas"].item() == pytest.approx(0.02)


def test_ols_effect_naive_es_diferencia_de_medias():
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({c: rng.normal(size=n) for c in Z_NUM})
    for c in Z_CAT:
        df[c] = rng.choice(["a", "b"], n)
    df[C.T] = rng.binomial(1, 0.3, n)
    df[C.Y] = rng.binomial(1, 0.1 + 0.05 * df[C.T])
    r = C.ols_effect(df, adjust=False)
    dm = df.loc[df[C.T] == 1, C.Y].mean() - df.loc[df[C.T] == 0, C.Y].mean()
    assert r["efecto_pp"] == pytest.approx(100 * dm)
