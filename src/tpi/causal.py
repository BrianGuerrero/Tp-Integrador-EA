"""Fase 4 — inferencia causal.

Incluye el DAG, la verificación formal del criterio backdoor y estimadores
implementados explícitamente (regresión backdoor, IPW, DML con cross-fitting,
refutaciones). El notebook corre además DoWhy y EconML y contrasta resultados
con estas implementaciones de referencia.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

from . import stats as S
from .config import COLLIDER, MEDIATOR, OUTCOME, SEED, TREATMENT, Z_CAT, Z_NUM

T, Y, M, C = TREATMENT, OUTCOME, MEDIATOR, COLLIDER
U = "gravedad_no_observada"
LOS = "time_in_hospital"


# =============================================================================
# DAG
# =============================================================================


def build_dag(include_los: bool = True) -> nx.DiGraph:
    """DAG estructural a nivel de variables (antes de crear dummies).

    Supuestos de dominio:
      * Z (contexto al ingreso) -> T y Z -> Y  (confusión observada)
      * T -> M -> Y  (medir HbA1c lleva a ajustar la medicación)
      * T -> C <- U -> Y  (C = cantidad de análisis: colisionador; U = gravedad latente)
      * T -> LOS -> Y y U -> LOS  (los días de internación son POSTERIORES al test)
      * U no causa T una vez condicionado en Z (supuesto de no confusión residual,
        evaluado luego con análisis de sensibilidad y E-value).
    """
    G = nx.DiGraph()
    for z in Z_NUM + Z_CAT:
        G.add_edges_from([(z, T), (z, Y)])
    G.add_edges_from([(T, Y), (T, M), (M, Y), (T, C), (U, C), (U, Y)])
    if include_los:
        G.add_edges_from([(T, LOS), (LOS, Y), (U, LOS)])
    G.add_edges_from([("number_diagnoses", M), ("diag_principal", M)])
    return G


def backdoor_check(G: nx.DiGraph, adj: set, t: str = T, y: str = Y) -> dict:
    """Criterio backdoor (Pearl): (i) Z sin descendientes de T, (ii) Z d-separa T e Y
    en el grafo sin las aristas que salen de T."""
    desc = nx.descendants(G, t)
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(t)))
    sep = nx.is_d_separator(G_bd, {t}, {y}, set(adj))
    return {"sin_descendientes_de_T": not (desc & set(adj)), "descendientes_incluidos": sorted(desc & set(adj)),
            "bloquea_backdoor": bool(sep), "valido": (not (desc & set(adj))) and bool(sep)}


def backdoor_paths(G: nx.DiGraph, t: str = T, y: str = Y, max_show: int = 5) -> list:
    """Caminos no dirigidos T ... Y que empiezan con una flecha HACIA T."""
    Gu = G.to_undirected()
    out = []
    for path in nx.all_simple_paths(Gu, t, y, cutoff=3):
        if G.has_edge(path[1], t):
            out.append(path)
    return out[:max_show], len(out)


# =============================================================================
# Diseño y estimadores de referencia
# =============================================================================


def z_matrix(df: pd.DataFrame, extra: list | None = None) -> pd.DataFrame:
    """Matriz de ajuste: Z numéricas + dummies de Z categóricas (base = moda)."""
    parts = [df[Z_NUM].astype(float)]
    for c in Z_CAT:
        base = df[c].mode()[0]
        parts.append(pd.get_dummies(df[c], prefix=c, dtype=float).drop(columns=f"{c}_{base}"))
    if extra:
        parts.append(df[extra].astype(float))
    Zm = pd.concat(parts, axis=1)
    Zm.columns = Zm.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    return Zm


def ols_effect(df: pd.DataFrame, extra: list | None = None, adjust: bool = True) -> dict:
    """Modelo lineal de probabilidad Y ~ T (+ Z) con errores HC3. Efecto en pp."""
    X = pd.concat([df[[T]].astype(float), z_matrix(df, extra)], axis=1) if adjust else df[[T]].astype(float)
    r = S.ols(X, df[Y])
    tab = r.table(robust=True).loc[T]
    return {"efecto_pp": 100 * tab.coef, "ci_low_pp": 100 * tab.ci_low, "ci_high_pp": 100 * tab.ci_high,
            "p_value": tab.p_value}


def crossfit_propensity(df: pd.DataFrame, seed: int = SEED, n_folds: int = 5) -> np.ndarray:
    """e(Z) = P(T=1|Z) con cross-fitting (cada obs. predicha por un modelo que no la vio)."""
    from sklearn.linear_model import LogisticRegression

    Zm = z_matrix(df).to_numpy()
    Zs = (Zm - Zm.mean(0)) / (Zm.std(0) + 1e-12)
    t = df[T].to_numpy()
    e = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Zs, t):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Zs[tr], t[tr])
        e[te] = m.predict_proba(Zs[te])[:, 1]
    return e


def ipw_ate(y, t, e, trim: tuple = (0.01, 0.99)) -> float:
    """IPW normalizado (Hájek): μ1 − μ0 con pesos 1/e y 1/(1−e), con recorte de e."""
    e = np.clip(e, *trim)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def ipw_bootstrap(y, t, e, n_boot: int = 300, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y)
    b = [ipw_ate(y[i], t[i], e[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return tuple(np.quantile(b, [0.025, 0.975]))


def smd_balance(df: pd.DataFrame, e: np.ndarray, cols: list) -> pd.DataFrame:
    """Diferencia de medias estandarizada antes y después de ponderar por IPW."""
    Zm = z_matrix(df)[cols] if set(cols) <= set(z_matrix(df).columns) else df[cols]
    t = df[T].to_numpy()
    e = np.clip(e, 0.01, 0.99)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    filas = []
    for c in cols:
        x = Zm[c].to_numpy(float)
        sd = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2) + 1e-12
        crudo = (x[t == 1].mean() - x[t == 0].mean()) / sd
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        filas.append({"variable": c, "SMD_crudo": crudo, "SMD_ponderado": (m1 - m0) / sd})
    return pd.DataFrame(filas)


def dml_plr(df: pd.DataFrame, X_cols: list | None = None, n_folds: int = 5, seed: int = SEED,
            n_estimators: int = 200, min_samples_leaf: int = 50) -> dict:
    """Double ML (modelo parcialmente lineal) con cross-fitting y Random Forest de nuisance.

    Ỹ = Y − Ê[Y|Z], T̃ = T − Ê[T|Z]; luego OLS de Ỹ sobre T̃·[1, X] (= LinearDML).
    Devuelve ATE (promedio del CATE), coeficientes del CATE lineal y los residuos.
    """
    Zm = z_matrix(df).to_numpy()
    y, t = df[Y].to_numpy(float), df[T].to_numpy(float)
    y_res, t_res = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Zm, t):
        my = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                   max_features=0.3, n_jobs=-1, random_state=seed).fit(Zm[tr], y[tr])
        mt = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    max_features=0.3, n_jobs=-1, random_state=seed).fit(Zm[tr], t[tr])
        y_res[te] = y[te] - my.predict(Zm[te])
        t_res[te] = t[te] - mt.predict_proba(Zm[te])[:, 1]
    out = {"y_res": y_res, "t_res": t_res}
    # ATE: regresión sin constante de Ỹ sobre T̃ con error HC0
    theta = np.sum(t_res * y_res) / np.sum(t_res**2)
    psi = (y_res - theta * t_res) * t_res
    se = np.sqrt(np.sum(psi**2)) / np.sum(t_res**2)
    out.update({"ate_pp": 100 * theta, "ci_low_pp": 100 * (theta - 1.96 * se),
                "ci_high_pp": 100 * (theta + 1.96 * se), "p_value": 2 * stats.norm.sf(abs(theta / se))})
    if X_cols:
        X = df[X_cols].to_numpy(float)
        D = np.column_stack([t_res, t_res[:, None] * X])
        r = S.ols(pd.DataFrame(D, columns=["T"] + [f"T:{c}" for c in X_cols]), y_res, add_const=False)
        out["cate_table"] = r.table(robust=True)
        out["cate_coef"] = r.beta
        out["cate_cov"] = _hc3_cov(D, r.resid, r.hat)
        cate = np.column_stack([np.ones(len(X)), X]) @ r.beta
        out["cate"] = cate
        out["ate_cate_pp"] = 100 * cate.mean()
    return out


def _hc3_cov(D, e, h):
    XtX_inv = np.linalg.inv(D.T @ D)
    meat = (D * (e**2 / (1 - h) ** 2)[:, None]).T @ D
    return XtX_inv @ meat @ XtX_inv


def cate_predict(coef, cov, X: np.ndarray, alpha: float = 0.05):
    """CATE lineal τ(x) = θ0 + θ'x con IC: Var = x̃' Σ x̃."""
    Xt = np.column_stack([np.ones(len(X)), X])
    tau = Xt @ coef
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xt, cov, Xt))
    z = stats.norm.ppf(1 - alpha / 2)
    return tau, tau - z * se, tau + z * se


# =============================================================================
# Refutaciones (réplica de las de DoWhy, sobre el estimador backdoor lineal)
# =============================================================================


def refute_placebo(df: pd.DataFrame, n_sim: int = 20, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    Zm = z_matrix(df)
    efs = []
    for _ in range(n_sim):
        tp = rng.permutation(df[T].to_numpy())
        r = S.ols(pd.concat([pd.Series(tp, name=T, index=df.index, dtype=float), Zm], axis=1), df[Y])
        efs.append(100 * r.beta[1])
    return {"nuevo_efecto_pp": float(np.mean(efs)), "sd": float(np.std(efs)), "simulaciones": efs}


def refute_random_common_cause(df: pd.DataFrame, n_sim: int = 20, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    Zm = z_matrix(df)
    efs = []
    for _ in range(n_sim):
        X = pd.concat([df[[T]].astype(float), Zm], axis=1).assign(w_random=rng.normal(size=len(df)))
        efs.append(100 * S.ols(X, df[Y]).beta[1])
    return {"nuevo_efecto_pp": float(np.mean(efs)), "sd": float(np.std(efs)), "simulaciones": efs}


def refute_subset(df: pd.DataFrame, frac: float = 0.8, n_sim: int = 20, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    efs = []
    for _ in range(n_sim):
        sub = df.iloc[rng.choice(len(df), int(frac * len(df)), replace=False)]
        efs.append(ols_effect(sub)["efecto_pp"])
    return {"nuevo_efecto_pp": float(np.mean(efs)), "sd": float(np.std(efs)), "simulaciones": efs}


def e_value(rr: float) -> float:
    """E-value (VanderWeele & Ding, 2017) para una razón de riesgos."""
    rr = 1 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1)))


# =============================================================================
# Simulación con efecto conocido (contraste de las trampas mediador/colisionador)
# =============================================================================


def simulate_known_effect(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, float]:
    """Mantiene Z y T reales y simula M, C, U e Y con un efecto causal total conocido.

    Y depende de Z (vía el riesgo base real), de T directamente (−1 pp), de M (−5 pp)
    y de U; C depende de T y de U. Así la confusión por Z es la del dataset real.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    Zm = z_matrix(df).to_numpy()
    Zs = (Zm - Zm.mean(0)) / (Zm.std(0) + 1e-12)
    p_base = LogisticRegression(max_iter=2000).fit(Zs, df[Y]).predict_proba(Zs)[:, 1]
    n = len(df)
    t = df[T].to_numpy()
    u = rng.normal(size=n)
    m1, m0 = rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n)

    def riesgo(tt, mm):
        return np.clip(p_base + 0.02 - 0.01 * tt - 0.05 * mm + 0.04 * u, 0.001, 0.999)

    p1, p0 = riesgo(1, m1), riesgo(0, m0)
    ate_true = float((p1 - p0).mean())
    sim = df.copy()
    sim[M] = np.where(t == 1, m1, m0)
    sim[Y] = rng.binomial(1, np.where(t == 1, p1, p0))
    sim[C] = 40 + 10 * t + 8 * u + rng.normal(0, 5, n)
    return sim, 100 * ate_true


# =============================================================================
# Política de decisión
# =============================================================================


def policy_table(cate: np.ndarray, cate_hi: np.ndarray, t_obs: np.ndarray, delta: float = 0.01) -> pd.DataFrame:
    """Reinternaciones evitadas estimadas (−Σ τ̂_i) bajo distintas políticas de asignación."""
    pol = {"Política actual (observada)": t_obs, "Medir a todos": np.ones_like(t_obs),
           f"Focalizada (CATE < −{100 * delta:.0f} pp)": (cate < -delta).astype(int),
           "Focalizada conservadora (IC sup. < 0)": (cate_hi < 0).astype(int)}
    filas = []
    for k, v in pol.items():
        evit = float(-(cate * v).sum())
        filas.append({"Política": k, "% pacientes": 100 * v.mean(), "Reinternaciones evitadas": evit,
                      "Evitadas cada 100 tests": 100 * evit / max(v.sum(), 1)})
    return pd.DataFrame(filas)
