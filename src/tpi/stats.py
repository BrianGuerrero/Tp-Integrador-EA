"""Inferencia clásica (Fase 1) y econometría de Gauss-Markov (Fase 2).

Todas las fórmulas están implementadas explícitamente con numpy/scipy para que
el cálculo sea auditable (y testeado en tests/test_stats.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ALPHA, SEED

# =============================================================================
# Fase 1 — contrastes de hipótesis
# =============================================================================


def two_proportion_test(x1: int, n1: int, x2: int, n2: int, alpha: float = ALPHA) -> dict:
    """Test z para diferencia de proporciones p1 - p2 (H0: p1 = p2).

    El estadístico usa la proporción combinada bajo H0; el IC usa el error
    estándar no combinado (Wald). Efecto: h de Cohen = 2·asin(√p1) − 2·asin(√p2).
    """
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "ci_low": p1 - p2 - zc * se,
            "ci_high": p1 - p2 + zc * se, "z": z, "p_value": p, "cohen_h": h}


def power_two_proportions(p1: float, p2: float, n1: int, n2: int, alpha: float = ALPHA) -> float:
    """Potencia (1-β) del test z bilateral para detectar p1 ≠ p2 con n1, n2."""
    h = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    n_eff = 1 / (1 / n1 + 1 / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(zc - h * np.sqrt(n_eff)) + stats.norm.cdf(-zc - h * np.sqrt(n_eff)))


def mde_two_proportions(p_base: float, n1: int, n2: int, alpha: float = ALPHA, power: float = 0.80) -> float:
    """Efecto mínimo detectable (en diferencia de proporciones) con potencia dada."""
    zc, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    h = (zc + zb) / np.sqrt(1 / (1 / n1 + 1 / n2))
    p_alt = np.sin(np.arcsin(np.sqrt(p_base)) - h / 2) ** 2
    return float(p_base - p_alt)


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def welch_t(a, b, alpha: float = ALPHA) -> dict:
    """Test t de Welch para medias independientes con IC de la diferencia."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = (a.mean() - b.mean()) / se
    tc = stats.t.ppf(1 - alpha / 2, df)
    diff = a.mean() - b.mean()
    return {"diff": diff, "ci_low": diff - tc * se, "ci_high": diff + tc * se, "t": t, "df": df,
            "p_value": 2 * stats.t.sf(abs(t), df), "cohen_d": cohen_d(a, b)}


def mann_whitney(a, b, alpha: float = ALPHA) -> dict:
    """Mann-Whitney U con correlación rank-biserial r = 2·AUC − 1 e IC por Hanley-McNeil (1982).

    AUC = U / (n_a·n_b) = P(A > B) + ½P(A = B): probabilidad de superioridad.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (na * nb)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (na - 1) * (q1 - auc**2) + (nb - 1) * (q2 - auc**2)) / (na * nb))
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"median_a": np.median(a), "median_b": np.median(b), "U": u, "p_value": p,
            "prob_superioridad": auc, "rank_biserial": 2 * auc - 1,
            "rb_ci_low": 2 * (auc - zc * se) - 1, "rb_ci_high": 2 * (auc + zc * se) - 1}


# =============================================================================
# Fase 1 — ANOVA y supuestos
# =============================================================================


def normality_summary(groups: dict, n_sub: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Asimetría, curtosis (exceso) y Shapiro-Wilk en la muestra completa vs. submuestra.

    Ilustra que con N grande Shapiro rechaza ante desvíos irrelevantes para el TCL.
    """
    rng = np.random.default_rng(seed)
    filas = []
    for k, x in groups.items():
        x = np.asarray(x, float)
        sub = rng.choice(x, min(n_sub, len(x)), replace=False)
        filas.append({"grupo": k, "n": len(x), "media": x.mean(), "desvío": x.std(ddof=1),
                      "asimetría": stats.skew(x), "curtosis_exceso": stats.kurtosis(x),
                      "|asim|<=1": abs(stats.skew(x)) <= 1,
                      "Shapiro p (n=5000)": stats.shapiro(x[:5000])[1],
                      f"Shapiro p (n={n_sub})": stats.shapiro(sub)[1]})
    return pd.DataFrame(filas)


def welch_anova(*samples) -> dict:
    """ANOVA de Welch (varianzas desiguales), fórmula de Welch (1951)."""
    k = len(samples)
    n = np.array([len(s) for s in samples], float)
    m = np.array([np.mean(s) for s in samples])
    v = np.array([np.var(s, ddof=1) for s in samples])
    w = n / v
    mw = np.sum(w * m) / np.sum(w)
    num = np.sum(w * (m - mw) ** 2) / (k - 1)
    lam = np.sum((1 - w / w.sum()) ** 2 / (n - 1))
    den = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    F = num / den
    df2 = (k**2 - 1) / (3 * lam)
    return {"F": F, "df1": k - 1, "df2": df2, "p_value": stats.f.sf(F, k - 1, df2)}


def anova_effect_sizes(*samples) -> dict:
    """η² y ω² de un ANOVA de un factor."""
    allx = np.concatenate(samples)
    gm = allx.mean()
    ssb = sum(len(s) * (np.mean(s) - gm) ** 2 for s in samples)
    sst = np.sum((allx - gm) ** 2)
    ssw = sst - ssb
    k, N = len(samples), len(allx)
    msw = ssw / (N - k)
    return {"eta2": ssb / sst, "omega2": (ssb - (k - 1) * msw) / (sst + msw)}


def games_howell(groups: dict) -> pd.DataFrame:
    """Post-hoc de Games-Howell: comparaciones de a pares robustas a heterocedasticidad.

    Controla la tasa de error por familia con la distribución del rango studentizado.
    """
    filas = []
    k = len(groups)
    for (ga, a), (gb, b) in combinations(groups.items(), 2):
        a, b = np.asarray(a, float), np.asarray(b, float)
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt((va + vb) / 2)
        df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        q = abs(a.mean() - b.mean()) / se
        p = stats.studentized_range.sf(q, k, df)
        filas.append({"A": ga, "B": gb, "dif_medias": a.mean() - b.mean(), "q": q, "df": df,
                      "p_ajustado": min(p, 1.0)})
    return pd.DataFrame(filas)


def bonferroni_pairwise(groups: dict, alpha: float = ALPHA) -> pd.DataFrame:
    """t de Welch de a pares con corrección de Bonferroni."""
    pares = list(combinations(groups.keys(), 2))
    filas = []
    for ga, gb in pares:
        r = welch_t(groups[ga], groups[gb])
        filas.append({"A": ga, "B": gb, "dif_medias": r["diff"], "p_crudo": r["p_value"],
                      "p_bonferroni": min(r["p_value"] * len(pares), 1.0),
                      "rechaza_H0": r["p_value"] * len(pares) < alpha})
    return pd.DataFrame(filas)


# =============================================================================
# Fase 2 — MCO, diagnósticos de Gauss-Markov, influencia, VIF
# =============================================================================


@dataclass
class OLSResult:
    names: list
    beta: np.ndarray
    se: np.ndarray
    se_hc3: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    hat: np.ndarray
    n: int
    p: int  # número de parámetros incluida la constante
    r2: float
    r2_adj: float
    F: float
    F_p: float
    sigma2: float

    def table(self, robust: bool = False, alpha: float = ALPHA) -> pd.DataFrame:
        se = self.se_hc3 if robust else self.se
        t = self.beta / se
        df = self.n - self.p
        tc = stats.t.ppf(1 - alpha / 2, df)
        return pd.DataFrame({"coef": self.beta, "se": se, "t": t, "p_value": 2 * stats.t.sf(np.abs(t), df),
                             "ci_low": self.beta - tc * se, "ci_high": self.beta + tc * se},
                            index=self.names)


def ols(X: pd.DataFrame, y, add_const: bool = True) -> OLSResult:
    """Mínimos cuadrados ordinarios: β̂ = (X'X)⁻¹X'y, con errores clásicos y HC3."""
    Xm = X.to_numpy(float)
    names = list(X.columns)
    if add_const:
        Xm = np.column_stack([np.ones(len(Xm)), Xm])
        names = ["const"] + names
    y = np.asarray(y, float)
    n, p = Xm.shape
    Q, R = np.linalg.qr(Xm)
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = Xm @ beta
    e = y - fitted
    hat = np.sum(Q**2, axis=1)  # h_ii = diag(X (X'X)⁻¹ X')
    sigma2 = e @ e / (n - p)
    Rinv = np.linalg.inv(R)
    XtX_inv = Rinv @ Rinv.T
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    # HC3 (MacKinnon & White, 1985): Ω = diag(e_i² / (1 - h_ii)²)
    meat = (Xm * (e**2 / (1 - hat) ** 2)[:, None]).T @ Xm
    se_hc3 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - e @ e / sst
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)
    F = (r2 / (p - 1)) / ((1 - r2) / (n - p))
    return OLSResult(names, beta, se, se_hc3, e, fitted, hat, n, p, r2, r2_adj, F,
                     stats.f.sf(F, p - 1, n - p), sigma2)


def breusch_pagan(res: OLSResult, Xaux: np.ndarray) -> dict:
    """Breusch-Pagan studentizado (Koenker): LM = n·R² de e² sobre X. H0: homocedasticidad."""
    e2 = res.resid**2
    Xa = np.column_stack([np.ones(len(e2)), Xaux])
    b = np.linalg.lstsq(Xa, e2, rcond=None)[0]
    u = e2 - Xa @ b
    r2 = 1 - u @ u / np.sum((e2 - e2.mean()) ** 2)
    lm = len(e2) * r2
    df = Xa.shape[1] - 1
    return {"LM": lm, "df": df, "p_value": stats.chi2.sf(lm, df)}


def white_test(res: OLSResult, Xnum: np.ndarray) -> dict:
    """Test de White (versión con cuadrados, sin productos cruzados, para numéricas)."""
    return breusch_pagan(res, np.column_stack([Xnum, Xnum**2]))


def jarque_bera(resid) -> dict:
    r = stats.jarque_bera(resid)
    return {"JB": r.statistic, "p_value": r.pvalue, "asimetria": stats.skew(resid),
            "curtosis": stats.kurtosis(resid, fisher=False)}


def durbin_watson(resid) -> float:
    """DW = Σ(e_t - e_{t-1})² / Σe_t² ; ≈2 sin autocorrelación de orden 1."""
    e = np.asarray(resid)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e**2))


def influence(res: OLSResult) -> pd.DataFrame:
    """Residuos estudentizados externamente, leverage y distancia de Cook."""
    e, h, n, p = res.resid, res.hat, res.n, res.p
    s2 = res.sigma2
    r_int = e / np.sqrt(s2 * (1 - h))
    # varianza sin la obs i: s²_(i) = [(n-p)s² - e_i²/(1-h_ii)] / (n-p-1)
    s2_i = ((n - p) * s2 - e**2 / (1 - h)) / (n - p - 1)
    r_ext = e / np.sqrt(s2_i * (1 - h))
    cook = (e**2 / (p * s2)) * (h / (1 - h) ** 2)
    return pd.DataFrame({"resid_stud_ext": r_ext, "resid_stud_int": r_int, "leverage": h, "cook": cook})


def vif(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R²_j), regresando cada predictor sobre los demás (con constante)."""
    out = {}
    for c in X.columns:
        r = ols(X.drop(columns=c), X[c])
        out[c] = 1 / (1 - r.r2)
    return pd.Series(out, name="VIF").sort_values(ascending=False)
