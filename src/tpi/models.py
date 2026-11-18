"""Fase 3 — torneo predictivo: preprocesamiento sin fuga, modelos, métricas y explicabilidad."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import OUTCOME, PRED_BIN, PRED_CAT, PRED_NUM, SEED, TEST_SIZE

FEATURES = PRED_NUM + PRED_BIN + PRED_CAT


def split(df: pd.DataFrame, seed: int = SEED):
    """Partición estratificada 80/20. El test no se toca hasta la evaluación final."""
    X, y = df[FEATURES], df[OUTCOME].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


def preprocessor(scale: bool = True) -> ColumnTransformer:
    """Transformaciones que se AJUSTAN dentro de cada fold (evita data leakage)."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, PRED_NUM),
         ("bin", "passthrough", PRED_BIN),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), PRED_CAT)],
        verbose_feature_names_out=False,
    )


def preprocessor_discreto(n_bins: int = 5) -> ColumnTransformer:
    """Todo a categorías ordinales para CategoricalNB (numéricas en quintiles)."""
    return ColumnTransformer(
        [("num", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                                  quantile_method="averaged_inverted_cdf"), PRED_NUM),
         ("bin", "passthrough", PRED_BIN),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), PRED_CAT)],
        verbose_feature_names_out=False,
    )


def make_pipe(model, pre=None) -> Pipeline:
    return Pipeline([("pre", pre if pre is not None else preprocessor()), ("model", model)])


# ---------------------------------------------------------------------------
# Inferencia sobre la logística: odds ratios con IC 95% (matriz de información)
# ---------------------------------------------------------------------------


def logit_design(X: pd.DataFrame) -> pd.DataFrame:
    """Matriz de diseño interpretable: numéricas en escala original + dummies (base = moda)."""
    num = X[PRED_NUM + PRED_BIN].astype(float)
    dums = []
    for c in PRED_CAT:
        base = X[c].mode()[0]
        dums.append(pd.get_dummies(X[c], prefix=c, dtype=float).drop(columns=f"{c}_{base}"))
    return pd.concat([num] + dums, axis=1)


def logit_odds_ratios(Xd: pd.DataFrame, y, alpha: float = 0.05) -> pd.DataFrame:
    """Logística sin penalización por máxima verosimilitud (Newton-Raphson).

    Var(β̂) = (X'WX)⁻¹ con W = diag(p̂(1-p̂)); OR = exp(β), IC = exp(β ± z·se).
    """
    X = np.column_stack([np.ones(len(Xd)), Xd.to_numpy(float)])
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = expit(X @ beta)
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = expit(X @ beta)
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    z = stats.norm.ppf(1 - alpha / 2)
    out = pd.DataFrame({"beta": beta, "se": se, "OR": np.exp(beta), "OR_ci_low": np.exp(beta - z * se),
                        "OR_ci_high": np.exp(beta + z * se),
                        "p_value": 2 * stats.norm.sf(np.abs(beta / se))},
                       index=["const"] + list(Xd.columns))
    return out


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def best_threshold_f1(y, p) -> float:
    """Umbral que maximiza F1 (se elige con predicciones out-of-fold de TRAIN)."""
    grid = np.quantile(p, np.linspace(0.50, 0.99, 100))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    return {"AUC-ROC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p),
            "Precision": precision_score(y, yhat, zero_division=0), "Recall": recall_score(y, yhat),
            "F1": f1_score(y, yhat), "F1-macro": f1_score(y, yhat, average="macro"),
            "F1-ponderado": f1_score(y, yhat, average="weighted"), "umbral": thr}


def bootstrap_auc(y, p, n_boot: int = 500, seed: int = SEED) -> tuple[float, float]:
    """IC 95% percentil del AUC en test por bootstrap estratificado."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(aucs, [0.025, 0.975]))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    pt, pp = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    counts = np.histogram(p, bins=np.quantile(p, np.linspace(0, 1, n_bins + 1)))[0]
    counts = counts[: len(pt)]
    return float(np.sum(counts * np.abs(pt - pp)) / counts.sum())


def mdi_por_variable(pipe: Pipeline) -> pd.Series:
    """Suma la importancia MDI de las dummies de una misma variable original."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    orig = [next((c for c in PRED_CAT if n.startswith(c + "_")), n) for n in names]
    return imp.groupby(orig).sum().sort_values(ascending=False)
