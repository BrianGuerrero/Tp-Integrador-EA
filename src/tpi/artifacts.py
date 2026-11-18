"""Serialización de los artefactos que consume la app (contrato de datos notebook -> app)."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, PCA_VARS, PRED_BIN, PRED_CAT, PRED_NUM, SEED


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, name: str):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / name).write_text(json.dumps(_to_py(obj), indent=2, ensure_ascii=False))


def export_all(df: pd.DataFrame, pca_pipe, pred_pipe, pred_metrics: dict, cate: dict, ate_table: pd.DataFrame,
               refutations: list, dag_edges: list, fuente_causal: str, n_sample: int = 4000):
    """Guarda todo lo que necesita la app.

    cate: {"X_cols": [...], "coef": [...], "cov": [[...]], "ate_pp":..., "ci_low_pp":..., "ci_high_pp":...}
    refutations: [{"nombre", "efecto_original_pp", "nuevo_efecto_pp", "criterio", "aprobado"}]
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_pipe, ARTIFACTS_DIR / "pca_pipeline.joblib", compress=3)
    joblib.dump(pred_pipe, ARTIFACTS_DIR / "modelo_predictivo.joblib", compress=3)

    # muestra para el módulo de exploración (proyección PCA incluida)
    smp = df.sample(min(n_sample, len(df)), random_state=SEED)
    pcs = pca_pipe.transform(smp[PCA_VARS])
    cols = PRED_NUM + PRED_BIN + PRED_CAT + ["readmit30", "edad_grupo", "diag_circ", "emergencia"]
    out = smp[list(dict.fromkeys(cols))].copy()
    out["PC1"], out["PC2"] = pcs[:, 0], pcs[:, 1]
    out.to_csv(ARTIFACTS_DIR / "muestra_exploracion.csv.gz", index=False)

    # valores de referencia (mediana / moda) para explicaciones locales
    ref = {c: float(df[c].median()) for c in PRED_NUM + PRED_BIN}
    ref.update({c: str(df[c].mode()[0]) for c in PRED_CAT})
    opciones = {c: sorted(df[c].dropna().unique().tolist()) for c in PRED_CAT}

    save_json({"creado": datetime.now().isoformat(timespec="seconds"), "n": len(df),
               "tasa_base": float(df.readmit30.mean()), "pred_num": PRED_NUM, "pred_bin": PRED_BIN,
               "pred_cat": PRED_CAT, "pca_vars": PCA_VARS, "referencia": ref, "opciones": opciones,
               "pca_var_explicada": pca_pipe.named_steps["pca"].explained_variance_ratio_.tolist(),
               "pred_metrics": pred_metrics, "fuente_causal": fuente_causal}, "meta.json")
    save_json(cate, "cate_lineal.json")
    save_json({"tabla": ate_table.reset_index().to_dict(orient="records")}, "ate.json")
    save_json(refutations, "refutaciones.json")
    save_json({"aristas": dag_edges}, "dag.json")
