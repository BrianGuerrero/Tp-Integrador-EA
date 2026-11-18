"""Contrato notebook -> app: los artefactos existen y tienen el esquema esperado."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from tpi.config import ARTIFACTS_DIR

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "meta.json").exists(),
                                reason="artefactos no generados (correr el notebook)")


def _j(n):
    return json.loads((ARTIFACTS_DIR / n).read_text(encoding="utf-8"))


def test_meta_tiene_claves():
    meta = _j("meta.json")
    for k in ["n", "tasa_base", "pred_num", "pred_bin", "pred_cat", "referencia", "opciones", "pred_metrics"]:
        assert k in meta


def test_cate_dimensiones_y_covarianza_psd():
    c = _j("cate_lineal.json")
    coef, cov = np.array(c["coef"]), np.array(c["cov"])
    assert len(coef) == len(c["X_cols"]) + 1
    assert cov.shape == (len(coef), len(coef))
    assert np.all(np.linalg.eigvalsh((cov + cov.T) / 2) >= -1e-12)


def test_refutaciones_tres_con_veredicto():
    r = _j("refutaciones.json")
    assert len(r) == 3 and all(isinstance(x["aprobado"], bool) for x in r)


def test_modelo_predice_probabilidades_validas():
    meta = _j("meta.json")
    modelo = joblib.load(ARTIFACTS_DIR / "modelo_predictivo.joblib")
    muestra = pd.read_csv(ARTIFACTS_DIR / "muestra_exploracion.csv.gz")
    p = modelo.predict_proba(muestra[meta["pred_num"] + meta["pred_bin"] + meta["pred_cat"]].head(50))[:, 1]
    assert np.all((p > 0) & (p < 1))
