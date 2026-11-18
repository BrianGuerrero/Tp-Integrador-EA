"""Controles de fuga de datos (data leakage) y de la partición."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tpi import models as Mdl
from tpi.config import PCA_VARS, TEST_SIZE
from tpi.data import load_dataset


@pytest.fixture(scope="module")
def particion():
    return Mdl.split(load_dataset())


def test_particion_disjunta_y_80_20(particion):
    Xtr, Xte, ytr, yte = particion
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert abs(len(Xte) / (len(Xtr) + len(Xte)) - TEST_SIZE) < 1e-3


def test_particion_estratificada(particion):
    _, _, ytr, yte = particion
    assert abs(ytr.mean() - yte.mean()) < 1e-3


def test_features_sin_target_ni_ids():
    prohibidas = {"readmit30", "readmitted", "encounter_id", "patient_nbr", "discharge_disposition_id"}
    assert prohibidas.isdisjoint(Mdl.FEATURES)


def test_escalado_aprendido_solo_en_train(particion):
    Xtr, Xte, ytr, _ = particion
    pipe = Mdl.make_pipe(LogisticRegression(max_iter=200)).fit(Xtr, ytr)
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, Xtr[scaler.feature_names_in_].mean().to_numpy(), rtol=1e-10)
    todo = np.vstack([Xtr[scaler.feature_names_in_], Xte[scaler.feature_names_in_]]).mean(0)
    assert not np.allclose(scaler.mean_, todo, rtol=1e-6)


def test_pca_sin_fuga(particion):
    Xtr, Xte, _, _ = particion
    pca = Pipeline([("s", StandardScaler()), ("p", PCA(random_state=0))]).fit(Xtr[PCA_VARS])
    assert np.abs(pca.transform(Xtr[PCA_VARS]).mean(0)).max() < 1e-10
    assert np.abs(pca.transform(Xte[PCA_VARS]).mean(0)).max() > 1e-4


def test_one_hot_tolera_categorias_nuevas(particion):
    Xtr, Xte, ytr, _ = particion
    pipe = Mdl.make_pipe(LogisticRegression(max_iter=200)).fit(Xtr, ytr)
    x = Xte.iloc[:1].copy()
    x["especialidad"] = "CategoriaInexistente"
    assert 0 < pipe.predict_proba(x)[0, 1] < 1
