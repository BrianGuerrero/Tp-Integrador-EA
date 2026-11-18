"""Integridad de la limpieza y de los contratos de datos."""

import pytest

from tpi.config import OUTCOME, TREATMENT
from tpi.data import ALTA_FALLECIDO_HOSPICIO, auditoria_centinelas, build_features, clean, grupo_cie9, load_raw


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def df(raw):
    return build_features(clean(raw)[0])


def test_dimensiones_base_original(raw):
    assert raw.shape == (101766, 50)


def test_una_fila_por_paciente(df):
    assert df.patient_nbr.is_unique


def test_sin_fallecidos_ni_hospicio(df):
    assert not df.discharge_disposition_id.isin(ALTA_FALLECIDO_HOSPICIO).any()


def test_sin_centinela_de_sexo(df):
    assert set(df.sexo.unique()) == {"Mujer", "Hombre"}


def test_n_analitico(df):
    assert len(df) == 69970


def test_centinelas_mapeados_a_desconocido(df):
    sentinela = df.admission_type_id.isin([5, 6, 8])
    assert (df.loc[sentinela, "tipo_admision"] == "Desconocido").all()
    assert not (df.loc[~sentinela, "tipo_admision"] == "Desconocido").any()


def test_none_de_hba1c_no_es_faltante(raw):
    assert (raw.A1Cresult == "None").sum() > 80000
    assert raw.A1Cresult.isna().sum() == 0


def test_tratamiento_y_resultado_binarios(df):
    for c in [TREATMENT, OUTCOME, "cambio_med"]:
        assert set(df[c].unique()) == {0, 1}


def test_sin_faltantes_en_variables_modeladas(df):
    from tpi.config import PRED_BIN, PRED_CAT, PRED_NUM

    assert df[PRED_NUM + PRED_BIN + PRED_CAT].isna().sum().sum() == 0


def test_auditoria_cuenta_centinelas(raw):
    a = auditoria_centinelas(raw)
    assert a.loc[(a.variable == "admission_source_id") & (a.codigo == 17), "n"].item() == 6781


@pytest.mark.parametrize("code,grupo", [("428", "Circulatorio"), ("250.83", "Diabetes"), ("V57", "Otro"),
                                       ("486", "Respiratorio"), ("820", "Lesiones"), (None, "Otro")])
def test_grupo_cie9(code, grupo):
    assert grupo_cie9(code) == grupo
