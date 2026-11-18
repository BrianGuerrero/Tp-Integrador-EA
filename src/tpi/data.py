"""Carga, auditoría forense y limpieza del dataset Diabetes 130-US Hospitals (UCI id 296).

Diccionario de códigos verificado contra IDs_mapping.csv (distribuido con el dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_CSV, UCI_ID

# ---------------------------------------------------------------------------
# Códigos centinela: valores numéricos que en realidad significan "faltante"
# ---------------------------------------------------------------------------
SENTINELAS = {
    "admission_type_id": {5: "Not Available", 6: "NULL", 8: "Not Mapped"},
    "admission_source_id": {
        9: "Not Available",
        15: "Not Available",
        17: "NULL",
        20: "Not Mapped",
        21: "Unknown/Invalid",
    },
    "discharge_disposition_id": {18: "NULL", 25: "Not Mapped", 26: "Unknown/Invalid"},
    "gender": {"Unknown/Invalid": "Unknown/Invalid"},
}

# Altas por fallecimiento u hospicio: no pueden reinternarse (Y=0 sin significado)
ALTA_FALLECIDO_HOSPICIO = [11, 13, 14, 19, 20, 21]

TIPO_ADMISION = {1: "Emergencia", 2: "Urgencia", 3: "Electiva", 4: "Otra", 7: "Otra"}
ORIGEN_ADMISION = {
    7: "Guardia",
    1: "Derivacion",
    2: "Derivacion",
    3: "Derivacion",
    4: "Traslado",
    5: "Traslado",
    6: "Traslado",
    10: "Traslado",
    22: "Traslado",
    25: "Traslado",
}
COBERTURA = {
    "MC": "Medicare",
    "MD": "Medicaid",
    "SP": "Particular",
    "HM": "Privada",
    "BC": "Privada",
    "CP": "Privada",
    "UN": "Privada",
    "CM": "Privada",
    "OG": "Privada",
    "PO": "Privada",
    "DM": "Privada",
    "CH": "Privada",
    "WC": "Privada",
    "OT": "Privada",
    "MP": "Privada",
    "SI": "Privada",
    "FR": "Privada",
}


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Lee el CSV original. Si no existe, lo descarga de UCI con `ucimlrepo`.

    "?" es el código de faltante del archivo original. `keep_default_na=False`
    evita que pandas convierta el texto "None" (= test no realizado) en NaN.
    """
    try:
        return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    except FileNotFoundError:
        from ucimlrepo import fetch_ucirepo

        raw = fetch_ucirepo(id=UCI_ID).data.original.replace("?", np.nan)
        for c in ["A1Cresult", "max_glu_serum"]:
            raw[c] = raw[c].fillna("None")
        return raw


def grupo_cie9(code) -> str:
    """Agrupa un código CIE-9 en las categorías usadas por Strack et al. (2014)."""
    if pd.isna(code):
        return "Otro"  # 10 casos: demasiado pocos para una categoría propia
    s = str(code)
    if s.startswith(("V", "E")):
        return "Otro"
    x = float(s)
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatorio"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratorio"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestivo"
    if int(x) == 250:
        return "Diabetes"
    if 800 <= x <= 999:
        return "Lesiones"
    if 710 <= x <= 739:
        return "Musculoesqueletico"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinario"
    if 140 <= x <= 239:
        return "Neoplasias"
    return "Otro"


def _especialidad(s: pd.Series) -> pd.Series:
    s = s.copy()
    s[s.str.startswith("Surgery", na=False)] = "Cirugia"
    mapa = {
        "InternalMedicine": "ClinicaMedica",
        "Family/GeneralPractice": "MedicinaGeneral",
        "Emergency/Trauma": "Emergencias",
        "Cardiology": "Cardiologia",
        "Cirugia": "Cirugia",
    }
    out = s.map(mapa)
    out[s.notna() & out.isna()] = "Otra"
    return out.fillna("Desconocido")


def auditoria_centinelas(raw: pd.DataFrame) -> pd.DataFrame:
    """Cuenta los valores centinela ocultos (faltantes disfrazados de código válido)."""
    filas = []
    for col, codigos in SENTINELAS.items():
        for cod, desc in codigos.items():
            n = int((raw[col] == cod).sum())
            filas.append({"variable": col, "codigo": cod, "significado": desc, "n": n,
                          "%": 100 * n / len(raw)})
    return pd.DataFrame(filas)


def clean(raw: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica los filtros de unidad de análisis y devuelve (df, log de pasos)."""
    log = []

    def paso(nombre, d):
        log.append({"paso": nombre, "N": len(d), "tasa_readmit30_%": 100 * (d.readmitted == "<30").mean()})

    df = raw.copy()
    paso("0. Base original (internaciones)", df)

    # 1) independencia: una internación por paciente (la primera en el tiempo)
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    paso("1. Primera internación por paciente", df)

    # 2) Y no definida: fallecidos u hospicio no pueden reinternarse
    df = df[~df.discharge_disposition_id.isin(ALTA_FALLECIDO_HOSPICIO)]
    paso("2. Excluye fallecidos / hospicio", df)

    # 3) centinela de sexo (3 registros "Unknown/Invalid"): se descartan
    df = df[df.gender != "Unknown/Invalid"]
    paso("3. Excluye sexo inválido (centinela)", df)

    if verbose:
        print(pd.DataFrame(log).to_string(index=False))
    return df.reset_index(drop=True), pd.DataFrame(log)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye T, Y, M y las covariables con nombres en castellano."""
    d = df.copy()
    # resultado, tratamiento, mediador
    d["readmit30"] = (d.readmitted == "<30").astype(int)
    d["hba1c_medida"] = (d.A1Cresult != "None").astype(int)
    d["cambio_med"] = (d.change == "Ch").astype(int)
    d["diabetesMed"] = (d.diabetesMed == "Yes").astype(int)
    d["insulina"] = (d.insulin != "No").astype(int)

    # edad: punto medio del decenio "[70-80)" -> 75
    d["edad"] = d.age.str.extract(r"\[(\d+)-").astype(float)[0] + 5
    d["edad_c"] = d.edad - 65  # edad centrada (para interacciones del CATE)
    d["edad_grupo"] = pd.cut(d.edad, [0, 50, 65, 80, 100], right=False,
                             labels=["<50", "50-64", "65-79", "80+"])

    d["sexo"] = d.gender.map({"Female": "Mujer", "Male": "Hombre"})
    d["raza"] = d.race.fillna("Desconocido")
    # centinelas -> "Desconocido" (no se pierden filas; se modela la falta)
    d["tipo_admision"] = d.admission_type_id.map(TIPO_ADMISION).fillna("Desconocido")
    d["origen_admision"] = d.admission_source_id.map(ORIGEN_ADMISION)
    validos = ~d.admission_source_id.isin(SENTINELAS["admission_source_id"].keys())
    d.loc[d.origen_admision.isna() & validos, "origen_admision"] = "Otro"
    d["origen_admision"] = d.origen_admision.fillna("Desconocido")
    d["especialidad"] = _especialidad(d.medical_specialty)
    d["cobertura"] = d.payer_code.map(COBERTURA).fillna("Desconocido")
    d["diag_principal"] = d.diag_1.map(grupo_cie9)
    d["diag_circ"] = (d.diag_principal == "Circulatorio").astype(int)
    d["emergencia"] = (d.tipo_admision == "Emergencia").astype(int)

    # colas pesadas: log1p de los conteos de visitas previas
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        d[f"log_{c}"] = np.log1p(d[c])

    # indicadores de faltante (para modelar MAR)
    d["falta_raza"] = d.race.isna().astype(int)
    d["falta_especialidad"] = d.medical_specialty.isna().astype(int)
    d["falta_cobertura"] = d.payer_code.isna().astype(int)
    return d


def load_dataset(verbose: bool = False) -> pd.DataFrame:
    """Pipeline completo: raw -> limpieza -> features."""
    raw = load_raw()
    df, _ = clean(raw, verbose=verbose)
    return build_features(df)
