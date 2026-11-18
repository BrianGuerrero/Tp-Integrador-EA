"""Configuración global del proyecto: rutas, semilla y contratos de datos entre fases."""

from pathlib import Path

SEED = 42
ALPHA = 0.05  # nivel de significación fijado a priori para todo el TP
TEST_SIZE = 0.20  # partición 80/20 exigida por la consigna
N_FOLDS = 5

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "diabetic_data.csv"
ARTIFACTS_DIR = ROOT / "app" / "artifacts"
FIG_DIR = ROOT / "reports" / "figures"

UCI_ID = 296
UCI_URL = "https://archive.ics.uci.edu/dataset/296"

# --- Roles causales (contrato de datos de la Fase 4) -----------------------
TREATMENT = "hba1c_medida"  # T: se midió HbA1c durante la internación
OUTCOME = "readmit30"  # Y: reinternación en < 30 días
MEDIATOR = "cambio_med"  # M: cambio de medicación diabética (T -> M -> Y)
COLLIDER = "num_lab_procedures"  # C: T -> C <- gravedad (U) -> Y
POST_TREATMENT_AMBIGUOUS = "time_in_hospital"  # días: puede ser consecuencia de T

# Confusores Z medidos al ingreso (pre-tratamiento)
Z_NUM = [
    "edad",
    "number_diagnoses",
    "log_number_outpatient",
    "log_number_emergency",
    "log_number_inpatient",
]
Z_CAT = [
    "sexo",
    "raza",
    "tipo_admision",
    "origen_admision",
    "especialidad",
    "cobertura",
    "diag_principal",
]

# --- Variables predictivas (Fase 3) ----------------------------------------
PRED_NUM = [
    "edad",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "log_number_outpatient",
    "log_number_emergency",
    "log_number_inpatient",
]
PRED_BIN = ["hba1c_medida", "cambio_med", "diabetesMed", "insulina"]
PRED_CAT = Z_CAT

# --- Variables numéricas para PCA (Fase 2) ---------------------------------
PCA_VARS = PRED_NUM
