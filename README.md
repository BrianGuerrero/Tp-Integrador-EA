# 🩺 Decision Intelligence: medición de HbA1c y reinternación a 30 días

![Python](https://img.shields.io/badge/python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange)
![DoWhy](https://img.shields.io/badge/causal-DoWhy%20%7C%20EconML-purple)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)
![Quality Gate](https://img.shields.io/badge/QA-pytest%20%2B%20ruff-brightgreen)

**TP Integrador — Estadística Aplicada y Modelos de Regresión (EAyMR 2026), UCA.**

🔗 **App desplegada:** https://tp-integrador-ea.streamlit.app/
🔗 **Repositorio:** https://github.com/BrianGuerrero/Tp-Integrador-EA

## Equipo y roles Scrum

| Rol | Integrante | Responsable de |
|---|---|---|
| 👑 Project Lead & Scrum Master | Brian Guerrero | arquitectura, issues/milestones, merge a `main` sólo con `[APROBADO]` de QA |
| ⚙️ Backend & ML / Causal Engineer | Matias Fox | `src/tpi/` (stats, models, causal), notebook, serialización |
| 🎨 Frontend & UI/UX Engineer | Brian Guerrero | `app/app.py`, visualizaciones, deploy |
| 🔍 QA & Validation Engineer | Joseba Esain | `tests/`, `ruff`, CI (`.github/workflows`), veredicto de PRs |

## Problema y preguntas

Un hospital quiere reducir las reinternaciones a 30 días de pacientes diabéticos. Dos preguntas distintas:

1. **Predicción** $P(Y \mid X)$: ¿quién se va a reinternar? → priorizar seguimiento post-alta.
2. **Intervención** $P(Y \mid do(T))$: si se obliga a medir HbA1c (marcador de atención a la diabetes), ¿cuántas reinternaciones se evitan, y en quiénes?

```
Z (contexto al ingreso) ──► T (HbA1c medida) ──► M (cambio de medicación) ──► Y (reinternación <30d)
        └───────────────────────────────────────────────────────────────────►┘
T ──► C (nº de análisis) ◄── U (gravedad, no observada) ──► Y        T ──► días de internación
```

## Resultados principales

| | Resultado |
|---|---|
| Datos | 101.766 internaciones → **69.970 pacientes** (1ª internación, sin fallecidos/hospicio); centinelas ocultos en ~10% de los tipos y ~7% de los orígenes de admisión |
| Fase 1 | Asociación cruda HbA1c–reinternación: −0,71 pp (p = 0,011) pero *h* = 0,025; MDE del diseño ≈ 0,8 pp. ANOVA de Welch + Games-Howell sobre polifarmacia por edad (η² = 1,6%) |
| Fase 2 | MCO de días de internación (R² ≈ 0,31); heterocedasticidad (BP) y no normalidad (JB) corregidas con log + HC3; sin puntos influyentes (Cook máx ≈ 0,001); PCA: PC1 = intensidad, PC2 = uso previo |
| Fase 3 | Random Forest ≈ Logística (AUC test ≈ 0,62); NB mal calibrado (corregido con isotónica); CART podado: regla raíz = internaciones previas; MDI vs permutación con experimento de ruido |
| Fase 4 | Efecto causal promedio ≈ **−0,2 pp, no significativo** (regresión, IPW, DML coinciden); el beneficio aparente era autoselección. Ajustar por días/mediador/colisionador fabrica efectos. CATE: beneficio en pacientes con internaciones previas no circulatorios |

## Estructura

```
├── README.md
├── BITACORA.md                      ← cuaderno de decisiones (5 entradas)
├── requirements.txt                 ← dependencias fijadas
├── pyproject.toml                   ← configuración de ruff y pytest
├── data/
│   ├── diabetic_data.csv            ← dataset original UCI (id 296)
│   └── download_data.py             ← descarga alternativa con ucimlrepo
├── src/tpi/
│   ├── config.py                    ← semilla, α, rutas, roles causales (contrato de datos)
│   ├── data.py                      ← carga, auditoría de centinelas, limpieza, features
│   ├── stats.py                     ← tests, ANOVA de Welch, Games-Howell, MCO, HC3, BP, Cook, VIF
│   ├── models.py                    ← pipelines sin fuga, odds ratios, métricas, MDI agregado
│   ├── causal.py                    ← DAG, criterio backdoor, IPW, DML, refutaciones, E-value
│   └── artifacts.py                 ← exportación notebook → app
├── notebooks/
│   ├── TP_Integrador_Final.ipynb    ← NOTEBOOK MAESTRO
│   └── TP_Integrador_Final.py       ← misma fuente en formato "percent" (diff legible)
├── app/
│   ├── app.py                       ← Streamlit: 3 módulos
│   ├── requirements.txt             ← dependencias del deploy
│   └── artifacts/                   ← modelos y resultados serializados
├── tests/                           ← pytest (estadística, fuga de datos, causal, contrato de artefactos)
├── scripts/build_notebook.py        ← .py → .ipynb (opcionalmente ejecutando)
└── .github/                         ← CI del Quality Gate + plantilla de PR con veredicto QA
```

## Reproducción paso a paso

```bash
git clone https://github.com/BrianGuerrero/Tp-Integrador-EA.git && cd Tp-Integrador-EA
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Quality Gate
ruff check src tests app scripts
pytest -q

# 2) Notebook maestro (≈ 20–30 min; el Random Forest y DML son lo más pesado)
jupyter notebook notebooks/TP_Integrador_Final.ipynb    # Kernel → Restart & Run All
#    la sección 14 regenera app/artifacts/

# 3) App
streamlit run app/app.py
```

**Informe editorial (PDF/HTML) con Quarto** — el notebook trae el encabezado YAML:

```bash
quarto render notebooks/TP_Integrador_Final.ipynb --to pdf    # o --to html
```

**Google Colab:** subir el repo (o `!git clone`), `%cd` a la carpeta, `!pip install -r requirements.txt`, y ejecutar el notebook desde `notebooks/`.

## Deploy en Streamlit Community Cloud

1. Subir el repo a GitHub (incluida la carpeta `app/artifacts/`).
2. En https://share.streamlit.io → *New app* → elegir el repo, rama `main` y archivo `app/app.py`.
3. *Advanced settings* → Python 3.13. Streamlit usa `app/requirements.txt`.
4. Pegar la URL pública arriba en este README.

## La app

| Módulo | Qué hace |
|---|---|
| 1 · Diagnóstico y exploración | filtros por edad/diagnóstico/HbA1c, KPIs, distribuciones, plano PCA y proyección de un caso nuevo |
| 2 · Predictivo (What-Is) | probabilidad de reinternación del Random Forest, banda de incertidumbre (P10–P90 de los árboles), percentil de riesgo y explicación local por oclusión |
| 3 · Simulación causal (What-If) | semáforo de las 3 refutaciones, CATE con IC para un perfil, curva de heterogeneidad, simulador de política do(T=1) con costos, DAG interactivo con verificación del criterio backdoor |

## Uso de IA generativa

Ver **Entrada 5** de [`BITACORA.md`](BITACORA.md).

## Referencia del dataset

Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 781670. Datos: UCI Machine Learning Repository, id 296 (CC BY 4.0).
