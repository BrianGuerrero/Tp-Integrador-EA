"""App de Decision Intelligence — TP Integrador EAyMR 2026 (UCA).

Tres módulos: (1) Diagnóstico y exploración con PCA, (2) Predictivo "What-Is",
(3) Simulación causal de políticas "What-If".

Ejecutar desde la raíz del repo:  streamlit run app/app.py
Requiere los artefactos de app/artifacts/ (los genera la sección 14 del notebook).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR / "artifacts"
sys.path.insert(0, str(APP_DIR.parent / "src"))

st.set_page_config(page_title="Decision Intelligence — HbA1c y reinternación", page_icon="🩺", layout="wide")


# =============================================================================
# Carga de artefactos
# =============================================================================


@st.cache_resource
def cargar_modelos():
    return joblib.load(ART / "modelo_predictivo.joblib"), joblib.load(ART / "pca_pipeline.joblib")


@st.cache_data
def cargar_datos():
    leer = lambda n: json.loads((ART / n).read_text(encoding="utf-8"))  # noqa: E731
    muestra = pd.read_csv(ART / "muestra_exploracion.csv.gz")
    return (muestra, leer("meta.json"), leer("cate_lineal.json"), leer("ate.json"), leer("refutaciones.json"),
            leer("dag.json"))


try:
    modelo, pca_pipe = cargar_modelos()
    muestra, meta, cate_m, ate_j, refs, dag_j = cargar_datos()
except FileNotFoundError:
    st.error("No se encontraron los artefactos en `app/artifacts/`. Ejecutá el notebook completo "
             "(sección 14 exporta los artefactos).")
    st.stop()

PRED_NUM, PRED_BIN, PRED_CAT = meta["pred_num"], meta["pred_bin"], meta["pred_cat"]
FEATURES = PRED_NUM + PRED_BIN + PRED_CAT
REF = meta["referencia"]
TASA_BASE = meta["tasa_base"]

ETIQ = {
    "edad": "Edad (años)", "time_in_hospital": "Días de internación",
    "num_lab_procedures": "Nº de análisis de laboratorio",
    "num_procedures": "Nº de procedimientos", "num_medications": "Nº de medicamentos",
    "number_diagnoses": "Nº de diagnósticos", "log_number_outpatient": "Consultas ambulatorias (último año)",
    "log_number_emergency": "Visitas a guardia (último año)", "log_number_inpatient": "Internaciones (último año)",
    "hba1c_medida": "Se midió HbA1c", "cambio_med": "Cambio de medicación diabética",
    "diabetesMed": "Recibe medicación diabética", "insulina": "Usa insulina", "sexo": "Sexo", "raza": "Raza",
    "tipo_admision": "Tipo de admisión", "origen_admision": "Origen de la admisión", "especialidad": "Especialidad",
    "cobertura": "Cobertura", "diag_principal": "Diagnóstico principal",
}
LOG_VARS = {"log_number_outpatient", "log_number_emergency", "log_number_inpatient"}


def etiqueta(c):
    return ETIQ.get(c, c)


# =============================================================================
# Utilidades
# =============================================================================


def prob_con_intervalo(x: pd.DataFrame) -> tuple[float, float, float]:
    """Probabilidad del Random Forest + banda P10–P90 de la distribución de los árboles."""
    p = float(modelo.predict_proba(x)[0, 1])
    xt = modelo.named_steps["pre"].transform(x)
    arboles = np.array([t.predict_proba(xt)[0, 1] for t in modelo.named_steps["model"].estimators_])
    return p, float(np.quantile(arboles, 0.10)), float(np.quantile(arboles, 0.90))


@st.cache_data
def probs_muestra() -> np.ndarray:
    return modelo.predict_proba(muestra[FEATURES])[:, 1]


def explicacion_local(x: pd.DataFrame) -> pd.Series:
    """Oclusión: cuánto cambia la probabilidad si cada variable vuelve a su valor de referencia (mediana/moda)."""
    p0 = float(modelo.predict_proba(x)[0, 1])
    filas = []
    for c in FEATURES:
        x2 = x.copy()
        x2[c] = REF[c]
        filas.append(p0 - float(modelo.predict_proba(x2)[0, 1]))
    return pd.Series(filas, index=[etiqueta(c) for c in FEATURES])


def cate_perfil(diag_circ: int, emergencia: int, edad: float, internaciones_previas: float):
    """τ(x) = θ'[1, x] con IC 95% a partir de la covarianza HC3 exportada."""
    xt = np.array([1.0, diag_circ, emergencia, edad - 65, np.log1p(internaciones_previas)])
    coef, cov = np.array(cate_m["coef"]), np.array(cate_m["cov"])
    tau = float(xt @ coef)
    se = float(np.sqrt(xt @ cov @ xt))
    return tau, tau - 1.96 * se, tau + 1.96 * se


def cate_vector(df: pd.DataFrame) -> np.ndarray:
    Xm = np.column_stack([np.ones(len(df)), df.diag_circ, df.emergencia, df.edad - 65, df.log_number_inpatient])
    return Xm @ np.array(cate_m["coef"])


# =============================================================================
# Barra lateral
# =============================================================================

st.sidebar.title("🩺 Decision Intelligence")
st.sidebar.caption("Diabetes 130-US Hospitals (UCI) · TP Integrador EAyMR 2026")
modulo = st.sidebar.radio("Módulo", ["1 · Diagnóstico y exploración", "2 · Predictivo (What-Is)",
                                     "3 · Simulación causal (What-If)", "Acerca de"])
st.sidebar.divider()
st.sidebar.metric("Pacientes (base analítica)", f"{meta['n']:,}".replace(",", "."))
st.sidebar.metric("Tasa de reinternación < 30 días", f"{100 * TASA_BASE:.1f}%")
st.sidebar.caption(f"Artefactos generados: {meta['creado']}")

# =============================================================================
# Módulo 1 — Diagnóstico y exploración
# =============================================================================

if modulo.startswith("1"):
    st.header("1 · Diagnóstico y exploración")
    st.caption(f"Muestra aleatoria de {len(muestra):,} pacientes de la base analítica.".replace(",", "."))
    c1, c2, c3 = st.columns(3)
    f_edad = c1.multiselect("Grupo etario", sorted(muestra.edad_grupo.dropna().unique()),
                            default=sorted(muestra.edad_grupo.dropna().unique()))
    f_diag = c2.multiselect("Diagnóstico principal", meta["opciones"]["diag_principal"],
                            default=meta["opciones"]["diag_principal"])
    f_hba = c3.selectbox("HbA1c", ["Todos", "Medida", "No medida"])
    sub = muestra[muestra.edad_grupo.isin(f_edad) & muestra.diag_principal.isin(f_diag)]
    if f_hba != "Todos":
        sub = sub[sub.hba1c_medida == (1 if f_hba == "Medida" else 0)]
    if sub.empty:
        st.warning("Ningún paciente cumple los filtros.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes filtrados", f"{len(sub):,}".replace(",", "."))
    k2.metric("Reinternación < 30 d", f"{100 * sub.readmit30.mean():.1f}%",
              f"{100 * (sub.readmit30.mean() - TASA_BASE):+.1f} pp vs. total", delta_color="inverse")
    k3.metric("% con HbA1c medida", f"{100 * sub.hba1c_medida.mean():.1f}%")
    k4.metric("Días de internación (media)", f"{sub.time_in_hospital.mean():.1f}")

    t1, t2 = st.tabs(["Variables", "Espacio PCA"])
    with t1:
        var = st.selectbox("Variable", PRED_NUM + PRED_CAT, format_func=etiqueta)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        if var in PRED_CAT:
            (sub.groupby(var).readmit30.mean() * 100).sort_values().plot.barh(ax=ax, color="#4C72B0")
            ax.axvline(100 * TASA_BASE, color="red", ls="--", label="tasa global")
            ax.set(xlabel="% reinternación", ylabel="", title=f"Reinternación por {etiqueta(var).lower()}")
            ax.legend()
        else:
            vals = np.expm1(sub[var]) if var in LOG_VARS else sub[var]
            for yv, col, lab in [(0, "#4C72B0", "no reinterna"), (1, "#DD8452", "reinterna")]:
                ax.hist(vals[sub.readmit30 == yv], bins=30, alpha=.5, density=True, color=col, label=lab)
            ax.set(xlabel=etiqueta(var), ylabel="densidad", title=f"Distribución de {etiqueta(var).lower()}")
            ax.legend()
        st.pyplot(fig)
        st.dataframe(sub[[var, "readmit30"]].describe().T if var not in PRED_CAT else
                     sub.groupby(var).agg(n=("readmit30", "size"), tasa=("readmit30", "mean")))
    with t2:
        ve = meta["pca_var_explicada"]
        st.markdown(f"**PC1** ≈ intensidad de la internación ({100 * ve[0]:.1f}% de varianza) · "
                    f"**PC2** ≈ uso previo del sistema de salud ({100 * ve[1]:.1f}%). "
                    "El escalado y la rotación se estimaron sólo con datos de entrenamiento.")
        color_por = st.selectbox("Colorear por", ["readmit30", "hba1c_medida", "diag_circ", "emergencia"])
        st.markdown("**Proyectar un caso nuevo** (resto de las variables en su mediana):")
        cc = st.columns(4)
        caso = {c: REF[c] for c in meta["pca_vars"]}
        caso["time_in_hospital"] = cc[0].slider("Días internación", 1, 14, 4)
        caso["num_medications"] = cc[1].slider("Medicamentos", 1, 80, 15)
        caso["log_number_inpatient"] = np.log1p(cc[2].slider("Internaciones previas", 0, 20, 0))
        caso["log_number_emergency"] = np.log1p(cc[3].slider("Visitas a guardia", 0, 20, 0))
        pc_caso = pca_pipe.transform(pd.DataFrame([caso])[meta["pca_vars"]])[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(sub.PC1, sub.PC2, c=sub[color_por], cmap="coolwarm", s=6, alpha=.5)
        ax.scatter(pc_caso[0], pc_caso[1], marker="*", s=400, c="gold", edgecolor="k", label="caso nuevo")
        ax.set(xlabel="PC1 — intensidad de la internación", ylabel="PC2 — uso previo del sistema",
               title=f"Pacientes en el plano de las 2 primeras componentes (color: {color_por})")
        ax.legend()
        plt.colorbar(sc)
        st.pyplot(fig)

# =============================================================================
# Módulo 2 — Predictivo
# =============================================================================

elif modulo.startswith("2"):
    st.header("2 · Predictivo (What-Is): riesgo de reinternación de un paciente")
    pm = meta["pred_metrics"]
    st.caption(f"Modelo: Random Forest (ganador del torneo). En Test: AUC = {pm['AUC-ROC']:.3f} "
               f"(IC95% {pm['AUC IC95% inf']:.3f}–{pm['AUC IC95% sup']:.3f}), Brier = {pm['Brier']:.4f}. "
               "Útil para **ordenar** pacientes por riesgo, no para clasificar sí/no.")
    with st.form("paciente"):
        st.subheader("Datos del paciente")
        a, b, c = st.columns(3)
        x = {}
        x["edad"] = a.slider("Edad", 5, 95, 65, step=10)
        x["time_in_hospital"] = a.slider("Días de internación", 1, 14, 4)
        x["num_lab_procedures"] = a.slider("Nº de análisis", 1, 130, 44)
        x["num_procedures"] = a.slider("Nº de procedimientos", 0, 6, 1)
        x["num_medications"] = b.slider("Nº de medicamentos", 1, 80, 15)
        x["number_diagnoses"] = b.slider("Nº de diagnósticos", 1, 16, 8)
        x["log_number_inpatient"] = np.log1p(b.number_input("Internaciones último año", 0, 25, 0))
        x["log_number_emergency"] = np.log1p(b.number_input("Visitas a guardia último año", 0, 50, 0))
        x["log_number_outpatient"] = np.log1p(b.number_input("Consultas ambulatorias último año", 0, 50, 0))
        x["hba1c_medida"] = int(c.checkbox("Se midió HbA1c"))
        x["cambio_med"] = int(c.checkbox("Cambio de medicación diabética"))
        x["diabetesMed"] = int(c.checkbox("Recibe medicación diabética", value=True))
        x["insulina"] = int(c.checkbox("Usa insulina", value=True))
        for col_ui, v in zip([c, c, a, a, b, b, c], PRED_CAT):
            opts = meta["opciones"][v]
            x[v] = col_ui.selectbox(etiqueta(v), opts, index=opts.index(REF[v]) if REF[v] in opts else 0)
        enviado = st.form_submit_button("Calcular riesgo", type="primary")

    xdf = pd.DataFrame([x])[FEATURES]
    p, lo, hi = prob_con_intervalo(xdf)
    pct = float((probs_muestra() < p).mean())
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de reinternación < 30 días", f"{100 * p:.1f}%",
              f"{100 * (p - TASA_BASE):+.1f} pp vs. tasa base", delta_color="inverse")
    m2.metric("Banda de incertidumbre (P10–P90 de los árboles)", f"{100 * lo:.1f}% – {100 * hi:.1f}%")
    m3.metric("Percentil de riesgo en la población", f"{100 * pct:.0f}")
    decil = min(int(pct * 10) + 1, 10)
    if decil >= 9:
        st.error(f"Decil de riesgo {decil}/10 → priorizar seguimiento post-alta (llamado a 48–72 h, turno precoz).")
    elif decil >= 6:
        st.warning(f"Decil de riesgo {decil}/10 → riesgo intermedio.")
    else:
        st.success(f"Decil de riesgo {decil}/10 → riesgo bajo.")

    st.subheader("¿Qué empuja la predicción para este paciente?")
    exp = explicacion_local(xdf).sort_values(key=np.abs, ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(exp.index, 100 * exp.values, color=np.where(exp.values > 0, "#DD8452", "#4C72B0"))
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel="aporte a la probabilidad (pp) respecto de un paciente de referencia",
           title="Explicación local por oclusión (naranja = aumenta el riesgo)")
    st.pyplot(fig)
    st.caption("Cada barra = cuánto cambia la probabilidad si esa variable tomara el valor de un paciente típico "
               "(mediana o moda). Es una explicación **predictiva**, no causal: ver módulo 3.")

# =============================================================================
# Módulo 3 — Simulación causal
# =============================================================================

elif modulo.startswith("3"):
    st.header("3 · Simulación causal de políticas (What-If): do(medir HbA1c = 1)")
    st.caption(f"Estimador del CATE: {meta['fuente_causal']}. Supone que no hay confusión no observada "
               "dado el conjunto de ajuste del DAG (criterio backdoor).")

    # --- Semáforo de refutaciones ---------------------------------------------
    st.subheader("Semáforo de confianza causal (refutaciones DoWhy)")
    cols = st.columns(len(refs))
    for col, r in zip(cols, refs):
        icono = "🟢" if r["aprobado"] else "🔴"
        col.markdown(f"### {icono} {r['nombre']}")
        col.metric("Nuevo efecto", f"{r['nuevo_efecto_pp']:+.2f} pp", f"original {r['efecto_original_pp']:+.2f} pp",
                   delta_color="off")
        col.caption(r["criterio"])
    n_ok = sum(r["aprobado"] for r in refs)
    msg = (f"{n_ok}/{len(refs)} refutaciones aprobadas. Atención: las refutaciones no descartan confusión no "
           "observada (E-value ≈ 1,2: un confusor oculto débil podría explicar el efecto promedio).")
    (st.success if n_ok == len(refs) else st.warning)(msg)

    t1, t2, t3 = st.tabs(["Efecto para un perfil (CATE)", "Simulador de política", "DAG y criterio backdoor"])
    with t1:
        a, b = st.columns([1, 2])
        circ = a.checkbox("Diagnóstico principal circulatorio")
        emerg = a.checkbox("Admisión de emergencia", value=True)
        edad = a.slider("Edad ", 25, 95, 65, step=10)
        previas = a.slider("Internaciones en el último año ", 0, 10, 1)
        delta = a.slider("Reducción mínima que justifica el protocolo (pp)", 0.0, 5.0, 1.0, 0.5) / 100
        tau, lo, hi = cate_perfil(int(circ), int(emerg), edad, previas)
        b.metric("Efecto estimado de medir HbA1c sobre P(reinternación)", f"{100 * tau:+.2f} pp",
                 f"IC95% [{100 * lo:+.2f}; {100 * hi:+.2f}] pp", delta_color="off")
        if hi < -delta:
            b.success("✅ PRIORIZAR: el beneficio supera el umbral en todo el intervalo de confianza.")
        elif tau < -delta:
            b.warning("🟡 PRIORIZAR CON CAUTELA: beneficio esperado mayor al umbral, pero con incertidumbre.")
        elif lo > 0:
            b.error("🔴 Efecto estimado desfavorable para este perfil.")
        else:
            b.info("ℹ️ PROTOCOLO ESTÁNDAR: no se estima un beneficio mayor al umbral.")
        xs = np.arange(0, 11)
        curva = np.array([cate_perfil(int(circ), int(emerg), edad, k) for k in xs])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(xs, 100 * curva[:, 0], "o-", color="#4C72B0")
        ax.fill_between(xs, 100 * curva[:, 1], 100 * curva[:, 2], alpha=.2)
        ax.axhline(0, color="k", lw=.8)
        ax.axhline(-100 * delta, color="green", ls="--", label="umbral δ")
        ax.scatter([previas], [100 * tau], s=150, c="gold", edgecolor="k", zorder=3, label="perfil elegido")
        ax.set(xlabel="internaciones en el último año", ylabel="CATE (pp)",
               title="Curva de heterogeneidad del efecto para este perfil")
        ax.legend()
        b.pyplot(fig)

        st.subheader("Efecto promedio (ATE): comparación de estimadores")
        tab = pd.DataFrame(ate_j["tabla"]).set_index("Estimador")
        st.dataframe(tab.round(2))

    with t2:
        st.markdown("Elegí a qué subgrupo aplicar la política **do(medir HbA1c = 1)**. Las cifras se escalan "
                    "de la muestra a la población analítica.")
        a, b, c = st.columns(3)
        min_prev = a.slider("Mínimo de internaciones previas", 0, 5, 1)
        excl_circ = a.checkbox("Excluir diagnóstico circulatorio", value=True)
        edad_min, edad_max = b.slider("Rango de edad", 5, 95, (5, 95), step=10)
        costo_prot = c.number_input("Costo del protocolo por paciente (USD)", 0, 5000, 100, step=50)
        costo_reint = c.number_input("Costo de una reinternación (USD)", 0, 100000, 15000, step=1000)
        tau_m = cate_vector(muestra)
        prev = np.expm1(muestra.log_number_inpatient).round()
        sel = (prev >= min_prev) & muestra.edad.between(edad_min, edad_max)
        if excl_circ:
            sel &= muestra.diag_circ == 0
        escala = meta["n"] / len(muestra)
        n_pol = sel.sum() * escala
        evit = -tau_m[sel].sum() * escala
        evit_todos = -tau_m.sum() * escala
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pacientes con protocolo", f"{n_pol:,.0f}".replace(",", "."), f"{100 * sel.mean():.0f}% del total",
                  delta_color="off")
        k2.metric("Reinternaciones evitadas (estimadas)", f"{evit:,.0f}".replace(",", "."))
        k3.metric("Evitadas cada 100 protocolos", f"{100 * evit / max(n_pol, 1):.2f}",
                  f"vs. {100 * evit_todos / meta['n']:.2f} si se mide a todos", delta_color="off")
        k4.metric("Retorno neto estimado (USD)", f"{evit * costo_reint - n_pol * costo_prot:,.0f}".replace(",", "."))
        fig, ax = plt.subplots(figsize=(8, 3.3))
        ax.hist(100 * tau_m[~sel], bins=40, alpha=.5, label="sin protocolo", color="#BAB0AC")
        ax.hist(100 * tau_m[sel], bins=40, alpha=.7, label="con protocolo", color="#59A14F")
        ax.axvline(0, color="k", lw=.8)
        ax.set(xlabel="CATE estimado (pp)", ylabel="pacientes (muestra)", title="Distribución del efecto estimado")
        ax.legend()
        st.pyplot(fig)
        st.caption("⚠️ Estimaciones observacionales: antes de implementar, validar con un ensayo aleatorizado "
                   "por conglomerados en el subgrupo elegido.")

    with t3:
        from tpi.causal import backdoor_check

        G = nx.DiGraph([tuple(e) for e in dag_j["aristas"]])
        T, Y = "hba1c_medida", "readmit30"
        candidatas = sorted(n for n in G.nodes if n not in (T, Y, "gravedad_no_observada"))
        Zdef = [n for n in candidatas if n not in ("cambio_med", "num_lab_procedures", "time_in_hospital")]
        ajuste = st.multiselect("Conjunto de ajuste Z (variables por las que se controla)", candidatas, default=Zdef)
        chk = backdoor_check(G, set(ajuste))
        if chk["valido"]:
            st.success("✅ El conjunto cumple el criterio backdoor: el efecto es identificable.")
        else:
            if not chk["sin_descendientes_de_T"]:
                st.error(f"❌ Incluye descendientes de T: {', '.join(chk['descendientes_incluidos'])} "
                         "(mediador / colisionador / post-tratamiento) → sesgo.")
            if not chk["bloquea_backdoor"]:
                st.error("❌ Quedan caminos backdoor abiertos (faltan confusores) → sesgo de confusión.")
        pos = nx.spring_layout(G, seed=3, k=1.2)
        pos.update({T: np.array([-1.2, 0]), Y: np.array([1.2, 0])})
        colores = ["#F28E2B" if n == T else "#59A14F" if n == Y else "#BAB0AC" if n == "gravedad_no_observada"
                   else "#E15759" if n in ("cambio_med", "num_lab_procedures", "time_in_hospital")
                   else "#4C72B0" if n in ajuste else "#A0CBE8" for n in G.nodes]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        nx.draw_networkx(G, pos, ax=ax, node_color=colores, node_size=1300, font_size=7, arrowsize=10,
                         edge_color="#888")
        ax.set_title("DAG causal (azul oscuro = ajustadas; rojo = descendientes de T; gris = latente)")
        ax.axis("off")
        st.pyplot(fig)

else:
    st.header("Acerca de")
    st.markdown("""
**Pregunta de negocio:** ¿conviene a un hospital medir HbA1c a los pacientes diabéticos internados para reducir
las reinternaciones a 30 días? ¿A quiénes?

* **Datos:** Diabetes 130-US Hospitals 1999–2008 (UCI, id 296), primera internación por paciente, sin fallecidos.
* **Predicción:** Random Forest con búsqueda de hiperparámetros y validación cruzada estratificada.
* **Causalidad:** DAG + criterio backdoor, DoWhy (regresión e IPW), Double ML (EconML) y tres refutaciones.

Herramienta académica de apoyo a la decisión: **no** es para uso clínico.
""")
    st.json({k: meta[k] for k in ("n", "tasa_base", "fuente_causal", "creado")})
