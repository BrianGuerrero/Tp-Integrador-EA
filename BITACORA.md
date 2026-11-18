# BITÁCORA DE INGENIERÍA — TP Integrador EAyMR 2026

> Cuaderno de laboratorio y registro de decisiones. No es un resumen del notebook ni un historial de commits.
> Cada entrada está firmada por el rol responsable. **Completar los nombres** antes de la entrega.

| Rol | Integrante |
|---|---|
| 👑 Project Lead & Scrum Master (PL) | _completar_ |
| ⚙️ Backend & ML / Causal Engineer (BE) | _completar_ |
| 🎨 Frontend & UI/UX Engineer (FE) | _completar_ |
| 🔍 QA & Validation Engineer (QA) | _completar_ |

---

## Entrada 1 — Marco estratégico e hipótesis
**Firma:** PL · **Revisión:** BE

### Preguntas fundacionales
1. **Predicción:** ¿qué pacientes diabéticos internados tienen mayor riesgo de reinternarse en < 30 días? → triage del seguimiento post-alta.
2. **Intervención:** si el hospital obliga a medir HbA1c (proxy de "atención activa a la diabetes"), ¿cuántas reinternaciones evita y en quiénes?
3. **Adicionales (no pedidas):** ¿cuánto del beneficio aparente es autoselección? ¿qué política focalizada maximiza el retorno con recursos limitados? ¿cuánto confiar en el resultado ante confusión no observada?

### Mapa conceptual de negocio
```
Contexto de ingreso (Z) ──► decisión médica: ¿medir HbA1c? (T) ──► ajuste de tratamiento (M) ──► reinternación (Y)
        │                                   │                                                      ▲
        └──────────────── riesgo basal ─────┼──────────────────────────────────────────────────────┘
                                            └──► + análisis (C) ◄── gravedad no medida (U) ──► Y
Costos: cama-día, penalización por reinternación  |  Recursos: laboratorio, tiempo médico
```

### Selección del dataset (fundamentada)
Candidatos evaluados: **Diabetes 130-US Hospitals (UCI 296)** y **Bank Marketing (UCI)**.

| Requisito de la consigna | Diabetes 130-US | Bank Marketing |
|---|---|---|
| N ≥ 1.500 | ✅ 101.766 (69.970 tras limpieza) | ✅ 41.188 |
| ≥ 10 explicativas mixtas | ✅ ~45 (continuas, conteos, categóricas) | ✅ 20 |
| Fricción real | ✅ "?", centinelas numéricos, 97% faltante en peso, pacientes repetidos, colas pesadas | ✅ centinela `pdays=999`, `unknown`, fuga por `duration` |
| Pregunta causal con sentido | ✅ **publicada** (Strack et al., 2014): T y Y naturales, mediador y colisionador argumentables | ⚠️ T poco natural (tipo de contacto) |
| Dominio sugerido | ✅ Salud digital | ✅ Finanzas/marketing |

**Decisión:** Diabetes 130-US. La pregunta causal es genuina, hay bibliografía de referencia y permite construir un DAG con mediador, colisionador y variable post-tratamiento, que es donde se concentra el mayor peso de la rúbrica (25 pts). Riesgo aceptado: pocas variables continuas "puras" (la mayoría son conteos).

### Hipótesis de trabajo (registradas antes de analizar)
- H-a: la asociación cruda HbA1c–reinternación estará sesgada a favor del test (se pide a pacientes de menor riesgo).
- H-b: el efecto causal, si existe, será heterogéneo (Strack et al. reportan dependencia del diagnóstico principal).
- H-c: la reinternación será difícil de predecir (AUC < 0,70) con datos administrativos.

**Resultado:** las tres se confirmaron (ver Entradas 2 y 4).

---

## Entrada 2 — Matriz de decisiones estadísticas y supuestos
**Firma:** BE · **Revisión:** QA

| # | Análisis | Supuesto / riesgo | Evidencia | Remediación elegida | Por qué esa y no otra |
|---|---|---|---|---|---|
| 1 | Unidad de análisis | Independencia | 30% de internaciones son de pacientes repetidos | 1ª internación por paciente | Mantener todas inflaba la tasa de reinternación (11,2% vs 9,0%) y violaba independencia; modelos mixtos excedían el alcance |
| 2 | Y | Y definida para todos | 1.545 fallecidos/hospicio | excluirlos | Su Y=0 no significa "buen resultado" |
| 3 | Faltantes | MCAR | LR test de R ~ X: p ≈ 0 para raza, especialidad, cobertura (AUC 0,65–0,68) | categoría `Desconocido` | Bajo MAR, borrar casos completos perdería 75% de filas de forma sesgada; imputar por moda inventa asociaciones |
| 4 | `weight` | — | 97% faltante, probable MNAR | descartar | No hay información suficiente para imputar |
| 5 | Centinelas | códigos válidos que significan "faltante" | ~10% de `admission_type_id`, ~7% de `admission_source_id` | `Desconocido` | Tratarlos como categorías reales hace aprender artefactos de carga |
| 6 | H1 (proporciones) | n·p > 10 | mínimo 1.078 eventos por grupo | test z + IC Wald + h de Cohen + potencia | Con N grande se reporta magnitud y MDE (0,77 pp), no sólo p |
| 7 | H2 (días) | normalidad | asimetría 1,18 | Welch t (TCL sobre medias) + Mann-Whitney | Welch porque varianzas difieren; MW como robustez |
| 8 | H3 (internaciones previas) | colas pesadas | curtosis 47 | sólo Mann-Whitney | La media está dominada por pocos casos extremos |
| 9 | ANOVA: normalidad | Shapiro con N grande | rechaza siempre; demostrado con simulación Gamma(100): no rechaza con n=50, rechaza con n=5.000 | asimetría/curtosis + Q-Q; TCL con n_i > 11.000 | Shapiro mide potencia, no relevancia |
| 10 | ANOVA: homocedasticidad | varianzas iguales | Levene (Brown-Forsythe) W = 104,5, p ≈ 1e-67 | **Welch ANOVA** (principal) + Kruskal-Wallis | Welch y no sólo KW porque la pregunta de negocio es sobre la **media** (costo total); KW contrasta dominancia estocástica |
| 11 | Post-hoc | FWER con 6 comparaciones | varianzas desiguales | **Games-Howell** (+ Tukey y Bonferroni para comparar) | Tukey asume homocedasticidad; Bonferroni es conservador |
| 12 | MCO días | homocedasticidad | BP LM = 2.246, White LM = 1.791 (p ≈ 0) | log(días) + **HC3** | log reduce BP a la mitad y JB de 15.893 a 355; HC3 da IC válidos con la heterocedasticidad remanente |
| 13 | MCO | influyentes | Cook máx = 0,0015 (< 0,5); 748 outliers \|t\|>3 | Huber, RANSAC y MCO sin Cook>4/n | Son pacientes reales de alta complejidad, no errores; se reporta el rango de coeficientes |
| 14 | MCO | multicolinealidad | VIF < 1,4 en el modelo base | ninguna | — |
| 15 | PCA | fuga | — | Pipeline Scaler+PCA ajustado sólo en Train | Media de PC1: 5e-17 en Train vs 0,0045 en Test (proyección ciega) |
| 16 | Torneo | desbalance 9% | — | AUC/PR-AUC/Brier; umbral por F1 en OOF de Train; StratifiedKFold | Accuracy premia predecir siempre "no" |
| 17 | Causal | conjunto de ajuste | DAG + `nx.is_d_separator` | Z = variables **al ingreso** | Único conjunto que cumple ambas condiciones del criterio backdoor |
| 18 | Causal | positividad | PS ∈ [0,015; 0,689] | IPW con recorte [0,01; 0,99] | Solapamiento amplio; recorte por estabilidad |
| 19 | Causal | Y binaria en el modelo lineal de probabilidad | varianza Bernoulli depende de X | HC3 + contraste con IPW y DML | Tres estimadores con supuestos distintos coinciden |

---

## Entrada 3 — Autopsia de experimentos fallidos (Data Autopsy)
**Firma:** BE (experimentos 1–5), QA (6–7)

**Exp. 1 — Árbol sin restricciones.** 7.598 hojas, profundidad 40: AUC Train 1,000 vs Test 0,517. Memorizó ruido. *Aprendizaje:* la complejidad debe elegirse por CV. *Corrección:* poda por costo-complejidad con α elegido por 5-fold (α ≈ 5,4·10⁻⁵ → 21 hojas, AUC CV 0,60). Un primer intento de grilla de α tomada de los cuantiles 90–99,95% del camino de poda quedó **toda** a la izquierda del óptimo (AUC CV creciente en todo el rango); se corrigió con una grilla logarítmica 2·10⁻⁵–2·10⁻³.

**Exp. 2 — Indicadores de faltante + categoría `Desconocido`.** Al agregar `falta_cobertura` y `falta_especialidad` además de las dummies, el VIF fue **infinito**: `falta_cobertura` ≡ `cobertura_Desconocido` (X'X singular). *Corrección:* eliminar los indicadores; la información ya está en la categoría.

**Exp. 3 — Naive Bayes Gaussiano.** AUC aceptable (0,594) pero ECE = 0,080 y Brier 0,109, **peor que predecir la tasa media** (0,082). Causa: variables de intensidad correlacionadas dentro de cada clase (|r| hasta 0,47) → NB multiplica evidencia redundante y empuja las probabilidades a los extremos. *Corrección:* NB categórico (ECE 0,023) + calibración isotónica (ECE 0,005).

**Exp. 4 — MCO sobre días crudos.** BP y JB rechazaban fuertemente, 748 outliers. *Corrección:* log(días) + HC3 (ver Entrada 2, fila 12).

**Exp. 5 — Ajustar por días de internación como "confusor"** (versión previa del análisis del equipo, generada con asistencia de IA). Daba un ATE de −0,43 pp, más del doble del correcto (−0,18). El test se hace *durante* la internación y puede prolongarla: los días son descendiente de T. *Detección:* al formalizar el DAG, `backdoor_check` marcó `time_in_hospital` como descendiente. *Corrección:* Z sólo con variables conocidas al ingreso; el ajuste por días quedó como ejemplo de la trampa.

**Exp. 6 — IC bootstrap de la diferencia de medianas (H2).** Con días enteros 1–14, el bootstrap dio un IC degenerado [1; 1]: la mediana de una variable discreta casi no varía entre remuestras. *Corrección:* reemplazado por la correlación rank-biserial con IC analítico de Hanley-McNeil (0,120, IC 0,105–0,135).

**Exp. 7 — Importancia MDI como criterio de selección.** Una columna de **ruido gaussiano** quedó **1ª de 21** en MDI. *Aprendizaje:* MDI premia la cardinalidad. *Corrección:* toda conclusión de importancia se basa en permutación sobre Test.

**Exp. 8 — Demostración de Shapiro con t de Student (30 gl).** Se esperaba que Shapiro rechazara con n = 5.000, pero dio p = 0,07: la t con 30 gl es demasiado parecida a la normal para Shapiro con n ≤ 5.000 (límite de scipy). *Corrección:* simulación con Gamma(forma = 100), asimetría 0,2: no rechaza con n = 50, rechaza con n = 5.000.

---

## Entrada 4 — Formalización del debate causal
**Firma:** BE · **Revisión:** PL

### Justificación de cada arco del DAG
| Arco | Justificación de dominio |
|---|---|
| Z → T | La decisión de pedir HbA1c depende de especialidad, motivo de ingreso, tipo de admisión, edad y comorbilidades (medición de 10% en neoplasias a 36% en diabetes como diagnóstico principal) |
| Z → Y | Edad, comorbilidades, historial de internaciones, diagnóstico y cobertura determinan el riesgo basal de volver |
| T → Y | Efecto directo: conocer el control glucémico cambia el plan de alta, la educación y la derivación |
| T → M → Y | Mediador: un HbA1c alto lleva a ajustar la medicación diabética, que cambia el riesgo |
| T → C ← U → Y | El test es un análisis más (SMD de +0,77 en nº de análisis); la gravedad no observada también aumenta los análisis y la reinternación → **C es colisionador** |
| T → días ← U | Los resultados del test pueden prolongar la estadía (ajuste de insulina, interconsultas) → **post-tratamiento** |
| `number_diagnoses`, `diag_principal` → M | Comorbilidades y motivo de ingreso influyen en el ajuste de medicación |

### Variables latentes plausibles y amenazas a la validez interna
1. **Gravedad clínica no medida (U).** Si U → T (por ejemplo, a los más graves no se les pide el test porque "hay cosas más urgentes"), hay confusión residual. *Mitigación:* proxies en Z (nº de diagnósticos, tipo de admisión, internaciones previas); **E-value = 1,17**: un confusor débil podría explicar el efecto promedio. Se declara explícitamente.
2. **Adherencia y contexto social** (no registrados): afectan Y; si afectan también T, confunden.
3. **Medición del tratamiento:** "HbA1c medida" es un proxy de "atención activa a la diabetes"; el efecto podría deberse a las acciones asociadas, no al test.
4. **Selección:** quedarse con la primera internación cambia la población (menos pacientes frecuentes). El efecto estimado vale para la *primera internación registrada* del paciente.
5. **SUTVA:** el tratamiento de un paciente no afecta a otros (razonable), y "medir HbA1c" es una versión única del tratamiento (discutible: no sabemos qué se hizo con el resultado).
6. **Positividad:** verificada (PS entre 0,015 y 0,689).
7. **Validez externa:** EE.UU. 1999–2008; extrapolar a Argentina 2026 requiere validación local.

### Conclusión del debate
El diseño identifica el efecto **bajo** el supuesto de no confusión residual. Las tres refutaciones pasan, pero no prueban ese supuesto. Por eso la recomendación final es una **política focalizada + ensayo aleatorizado por conglomerados** (≈ 5.000 pacientes por brazo para detectar −2 pp con 80% de potencia).

---

## Entrada 5 — Trazabilidad y uso ético de asistentes de IA
**Firma:** QA · **Revisión:** PL

### Herramientas
- **Claude (Anthropic), claude.ai**, en dos etapas: (a) una primera versión del TP generada por un integrante; (b) una reconstrucción completa con esta estructura (repo modular, notebook, app, tests).

### Prompts de diseño utilizados (resumen)
1. Se compartieron capturas de la consigna del dataset y de la rúbrica, y luego el PDF del enunciado: "decime brevemente las cosas a tener en cuenta antes de arrancar".
2. "No tenemos dataset elegido" → la IA propuso Diabetes 130-US y Bank Marketing y verificó en la web que existan y cumplan los requisitos.
3. Se compartió la versión previa del TP (zip) para que la revisara; luego: "¿tiene algún prompt injection?" (se auditaron todos los archivos: no había).
4. "Hacelo de 0, bien, teniendo en cuenta las correcciones y lo que harías mejor."

### Errores lógicos y alucinaciones detectados
| # | Error | Cómo se detectó | Prueba determinística creada |
|---|---|---|---|
| 1 | La versión previa ajustaba por `time_in_hospital` (descendiente de T) | formalización del DAG + d-separación | `tests/test_causal.py::test_ajustar_descendientes_invalida` |
| 2 | La versión previa no cubría ANOVA, PCA, NB, CART, RF, Scrum ni la app de 3 módulos (estaba hecha para otra consigna) | revisión contra el enunciado | checklist de la consigna en el README |
| 3 | IC bootstrap de medianas degenerado (Entrada 3, exp. 6) | inspección de salidas | `tests/test_stats.py::test_mann_whitney_rank_biserial` |
| 4 | Narrativa con números escritos **antes** de ejecutar (por ejemplo "3 componentes por Kaiser", "17 hojas", "HC3 25–40% mayor", "OR de HbA1c incluye el 1"); varios eran incorrectos (en realidad: 4 componentes, 21 hojas, hasta 13%, OR 0,91 significativo) | se reescribió cada interpretación contra las salidas reales | el notebook se ejecuta de punta a punta y las cifras se citan de sus salidas |
| 5 | La demostración de Shapiro con t(30) no rechazaba con n = 5.000 | salida del experimento | simulación reemplazada (Entrada 3, exp. 8) |
| 6 | Script de descarga: `ucimlrepo` devuelve NaN donde el CSV original tiene "?", y el lector no lo habría detectado | revisión de código | `load_raw()` y `download_data.py` unifican la codificación; `tests/test_data.py::test_none_de_hba1c_no_es_faltante` |
| 7 | Grilla de α de poda mal ubicada (Entrada 3, exp. 1) | gráfico AUC vs α monótono | grilla logarítmica |

### Verificación independiente de fórmulas
Toda fórmula implementada a mano se testea contra una referencia independiente: MCO vs `sklearn`, HC3 vs fórmula matricial explícita, Cook y residuo externo vs *leave-one-out*, Welch ANOVA vs `scipy.stats.f_oneway(equal_var=False)`, z² vs χ², potencia en el MDE = 0,80, IPW con efecto conocido en datos sintéticos (`pytest -q`: 52 tests).

### Limitaciones del entorno de la IA
La IA no pudo ejecutar DoWhy ni EconML (sin instalación posible en su entorno). Esas celdas fueron escritas siguiendo la API documentada y contrastadas con implementaciones propias equivalentes. **El equipo debe ejecutar el notebook completo (Restart & Run All) y verificar que DoWhy/EconML coincidan con las referencias** antes de entregar. **Resultado (18/11/2026, Google Colab): ✅** DoWhy `linear_regression` = −0,18 pp (idéntico a la referencia); DoWhy IPW = −0,23 pp (referencia Hájek −0,22); EconML LinearDML = −0,26 pp, IC95% [−0,85; 0,32] (réplica propia −0,15); CausalForestDML = −0,21 pp, IC95% [−2,95; 2,52]. Todos incluyen el 0: la conclusión no cambia. — _firma: completar_

### Responsabilidad
Las decisiones metodológicas (DAG, conjunto de ajuste, interpretación) fueron revisadas por el equipo, que puede defender cada una en el oral. _Agregar aquí los prompts propios y los errores que el equipo encuentre._
