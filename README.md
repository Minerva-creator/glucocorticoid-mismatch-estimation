# Machine Learning y Modelado Híbrido para la Estimación de Desalineación Fisiológica en Insuficiencia Suprarrenal mediante wearables

Sistema experimental de soporte contextual de decisión que estima, a partir de señales
de wearables y farmacocinética simplificada, situaciones de posible desalineación
entre la demanda fisiológica y la cobertura glucocorticoide en pacientes con
insuficiencia suprarrenal.

> **Este proyecto no es un dispositivo médico, no realiza diagnóstico clínico y no
> recomienda dosis terapéuticas.** Ver la sección [Alcance y limitaciones](#alcance-y-limitaciones).

## Contexto y motivación

La insuficiencia suprarrenal obliga a sustituir de forma externa los glucocorticoides
que las glándulas suprarrenales dejan de producir. Esta sustitución debe seguir, de
forma aproximada, el ritmo circadiano natural del cortisol endógeno, y debe además
ajustarse ante situaciones de mayor demanda fisiológica (actividad física, enfermedad,
estrés psicológico), donde una persona sin insuficiencia suprarrenal aumentaría su
producción de cortisol de forma autónoma.

En la práctica clínica, este ajuste depende en gran medida del criterio del propio
paciente, sin apoyo continuo que le ayude a valorar si su pauta de medicación está
cubriendo razonablemente su demanda fisiológica en cada momento del día. Los
wearables de consumo permiten registrar de forma continua y no invasiva señales
fisiológicas relacionadas con la activación del organismo (frecuencia cardíaca,
variabilidad de la frecuencia cardíaca, actividad, sueño), que pueden emplearse como
proxies de esa demanda.

Este proyecto explora si, combinando esas señales con un modelo farmacocinético
simplificado de la cobertura glucocorticoide, es posible construir un sistema de
soporte contextual de decisión que señale posibles situaciones de desalineación entre
demanda y cobertura, como apoyo informativo adicional y en ningún caso como sustituto
del criterio clínico.

## Arquitectura del sistema

El sistema se organiza en cuatro capas conceptuales, encadenadas:

**Capa 1 — StressLoad(t).** Estima la activación fisiológica o carga de estrés a
partir de señales derivadas de wearables (frecuencia cardíaca y su variabilidad,
principalmente). Se entrena y valida con el dataset público WESAD (ver
[Datos](#datos)) mediante un modelo Random Forest, mediante validación cruzada del
tipo *leave-one-subject-out*.

**Capa 2 — Demand(t).** Representa la demanda fisiológica relativa de
glucocorticoides. Es una variable latente, no observable directamente: se construye
combinando un componente circadiano (el patrón natural de cortisol a lo largo del
día) con StressLoad(t) como señal de demanda adicional ante activación fisiológica.

**Capa 3 — Coverage(t).** Representa la cobertura glucocorticoide estimada, derivada
de la dosis, el horario de la toma y un modelo farmacocinético simplificado
(compartimental, de absorción y eliminación de primer orden).

**Capa 4 — Risk(t).** Representa la posible desalineación contextual entre demanda y
cobertura:

```
Risk(t) = Demand(t) − Coverage(t)
```

Un valor de Risk(t) por encima de cero indica que la demanda estimada supera a la
cobertura estimada en ese instante, es decir, una posible situación de
infra-cobertura.

Risk(t) se expresa como un número con unidades (mg-equivalente/hora), pero esa
precisión aparente no debe confundirse con exactitud referencial. El valor hereda,
sin corregirla, la incertidumbre explicitada de sus dos términos: Demand(t) y
Coverage(t). Además, tiene la incertidumbre añadida de que Demand(t) deja fuera
explícitamente otros factores con efecto fisiológico plausible sobre la demanda real
(enfermedad intercurrente, calidad de sueño, actividad física acumulada, etc.).
Risk(t) es, por tanto, la salida de un modelo deliberadamente parcial, no una
reconstrucción completa de los factores que determinarían una desalineación real.

En consecuencia, el valor puntual de Risk(t) no debe interpretarse como una cantidad
clínica verificable, sino como una hipótesis relativa generada dentro de los límites
del modelo: útil para razonar sobre dirección (¿hacia infra- o sobre-cobertura?) y
magnitud comparativa entre instantes o escenarios, no como una medida absoluta
trasladable fuera de ese marco.

### Wearables y proxies fisiológicos

Los wearables empleados en este proyecto no miden cortisol, ACTH, ni ninguna otra
variable endocrina. Miden frecuencia cardíaca, variabilidad de la frecuencia
cardíaca, actividad, sueño y otras señales fisiológicas indirectas, que se
interpretan en este sistema como proxies de activación fisiológica, no como
biomarcadores endocrinos directos.

### Dónde encontrar cada capa

| Capa | Notebook(s) |
|---|---|
| Preprocesamiento de WESAD | `notebooks/01_WESAD_preprocesamiento.ipynb` |
| StressLoad(t) | `notebooks/02_StressLoad.ipynb` |
| Demand(t) | `notebooks/03_Demand.ipynb` |
| Coverage(t) | `notebooks/04_Coverage.ipynb` |
| Validación con escenarios sintéticos | `notebooks/05_Escenarios_sinteticos.ipynb` |
| Preprocesamiento del caso n-of-1 | `notebooks/06_n_of_1_preprocesamiento.ipynb` |
| Integración final y cálculo de Risk(t) | `notebooks/07_n_of_1_calculo_validacion.ipynb` |

## Alcance y limitaciones

### Qué es y qué no es este proyecto

Este proyecto es un sistema experimental de soporte contextual de decisión que estima
posibles situaciones de desalineación entre demanda fisiológica y cobertura
glucocorticoide, a partir de señales de wearables y un modelo farmacocinético
simplificado.

**Este proyecto no es:**

- un dispositivo médico,
- un sistema de diagnóstico clínico,
- un sistema de predicción de crisis adrenal,
- una herramienta que recomiende dosis terapéuticas,
- ni un sustituto del criterio clínico o del seguimiento médico habitual.

Ningún resultado de este sistema debe usarse para tomar decisiones de dosificación
sin la supervisión de un profesional sanitario.

### Naturaleza de las estimaciones

Como se detalla en [Risk(t)](#arquitectura-del-sistema), las salidas de este sistema
son estimaciones relativas dentro de los límites de un modelo deliberadamente
parcial, no cantidades clínicas verificables. Esto aplica también, de forma análoga,
a Demand(t) y StressLoad(t): son variables latentes inferidas, no medidas
directamente por ningún sensor.

### Limitaciones metodológicas

- **Generalización desde WESAD.** El modelo de StressLoad(t) se entrena y valida con
  11 sujetos de WESAD, mediante validación cruzada *leave-one-subject-out*. La
  variabilidad de rendimiento entre sujetos es alta (F1 macro medio 0.73, desviación
  estándar 0.22), lo que indica que el modelo generaliza de forma desigual según el
  sujeto, y que su comportamiento en población general fuera de WESAD no está
  garantizado.
- **Caso n-of-1.** La validación final del sistema completo se realiza sobre los
  datos de una única persona, durante un periodo acotado. Los resultados no son
  generalizables a otros pacientes con insuficiencia suprarrenal sin validación
  adicional.
- **Cobertura parcial de los datos.** No todos los minutos del periodo analizado
  cuentan con una estimación válida de StressLoad(t) o Demand(t) (ver notebooks para
  cifras exactas de cobertura), por huecos en el registro del wearable o por
  encontrarse fuera del dominio de actividad del modelo.
- **Sin validación clínica.** Este sistema no ha sido evaluado en un entorno clínico
  ni comparado contra mediciones hormonales de referencia (cortisol sérico o
  salival). No cuenta con ninguna certificación ni marcado como producto sanitario.

### Limitaciones de reproducibilidad

Ver la sección [Datos](#datos) para las limitaciones de reproducibilidad derivadas de
la falta de redistribución de WESAD y de los datos crudos del wearable personal.

## Estructura del repositorio

```
glucocorticoid-mismatch-estimation/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── LICENSE
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_WESAD_preprocesamiento.ipynb
│   ├── 02_StressLoad.ipynb
│   ├── 03_Demand.ipynb
│   ├── 04_Coverage.ipynb
│   ├── 05_Escenarios_sinteticos.ipynb
│   ├── 06_n_of_1_preprocesamiento.ipynb
│   ├── 07_n_of_1_calculo_validacion.ipynb
│   ├── formulario_utils.py
│   ├── MVA_config_rutas.json
│   ├── WESAD_config_rutas.json
│   └── confirmed_omissions.json
│
├── models/
│   ├── stressload_camino_B/
│   │   ├── modelo_stress_camino_B_final.pkl
│   │   └── columnas_camino_B.json
│   └── n_of_1_extended/
│       └── baseline_crhr_mcmc_extended.parquet
│
├── data/
│   ├── raw/            (no incluido — ver Datos)
│   └── processed/
│       ├── MVA_n_of_1_extended.parquet
│       └── MVA_n_of_1_risk_extended.parquet
│
└── docs/
    ├── Extraccion_Revision_Bibliografica_TFM.xlsx
    └── Prompts_utilizadas.docx
```

## Datos

### Qué se incluye en este repositorio

- `data/processed/MVA_n_of_1_extended.parquet` y `MVA_n_of_1_risk_extended.parquet`:
  matriz personal minuto a minuto del caso n-of-1, con las variables derivadas de
  cada capa (StressLoad, Demand, Coverage, Risk). Publicados con consentimiento
  expreso de la persona cuyos datos contienen.
- `data/raw/20260813/`: exports crudos de frecuencia cardíaca y actividad del
  wearable personal, en formato CSV.
- `data/raw/Formulario EMA/Tomas_hidrocortisona_20260830.xlsx`: registro de tomas de
  hidrocortisona (dosis y horario) del caso n-of-1.
- `models/`: modelo entrenado de StressLoad(t) (Camino B) y sus artefactos asociados.

### Qué NO se incluye

- **WESAD** (dataset público de terceros usado para entrenar y validar StressLoad(t)):
  no se redistribuye en este repositorio. Ver
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) para la cita completa y las
  instrucciones de descarga desde la fuente oficial.
- Ningún dato derivado de WESAD que permita reconstruir la etiqueta de condición
  experimental o la señal fisiológica por sujeto (por ejemplo, `df_loso_wesad.parquet`)
  se publica, por la misma razón.

### Diccionario de datos

Cada parquet en `data/processed/` contiene entre 40 y 55 columnas. Se agrupan por
capa de la arquitectura:

- **Señales de wearable:** `heart_rate_bpm`, `steps`, `spo2_pct`,
  `ambient_light_lux`, `temp_diff_celsius`, `estado` (categoría de actividad).
- **StressLoad:** `delta_hr*`, `hr_esperado*`, `StressLoad_pred_*`,
  `StressLoad_proba_*`. Proxies fisiológicos derivados de wearables, no mediciones
  endocrinas.
- **Demand:** `demand_circadian`, `demand_total`, `demand_completo`. Variable
  latente inferida, no observable directamente (ver
  [Arquitectura del sistema](#arquitectura-del-sistema)).
- **Coverage:** `coverage`, `coverage_sin_refuerzo`.
- **Risk:** `risk_total`, `risk_sin_refuerzo`.

### Reproducibilidad

El preprocesamiento de WESAD (`01_WESAD_preprocesamiento.ipynb`) requiere descargar
el dataset por separado (ver arriba). El resto del pipeline, desde
`02_StressLoad.ipynb` hasta `07_n_of_1_calculo_validacion.ipynb`, es ejecutable de
principio a fin con los datos incluidos en este repositorio.

## Instalación y reproducción

### Requisitos

- Python 3.13
- Git

### Instalación

```bash
git clone https://github.com/Minerva-creator/glucocorticoid-mismatch-estimation.git
cd glucocorticoid-mismatch-estimation
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Para reproducir el entorno exacto usado en el desarrollo, en vez de solo las
dependencias directas:

```bash
pip install -r requirements-lock.txt
```

### Orden de ejecución de los notebooks

```
01_WESAD_preprocesamiento.ipynb        # requiere descargar WESAD (ver Datos)
02_StressLoad.ipynb                    # entrena el modelo de Capa 1; requiere WESAD
03_Demand.ipynb                        # exploración/validación de Capa 2;
                                        # depende del artefacto generado por 02
04_Coverage.ipynb                      # exploración/validación de Capa 3
05_Escenarios_sinteticos.ipynb         # validación con datos sintéticos
06_n_of_1_preprocesamiento.ipynb       # construye la matriz personal
07_n_of_1_calculo_validacion.ipynb     # integra todo, calcula Risk(t)
```

Cada notebook lee su configuración de rutas desde `MVA_config_rutas.json` o
`WESAD_config_rutas.json`, ambos en `notebooks/`, con rutas relativas a la raíz del
repositorio. No es necesario modificarlos para ejecutar el proyecto tal como está
publicado.

## Cita de este trabajo

Si este proyecto resulta de utilidad para trabajos posteriores, puede citarse como:

> Villegas Alcaide, M. (2026). *Machine Learning y Modelado Híbrido para la
> Estimación de Desalineación Fisiológica en Insuficiencia Suprarrenal mediante
> wearables* [Trabajo de Fin de Máster]. Universidad Internacional de Valencia (VIU).

## Licencia

El código de este repositorio se distribuye bajo licencia MIT (ver [LICENSE](LICENSE)).

Los datos en `data/`, los modelos en `models/` y los documentos en `docs/` **no**
están cubiertos por esta licencia. Su uso queda sujeto a lo indicado en
[Datos](#datos) y [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
