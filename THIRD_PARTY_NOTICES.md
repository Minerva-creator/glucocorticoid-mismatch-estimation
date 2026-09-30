# Avisos de terceros

Este proyecto utiliza datos y referencias de terceros. Este documento declara su procedencia y las condiciones bajo las que se han usado.

## WESAD (Wearable Stress and Affect Detection)

Este proyecto utiliza el dataset WESAD para entrenar y validar el modelo de Capa 1 (StressLoad).

**Cita:**

> Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proceedings of the 20th ACM International Conference on Multimodal Interaction*, 400–408. https://doi.org/10.1145/3242969.3242985

**Condiciones de uso:** según se declara en la fuente oficial, WESAD puede usarse con fines científicos y no comerciales, citando a los autores. Estas condiciones autorizan el uso de los datos, pero no su redistribución.

**Por ese motivo:**
- Los datos originales de WESAD **no se distribuyen en este repositorio**.
- Para reproducir los notebooks `01_WESAD_preprocesamiento.ipynb` y `02_StressLoad.ipynb`, es necesario descargar el dataset desde la fuente oficial: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html y colocarlo en `data/raw/WESAD/`, siguiendo la estructura de carpetas que espera `notebooks/WESAD_config_rutas.json`.
- data/raw/WESAD/
├── S2/
│   ├── S2.pkl
│   └── S2_E4_Data/
├── S3/
│   ├── S3.pkl
│   └── S3_E4_Data/
├── S4/
│   ├── S4.pkl
│   └── S4_E4_Data/
...
├── S16/
│   ├── S16.pkl
│   └── S16_E4_Data/
└── S17/
    ├── S17.pkl
    └── S17_E4_Data/
- Por el mismo motivo, tampoco se publica `df_loso_wesad.parquet`, al conservar identificadores de sujeto y la etiqueta de condición experimental (el contenido sustantivo del dataset), aunque incluya variables derivadas propias.

## Otras dependencias

Las librerías de terceros utilizadas (scikit-learn, XGBoost, pandas, etc.) se listan con su versión exacta en `requirements.txt`, cada una bajo su propia licencia de código abierto (BSD-3-Clause o Apache 2.0 según el paquete).