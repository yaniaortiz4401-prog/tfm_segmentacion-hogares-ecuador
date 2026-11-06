# Segmentación de hogares ecuatorianos según su perfil socioeconómico y financiero

Código y datos del Trabajo Fin de Máster **«Segmentación de hogares ecuatorianos según su perfil
socioeconómico y financiero mediante técnicas de clustering y visualización de datos, como base para
la identificación de segmentos potenciales de clientes bancarios»** (Máster en Visual Analytics y Big
Data, UNIR; Tipo 1: piloto experimental).

**Autora:** Yania Melissa Peña Ortiz · **Directora:** Amaya Carriba Pérez

## Qué hace este repositorio

Toma la **ENIGHUR 2025** (INEC, Ecuador) como *proxy* de clientes bancarios —en Ecuador no existen
datos abiertos de clientes de banca— y ejecuta el pipeline:

`limpieza → EDA → ingeniería de variables → StandardScaler → PCA (10 comp.) → K-Means (k = 6) → validación → caracterización`

Resultado: 6 segmentos de hogares (35 397 hogares) y su lectura como segmentos potenciales de clientes.

| Segmento | Hogares | % población (ponderado) |
|---|---:|---:|
| 0 Consolidados ahorradores | 9 562 | 26,7 |
| 1 Familias con gasto educativo | 3 333 | 11,1 |
| 2 Gasto en vivienda y servicios | 4 971 | 15,6 |
| 3 Nicho de alcohol y tabaco | 565 | 1,8 |
| 4 Subsistencia (ingreso bajo) | 13 274 | 35,7 |
| 5 Hogares con gasto en salud | 3 692 | 9,1 |

> **Limitaciones (importantes).** La silhouette es baja (≈ 0,14) y la estabilidad bajo remuestreo es
> moderada: los segmentos son una *partición útil de un continuo*, no grupos naturalmente separados.
> Sexo y etnia **no** entran al algoritmo (equidad algorítmica). Los hogares de la ENIGHUR no son
> clientes reales de ningún banco; los segmentos requieren validación con datos propios de una entidad.

## Estructura

```
├── README.md · LICENSE · requirements.txt · run_all.py
├── src/
│   ├── config.py                  # parámetros, rutas, variables, nombres de segmentos
│   ├── utils.py                   # lectura del CSV crudo y guardado de figuras/tablas
│   ├── 00_inspeccion_fuentes.py   # (opcional) inspección de los 6 CSV crudos
│   ├── 01_preparar_datos.py       # limpieza y variables derivadas
│   ├── 02_eda.py                  # análisis exploratorio
│   ├── 03_matriz_clustering.py    # transformaciones + estandarización (18 variables)
│   ├── 04_clustering.py           # PCA, selección de k, K-Means final
│   ├── 05_validacion.py           # ANOVA/η², χ²/V de Cramér, bootstrap, Jaccard, Ward, GMM, DBSCAN
│   ├── 05b_caso_base.py           # caso base (solo ingreso / sin PCA) vs. experimental
│   └── 06_caracterizacion.py      # perfiles, indicadores bancarios y figuras finales
├── tests/test_pipeline.py         # 8 pruebas de integridad
├── data/raw/                      # CSV de entrada (ver «Datos»)
├── data/processed/                # datasets derivados (se regeneran)
└── outputs/{figures,tables}/      # resultados
```

## Instalación y ejecución

```bash
git clone https://github.com/yaniaortiz4401-prog/tfm_segmentacion-hogares-ecuador
cd tfm_segmentacion-hogares-ecuador
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # ~10 min (la validación por bootstrap es la parte lenta)
pytest -q                  # 8 pruebas; ejecutar DESPUÉS de run_all.py
```

Si los datos están en otra carpeta: `TFM_DATA_RAW=/ruta/a/los/csv python run_all.py`
(PowerShell: `$env:TFM_DATA_RAW="C:\ruta"; python run_all.py`).

Reproducibilidad: `RANDOM_STATE = 42` en `src/config.py`; con las mismas versiones de librerías las
etiquetas de segmento son idénticas entre ejecuciones.

## Datos

Fuente: INEC, *Encuesta Nacional de Ingresos y Gastos de los Hogares Urbanos y Rurales (ENIGHUR) 2025*,
datos abiertos (<https://www.ecuadorencifras.gob.ec>). Este pipeline solo necesita
`ENIGHUR25_HOGARES_AGREGADOS.csv` (31 MB, ya incluido en `data/raw/` si se copia allí). Los otros cinco
archivos (gastos por hogar, regalos, ingresos de hogares y personas; ~900 MB) se usaron en el control de
consistencia y **no** se suben al repositorio. Los datos son anonimizados por el INEC; el análisis es a
nivel de hogar y no intenta reidentificar a nadie.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos pertenecen al INEC y se rigen por sus términos de uso.

## Cómo citar

Peña Ortiz, Y. M. (2026). Segmentación de hogares ecuatorianos según su perfil socioeconómico y financiero… [Trabajo Fin de Máster, UNIR]. Código: https://github.com/yaniaortiz4401-prog/tfm_segmentacion-hogares-ecuador
