"""Configuración central del proyecto (parámetros, rutas y constantes)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("TFM_DATA_RAW", ROOT / "data" / "raw"))
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

ARCHIVO_HOGARES = "ENIGHUR25_HOGARES_AGREGADOS.csv"

# --- Parámetros del experimento ---
RANDOM_STATE = 42
K_FINAL = 6
K_RANGO = range(2, 9)
N_PCA = 10
N_SIL_MUESTRA = 5000      # submuestra para silhouette (coste O(n^2))
B_BOOTSTRAP = 100         # remuestreos bootstrap para estabilidad
FRAC_BOOTSTRAP = 0.8      # fracción de hogares en cada remuestreo
N_JERARQUICO = 4000       # submuestra para Ward (memoria O(n^2))
N_SEMILLAS = 20           # semillas para estabilidad de inicialización
N_JACCARD = 30            # remuestreos para Jaccard por segmento
WINSOR_P = (0.01, 0.99)   # percentiles de winsorización de la tasa de ahorro

# --- Divisiones COICOP (d1..d13) del archivo agregado ---
DIVISIONES = {
    "d1": "alimentos_bebidas",
    "d2": "bebidas_alcoholicas_tabaco",
    "d3": "prendas_vestir_calzado",
    "d4": "vivienda_agua_electricidad",
    "d5": "muebles_hogar",
    "d6": "salud",
    "d7": "transporte",
    "d8": "comunicaciones",
    "d9": "recreacion_cultura",
    "d10": "educacion",
    "d11": "restaurantes_hoteles",
    "d12": "bienes_serv_diversos",
    "d13": "otros_gastos",
}
CATEGORIAS = list(DIVISIONES.values())
PCT_COLS = [f"pct_{c}" for c in CATEGORIAS]

# --- Variables del algoritmo (18) ---
FEATURES_CLUSTER = (
    ["log_ing_cor_per", "tasa_ahorro_w", "NUMPERS", "nivins_j", "tiene_auto"] + PCT_COLS
)
Z_COLS = [f"z_{c}" for c in FEATURES_CLUSTER]

# --- Variables excluidas del algoritmo (solo caracterización / equidad) ---
VARS_SENSIBLES = ["sexo_j", "etnia_j"]

# --- Columnas del dataset limpio (orden de salida) ---
COLS_BASE = [
    "Identif_hog", "AREA", "REGION", "PROVINCIA", "NCIUDAD", "NUMPERS", "sexo_j",
    "gedad_j", "nivins_j", "etnia_j", "cant_auto", "cant_urb", "ing_cor_tot",
    "ing_cor_per", "ing_mon_cor", "ing_no_mon", "dec_nac_per", "gas_cor_tot",
    "gas_cor_per", "gas_mon_cor", "gas_nm_cor", "dec_gas_per_nac",
]

REGIONES = {1: "Costa", 2: "Sierra", 3: "Amazonía", 4: "Insular"}
AREAS = {1: "Urbana", 2: "Rural"}
SEXO = {1: "Hombre", 2: "Mujer"}
ETNIA = {1: "Indígena", 2: "Afroecuatoriano", 3: "Mestizo", 4: "Blanco", 5: "Montubio/otro"}

# Nombres de segmento (se fijan tras la caracterización; ver 06_caracterizacion.py)
SEGMENTOS = {
    0: "Consolidados ahorradores",
    1: "Familias con gasto educativo",
    2: "Gasto en vivienda y servicios",
    3: "Nicho de alcohol y tabaco",
    4: "Subsistencia (ingreso bajo)",
    5: "Hogares con gasto en salud",
}
COLORES = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd", 5: "#8c564b"}
