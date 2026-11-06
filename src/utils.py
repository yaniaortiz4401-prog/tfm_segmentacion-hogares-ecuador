"""Utilidades de lectura y escritura compartidas por los scripts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as cfg


def leer_hogares_crudo(ruta=None):
    """Lee el archivo agregado de hogares de la ENIGHUR 2025.

    El CSV usa ';' como separador, coma decimal, BOM UTF-8 y CRLF. Se lee todo
    como texto y se convierte columna a columna para tolerar celdas vacías.
    """
    ruta = Path(ruta) if ruta else cfg.DATA_RAW / cfg.ARCHIVO_HOGARES
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if c == "NCIUDAD":
            df[c] = df[c].str.strip()
        else:
            df[c] = a_float(df[c])
    return df


def a_float(serie):
    """Convierte texto con coma decimal a float; vacíos -> NaN."""
    return pd.to_numeric(serie.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")


def guardar_fig(fig, nombre, dpi=200):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = cfg.FIG_DIR / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def guardar_tabla(df, nombre, index=True):
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    ruta = cfg.TAB_DIR / nombre
    df.to_csv(ruta, index=index, encoding="utf-8-sig")
    return ruta


def guardar_json(obj, nombre):
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    ruta = cfg.TAB_DIR / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return ruta
