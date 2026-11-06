"""Paso 1. Limpieza y construcción del dataset analítico de hogares.

Entrada : data/raw/ENIGHUR25_HOGARES_AGREGADOS.csv
Salida  : data/processed/ENIGHUR2025_dataset_clustering.csv
          outputs/tables/01_resumen_limpieza.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as cfg
from src.utils import leer_hogares_crudo, guardar_json


def main():
    raw = leer_hogares_crudo()
    n0 = len(raw)
    resumen = {"filas_crudas": n0, "columnas_crudas": raw.shape[1]}

    # 1) Identificador único
    assert raw["Identif_hog"].is_unique, "Identif_hog duplicado"

    # 2) Registros sin ingreso corriente total no permiten calcular ahorro
    sin_ing = raw["ing_cor_tot"].isna() | raw["gas_cor_tot"].isna()
    resumen["eliminados_sin_ingreso_o_gasto"] = int(sin_ing.sum())
    df = raw.loc[~sin_ing].copy()

    # 2b) Identidades contables del archivo agregado (control de consistencia interna)
    def maxdif(a, b):
        return float((a.fillna(0) - b.fillna(0)).abs().max())
    suma_d = df[list(cfg.DIVISIONES)].fillna(0).sum(axis=1)
    resumen["identidades"] = {
        "ing_cor_tot = ing_mon_cor + ing_no_mon": maxdif(df.ing_cor_tot, df.ing_mon_cor.fillna(0) + df.ing_no_mon.fillna(0)),
        "gas_cor_tot = gas_mon_cor + gas_nm_cor": maxdif(df.gas_cor_tot, df.gas_mon_cor.fillna(0) + df.gas_nm_cor.fillna(0)),
        "gas_mon_cor = gas_gru_cor + ot_gas_mon": maxdif(df.gas_mon_cor, df.gas_gru_cor.fillna(0) + df.ot_gas_mon.fillna(0)),
        "sum(d1..d13) = gas_gru_cor": maxdif(suma_d, df.gas_gru_cor),
        "ing_cor_per = ing_cor_tot / NUMPERS": maxdif(df.ing_cor_per, df.ing_cor_tot / df.NUMPERS),
    }
    resumen["pct_gasto_monetario_en_otros"] = float(df.ot_gas_mon.fillna(0).sum() / df.gas_mon_cor.sum())

    # 3) Ingreso y gasto estrictamente positivos
    invalido = (df["ing_cor_tot"] <= 0) | (df["gas_cor_tot"] <= 0)
    resumen["eliminados_no_positivos"] = int(invalido.sum())
    df = df.loc[~invalido].copy()

    # 4) Divisiones de gasto (d1..d13) con nombres legibles
    for d, nombre in cfg.DIVISIONES.items():
        df[nombre] = df[d].fillna(0.0)

    # 5) Variables derivadas
    df["tasa_ahorro"] = (df["ing_cor_tot"] - df["gas_cor_tot"]) / df["ing_cor_tot"]
    # En el crudo, la ausencia de auto/inmueble urbano es una celda vacía: equivale a 0
    for c in ["cant_auto", "cant_urb", "ing_no_mon", "gas_nm_cor"]:
        df[c] = df[c].fillna(0.0)
    df["tiene_auto"] = (df["cant_auto"] > 0).astype(int)
    for nombre in cfg.CATEGORIAS:
        # Participación sobre el gasto monetario corriente. Si el hogar no tuvo gasto
        # monetario (denominador 0) la participación queda indefinida (NaN) y se
        # imputa con 0 en el paso 03 (el hogar no destina nada a ninguna categoría).
        den = df["gas_mon_cor"].replace(0, np.nan)
        df[f"pct_{nombre}"] = df[nombre] / den

    cols = cfg.COLS_BASE + cfg.CATEGORIAS + ["Fexp", "tasa_ahorro", "tiene_auto"] + cfg.PCT_COLS
    out = df[cols].reset_index(drop=True)

    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_clustering.csv", index=False)
    resumen.update({"filas_finales": len(out), "columnas_finales": out.shape[1],
                    "hogares_sin_gasto_monetario": int((out["gas_mon_cor"] == 0).sum()),
                    "nulos_en_variables_base": int(out[["ing_cor_per", "gas_cor_per", "tasa_ahorro", "NUMPERS", "nivins_j", "tiene_auto", "Fexp"]].isna().sum().sum())})
    guardar_json(resumen, "01_resumen_limpieza.json")
    print(resumen)


if __name__ == "__main__":
    main()
