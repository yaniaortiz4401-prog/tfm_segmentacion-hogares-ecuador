"""Paso 3. Ingeniería de variables y matriz estandarizada para clustering.

Entrada : data/processed/ENIGHUR2025_dataset_clustering.csv
Salida  : data/processed/ENIGHUR2025_dataset_listo_clustering.csv (18 variables + z-scores)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.utils import guardar_json


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_clustering.csv")

    # Transformaciones
    df["log_ing_cor_per"] = np.log1p(df["ing_cor_per"])
    lo = df["tasa_ahorro"].quantile(cfg.WINSOR_P[0], interpolation="lower")
    hi = df["tasa_ahorro"].quantile(cfg.WINSOR_P[1], interpolation="higher")
    df["tasa_ahorro_w"] = df["tasa_ahorro"].clip(lo, hi)
    df[cfg.PCT_COLS] = df[cfg.PCT_COLS].fillna(0.0)

    cols = ["Identif_hog", "AREA", "REGION", "PROVINCIA", "NCIUDAD", "sexo_j", "etnia_j", "gedad_j",
            "ing_cor_per", "gas_cor_per", "tasa_ahorro", "Fexp", "log_ing_cor_per", "tasa_ahorro_w",
            "NUMPERS", "nivins_j", "tiene_auto"] + cfg.PCT_COLS
    out = df[cols].copy()
    assert out[cfg.FEATURES_CLUSTER].isna().sum().sum() == 0

    Z = StandardScaler().fit_transform(out[cfg.FEATURES_CLUSTER])
    out[cfg.Z_COLS] = Z
    out.to_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_listo_clustering.csv", index=False)
    guardar_json({"filas": len(out), "variables_algoritmo": cfg.FEATURES_CLUSTER,
                  "winsor_inferior": float(lo), "winsor_superior": float(hi)}, "03_resumen_matriz.json")
    print("Matriz lista:", out.shape, f"winsor=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
