"""Paso 2. Análisis exploratorio de datos (EDA).

Salidas: outputs/tables/02_*.csv y outputs/figures/fig01..fig03
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.utils import guardar_fig, guardar_tabla


def media_ponderada(x, w):
    return float(np.average(x, weights=w))


def cuantil_ponderado(x, w, q):
    o = np.argsort(x.values)
    xs, ws = x.values[o], w.values[o]
    c = np.cumsum(ws) / ws.sum()
    return float(xs[np.searchsorted(c, q)])


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_clustering.csv")
    w = df["Fexp"]

    # --- Tabla 1: estadísticos descriptivos de las variables clave ---
    vars_clave = ["NUMPERS", "ing_cor_per", "gas_cor_per", "tasa_ahorro", "nivins_j"]
    filas = []
    for v in vars_clave:
        x = df[v]
        filas.append({
            "variable": v, "n": int(x.notna().sum()), "media": x.mean(), "desv_est": x.std(),
            "min": x.min(), "p25": x.quantile(.25), "mediana": x.median(),
            "p75": x.quantile(.75), "max": x.max(),
            "media_ponderada": media_ponderada(x, w),
            "mediana_ponderada": cuantil_ponderado(x, w, .5),
            "asimetria": x.skew(), "curtosis": x.kurt(),
        })
    guardar_tabla(pd.DataFrame(filas), "02_descriptivos.csv", index=False)

    # --- Tabla 2: composición muestral y poblacional ---
    comp = []
    for var, dic in [("REGION", cfg.REGIONES), ("AREA", cfg.AREAS), ("sexo_j", cfg.SEXO), ("etnia_j", cfg.ETNIA)]:
        n = df[var].value_counts().sort_index()
        pw = df.groupby(var)["Fexp"].sum()
        for k in n.index:
            comp.append({"variable": var, "categoria": dic.get(k, k), "hogares_muestra": int(n[k]),
                         "pct_muestra": 100 * n[k] / len(df), "hogares_expandidos": float(pw[k]),
                         "pct_poblacion": 100 * pw[k] / pw.sum()})
    guardar_tabla(pd.DataFrame(comp), "02_composicion.csv", index=False)

    # --- Tabla 3: gasto promedio por categoría ---
    gasto = pd.DataFrame({
        "categoria": cfg.CATEGORIAS,
        "gasto_medio_hogar": [df[c].mean() for c in cfg.CATEGORIAS],
        "participacion_media_pct": [100 * df[f"pct_{c}"].mean() for c in cfg.CATEGORIAS],
        "pct_hogares_con_gasto": [100 * (df[c] > 0).mean() for c in cfg.CATEGORIAS],
    })
    guardar_tabla(gasto, "02_gasto_categorias.csv", index=False)

    # --- Tabla 4: ingreso per cápita por decil nacional ---
    dec = df.groupby("dec_nac_per").agg(hogares=("Identif_hog", "size"), ingreso_pc=("ing_cor_per", "mean"),
                                        gasto_pc=("gas_cor_per", "mean"), ahorro=("tasa_ahorro", "mean"))
    guardar_tabla(dec, "02_deciles.csv")

    # --- Tabla 5: asimetría antes/después de transformar ---
    tr = pd.DataFrame({
        "variable": ["ing_cor_per", "log1p(ing_cor_per)", "tasa_ahorro", "tasa_ahorro (winsor.)"],
        "asimetria": [df.ing_cor_per.skew(), np.log1p(df.ing_cor_per).skew(), df.tasa_ahorro.skew(),
                      df.tasa_ahorro.clip(df.tasa_ahorro.quantile(cfg.WINSOR_P[0], interpolation="lower"),
                                          df.tasa_ahorro.quantile(cfg.WINSOR_P[1], interpolation="higher")).skew()],
    })
    guardar_tabla(tr, "02_transformaciones.csv", index=False)

    # --- Figura 1: distribución del ingreso per cápita (original vs log) ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df.ing_cor_per.clip(upper=df.ing_cor_per.quantile(.99)), bins=60, color="#4c78a8")
    ax[0].set_title("Ingreso corriente per cápita (USD/mes, tope p99)")
    ax[0].set_xlabel("USD"); ax[0].set_ylabel("Hogares")
    ax[1].hist(np.log1p(df.ing_cor_per), bins=60, color="#54a24b")
    ax[1].set_title("log(1 + ingreso per cápita)")
    ax[1].set_xlabel("log USD")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    guardar_fig(fig, "fig01_distribucion_ingreso.png")

    # --- Figura 2: tasa de ahorro ---
    lo, hi = df.tasa_ahorro.quantile(cfg.WINSOR_P[0], interpolation="lower"), df.tasa_ahorro.quantile(cfg.WINSOR_P[1], interpolation="higher")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df.tasa_ahorro.clip(-3, 1), bins=80, color="#e45756")
    ax[0].set_title("Tasa de ahorro (recortada a [-3, 1] para graficar)")
    ax[0].set_xlabel("(ingreso - gasto) / ingreso")
    ax[1].hist(df.tasa_ahorro.clip(lo, hi), bins=80, color="#72b7b2")
    ax[1].set_title(f"Tasa de ahorro winsorizada [{lo:.2f}; {hi:.2f}]")
    ax[1].set_xlabel("tasa de ahorro")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.set_ylabel("Hogares")
    fig.tight_layout()
    guardar_fig(fig, "fig02_tasa_ahorro.png")

    # --- Figura 3: matriz de correlaciones de las variables del algoritmo ---
    X = pd.DataFrame({"log_ing_cor_per": np.log1p(df.ing_cor_per),
                      "tasa_ahorro_w": df.tasa_ahorro.clip(lo, hi),
                      "NUMPERS": df.NUMPERS, "nivins_j": df.nivins_j, "tiene_auto": df.tiene_auto})
    for c in cfg.PCT_COLS:
        X[c] = df[c].fillna(0)
    corr = X.corr()
    guardar_tabla(corr, "02_correlaciones.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    etiq = [c.replace("pct_", "% ").replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(etiq))); ax.set_xticklabels(etiq, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(etiq))); ax.set_yticklabels(etiq, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8, label="Correlación de Pearson")
    ax.set_title("Correlaciones entre las 18 variables del algoritmo")
    fig.tight_layout()
    guardar_fig(fig, "fig03_correlaciones.png")
    print("EDA lista")


if __name__ == "__main__":
    main()
