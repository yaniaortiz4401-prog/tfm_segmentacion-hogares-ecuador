"""Paso 5b. Caso base vs caso experimental.

Compara, en el mismo espacio estandarizado de las 18 variables:
  * Base A: segmentación solo por ingreso per cápita (6 grupos de igual tamaño, cuantiles);
  * Base B: K-Means (k=6) directamente sobre las 18 variables, sin PCA;
  * Experimental: PCA (10 comp.) + K-Means (k=6).
Salida: outputs/tables/05b_caso_base.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src import config as cfg
from src.utils import guardar_tabla


def eta2(v, g):
    m = v.mean()
    ssb = sum(len(v[g == c]) * (v[g == c].mean() - m) ** 2 for c in np.unique(g))
    return ssb / ((v - m) ** 2).sum()


def inercias_semillas(X, y_ref):
    """Inercia y ARI de 20 inicializaciones aleatorias (n_init=1) frente a la solución de referencia."""
    filas = []
    for sd in range(cfg.N_SEMILLAS):
        km = KMeans(cfg.K_FINAL, n_init=1, random_state=1000 + sd).fit(X)
        filas.append({"semilla": 1000 + sd, "inercia": km.inertia_,
                      "ARI_vs_referencia": adjusted_rand_score(y_ref, km.labels_)})
    return pd.DataFrame(filas)


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_hogares_con_clusters.csv")
    Z = df[cfg.Z_COLS].values
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    idx = rng.choice(len(Z), cfg.N_SIL_MUESTRA, replace=False)
    y_exp = df["cluster"].values

    y_ing = pd.qcut(df["ing_cor_per"], cfg.K_FINAL, labels=False).values
    y_z = KMeans(cfg.K_FINAL, n_init=10, random_state=cfg.RANDOM_STATE).fit(Z).labels_

    filas = []
    for nombre, y in [("Base A: solo ingreso (6 cuantiles)", y_ing),
                      ("Base B: K-Means sin PCA (18 variables)", y_z),
                      ("Experimental: PCA + K-Means", y_exp)]:
        e = [eta2(df[v].values, y) for v in cfg.FEATURES_CLUSTER]
        filas.append({
            "caso": nombre,
            "silhouette": silhouette_score(Z[idx], y[idx]),
            "davies_bouldin": davies_bouldin_score(Z, y),
            "calinski_harabasz": calinski_harabasz_score(Z, y),
            "eta2_medio_18_variables": float(np.mean(e)),
            "eta2_medio_participaciones_gasto": float(np.mean([eta2(df[c].values, y) for c in cfg.PCT_COLS])),
            "variables_eta2_grande": int(sum(x >= .14 for x in e)),
            "ARI_vs_experimental": adjusted_rand_score(y_exp, y),
        })
    X = np.load(cfg.DATA_PROC / "pca_10.npy")
    ine = inercias_semillas(X, y_exp)
    guardar_tabla(ine, "05b_inercias_semillas.csv", index=False)
    print("Inercia de referencia (n_init=10): %.0f | semillas: min %.0f, mediana %.0f, max %.0f" % (
        KMeans(cfg.K_FINAL, n_init=10, random_state=cfg.RANDOM_STATE).fit(X).inertia_,
        ine.inercia.min(), ine.inercia.median(), ine.inercia.max()))
    out = pd.DataFrame(filas)
    guardar_tabla(out, "05b_caso_base.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
