"""Paso 4. PCA, selección de k y K-Means final.

Salidas: data/processed/ENIGHUR2025_hogares_con_clusters.csv
         outputs/tables/04_*.csv|json ; outputs/figures/fig04..fig05
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src import config as cfg
from src.utils import guardar_fig, guardar_json, guardar_tabla


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_listo_clustering.csv")
    Z = df[cfg.Z_COLS].values

    # --- PCA ---
    pca_full = PCA(random_state=cfg.RANDOM_STATE).fit(Z)
    ev = pca_full.explained_variance_ratio_
    tabla_pca = pd.DataFrame({"componente": np.arange(1, len(ev) + 1), "varianza": ev,
                              "acumulada": ev.cumsum(), "autovalor": pca_full.explained_variance_})
    guardar_tabla(tabla_pca, "04_pca_varianza.csv", index=False)
    pca = PCA(n_components=cfg.N_PCA, random_state=cfg.RANDOM_STATE).fit(Z)
    X = pca.transform(Z)
    cargas = pd.DataFrame(pca.components_[:4].T, index=cfg.FEATURES_CLUSTER, columns=[f"PC{i}" for i in range(1, 5)])
    guardar_tabla(cargas, "04_pca_cargas.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 19), ev * 100, color="#4c78a8", label="Individual")
    ax.plot(range(1, 19), ev.cumsum() * 100, "o-", color="#e45756", label="Acumulada")
    ax.axvline(cfg.N_PCA + .5, ls="--", color="gray"); ax.set_xlabel("Componente principal")
    ax.set_ylabel("% de varianza"); ax.set_title("Varianza explicada por componente"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    guardar_fig(fig, "fig04_pca_varianza.png")

    # --- Selección de k ---
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    idx = rng.choice(len(X), cfg.N_SIL_MUESTRA, replace=False)
    filas = []
    for k in cfg.K_RANGO:
        km = KMeans(k, n_init=10, random_state=cfg.RANDOM_STATE).fit(X)
        filas.append({"k": k, "inercia": km.inertia_,
                      "silhouette": silhouette_score(X[idx], km.labels_[idx]),
                      "davies_bouldin": davies_bouldin_score(X, km.labels_),
                      "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                      "tam_min": int(np.bincount(km.labels_).min()),
                      "tam_max": int(np.bincount(km.labels_).max())})
    sel = pd.DataFrame(filas)
    guardar_tabla(sel, "04_seleccion_k.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
    for a, (col, tit) in zip(ax, [("inercia", "Inercia (codo)"), ("silhouette", "Silhouette ↑"),
                                  ("davies_bouldin", "Davies–Bouldin ↓"), ("calinski_harabasz", "Calinski–Harabasz ↑")]):
        a.plot(sel.k, sel[col], "o-", color="#4c78a8"); a.axvline(cfg.K_FINAL, ls="--", color="#e45756")
        a.set_title(tit); a.set_xlabel("k"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); guardar_fig(fig, "fig05_seleccion_k.png")

    # --- K-Means final ---
    km = KMeans(cfg.K_FINAL, n_init=10, random_state=cfg.RANDOM_STATE).fit(X)
    df["cluster"] = km.labels_
    df["pc1"], df["pc2"] = X[:, 0], X[:, 1]
    df.to_csv(cfg.DATA_PROC / "ENIGHUR2025_hogares_con_clusters.csv", index=False)
    np.save(cfg.DATA_PROC / "pca_10.npy", X)

    sil_final = silhouette_score(X[idx], km.labels_[idx])
    guardar_json({
        "k": cfg.K_FINAL, "n_pca": cfg.N_PCA,
        "varianza_acumulada_pca": float(ev[:cfg.N_PCA].sum()),
        "inercia": float(km.inertia_), "silhouette": float(sil_final),
        "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
        "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_)),
        "tamanos": {int(k): int(v) for k, v in pd.Series(km.labels_).value_counts().sort_index().items()},
    }, "04_clustering_final.json")
    print(pd.Series(km.labels_).value_counts().sort_index().to_dict(), "sil=%.3f" % sil_final)


if __name__ == "__main__":
    main()
