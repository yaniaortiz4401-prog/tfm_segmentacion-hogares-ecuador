"""Paso 5. Validación interna, externa y de estabilidad de la segmentación.

Salidas: outputs/tables/05_*.csv|json ; outputs/figures/fig06
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

from src import config as cfg
from src.utils import guardar_fig, guardar_json, guardar_tabla


def anova_eta2(valores, grupos):
    """ANOVA de un factor; devuelve F, p y eta cuadrado."""
    g = [valores[grupos == c] for c in np.unique(grupos)]
    F, p = stats.f_oneway(*g)
    media = valores.mean()
    ss_b = sum(len(x) * (x.mean() - media) ** 2 for x in g)
    ss_t = ((valores - media) ** 2).sum()
    return F, p, ss_b / ss_t


def cramer_v(tabla):
    chi2, p, dof, _ = stats.chi2_contingency(tabla)
    n = tabla.values.sum()
    k = min(tabla.shape) - 1
    return chi2, p, dof, np.sqrt(chi2 / (n * k))


def jaccard_max(a, b, k):
    """Jaccard máximo de cada cluster de `a` contra los de `b` (misma muestra)."""
    res = []
    for i in range(k):
        A = a == i
        res.append(max((A & (b == j)).sum() / (A | (b == j)).sum() for j in range(k)))
    return np.array(res)


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_hogares_con_clusters.csv")
    X = np.load(cfg.DATA_PROC / "pca_10.npy")
    y = df["cluster"].values
    k = cfg.K_FINAL
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    res = {}

    # 1) ANOVA por variable del algoritmo ----------------------------------
    filas = []
    for v in cfg.FEATURES_CLUSTER:
        F, p, e2 = anova_eta2(df[v].values, y)
        filas.append({"variable": v, "F": F, "p": p, "eta2": e2,
                      "efecto": "grande" if e2 >= .14 else "mediano" if e2 >= .06 else "pequeño"})
    anova = pd.DataFrame(filas).sort_values("eta2", ascending=False)
    guardar_tabla(anova, "05_anova_eta2.csv", index=False)

    # 2) Chi-cuadrado y V de Cramér contra variables NO usadas ------------------
    filas = []
    for v in ["REGION", "AREA", "sexo_j", "etnia_j", "gedad_j"]:
        tabla = pd.crosstab(df[v], df["cluster"])
        chi2, p, dof, V = cramer_v(tabla)
        filas.append({"variable": v, "chi2": chi2, "gl": dof, "p": p, "V_Cramer": V})
    guardar_tabla(pd.DataFrame(filas), "05_chi2_cramer.csv", index=False)

    # 3) Bootstrap (ARI contra la partición completa) ---------------------------
    aris = []
    for b in range(cfg.B_BOOTSTRAP):
        idx = rng.choice(len(X), int(cfg.FRAC_BOOTSTRAP * len(X)), replace=False)
        kmb = KMeans(k, n_init=5, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(y[idx], kmb.labels_))
    aris = np.array(aris)
    res["bootstrap_ari"] = {"media": aris.mean(), "sd": aris.std(), "min": aris.min(),
                            "p5": np.percentile(aris, 5), "max": aris.max()}

    # 4) Estabilidad frente a la semilla ----------------------------------------
    aris_s = []
    for s in range(cfg.N_SEMILLAS):
        km = KMeans(k, n_init=1, random_state=1000 + s).fit(X)
        aris_s.append(adjusted_rand_score(y, km.labels_))
    aris_s = np.array(aris_s)
    res["semillas_ari"] = {"media": aris_s.mean(), "sd": aris_s.std(), "min": aris_s.min(), "max": aris_s.max()}

    # Igual, pero con el procedimiento de producción (n_init=10, mejor de 10 inicializaciones)
    aris_p = np.array([adjusted_rand_score(y, KMeans(k, n_init=10, random_state=2000 + s).fit(X).labels_)
                       for s in range(cfg.N_SEMILLAS)])
    res["semillas_ari_n_init10"] = {"media": aris_p.mean(), "sd": aris_p.std(), "min": aris_p.min(),
                                    "max": aris_p.max(), "n_iguales_a_referencia": int((aris_p > .95).sum())}

    # 5) Jaccard por segmento ---------------------------------------------------
    J = np.zeros((cfg.N_JACCARD, k))
    for b in range(cfg.N_JACCARD):
        idx = rng.choice(len(X), int(cfg.FRAC_BOOTSTRAP * len(X)), replace=False)
        kmb = KMeans(k, n_init=5, random_state=500 + b).fit(X[idx])
        J[b] = jaccard_max(y[idx], kmb.labels_, k)
    jac = pd.DataFrame({"cluster": range(k), "jaccard_medio": J.mean(0), "jaccard_min": J.min(0),
                        "tamano": np.bincount(y)})
    jac["estabilidad"] = np.where(jac.jaccard_medio >= .85, "muy estable",
                                  np.where(jac.jaccard_medio >= .75, "estable",
                                           np.where(jac.jaccard_medio >= .60, "moderada", "inestable")))
    guardar_tabla(jac, "05_jaccard.csv", index=False)

    # 6) Algoritmos alternativos --------------------------------------------------
    idx_sil = rng.choice(len(X), cfg.N_SIL_MUESTRA, replace=False)
    comp = []
    def reg(nombre, lab, muestra=None):
        Xm, lm = (X, lab) if muestra is None else (X[muestra], lab)
        if len(np.unique(lm)) < 2:
            comp.append({"algoritmo": nombre, "clusters": len(np.unique(lm)), "silhouette": np.nan,
                         "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "ARI_vs_kmeans": np.nan}); return
        ii = rng.choice(len(Xm), min(cfg.N_SIL_MUESTRA, len(Xm)), replace=False)
        ref = y if muestra is None else y[muestra]
        comp.append({"algoritmo": nombre, "clusters": len(np.unique(lm)),
                     "silhouette": silhouette_score(Xm[ii], lm[ii]),
                     "davies_bouldin": davies_bouldin_score(Xm, lm),
                     "calinski_harabasz": calinski_harabasz_score(Xm, lm),
                     "ARI_vs_kmeans": adjusted_rand_score(ref, lm)})
    reg("K-Means (k=6)", y)
    sub = rng.choice(len(X), cfg.N_JERARQUICO, replace=False)
    reg("Ward jerárquico (k=6, submuestra)", AgglomerativeClustering(k, linkage="ward").fit_predict(X[sub]), sub)
    reg("Mezcla gaussiana (k=6)", GaussianMixture(k, covariance_type="full", random_state=cfg.RANDOM_STATE, n_init=2).fit_predict(X))
    db = DBSCAN(eps=1.5, min_samples=50).fit(X[sub]).labels_
    n_db = len(set(db)) - (1 if -1 in db else 0)
    res["dbscan"] = {"clusters_encontrados": n_db, "ruido_pct": float(100 * (db == -1).mean())}
    if n_db >= 2:
        m = db != -1
        comp.append({"algoritmo": "DBSCAN (eps=1.5, minPts=50)", "clusters": n_db,
                     "silhouette": silhouette_score(X[sub][m], db[m]), "davies_bouldin": np.nan,
                     "calinski_harabasz": np.nan, "ARI_vs_kmeans": adjusted_rand_score(y[sub][m], db[m])})
    else:
        comp.append({"algoritmo": "DBSCAN (eps=1.5, minPts=50)", "clusters": n_db, "silhouette": np.nan,
                     "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "ARI_vs_kmeans": np.nan})
    guardar_tabla(pd.DataFrame(comp), "05_comparacion_algoritmos.csv", index=False)

    # Figura 6: estabilidad -------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist(aris, bins=15, color="#4c78a8"); ax[0].set_title(f"Bootstrap ARI (B={cfg.B_BOOTSTRAP})")
    ax[1].hist(aris_s, bins=10, color="#54a24b"); ax[1].set_title(f"ARI entre semillas (n={cfg.N_SEMILLAS})")
    ax[2].bar(jac.cluster, jac.jaccard_medio, color=[cfg.COLORES[i] for i in jac.cluster])
    ax[2].axhline(.75, ls="--", color="gray"); ax[2].set_ylim(0, 1); ax[2].set_title("Jaccard medio por segmento")
    ax[2].set_xlabel("Segmento")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); guardar_fig(fig, "fig06_estabilidad.png")

    guardar_json(res, "05_resumen_validacion.json")
    print(json_pp(res))


def json_pp(d):
    return {k: {kk: (round(float(vv), 4) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} for k, v in d.items()}


if __name__ == "__main__":
    main()
