"""Paso 6. Caracterización de los segmentos y figuras finales.

Salidas: outputs/tables/06_*.csv ; outputs/figures/fig07..fig10
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.utils import guardar_fig, guardar_tabla


def wmean(x, w):
    return float(np.average(x, weights=w))


def main():
    df = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_hogares_con_clusters.csv")
    k = cfg.K_FINAL
    nombres = [cfg.SEGMENTOS[i] for i in range(k)]

    # --- Perfil numérico por segmento (medias simples y ponderadas) ---
    filas = []
    for c in range(k):
        s = df[df.cluster == c]
        w = s.Fexp
        fila = {"cluster": c, "segmento": cfg.SEGMENTOS[c], "hogares": len(s),
                "pct_muestra": 100 * len(s) / len(df),
                "pct_poblacion": 100 * w.sum() / df.Fexp.sum(),
                "ing_pc_media": s.ing_cor_per.mean(), "ing_pc_mediana": s.ing_cor_per.median(),
                "gas_pc_media": s.gas_cor_per.mean(),
                "tasa_ahorro_media_w": s.tasa_ahorro_w.mean(),
                "pct_con_ahorro_positivo": 100 * (s.tasa_ahorro > 0).mean(),
                "personas_hogar": s.NUMPERS.mean(), "nivel_instruccion": s.nivins_j.mean(),
                "pct_con_auto": 100 * s.tiene_auto.mean(), "grupo_edad_jefe": s.gedad_j.mean(),
                "pct_rural": 100 * (s.AREA == 2).mean()}
        for r, nom in cfg.REGIONES.items():
            fila[f"pct_{nom.lower()}"] = 100 * (s.REGION == r).mean()
        for col in cfg.PCT_COLS:
            fila[col] = 100 * s[col].mean()
        filas.append(fila)
    perfil = pd.DataFrame(filas)
    guardar_tabla(perfil, "06_perfil_segmentos.csv", index=False)

    # --- z-medios por segmento (perfil estandarizado) ---
    zmed = df.groupby("cluster")[cfg.Z_COLS].mean()
    zmed.index = nombres
    guardar_tabla(zmed.round(3), "06_z_medios.csv")

    # --- Composición demográfica (variables NO usadas por el algoritmo) ---
    demo = []
    for var, dic in [("sexo_j", cfg.SEXO), ("etnia_j", cfg.ETNIA)]:
        t = pd.crosstab(df.cluster, df[var], normalize="index") * 100
        t.columns = [dic.get(c, c) for c in t.columns]; t.index = nombres
        t.insert(0, "variable", var)
        demo.append(t)
    guardar_tabla(pd.concat(demo), "06_composicion_demografica.csv")

    # --- Participación de cada segmento por región y área ---
    reg = pd.crosstab(df.REGION, df.cluster, normalize="index") * 100
    reg.index = [cfg.REGIONES[i] for i in reg.index]; reg.columns = nombres
    guardar_tabla(reg, "06_segmentos_por_region.csv")

    # --- Distribución de cada segmento por decil nacional de ingreso per cápita ---
    base = pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_clustering.csv", usecols=["Identif_hog", "dec_nac_per"])
    dd = df[["Identif_hog", "cluster"]].merge(base, on="Identif_hog")
    dec = pd.crosstab(dd.cluster, dd.dec_nac_per.astype(int), normalize="index") * 100
    dec.index = nombres; dec.columns = [f"D{c}" for c in dec.columns]
    guardar_tabla(dec.round(2), "06_deciles_por_segmento.csv")

    # --- Regla simple para el nicho de alcohol y tabaco (diagnóstico de la inestabilidad) ---
    regla = []
    for umbral in (0.05, 0.08, 0.10):
        m = df.pct_bebidas_alcoholicas_tabaco > umbral
        regla.append({"umbral_participacion": umbral, "hogares_sobre_umbral": int(m.sum()),
                      "pct_en_segmento3": 100 * (df.loc[m, "cluster"] == 3).mean(),
                      "pct_del_segmento3_cubierto": 100 * m[df.cluster == 3].mean()})
    guardar_tabla(pd.DataFrame(regla), "06_regla_nicho_alcohol.csv", index=False)

    # --- Indicadores orientados a producto bancario ---
    banca = pd.DataFrame({
        "segmento": nombres,
        "pct_poblacion": perfil.pct_poblacion.values,
        "ing_pc_mediana": perfil.ing_pc_mediana.values,
        "pct_ahorro_positivo": perfil.pct_con_ahorro_positivo.values,
        "tasa_ahorro_media": perfil.tasa_ahorro_media_w.values,
        "pct_con_auto": perfil.pct_con_auto.values,
        "pct_gasto_educacion": perfil.pct_educacion.values,
        "pct_gasto_salud": perfil.pct_salud.values,
        "pct_gasto_transporte": perfil.pct_transporte.values,
        "pct_gasto_vivienda": perfil.pct_vivienda_agua_electricidad.values,
    })
    guardar_tabla(banca, "06_indicadores_bancarios.csv", index=False)

    # === Figura 7: composición del gasto por segmento (barras horizontales apiladas) ===
    comp = perfil.set_index("segmento")[cfg.PCT_COLS]
    comp.columns = [c.replace("pct_", "").replace("_", " ") for c in comp.columns]
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 13))
    fig, ax = plt.subplots(figsize=(11, 5))
    izq = np.zeros(k)
    for j, col in enumerate(comp.columns):
        ax.barh(range(k), comp[col].values, left=izq, color=pal[j], label=col, edgecolor="white", linewidth=.4)
        izq += comp[col].values
    ax.set_yticks(range(k)); ax.set_yticklabels(comp.index); ax.invert_yaxis()
    ax.set_xlabel("% del gasto monetario corriente"); ax.set_xlim(0, 100)
    ax.set_title("Composición del gasto por segmento")
    ax.legend(ncol=1, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    guardar_fig(fig, "fig07_composicion_gasto.png")

    # === Figura 8: mapa de calor de z-medios ===
    fig, ax = plt.subplots(figsize=(11, 4.6))
    M = zmed.values
    im = ax.imshow(M, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(k)); ax.set_yticklabels(nombres, fontsize=9)
    ax.set_xticks(range(len(cfg.FEATURES_CLUSTER)))
    ax.set_xticklabels([c.replace("pct_", "% ").replace("_", " ") for c in cfg.FEATURES_CLUSTER], rotation=70, ha="right", fontsize=8)
    for i in range(k):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 1:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 1.8 else "black")
    fig.colorbar(im, ax=ax, label="Media estandarizada (z)", shrink=.9)
    ax.set_title("Perfil estandarizado de cada segmento (se anotan |z| ≥ 1)")
    guardar_fig(fig, "fig08_heatmap_segmentos.png")

    # === Figura 9: proyección PCA ===
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    ii = rng.choice(len(df), 8000, replace=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for c in range(k):
        m = df.iloc[ii].cluster == c
        ax.scatter(df.iloc[ii][m].pc1, df.iloc[ii][m].pc2, s=5, alpha=.5, color=cfg.COLORES[c], label=cfg.SEGMENTOS[c])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=3, fontsize=8, frameon=False)
    ax.set_title("Hogares en el plano PC1–PC2 (muestra de 8 000)")
    ax.spines[["top", "right"]].set_visible(False)
    guardar_fig(fig, "fig09_pca_clusters.png")

    # === Figura 10: región y área por segmento (composición interna de cada segmento) ===
    regseg = pd.crosstab(df.cluster, df.REGION, normalize="index") * 100
    regseg.index = nombres; regseg.columns = [cfg.REGIONES[c] for c in regseg.columns]
    ar = pd.crosstab(df.cluster, df.AREA, normalize="index") * 100
    ar.index = nombres; ar.columns = [cfg.AREAS[c] for c in ar.columns]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    regseg.plot(kind="barh", stacked=True, ax=ax[0], colormap="Set2", width=.8)
    ax[0].set_title("Región (% de los hogares del segmento)")
    ar.plot(kind="barh", stacked=True, ax=ax[1], color=["#4c78a8", "#f58518"], width=.8)
    ax[1].set_title("Área (% de los hogares del segmento)"); ax[1].set_yticklabels([])
    for a_ in ax:
        a_.invert_yaxis(); a_.spines[["top", "right"]].set_visible(False); a_.set_xlabel("%"); a_.set_xlim(0, 100)
        a_.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.tight_layout(); guardar_fig(fig, "fig10_region_area.png")
    print(perfil[["segmento", "hogares", "pct_poblacion", "ing_pc_mediana"]].round(1).to_string())


if __name__ == "__main__":
    main()
