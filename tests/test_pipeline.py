"""Pruebas de integridad del pipeline (requieren haber ejecutado run_all.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import config as cfg


@pytest.fixture(scope="module")
def listo():
    return pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_dataset_listo_clustering.csv")


@pytest.fixture(scope="module")
def clusters():
    return pd.read_csv(cfg.DATA_PROC / "ENIGHUR2025_hogares_con_clusters.csv")


def test_numero_de_hogares(listo):
    assert len(listo) == 35397


def test_identificador_unico(listo):
    assert listo["Identif_hog"].is_unique


def test_participaciones_de_gasto_acotadas(listo):
    s = listo[cfg.PCT_COLS].sum(axis=1)
    assert (s <= 1 + 1e-9).all()
    assert s.median() > 0.98


def test_sin_nulos_en_variables_del_algoritmo(listo):
    assert listo[cfg.FEATURES_CLUSTER + cfg.Z_COLS].isna().sum().sum() == 0


def test_variables_estandarizadas(listo):
    assert np.allclose(listo[cfg.Z_COLS].mean(), 0, atol=1e-9)
    assert np.allclose(listo[cfg.Z_COLS].std(ddof=0), 1, atol=1e-9)


def test_seis_clusters(clusters):
    assert clusters["cluster"].nunique() == cfg.K_FINAL
    assert set(clusters["cluster"]) == set(range(cfg.K_FINAL))


def test_tasa_ahorro_winsorizada_en_rango(listo):
    assert listo["tasa_ahorro_w"].between(-1, 1, inclusive="neither").all()


def test_variables_sensibles_fuera_del_algoritmo():
    for v in cfg.VARS_SENSIBLES:
        assert v not in cfg.FEATURES_CLUSTER
        assert f"z_{v}" not in cfg.Z_COLS
