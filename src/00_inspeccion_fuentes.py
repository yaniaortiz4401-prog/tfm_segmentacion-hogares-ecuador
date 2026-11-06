"""Paso 0 (opcional). Inspección rápida de las fuentes crudas de la ENIGHUR 2025.

Lee solo la cabecera y cuenta filas en modo streaming (los archivos de gasto
tienen más de 3 500 columnas y 250 MB; cargarlos con pandas agota la memoria).
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg


def inspeccionar(ruta):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(ruta, encoding=enc, newline="") as f:
                primera = f.readline()
                sep = ";" if primera.count(";") >= primera.count(",") else ","
                f.seek(0)
                lector = csv.reader(f, delimiter=sep)
                cab = next(lector)
                n = sum(1 for _ in lector)
            return enc, len(cab), n
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {ruta}")


if __name__ == "__main__":
    for ruta in sorted(Path(cfg.DATA_RAW).glob("*.csv")):
        enc, nc, nf = inspeccionar(ruta)
        print(f"{ruta.name:45s} codificación={enc:10s} columnas={nc:5d} filas={nf:7d} tamaño={ruta.stat().st_size/1e6:7.1f} MB")
