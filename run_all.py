"""Ejecuta el pipeline completo en orden: python run_all.py

Si los CSV crudos no están en data/raw/, indique la carpeta con la variable
de entorno TFM_DATA_RAW (por ejemplo: TFM_DATA_RAW=/ruta/a/ENIGHUR python run_all.py).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASOS = ["01_preparar_datos", "02_eda", "03_matriz_clustering", "04_clustering",
         "05_validacion", "05b_caso_base", "06_caracterizacion"]

if __name__ == "__main__":
    t0 = time.time()
    for p in PASOS:
        print(f"\n=== {p} ===", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "src" / f"{p}.py")], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Falló el paso {p}")
    print(f"\nPipeline completo en {time.time() - t0:.0f} s")
