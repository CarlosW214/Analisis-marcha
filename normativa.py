"""Marcha ideal contra la que se compara la prueba (patrón normal, media ± DE).

Dos fuentes:
  - Por omisión, datos/normativa_uci.csv: 10 adultos sanos en caminadora
    (Helwig & Hsiao-Wecksler, 2016, UCI Machine Learning Repository, CC BY 4.0).
    La banda es la variabilidad entre sujetos.
  - Una carpeta de resultados de analisis_marcha.py (p. ej. un sujeto sano grabado
    con este mismo sistema y la misma cámara). La banda es la variabilidad entre sus
    ciclos. Tiene la ventaja de usar exactamente las mismas definiciones de ángulo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTICULACIONES = ("cadera", "rodilla", "tobillo")
RUTA_UCI = Path(__file__).resolve().parent / "datos" / "normativa_uci.csv"
CITA_UCI = "Helwig & Hsiao-Wecksler (2016), UCI Multivariate Gait Data, CC BY 4.0"


def cargar(referencia=None):
    """{'nombre', 'detalle', 'curvas': {art: (media[101], de[101])}, 'apoyo_pct': float | None}."""
    if referencia is None:
        tabla = pd.read_csv(RUTA_UCI, comment="#")
        return {
            "nombre": "Normal: adultos sanos (n = 10)",
            "detalle": f"media ± 1 DE entre sujetos · {CITA_UCI}",
            "curvas": {a: (tabla[f"{a}_media"].to_numpy(float), tabla[f"{a}_de"].to_numpy(float))
                       for a in ARTICULACIONES},
            "apoyo_pct": None,
        }
    carpeta = Path(referencia)
    tabla = pd.read_csv(carpeta / "ciclos_normalizados.csv")
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    if resumen.get("vista") != "lateral":
        raise ValueError("La marcha de referencia debe ser un análisis lateral (ángulos sagitales).")
    pierna = resumen["pierna_grabada"]
    info = resumen["por_pierna"][pierna]
    nombre = Path(resumen["fuente"]).name
    return {
        "nombre": f"Referencia: {nombre}",
        "detalle": f"media ± 1 DE de {info['ciclos']} ciclos, pierna {pierna}",
        "curvas": {a: (tabla[f"{pierna}_angulo_{a}_media"].to_numpy(float),
                       tabla[f"{pierna}_angulo_{a}_de"].to_numpy(float)) for a in ARTICULACIONES},
        "apoyo_pct": info.get("apoyo_pct"),
    }


def metricas(media, de_sujeto, normal_media, normal_de):
    """Comparación de una curva media de la prueba contra la marcha ideal (101 puntos)."""
    dif = media - normal_media
    fuera = np.abs(dif) > normal_de
    return {
        "min": float(np.min(media)), "max": float(np.max(media)), "rango": float(np.ptp(media)),
        "normal_min": float(np.min(normal_media)), "normal_max": float(np.max(normal_media)),
        "normal_rango": float(np.ptp(normal_media)),
        "dif_media": float(np.mean(dif)),
        "dif_rms": float(np.sqrt(np.mean(dif ** 2))),
        "correlacion": float(np.corrcoef(media, normal_media)[0, 1]),
        "fuera_banda_pct": float(100.0 * np.mean(fuera)),
        "de_media_prueba": float(np.mean(de_sujeto)),
    }
