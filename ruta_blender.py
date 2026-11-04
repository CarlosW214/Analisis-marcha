"""Localiza blender.exe y recuerda la ruta elegida por el usuario (una por computadora)."""
import os
import re
import shutil
from glob import glob
from pathlib import Path

# Se guarda junto al entorno del programa, no en el repositorio: cada computadora tiene su ruta.
GUARDADA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AnalisisMarcha" / "blender.txt"

PATRONES = [
    r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
    r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
    r"C:\Program Files\Steam\steamapps\common\Blender\blender.exe",
]


def _version(p):
    return [int(v) for v in re.findall(r"\d+", Path(p).parent.name)]


def es_valida(ruta):
    return bool(ruta) and Path(ruta).is_file() and Path(ruta).name.lower() == "blender.exe"


def encontrar():
    """Ruta de blender.exe o None: guardada > PATH > instalaciones conocidas (la versión más nueva)."""
    try:
        guardada = GUARDADA.read_text(encoding="utf-8").strip()
        if es_valida(guardada):
            return guardada
    except OSError:
        pass
    if shutil.which("blender"):
        return shutil.which("blender")
    for patron in PATRONES:
        candidatos = sorted(glob(patron), key=_version)
        if candidatos:
            return candidatos[-1]
    return None


def guardar(ruta):
    GUARDADA.parent.mkdir(parents=True, exist_ok=True)
    GUARDADA.write_text(str(ruta), encoding="utf-8")
