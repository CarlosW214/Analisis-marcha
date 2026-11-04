"""Experimento 3, paso 1: exporta la marcha registrada a BVH (rotaciones + posiciones).

    py exportar_bvh.py resultados/21 [--estatura-m 1.70]

Esqueleto humanoide estándar (Hips, Spine, Chest, Neck, Head, brazos y piernas),
en metros, con Y hacia arriba, el sujeto caminando hacia +Z y su izquierda en +X:
  - Hips (raíz): posición medida del marcador de cadera (altura y avance).
  - Piernas: flexión sagital (rotación X) de cadera, rodilla y tobillo de cada pierna.
  - Tronco y brazos: postura fija, sujetando los barandales (no se midieron).
Las longitudes de muslo y pierna son las medianas medidas entre marcadores; el
resto del cuerpo sigue las proporciones antropométricas de Winter (2009).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANALES_RAIZ = "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
CANALES = "CHANNELS 3 Zrotation Xrotation Yrotation"
POSTURA_BRAZOS = {"Arm": -50.0, "ForeArm": -40.0}   # flexión de hombro y codo [°]


def _jerarquia(H, muslo, pierna, torso):
    """(nombre, offset, hijos); los "End" son End Site."""
    tobillo_h, pie = 0.039 * H, 0.152 * H
    cadera_x = 0.052 * H
    hombro_x, brazo, antebrazo, mano = 0.129 * H, 0.186 * H, 0.146 * H, 0.108 * H

    def extremidad_inferior(lado, s):
        return (f"{lado}UpLeg", (s * cadera_x, 0.0, 0.0), [
            (f"{lado}Leg", (0.0, -muslo, 0.0), [
                (f"{lado}Foot", (0.0, -pierna, 0.0), [("End", (0.0, -tobillo_h, 0.75 * pie), [])])])])

    def extremidad_superior(lado, s):
        return (f"{lado}Shoulder", (s * 0.02, 0.35 * torso - 0.02, 0.0), [
            (f"{lado}Arm", (s * (hombro_x - 0.02), 0.0, 0.0), [
                (f"{lado}ForeArm", (0.0, -brazo, 0.0), [
                    (f"{lado}Hand", (0.0, -antebrazo, 0.0), [("End", (0.0, -mano, 0.0), [])])])])])

    return ("Hips", (0.0, 0.0, 0.0), [
        ("Spine", (0.0, 0.30 * torso, 0.0), [
            ("Chest", (0.0, 0.35 * torso, 0.0), [
                ("Neck", (0.0, 0.35 * torso, 0.0), [
                    ("Head", (0.0, 0.052 * H, 0.0), [("End", (0.0, 0.13 * H, 0.0), [])])]),
                extremidad_superior("Left", 1), extremidad_superior("Right", -1)])]),
        extremidad_inferior("Right", -1), extremidad_inferior("Left", 1)])


def _escribir_jerarquia(nodo, nivel, lineas, orden):
    nombre, offset, hijos = nodo
    sangria = "  " * nivel
    off = "OFFSET " + " ".join(f"{v:.6f}" for v in offset)
    if nombre == "End":
        lineas += [f"{sangria}End Site", f"{sangria}{{", f"{sangria}  {off}", f"{sangria}}}"]
        return
    lineas += [f"{sangria}{'ROOT' if nivel == 0 else 'JOINT'} {nombre}", f"{sangria}{{", f"{sangria}  {off}",
               f"{sangria}  {CANALES_RAIZ if nivel == 0 else CANALES}"]
    orden.append(nombre)
    for h in hijos:
        _escribir_jerarquia(h, nivel + 1, lineas, orden)
    lineas.append(f"{sangria}}}")


def exportar(carpeta, estatura_m=None):
    carpeta = Path(carpeta)
    datos = pd.read_csv(carpeta / "datos_marcha.csv")
    with open(carpeta / "resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)
    if resumen.get("vista") != "lateral":
        raise ValueError("El BVH se genera a partir de un análisis lateral (ángulos sagitales).")

    medida = resumen["pierna_grabada"]
    escala = resumen["escala"]
    if escala.get("m_por_px"):
        # Escala del análisis; la estatura solo define las proporciones del tronco y brazos.
        m_por_px, metodo = escala["m_por_px"], escala["metodo"]
        H = estatura_m or m_por_px * escala["pierna_px"] / 0.246
    else:
        H = estatura_m or 1.70
        m_por_px = 0.246 * H / escala["pierna_px"]
        metodo = f"estatura {'indicada' if estatura_m else 'supuesta'} de {H:.2f} m (pierna = 0.246 H)"

    xy = {a: datos[[f"{medida}_{a}_x", f"{medida}_{a}_y"]].to_numpy() for a in ("cadera", "rodilla", "tobillo")}
    muslo = float(np.median(np.linalg.norm(xy["rodilla"] - xy["cadera"], axis=1))) * m_por_px
    pierna = float(np.median(np.linalg.norm(xy["tobillo"] - xy["rodilla"], axis=1))) * m_por_px
    direccion = 1 if resumen["direccion_marcha"] == "derecha" else -1
    tobillo_h = 0.039 * H
    y_suelo = float(xy["tobillo"][:, 1].max())
    x_ref = float(xy["cadera"][:, 0].mean())
    raiz_y = (y_suelo - xy["cadera"][:, 1]) * m_por_px + tobillo_h
    raiz_z = direccion * (xy["cadera"][:, 0] - x_ref) * m_por_px
    torso = max(0.25, 0.818 * H - float(raiz_y.mean()))

    lineas, orden = ["HIERARCHY"], []
    _escribir_jerarquia(_jerarquia(H, muslo, pierna, torso), 0, lineas, orden)
    n = len(datos)
    lineas += ["MOTION", f"Frames: {n}", f"Frame Time: {1.0 / resumen['fps']:.7f}"]

    # Rotación X de cada hueso (BVH, regla de la mano derecha): flexión de cadera = -X,
    # flexión de rodilla = +X, dorsiflexión de tobillo = -X.
    rot_x = {}
    for lado, pierna_csv in (("Right", "derecha"), ("Left", "izquierda")):
        rot_x[f"{lado}UpLeg"] = -datos[f"{pierna_csv}_angulo_cadera"].to_numpy()
        rot_x[f"{lado}Leg"] = datos[f"{pierna_csv}_angulo_rodilla"].to_numpy()
        rot_x[f"{lado}Foot"] = -datos[f"{pierna_csv}_angulo_tobillo"].to_numpy()
        for hueso, grados in POSTURA_BRAZOS.items():
            rot_x[f"{lado}{hueso}"] = np.full(n, grados)
    for k in range(n):
        valores = []
        for nombre in orden:
            if nombre == "Hips":
                valores += [0.0, raiz_y[k], raiz_z[k], 0.0, 0.0, 0.0]
            else:
                valores += [0.0, rot_x[nombre][k] if nombre in rot_x else 0.0, 0.0]
        lineas.append(" ".join(f"{v:.5f}" for v in valores))

    ruta_bvh = carpeta / "marcha.bvh"
    ruta_bvh.write_text("\n".join(lineas) + "\n", encoding="ascii")
    meta = {
        "estatura_m": H, "m_por_px": m_por_px, "metodo_escala": metodo,
        "muslo_m": muslo, "pierna_m": pierna, "tobillo_h_m": tobillo_h, "pie_m": 0.152 * H,
        "cadera_x_m": 0.052 * H, "pierna_medida": medida, "direccion": direccion,
        "y_suelo_px": y_suelo, "x_ref_px": x_ref, "fps": resumen["fps"], "cuadros": n,
        "fotograma_inicial": int(datos["fotograma"].iloc[0]),
    }
    (carpeta / "marcha_bvh.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta_bvh, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", type=Path, help="carpeta de resultados de analisis_marcha.py (vista lateral)")
    p.add_argument("--estatura-m", type=float, default=None,
                   help="estatura del sujeto (por omisión 1.70 m si el análisis no tiene escala)")
    args = p.parse_args()
    ruta, meta = exportar(args.carpeta, args.estatura_m)
    print(f"BVH: {ruta}  ({meta['cuadros']} cuadros, muslo {meta['muslo_m']:.3f} m, pierna {meta['pierna_m']:.3f} m,"
          f" escala por {meta['metodo_escala']})")


if __name__ == "__main__":
    main()
