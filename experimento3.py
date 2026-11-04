"""Experimento 3: visualización 3D de la marcha registrada.

    py experimento3.py resultados/21                   # todo el video
    py experimento3.py resultados/21 --cuadros 1:300   # solo los primeros 10 s en el render

1. Exporta datos_marcha.csv a marcha.bvh (rotaciones + posición de la cadera).
2. Ejecuta Blender en segundo plano (blender_marcha.py): modelo humanoide, validación de los
   ángulos del modelo, marcha_3d.blend / marcha_3d.fbx y render de las cámaras lateral y 3/4.
3. Genera validacion_modelo.png, comparacion_3d.mp4 (video | modelo lateral | modelo 3D) y
   comparacion_3d.png (fases de un ciclo, video contra modelo).
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import exportar_bvh
import graficas
import ruta_blender
from deteccion import abrir_fuente, leer_fotogramas

plt = graficas.plt

AQUI = Path(__file__).resolve().parent
FASES = [(0, "contacto inicial"), (15, "respuesta a carga"), (30, "apoyo medio"), (50, "apoyo terminal"),
         (62, "despegue"), (75, "balanceo medio"), (90, "balanceo final")]


def buscar_blender(ruta=None):
    if ruta:
        if not ruta_blender.es_valida(ruta):
            raise FileNotFoundError(f"No existe blender.exe en {ruta}")
        ruta_blender.guardar(ruta)          # las siguientes veces ya no hace falta --blender
        return str(ruta)
    encontrada = ruta_blender.encontrar()
    if encontrada is None:
        raise FileNotFoundError("No se encontró Blender; indique la ruta con --blender")
    return encontrada


def ejecutar_blender(blender, carpeta, cuadros):
    cmd = [blender, "--background", "--factory-startup", "--python", str(AQUI / "blender_marcha.py"), "--",
           "--carpeta", str(carpeta)]
    if cuadros:
        cmd += ["--render", cuadros]
    salida = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "LISTO" not in salida.stdout:
        print(salida.stdout[-3000:], salida.stderr[-3000:])
        raise RuntimeError("Blender no terminó correctamente")
    return json.loads((carpeta / "validacion_modelo.json").read_text(encoding="utf-8"))


def grafica_validacion(carpeta, meta, val, segundos=6.0):
    """Ángulos calculados del video contra los medidos en el modelo 3D (pierna grabada)."""
    v = pd.read_csv(carpeta / "validacion_modelo.csv")
    pierna = meta["pierna_medida"]
    t = (v["fotograma_video"] - v["fotograma_video"].iloc[0]) / meta["fps"]
    sel = t <= segundos
    fig, ejes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, art in zip(ejes, ("cadera", "rodilla", "tobillo")):
        ax.plot(t[sel], v.loc[sel, f"{pierna}_calculado_{art}"], color=graficas.COLOR_PIERNA[pierna], linewidth=5,
                alpha=0.35, label="Calculado del video")
        ax.plot(t[sel], v.loc[sel, f"{pierna}_modelo_{art}"], color=graficas.TINTA, linewidth=1.2,
                label="Medido en el modelo 3D (Blender)")
        texto = f"error máx. {val['error_angular_max_deg'][f'{pierna}_{art}']:.1e}°"
        if f"{pierna}_{art}" in val["error_posicion_rms_cm"]:
            texto += f" · posición RMS {val['error_posicion_rms_cm'][f'{pierna}_{art}']:.2f} cm"
        ax.set_title(f"{art.capitalize()} · pierna {pierna} · {texto}", loc="left", fontsize=9.5)
        ax.set_ylabel("Ángulo [°]")
    ejes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.12), ncol=2, fontsize=8)
    ejes[-1].set_xlabel("Tiempo [s]")
    fig.suptitle("Validación del modelo 3D: ángulos del esqueleto contra los calculados", x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(carpeta / "validacion_modelo.png", dpi=130)
    plt.close(fig)


def _leer_video(ruta):
    cap = cv2.VideoCapture(str(ruta))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def _rotulo(img, texto, y, escala=0.7, color=(30, 30, 30)):
    cv2.putText(img, texto, (14, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


def ruta_video(resumen, explicita=None):
    candidatos = [explicita, resumen.get("ruta_video"), AQUI / resumen["fuente"], Path(resumen["fuente"])]
    for c in candidatos:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError("No se encontró el video original; indíquelo con --video")


def comparar(carpeta, meta, video_original, cuadro_inicial):
    """Video lado a lado y lámina de fases del ciclo."""
    datos = pd.read_csv(carpeta / "datos_marcha.csv")
    pierna = meta["pierna_medida"]
    lat, per = carpeta / "render_lateral.mp4", carpeta / "render_perspectiva.mp4"
    n = int(cv2.VideoCapture(str(lat)).get(cv2.CAP_PROP_FRAME_COUNT))
    filas = datos.iloc[cuadro_inicial - 1: cuadro_inicial - 1 + n].reset_index(drop=True)
    cap, fps, grados, es_camara = abrir_fuente(video_original)
    video = leer_fotogramas(cap, grados, fps, es_camara, int(filas["fotograma"].iloc[0]),
                            int(filas["fotograma"].iloc[-1]) + 1)

    ciclos = pd.read_csv(carpeta / "ciclos.csv")
    ciclos = ciclos[(ciclos["pierna"] == pierna) & (ciclos["contacto_inicial_s"] >= filas["tiempo_s"].iloc[0])
                    & (ciclos["contacto_siguiente_s"] <= filas["tiempo_s"].iloc[-1])]
    fases = {}
    if len(ciclos):
        c = ciclos.iloc[min(1, len(ciclos) - 1)]
        for pct, nombre in FASES:
            tf = c["contacto_inicial_s"] + pct / 100 * c["duracion_s"]
            fases[int(np.argmin(np.abs(filas["tiempo_s"] - tf)))] = (pct, nombre)

    ancho, alto = 540, 960
    escritor = cv2.VideoWriter(str(carpeta / "comparacion_3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"],
                               (3 * ancho, alto + 110))
    capturas = {}
    for k, (orig, r_lat, r_per) in enumerate(zip(video, _leer_video(lat), _leer_video(per))):
        orig = cv2.resize(orig[2], (ancho, alto), interpolation=cv2.INTER_AREA)
        lienzo = np.full((alto + 110, 3 * ancho, 3), 250, np.uint8)
        lienzo[50:50 + alto] = np.hstack([orig, r_lat, r_per])
        for j, titulo in enumerate(("Video original", "Modelo 3D - vista lateral", "Modelo 3D - perspectiva")):
            cv2.putText(lienzo, titulo, (j * ancho + 14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2,
                        cv2.LINE_AA)
        fila = filas.iloc[k]
        texto = (f"t = {fila['tiempo_s']:5.2f} s   pierna {pierna}:  cadera {fila[f'{pierna}_angulo_cadera']:6.1f}   "
                 f"rodilla {fila[f'{pierna}_angulo_rodilla']:6.1f}   tobillo {fila[f'{pierna}_angulo_tobillo']:6.1f} grados")
        _rotulo(lienzo, texto, alto + 92)
        escritor.write(lienzo)
        if k in fases:
            capturas[k] = (cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), cv2.cvtColor(r_lat, cv2.COLOR_BGR2RGB), fila)
    escritor.release()

    if capturas:
        fig, ejes = plt.subplots(2, len(capturas), figsize=(2.3 * len(capturas), 8.6))
        for col, (k, (img_v, img_m, fila)) in enumerate(sorted(capturas.items())):
            pct, nombre = fases[k]
            ejes[0, col].imshow(img_v)
            ejes[1, col].imshow(img_m)
            ejes[0, col].set_title(f"{pct} % · {nombre}", fontsize=9)
            ejes[1, col].set_xlabel(f"cadera {fila[f'{pierna}_angulo_cadera']:.0f}°\nrodilla "
                                    f"{fila[f'{pierna}_angulo_rodilla']:.0f}°\ntobillo {fila[f'{pierna}_angulo_tobillo']:.0f}°",
                                    fontsize=8.5)
            for ax in ejes[:, col]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
        ejes[0, 0].set_ylabel("Video original")
        ejes[1, 0].set_ylabel("Modelo 3D (Blender)")
        fig.suptitle(f"Un ciclo de marcha · {Path(video_original).name} · pierna {pierna}", x=0.01, ha="left",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(carpeta / "comparacion_3d.png", dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", type=Path, help="carpeta de resultados de analisis_marcha.py (vista lateral)")
    p.add_argument("--cuadros", default="todo", help="cuadros de Blender a renderizar (p. ej. 1:300) o 'todo'")
    p.add_argument("--sin-render", action="store_true", help="solo BVH, .blend, .fbx y validación")
    p.add_argument("--estatura-m", type=float, default=None, help="estatura del sujeto (1.70 m si no hay escala)")
    p.add_argument("--blender", default=None, help="ruta de blender.exe")
    p.add_argument("--video", default=None, help="video original (por omisión, el del análisis)")
    args = p.parse_args()
    carpeta = args.carpeta.resolve()
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    video = None if args.sin_render else ruta_video(resumen, args.video)

    ruta_bvh, meta = exportar_bvh.exportar(carpeta, args.estatura_m)
    print(f"1. BVH: {ruta_bvh} (escala por {meta['metodo_escala']})")
    blender = buscar_blender(args.blender)
    print(f"2. Blender: {blender}")
    val = ejecutar_blender(blender, carpeta, None if args.sin_render else args.cuadros)
    grafica_validacion(carpeta, meta, val)
    print("   Error angular máximo del modelo [°]:", val["error_angular_max_deg"])
    print("   Error de posición RMS [cm]:", val["error_posicion_rms_cm"])
    if not args.sin_render:
        inicial = 1 if args.cuadros == "todo" else int(args.cuadros.split(":")[0])
        comparar(carpeta, meta, video, inicial)
        print(f"3. Comparación: {carpeta / 'comparacion_3d.mp4'}")
    print(f"Resultados en: {carpeta}")


if __name__ == "__main__":
    main()
