"""Video que muestra cómo se miden las alturas en la vista frontal (video_alturas.mp4).

    py video_alturas.py resultados/Marcha_frontal_ejemplo_sin_marcadores

Sobre cada cuadro del video original dibuja:
  - el "suelo" de ese cuadro: una línea en el tobillo apoyado (el más bajo de la imagen);
  - una línea vertical del suelo a cada articulación, con el cambio de altura respecto a la
    postura de pie (en m), y una marca blanca donde estaba esa articulación de pie;
  - el tronco (hombros a caderas), que solo sirve para seguir la escala;
  - a la derecha, la tabla de valores, la escala del cuadro y los contactos iniciales.

Se genera solo al terminar un análisis frontal, porque la postura de pie y la escala se
conocen hasta haber visto todo el video. Solo usa texto ASCII (OpenCV no tiene acentos).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from deteccion import ARTICULACIONES, COLOR_DIBUJO, PIERNAS, abrir_fuente, leer_fotogramas
from interfaz import COLOR_PIERNA_BGR, FONDO_PANEL, FUENTE, TEXTO, TEXTO_2

ALTO_SALIDA = 720
ANCHO_PANEL = 430
SUELO = (0, 215, 255)
BLANCO = (255, 255, 255)
GRIS = (170, 170, 170)
NOMBRE = {"cadera": "Cadera", "rodilla": "Rodilla", "tobillo": "Tobillo"}


def _video_de(carpeta, resumen):
    for ruta in (resumen.get("ruta_video"), carpeta / "captura.mp4"):
        if ruta and Path(ruta).is_file():
            return str(ruta)
    return None


def _texto(img, txt, xy, escala, color, grosor=1, fondo=True):
    """Texto con un contorno oscuro para que se lea sobre cualquier fondo."""
    if fondo:
        cv2.putText(img, txt, xy, FUENTE, escala, (20, 20, 20), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, txt, xy, FUENTE, escala, color, grosor, cv2.LINE_AA)


def _punteada(img, p0, p1, color, grosor, paso=10):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = max(1, int(np.linalg.norm(p1 - p0) / paso))
    for k in range(0, n, 2):
        a, b = p0 + (p1 - p0) * k / n, p0 + (p1 - p0) * min(k + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, grosor, cv2.LINE_AA)


def _dibujar_cuadro(img, fila, hombros, de_pie, laterales, escala_px):
    """Dibuja en `img` (resolución original) las mediciones de un cuadro."""
    g = escala_px                                        # grosores y tamaños según el alto del video
    m_por_px = fila["escala_m_por_px"]
    y_suelo = max(fila[f"{p}_tobillo_y"] for p in PIERNAS)
    xs = [fila[f"{p}_{a}_x"] for p in PIERNAS for a in ARTICULACIONES]
    apoyado = max(PIERNAS, key=lambda p: fila[f"{p}_tobillo_y"])

    # Tronco: solo da la escala (sigue el tamaño de la persona si se acerca o se aleja).
    if hombros is not None:
        hx, hy = hombros
        cx = 0.5 * (fila["derecha_cadera_x"] + fila["izquierda_cadera_x"])
        cy = 0.5 * (fila["derecha_cadera_y"] + fila["izquierda_cadera_y"])
        _punteada(img, (hx, hy), (cx, cy), GRIS, max(1, g // 2), paso=4 * g)
        _texto(img, "tronco: escala", (int(hx + 6 * g), int(hy + 4 * g)), 0.18 * g, GRIS, max(1, g // 3))

    # Suelo del cuadro: el tobillo apoyado.
    x0, x1 = int(min(xs) - 45 * g), int(max(xs) + 45 * g)
    cv2.line(img, (x0, int(y_suelo)), (x1, int(y_suelo)), SUELO, max(2, g), cv2.LINE_AA)
    _texto(img, "suelo", (x1 + 4 * g, int(y_suelo + 4 * g)), 0.28 * g, SUELO, max(1, g // 2))

    for pierna in PIERNAS:
        afuera = laterales[pierna]
        borde = (min if afuera < 0 else max)(fila[f"{pierna}_{a}_x"] for a in ARTICULACIONES)
        for k, art in enumerate(ARTICULACIONES):
            xj, yj = fila[f"{pierna}_{art}_x"], fila[f"{pierna}_{art}_y"]
            xl = borde + afuera * (14 + 11 * k) * g          # una columna por articulación, hacia afuera
            color = COLOR_DIBUJO[art]
            _punteada(img, (xj, yj), (xl, yj), color, max(1, g // 2), paso=3 * g)
            cv2.line(img, (int(xl), int(y_suelo)), (int(xl), int(yj)), color, max(2, g), cv2.LINE_AA)
            # Dónde estaba esta articulación de pie (misma escala del cuadro).
            ref = de_pie[pierna].get(f"altura_suelo_{art}")
            if ref is not None:
                yr = y_suelo - ref / m_por_px
                cv2.line(img, (int(xl - 5 * g), int(yr)), (int(xl + 5 * g), int(yr)), BLANCO, max(2, g // 2 + 1),
                         cv2.LINE_AA)
            if art == "tobillo" and pierna == apoyado:
                continue                                  # el apoyado es el suelo: 0 por definición
            cambio = fila[f"{pierna}_{art}_cambio_altura_m"]
            txt = f"{cambio:+.3f}"
            (w, _), _ = cv2.getTextSize(txt, FUENTE, 0.25 * g, max(1, g // 2))
            xt = xl + afuera * 4 * g - (w if afuera < 0 else 0)
            _texto(img, txt, (int(xt), int(yj - 3 * g)), 0.25 * g, color, max(1, g // 2))
        for art in ARTICULACIONES:
            p = (int(fila[f"{pierna}_{art}_x"]), int(fila[f"{pierna}_{art}_y"]))
            cv2.circle(img, p, 5 * g, BLANCO, -1, cv2.LINE_AA)
            cv2.circle(img, p, 4 * g, COLOR_DIBUJO[art], -1, cv2.LINE_AA)
    return apoyado


def _panel(alto, t, fila, resumen, de_pie_ahora, apoyado, contacto):
    panel = np.full((alto, ANCHO_PANEL, 3), FONDO_PANEL, np.uint8)
    y = 36
    cv2.putText(panel, "Como se miden las alturas", (16, y), FUENTE, 0.75, TEXTO, 2, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, f"t = {t:6.2f} s   cuadro {int(fila['fotograma'])}", (16, y), FUENTE, 0.5, TEXTO_2, 1,
                cv2.LINE_AA)
    y += 34
    if de_pie_ahora:
        cv2.rectangle(panel, (12, y - 22), (ANCHO_PANEL - 12, y + 10), (60, 60, 60), -1)
        cv2.putText(panel, "DE PIE: referencia (cambio = 0)", (20, y), FUENTE, 0.6, BLANCO, 2, cv2.LINE_AA)
    elif contacto:
        cv2.rectangle(panel, (12, y - 22), (ANCHO_PANEL - 12, y + 10), COLOR_PIERNA_BGR[contacto], -1)
        cv2.putText(panel, f"Contacto inicial {contacto}", (20, y), FUENTE, 0.6, BLANCO, 2, cv2.LINE_AA)
    elif resumen.get("reposo_s") is None:
        cv2.rectangle(panel, (12, y - 22), (ANCHO_PANEL - 12, y + 10), (30, 90, 160), -1)
        cv2.putText(panel, "De pie ESTIMADO (no empieza quieta)", (20, y), FUENTE, 0.55, BLANCO, 1, cv2.LINE_AA)
    y += 40

    cv2.putText(panel, "Cambio de altura vs. de pie [m]", (16, y), FUENTE, 0.55, TEXTO_2, 1, cv2.LINE_AA)
    y += 30
    cv2.putText(panel, "Der", (170, y), FUENTE, 0.6, COLOR_PIERNA_BGR["derecha"], 2, cv2.LINE_AA)
    cv2.putText(panel, "Izq", (300, y), FUENTE, 0.6, COLOR_PIERNA_BGR["izquierda"], 2, cv2.LINE_AA)
    for art in ARTICULACIONES:
        y += 32
        cv2.circle(panel, (24, y - 6), 8, COLOR_DIBUJO[art], -1, cv2.LINE_AA)
        cv2.putText(panel, NOMBRE[art], (40, y), FUENTE, 0.6, TEXTO, 1, cv2.LINE_AA)
        for x, p in ((150, "derecha"), (280, "izquierda")):
            v = fila[f"{p}_{art}_cambio_altura_m"]
            cv2.putText(panel, f"{v:+.3f}", (x, y), FUENTE, 0.65, COLOR_PIERNA_BGR[p], 2, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, f"Pie apoyado (suelo): {apoyado}", (16, y), FUENTE, 0.5, SUELO, 1, cv2.LINE_AA)

    y += 40
    cv2.putText(panel, "Altura sobre el pie apoyado [m]", (16, y), FUENTE, 0.55, TEXTO_2, 1, cv2.LINE_AA)
    for art in ("cadera", "rodilla"):
        y += 28
        cv2.putText(panel, NOMBRE[art], (40, y), FUENTE, 0.55, TEXTO, 1, cv2.LINE_AA)
        for x, p in ((150, "derecha"), (280, "izquierda")):
            v = fila[f"{p}_{art}_altura_sobre_pie_apoyado_m"]
            cv2.putText(panel, f"{v:.3f}", (x, y), FUENTE, 0.55, COLOR_PIERNA_BGR[p], 1, cv2.LINE_AA)

    y += 40
    cv2.putText(panel, "Escala", (16, y), FUENTE, 0.55, TEXTO_2, 1, cv2.LINE_AA)
    lineas = [
        f"estatura {resumen['estatura_m']:g} m -> cadera-tobillo",
        f"de pie = 0.491 x estatura = {0.491 * resumen['estatura_m']:.3f} m",
        f"este cuadro: {1000 * fila['escala_m_por_px']:.2f} mm por pixel",
        "(el tronco sigue si se acerca o aleja)",
    ]
    for txt in lineas:
        y += 24
        cv2.putText(panel, txt, (16, y), FUENTE, 0.48, TEXTO, 1, cv2.LINE_AA)

    leyenda = [
        (SUELO, "linea amarilla: suelo del cuadro"),
        (TEXTO, "linea vertical: altura de la articulacion"),
        (BLANCO, "marca blanca: donde estaba de pie"),
    ]
    y = alto - 22 - 24 * (len(leyenda) - 1)
    for color, txt in leyenda:
        cv2.putText(panel, txt, (16, y), FUENTE, 0.45, color, 1, cv2.LINE_AA)
        y += 24
    return panel


def generar(carpeta):
    """Crea video_alturas.mp4 en la carpeta de un análisis frontal. Regresa la ruta o None."""
    carpeta = Path(carpeta)
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    if resumen.get("vista") != "frontal" or not resumen.get("altura_relativa"):
        return None
    video = _video_de(carpeta, resumen)
    if video is None:
        print("Aviso: no se encontró el video original; no se genera video_alturas.mp4.")
        return None
    datos = pd.read_csv(carpeta / "datos_marcha.csv")
    por_cuadro = {int(f): fila for f, fila in zip(datos["fotograma"], datos.to_dict("records"))}
    crudo = pd.read_csv(carpeta / "detecciones_crudas.csv")
    hombros = {}
    if "derecha_hombro_x" in crudo:
        for f, fila in zip(crudo["fotograma"].round().astype(int), crudo.to_dict("records")):
            hx = np.nanmean([fila["derecha_hombro_x"], fila["izquierda_hombro_x"]])
            hy = np.nanmean([fila["derecha_hombro_y"], fila["izquierda_hombro_y"]])
            if np.isfinite(hx) and np.isfinite(hy):
                hombros[f] = (hx, hy)
    ciclos = pd.read_csv(carpeta / "ciclos.csv")
    contactos = [(r.contacto_inicial_s, r.pierna) for r in ciclos.itertuples()]
    contactos += [(r.contacto_siguiente_s, r.pierna) for r in ciclos.itertuples()]
    reposo = resumen.get("reposo_s")
    s = -1 if resumen.get("frente", "camara") == "camara" else 1
    laterales = {"derecha": s, "izquierda": -s}

    cap, fps, grados, es_camara = abrir_fuente(video, resumen.get("rotacion", "auto"))
    primero, ultimo = min(por_cuadro), max(por_cuadro)
    ruta, escritor = carpeta / "video_alturas.mp4", None
    total = ultimo - primero + 1
    for indice, t, img in leer_fotogramas(cap, grados, fps, es_camara, primero, ultimo + 1):
        fila = por_cuadro.get(indice)
        if fila is None:
            continue
        g = max(1, round(img.shape[0] / 360))
        apoyado = _dibujar_cuadro(img, fila, hombros.get(indice), resumen["de_pie_m"], laterales, g)
        f = ALTO_SALIDA / img.shape[0]
        img = cv2.resize(img, (int(round(img.shape[1] * f)) // 2 * 2, ALTO_SALIDA), interpolation=cv2.INTER_AREA)
        tt = fila["tiempo_s"]
        de_pie_ahora = reposo is not None and reposo[0] <= tt <= reposo[1]
        contacto = next((p for tc, p in contactos if abs(tt - tc) <= 0.1), None)
        cuadro = np.hstack([img, _panel(ALTO_SALIDA, tt, fila, resumen, de_pie_ahora, apoyado, contacto)])
        if escritor is None:
            escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, cuadro.shape[1::-1])
        escritor.write(cuadro)
        hechos = indice - primero + 1
        if hechos % 100 == 0:
            print(f"Procesando video de alturas: cuadro {hechos} de {total} ({100 * hechos / total:.0f} %)",
                  flush=True)
    cap.release()
    if escritor is None:
        return None
    escritor.release()
    return ruta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", type=Path, help="carpeta de resultados de un análisis frontal")
    ruta = generar(p.parse_args().carpeta)
    print(f"Video de alturas: {ruta}" if ruta else "No se generó el video (¿es un análisis frontal?).")


if __name__ == "__main__":
    main()
