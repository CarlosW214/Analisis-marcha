"""Interfaz de verificación en tiempo real: marcadores sobre el video y panel de ángulos.

Solo se usa texto ASCII porque las fuentes de OpenCV no tienen acentos.
"""
from collections import deque

import cv2
import numpy as np

from deteccion import COLOR_DIBUJO

COLOR_PIERNA_BGR = {"derecha": (214, 120, 42), "izquierda": (52, 104, 235)}
FONDO_PANEL = (24, 24, 24)
TEXTO = (235, 235, 235)
TEXTO_2 = (150, 150, 150)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


SEGMENTOS = (("hombro", "cadera"), ("cadera", "rodilla"), ("rodilla", "tobillo"),
             ("tobillo", "talon"), ("talon", "punta"), ("punta", "tobillo"))


def dibujar_pierna(fotograma, det, color_linea=(255, 255, 255)):
    """Círculos en las articulaciones detectadas y segmentos tronco / muslo / pierna / pie.

    Hombro, talón y punta solo existen sin marcadores; se dibujan más pequeños.
    """
    puntos = {a: (int(d[0]), int(d[1])) for a, d in det.items() if d is not None}
    for a, b in SEGMENTOS:
        if a in puntos and b in puntos:
            cv2.line(fotograma, puntos[a], puntos[b], color_linea, 6 if a != "hombro" else 3, cv2.LINE_AA)
    for art, p in puntos.items():
        if art in COLOR_DIBUJO:
            cv2.circle(fotograma, p, 22, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(fotograma, p, 16, COLOR_DIBUJO[art], -1, cv2.LINE_AA)
        else:
            cv2.circle(fotograma, p, 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(fotograma, p, 7, color_linea, -1, cv2.LINE_AA)
    return fotograma


class Interfaz:
    """Panel lateral con los valores actuales y su historia reciente (ventana deslizante)."""

    def __init__(self, vista, fps, segundos=4.0, ancho=470, dos_piernas=False):
        self.vista = vista
        self.ancho = ancho
        self.n = max(2, int(round(segundos * fps)))
        self.segundos = segundos
        der, izq = COLOR_PIERNA_BGR["derecha"], COLOR_PIERNA_BGR["izquierda"]
        if vista == "lateral" and dos_piernas:      # sin marcadores: las dos piernas, tobillo con el pie
            self.graficas = [
                (titulo, rango, [(f"derecha_{art}", "Der", der), (f"izquierda_{art}", "Izq", izq)])
                for titulo, rango, art in (("Cadera flex(+)/ext(-)", (-30, 40), "cadera"),
                                           ("Rodilla flexion", (-10, 80), "rodilla"),
                                           ("Tobillo dorsi(+)/planti(-)", (-40, 30), "tobillo"))
            ]
        elif vista == "frontal":
            # (título, rango, [(clave, etiqueta, color)])
            self.graficas = [
                ("Cadera abd(+)/add(-)", (-20, 20), [("derecha_cadera", "Der", COLOR_PIERNA_BGR["derecha"]),
                                                    ("izquierda_cadera", "Izq", COLOR_PIERNA_BGR["izquierda"])]),
                ("Rodilla varo(+)/valgo(-)", (-20, 20), [("derecha_rodilla", "Der", COLOR_PIERNA_BGR["derecha"]),
                                                        ("izquierda_rodilla", "Izq", COLOR_PIERNA_BGR["izquierda"])]),
            ]
        else:
            self.graficas = [
                ("Cadera flex(+)/ext(-)", (-30, 40), [("cadera", "", COLOR_DIBUJO["cadera"])]),
                ("Rodilla flexion", (-10, 80), [("rodilla", "", COLOR_DIBUJO["rodilla"])]),
                ("Tobillo dorsi(+)/planti(-)", (-30, 70), [("tobillo", "", COLOR_DIBUJO["tobillo"])]),
            ]
        self.hist = {clave: deque(maxlen=self.n) for _, _, series in self.graficas for clave, _, _ in series}

    def agregar(self, valores):
        for clave, h in self.hist.items():
            v = valores.get(clave)
            h.append(np.nan if v is None else v)

    def componer(self, vista_video, t, indice, valores, estado, grabando=False):
        """Regresa la imagen [video | panel] lista para mostrarse o guardarse."""
        alto = vista_video.shape[0]
        panel = np.full((alto, self.ancho, 3), FONDO_PANEL, np.uint8)
        cv2.putText(panel, "Analisis de marcha", (18, 38), FUENTE, 0.9, TEXTO, 2, cv2.LINE_AA)
        cv2.putText(panel, f"vista {self.vista}   t = {t:6.2f} s   cuadro {indice}", (18, 66), FUENTE, 0.5,
                    TEXTO_2, 1, cv2.LINE_AA)
        if grabando:
            cv2.circle(panel, (self.ancho - 70, 32), 9, (40, 40, 230), -1, cv2.LINE_AA)
            cv2.putText(panel, "REC", (self.ancho - 55, 39), FUENTE, 0.6, (40, 40, 230), 2, cv2.LINE_AA)
        y = 84
        bloque = (alto - y - 56) // len(self.graficas)
        for titulo, rango, series in self.graficas:
            self._bloque(panel, y, bloque, titulo, rango, series, valores)
            y += bloque
        cv2.putText(panel, estado, (18, alto - 32), FUENTE, 0.45, TEXTO_2, 1, cv2.LINE_AA)
        cv2.putText(panel, "q: terminar   espacio: pausa", (18, alto - 12), FUENTE, 0.45, TEXTO_2, 1,
                    cv2.LINE_AA)
        return np.hstack([vista_video, panel])

    def _bloque(self, panel, y0, alto, titulo, rango, series, valores):
        x0, ancho = 18, self.ancho - 36
        cv2.putText(panel, titulo, (x0, y0 + 20), FUENTE, 0.55, TEXTO_2, 1, cv2.LINE_AA)
        for k, (clave, etiqueta, color) in enumerate(series):
            v = valores.get(clave)
            txt = f"{etiqueta} {v:6.1f}" if v is not None else f"{etiqueta}   --"
            cv2.putText(panel, txt.strip(), (x0 + k * 200, y0 + 56), FUENTE, 1.0, color, 2, cv2.LINE_AA)
        gy, gh = y0 + 70, alto - 86
        if gh < 20:
            return
        lo, hi = rango
        cv2.rectangle(panel, (x0, gy), (x0 + ancho, gy + gh), (60, 60, 60), 1)

        def ypix(val):
            return gy + gh - (np.clip(val, lo, hi) - lo) / (hi - lo) * gh

        if lo < 0 < hi:
            cv2.line(panel, (x0, int(ypix(0))), (x0 + ancho, int(ypix(0))), (70, 70, 70), 1)
        cv2.putText(panel, f"{hi}", (x0 + 4, gy + 14), FUENTE, 0.4, TEXTO_2, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{lo}", (x0 + 4, gy + gh - 4), FUENTE, 0.4, TEXTO_2, 1, cv2.LINE_AA)
        cv2.putText(panel, f"-{self.segundos:.0f} s", (x0 + 4, gy + gh + 14), FUENTE, 0.4, TEXTO_2, 1,
                    cv2.LINE_AA)
        for clave, _, color in series:
            v = np.array(self.hist[clave], float)
            xs = x0 + ancho - (v.size - 1 - np.arange(v.size)) * ancho / (self.n - 1)
            validos = np.isfinite(v)
            # Se dibuja cada tramo continuo por separado para no unir huecos.
            cortes = np.flatnonzero(np.diff(validos.astype(int)) != 0) + 1
            for tramo in np.split(np.arange(v.size), cortes):
                if tramo.size > 1 and validos[tramo[0]]:
                    pts = np.column_stack([xs[tramo], ypix(v[tramo])]).astype(np.int32)
                    cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)
