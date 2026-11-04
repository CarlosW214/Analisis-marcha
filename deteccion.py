"""Detección de los marcadores de color de cadera, rodilla y tobillo.

Equivalencias con el código MATLAB del proyecto:
  - segmentar_rgb_matlab()  <- reglas if/elseif de pp3.m y proyecto2.m
  - DetectorMarcadores      <- bwlabel + regionprops('Centroid') de prueba1.m,
                               más seguimiento por proximidad entre cuadros
  - leer_fotogramas()       <- VideoReader / readFrame (con la rotación del
                               video aplicada, como lo hace MATLAB)

Vista lateral: un marcador de cada color (la pierna del lado de la cámara).
Vista frontal: dos marcadores de cada color, uno por pierna.
"""
import json
import time
from itertools import permutations

import cv2
import numpy as np

ARTICULACIONES = ("cadera", "rodilla", "tobillo")
PIERNAS = ("derecha", "izquierda")

# Color con que se dibuja cada articulación (BGR), igual al de su marcador.
COLOR_DIBUJO = {"cadera": (255, 120, 0), "rodilla": (60, 20, 230), "tobillo": (40, 200, 40)}

# Umbrales HSV de OpenCV (H: 0-179, S y V: 0-255) medidos en 21.MP4 y 23.MP4:
#   cadera  = marcador azul   H~108 S~205 V~215
#   rodilla = marcador rojo   H~172 S~205 V~195  (el rojo del tenis está en H~178)
#   tobillo = marcador verde  H~76  S~210 V~160
# Con otra iluminación se recalibran con calibrar_hsv.py.
UMBRALES_HSV = {
    "cadera": [((100, 150, 120), (118, 255, 255))],
    "rodilla": [((164, 150, 110), (176, 255, 255))],
    "tobillo": [((65, 140, 90), (88, 255, 255))],
}

_ROTACIONES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def cargar_umbrales(ruta):
    """Lee un JSON {articulación: [[[h, s, v], [h, s, v]], ...]} (lo genera calibrar_hsv.py)."""
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return {art: [(tuple(bajo), tuple(alto)) for bajo, alto in datos[art]] for art in ARTICULACIONES}


def guardar_umbrales(umbrales, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({a: [[list(b), list(c)] for b, c in r] for a, r in umbrales.items()}, f, indent=2)


def segmentar_rgb_matlab(bgr):
    """Reglas RGB originales de pp3.m, vectorizadas (el orden if/elseif se respeta).

    Ojo: los umbrales se ajustaron con el decodificador de MATLAB; OpenCV
    convierte YUV->RGB de forma un poco distinta y el marcador rojo sale
    incompleto. Se conserva para comparar contra el código original.
    """
    b, g, r = cv2.split(bgr)
    rodilla = (r >= 200) & (b <= 97) & (g <= 63)
    cadera = (b >= 150) & (r <= 47) & ~rodilla
    tobillo = (g >= 120) & (r <= 28) & ~rodilla & ~cadera
    return {
        "cadera": cadera.astype(np.uint8) * 255,
        "rodilla": rodilla.astype(np.uint8) * 255,
        "tobillo": tobillo.astype(np.uint8) * 255,
    }


def segmentar_hsv(bgr, umbrales=UMBRALES_HSV):
    """Umbralización HSV + apertura/cierre morfológico para limpiar cada máscara."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascaras = {}
    for nombre, rangos in umbrales.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for bajo, alto in rangos:
            m |= cv2.inRange(hsv, bajo, alto)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        mascaras[nombre] = m
    return mascaras


class DetectorMarcadores:
    """Obtiene el centroide de cada marcador en un fotograma.

    De cada máscara se toman las regiones conexas (bwlabel). En vista lateral
    se conserva la más grande; en vista frontal, las dos más grandes, una por
    pierna. Si hay varias candidatas se elige por cercanía a la posición del
    cuadro anterior (seguimiento por proximidad).

    frente: en vista frontal, "camara" si el sujeto camina hacia la cámara
    (su pierna derecha aparece a la izquierda de la imagen) o "espalda".
    """

    def __init__(self, modo="hsv", escala=0.5, area_min=400, umbrales=None, vista="lateral",
                 frente="camara"):
        self.modo = modo
        self.escala = escala          # se procesa una copia reducida para ir más rápido
        self.area_min = area_min      # en px de la imagen original
        self.umbrales = umbrales or UMBRALES_HSV
        self.vista = vista
        self.frente = frente
        self.previo = {}

    def mascaras(self, bgr):
        if self.escala != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.escala, fy=self.escala, interpolation=cv2.INTER_AREA)
        if self.modo == "rgb":
            return segmentar_rgb_matlab(bgr)
        return segmentar_hsv(bgr, self.umbrales)

    def detectar(self, bgr):
        """Lateral: {articulación: (x, y, área) | None}.
        Frontal: {pierna: {articulación: (x, y, área) | None}}. Coordenadas en px originales."""
        mascaras = self.mascaras(bgr)
        if self.vista == "frontal":
            ancho = bgr.shape[1] * self.escala
            res = {p: {} for p in PIERNAS}
            for art, m in mascaras.items():
                for pierna, det in self._asignar_piernas(art, self._regiones(m, 2), ancho).items():
                    res[pierna][art] = self._a_original((pierna, art), det)
            return res
        res = {}
        for art, m in mascaras.items():
            regiones = self._regiones(m, None)
            det = None
            if regiones:
                prev = self.previo.get(art)
                if prev is not None and len(regiones) > 1:
                    det = min(regiones, key=lambda r: np.hypot(r[0] - prev[0], r[1] - prev[1]))
                else:
                    det = regiones[0]
            res[art] = self._a_original(art, det)
        return res

    def _a_original(self, clave, det):
        self.previo[clave] = None if det is None else det[:2]
        if det is None:
            return None
        x, y, area = det
        return x / self.escala, y / self.escala, area / self.escala ** 2

    def _regiones(self, mascara, maximo):
        """Regiones (x, y, área) ordenadas de mayor a menor que superan el área mínima.

        Lateral: solo las de tamaño parecido a la mayor (>= 50 %), para no
        confundir el marcador con manchas pequeñas del mismo color.
        """
        n, _, stats, centros = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        if n <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA]
        relativo = 0.3 if maximo else 0.5
        minimo = max(self.area_min * self.escala ** 2, relativo * areas.max())
        orden = [j for j in np.argsort(-areas) if areas[j] >= minimo][:maximo]
        return [(centros[j + 1][0], centros[j + 1][1], float(areas[j])) for j in orden]

    def _asignar_piernas(self, art, regiones, ancho):
        previos = [self.previo.get((p, art)) for p in PIERNAS]
        asignacion = dict.fromkeys(PIERNAS)
        if not regiones:
            return asignacion
        if all(p is not None for p in previos):
            # Permutación que minimiza la distancia total a las posiciones anteriores.
            mejor = min(
                permutations(range(len(PIERNAS)), len(regiones)),
                key=lambda perm: sum(np.hypot(regiones[i][0] - previos[k][0], regiones[i][1] - previos[k][1])
                                     for i, k in enumerate(perm)),
            )
            for i, k in enumerate(mejor):
                asignacion[PIERNAS[k]] = regiones[i]
            return asignacion
        # Sin historial: por posición horizontal en la imagen.
        de_frente = self.frente == "camara"
        if len(regiones) == 2:
            izq_img, der_img = sorted(regiones, key=lambda r: r[0])
            asignacion["derecha"], asignacion["izquierda"] = (izq_img, der_img) if de_frente else (der_img, izq_img)
        else:
            r = regiones[0]
            en_izquierda_img = r[0] < ancho / 2
            asignacion["derecha" if en_izquierda_img == de_frente else "izquierda"] = r
        return asignacion


def abrir_fuente(fuente, rotacion="auto"):
    """Abre un archivo de video o una cámara (índice entero) para tiempo real.

    Los MP4 del celular traen la rotación como metadato (270° en 21.MP4 y
    23.MP4); se desactiva la autorrotación de OpenCV y se aplica a mano para
    obtener siempre el video vertical que entrega MATLAB.
    """
    es_camara = str(fuente).isdigit()
    if es_camara:
        cap = cv2.VideoCapture(int(fuente))
    else:
        cap = cv2.VideoCapture(str(fuente))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir la fuente de video: {fuente}")
    if rotacion == "auto":
        grados = 0 if es_camara else int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    else:
        grados = int(rotacion) % 360
    if grados not in _ROTACIONES:
        raise ValueError(f"Rotación no soportada: {grados}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps, grados, es_camara


def rotar(fotograma, grados):
    codigo = _ROTACIONES[grados]
    return fotograma if codigo is None else cv2.rotate(fotograma, codigo)


def leer_fotogramas(cap, grados, fps, es_camara, inicio=0, fin=None):
    """Generador de (índice, tiempo_s, fotograma BGR ya rotado)."""
    indice = 0
    if not es_camara:
        while indice < inicio and cap.grab():
            indice += 1
    t0 = time.perf_counter()
    while fin is None or indice < fin:
        ok, fotograma = cap.read()
        if not ok:
            break
        fotograma = rotar(fotograma, grados)
        t = time.perf_counter() - t0 if es_camara else indice / fps
        yield indice, t, fotograma
        indice += 1
