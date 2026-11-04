"""Detección de articulaciones sin marcadores (Experimento 1, opción 1 de la práctica).

Modelos preentrenados de estimación de pose humana en 2D:

  - "rtmpose"   RTMPose-m con 26 puntos (Halpe26, incluye talón y dedo gordo), de
                OpenMMLab, ejecutado con ONNX Runtime mediante rtmlib. Es el modelo
                por omisión: en 21.MP4 sus ángulos de la pierna cercana a la cámara
                coinciden con los de los marcadores (r = 0.99).
  - "mediapipe" MediaPipe Pose Landmarker heavy (Google), 33 puntos. Más rápido, pero
                en vista lateral confunde la pierna cercana con la lejana cuando se
                cruzan (rodilla con error de ~15° hacia el 30 % del ciclo).

Ambos entregan, por cuadro y por pierna, la posición (x, y) en px de la imagen
original y la confianza (0-1) de: hombro, cadera, rodilla, tobillo, talón y punta
del pie. Con el talón y la punta se mide el pie, así que el ángulo del tobillo ya no
supone el pie horizontal como con los marcadores.

Los modelos se descargan la primera vez (RTMPose en ~/.cache/rtmlib, MediaPipe en
modelos/); después funcionan sin internet.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PIERNAS = ("derecha", "izquierda")
PUNTOS = ("hombro", "cadera", "rodilla", "tobillo", "talon", "punta")

# Índice de cada punto en la salida de cada modelo, en el orden de PUNTOS.
_INDICES = {
    "rtmpose": {"derecha": (6, 12, 14, 16, 25, 21), "izquierda": (5, 11, 13, 15, 24, 20)},
    "mediapipe": {"derecha": (12, 24, 26, 28, 30, 32), "izquierda": (11, 23, 25, 27, 29, 31)},
}
MODELOS = tuple(_INDICES)
NOMBRE_MODELO = {"rtmpose": "RTMPose-m (Halpe26)", "mediapipe": "MediaPipe Pose heavy"}

_CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
_URL_MEDIAPIPE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def _descargar(url, destino):
    if not destino.is_file():
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando el modelo {destino.name} (solo la primera vez)...")
        temporal = destino.with_suffix(".parcial")
        urllib.request.urlretrieve(url, temporal)
        temporal.replace(destino)
    return destino


def preparar_modelo(modelo="rtmpose"):
    """Descarga (si hace falta) y carga el modelo; regresa la función de inferencia."""
    if modelo == "rtmpose":
        from rtmlib import BodyWithFeet
        return BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")
    if modelo == "mediapipe":
        return _descargar(_URL_MEDIAPIPE, _CARPETA_MODELOS / "pose_landmarker_heavy.task")
    raise ValueError(f"Modelo desconocido: {modelo}")


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


class DetectorPose:
    """Puntos del miembro inferior de una persona por cuadro, sin marcadores.

    detectar(bgr, t_s) -> {pierna: {punto: (x, y, confianza) | None}} en px originales.
    Un punto con confianza menor que `confianza_min` se marca como no detectado
    (se interpola después, igual que un marcador perdido).

    RTMPose necesita primero la caja de la persona (detector YOLOX). El detector es lo
    más lento, así que se ejecuta cada `cada` cuadros; entre una detección y otra la
    caja se desplaza con el centro de la cadera, que el modelo de pose sí mide en
    cada cuadro. Si hay varias personas se sigue a la de la detección anterior
    (mayor traslape) o, al inicio, a la más grande.
    """

    def __init__(self, modelo="rtmpose", cada=5, confianza_min=0.3):
        self.modelo = modelo
        self.cada = max(1, int(cada))
        self.confianza_min = confianza_min
        self.indices = _INDICES[modelo]
        self.n = 0
        self.caja = None              # caja de la persona [x0, y0, x1, y1]
        self.centro_caja = None       # centro de la cadera cuando se detectó la caja
        if modelo == "rtmpose":
            self.red = preparar_modelo(modelo)
        else:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            self._mp = mp
            opciones = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(preparar_modelo(modelo))),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            self.red = vision.PoseLandmarker.create_from_options(opciones)
            self._t_ms = -1

    def detectar(self, bgr, t_s):
        puntos = self._rtmpose(bgr) if self.modelo == "rtmpose" else self._mediapipe(bgr, t_s)
        self.n += 1
        vacio = {p: dict.fromkeys(PUNTOS) for p in PIERNAS}
        if puntos is None:
            return vacio
        for pierna in PIERNAS:
            for nombre, i in zip(PUNTOS, self.indices[pierna]):
                x, y, c = puntos[i]
                if c >= self.confianza_min and np.isfinite(x) and np.isfinite(y):
                    vacio[pierna][nombre] = (float(x), float(y), float(c))
        return vacio

    def _rtmpose(self, bgr):
        if self.caja is None or self.n % self.cada == 0:
            cajas = np.asarray(self.red.det_model(bgr), float).reshape(-1, 4)
            if len(cajas):
                if self.caja is not None:
                    self.caja = max(cajas, key=lambda c: _iou(c, self.caja))
                else:
                    self.caja = max(cajas, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                self.centro_caja = None
            elif self.caja is None:
                return None
        kp, sc = self.red.pose_model(bgr, bboxes=[self.caja])
        puntos = np.column_stack([kp[0], sc[0]])
        # Desplaza la caja con el centro de la cadera hasta la siguiente detección.
        caderas = puntos[[11, 12]]
        if (caderas[:, 2] > self.confianza_min).all():
            centro = caderas[:, :2].mean(axis=0)
            if self.centro_caja is None:
                self.centro_caja = centro
            else:
                dx, dy = centro - self.centro_caja
                self.caja = self.caja + np.array([dx, dy, dx, dy])
                self.centro_caja = centro
        return puntos

    def _mediapipe(self, bgr, t_s):
        alto, ancho = bgr.shape[:2]
        f = min(1.0, 960.0 / max(alto, ancho))      # el modelo trabaja a 256 px; basta con 960
        if f < 1.0:
            bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        imagen = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t_ms = max(int(round(t_s * 1000)), self._t_ms + 1)    # el modo VIDEO exige tiempos crecientes
        self._t_ms = t_ms
        r = self.red.detect_for_video(imagen, t_ms)
        if not r.pose_landmarks:
            return None
        return np.array([(p.x * ancho, p.y * alto, p.visibility) for p in r.pose_landmarks[0]])


def corregir_intercambios(crudo, puntos=("rodilla", "tobillo", "talon", "punta"), razon=0.5):
    """Corrige cuadros donde el modelo intercambió la pierna derecha y la izquierda.

    Compara, contra el cuadro anterior ya corregido, la distancia total de los puntos
    de ambas piernas con sus etiquetas y con las etiquetas intercambiadas; si
    intercambiadas quedan claramente más cerca (menos de `razon` veces), se
    intercambian. Cuando las piernas se cruzan las dos opciones cuestan casi lo mismo
    y no se toca nada. Regresa (crudo corregido, número de cuadros corregidos).
    """
    crudo = crudo.copy()
    cols = {p: [f"{p}_{a}_{c}" for a in puntos for c in ("x", "y")] for p in PIERNAS}
    conf = {p: [f"{p}_{a}_conf" for a in puntos] for p in PIERNAS}
    # copy=True: con pandas >= 3 (copy-on-write) to_numpy() puede ser de solo lectura.
    der = crudo[cols["derecha"]].to_numpy(float, copy=True)
    izq = crudo[cols["izquierda"]].to_numpy(float, copy=True)
    cd = crudo[conf["derecha"]].to_numpy(float, copy=True)
    ci = crudo[conf["izquierda"]].to_numpy(float, copy=True)
    previo, corregidos = None, 0
    for k in range(len(der)):
        if not (np.isfinite(der[k]).all() and np.isfinite(izq[k]).all()):
            continue
        if previo is not None:
            pd_, pi_ = previo
            igual = np.linalg.norm((der[k] - pd_).reshape(-1, 2), axis=1).sum() + \
                np.linalg.norm((izq[k] - pi_).reshape(-1, 2), axis=1).sum()
            cruzado = np.linalg.norm((der[k] - pi_).reshape(-1, 2), axis=1).sum() + \
                np.linalg.norm((izq[k] - pd_).reshape(-1, 2), axis=1).sum()
            if cruzado < razon * igual:
                der[k], izq[k] = izq[k].copy(), der[k].copy()
                cd[k], ci[k] = ci[k].copy(), cd[k].copy()
                corregidos += 1
        previo = (der[k], izq[k])
    crudo[cols["derecha"]], crudo[cols["izquierda"]] = der, izq
    crudo[conf["derecha"]], crudo[conf["izquierda"]] = cd, ci
    return crudo, corregidos
