"""Calibración interactiva de los umbrales HSV de los marcadores.

Experimento 1, verificación en tiempo real: "ajustar parámetros de detección según
condiciones" (iluminación, fondo, ropa, otros marcadores).

    py calibrar_hsv.py ../21.MP4            # o 0 para la cámara
    py analisis_marcha.py ../21.MP4 --umbrales umbrales_hsv.json

Teclas: 1/2/3 = cadera/rodilla/tobillo · clic sobre el video = tomar el color de ese punto ·
        espacio = reproducir/pausa · s = guardar · r = valores de fábrica · q = salir.
"""
import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

from deteccion import (ARTICULACIONES, COLOR_DIBUJO, UMBRALES_HSV, DetectorMarcadores, abrir_fuente,
                       cargar_umbrales, guardar_umbrales, rotar)

VENTANA = "Calibracion HSV"
CONTROLES = (("H min", 179), ("H max", 179), ("S min", 255), ("S max", 255), ("V min", 255), ("V max", 255))


def a_controles(rangos):
    (h0, s0, v0), (h1, s1, v1) = rangos[0]
    if len(rangos) > 1:          # rojo que cruza H = 0: [h0, 179] U [0, h1]
        h1 = rangos[1][1][0]
    return [h0, h1, s0, s1, v0, v1]


def a_rangos(c):
    h0, h1, s0, s1, v0, v1 = (int(v) for v in c)
    if h0 <= h1:
        return [((h0, s0, v0), (h1, s1, v1))]
    return [((h0, s0, v0), (179, s1, v1)), ((0, s0, v0), (h1, s1, v1))]


def muestrear(hsv, x, y, r=4):
    """Rango HSV alrededor del color mediano del parche donde se hizo clic."""
    parche = hsv[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
    h, s, v = np.median(parche, axis=0).astype(int)
    return [(h - 8) % 180, (h + 8) % 180, max(0, s - 70), 255, max(0, v - 80), 255]


def componer(fotograma, detector, art, escala):
    """[video con las detecciones | máscara del marcador seleccionado]."""
    det = detector.detectar(fotograma)
    grupos = det.values() if detector.vista == "frontal" else [det]
    vista = fotograma.copy()
    for d in grupos:
        for nombre, p in d.items():
            if p is not None:
                radio = int(np.sqrt(p[2] / np.pi))
                cv2.circle(vista, (int(p[0]), int(p[1])), radio + 6, COLOR_DIBUJO[nombre], 4, cv2.LINE_AA)
    vista = cv2.resize(vista, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    mascara = cv2.resize(detector.mascaras(fotograma)[art], vista.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(vista)
    color[mascara > 0] = COLOR_DIBUJO[art]
    areas = [d[art][2] for d in grupos if d[art] is not None]
    rangos = " + ".join(f"H {b[0]}-{a[0]} S {b[1]}-{a[1]} V {b[2]}-{a[2]}" for b, a in detector.umbrales[art])
    for y, texto in ((30, f"Marcador: {art} (1/2/3 para cambiar)"), (58, rangos),
                     (86, "Area: " + (", ".join(f"{a:.0f} px" for a in areas) if areas else "no detectado")),
                     (color.shape[0] - 16, "clic = tomar color   s = guardar   q = salir")):
        cv2.putText(color, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return np.hstack([vista, color])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fuente", help="video o índice de cámara")
    p.add_argument("--salida", type=Path, default=Path("umbrales_hsv.json"), help="JSON de umbrales a guardar")
    p.add_argument("--umbrales", type=Path, default=None, help="JSON inicial (por omisión, los de fábrica)")
    p.add_argument("--vista", choices=("lateral", "frontal"), default="lateral")
    p.add_argument("--cuadro", type=int, default=0, help="cuadro inicial del video")
    p.add_argument("--rotacion", default="auto", choices=("auto", "0", "90", "180", "270"))
    p.add_argument("--escala", type=float, default=0.4, help="tamaño de la ventana")
    p.add_argument("--prueba", type=Path, default=None,
                   help="sin ventana: guarda la vista del cuadro indicado en esta imagen y termina")
    args = p.parse_args()

    cap, _, grados, es_camara = abrir_fuente(args.fuente, args.rotacion)
    umbrales = cargar_umbrales(args.umbrales) if args.umbrales else copy.deepcopy(UMBRALES_HSV)
    detector = DetectorMarcadores(umbrales=umbrales, vista=args.vista)
    total = 0 if es_camara else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def leer(n=None):
        if n is not None and total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        if not ok:
            raise IOError("No se pudo leer el cuadro")
        return rotar(f, grados)

    if args.prueba:
        imagen = componer(leer(args.cuadro), detector, ARTICULACIONES[0], args.escala)
        cv2.imwrite(str(args.prueba), imagen)
        guardar_umbrales(umbrales, args.salida)
        print(f"Vista guardada en {args.prueba}; umbrales en {args.salida}")
        return

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    for nombre, maximo in CONTROLES:
        cv2.createTrackbar(nombre, VENTANA, 0, maximo, lambda v: None)
    if total:
        cv2.createTrackbar("Cuadro", VENTANA, min(args.cuadro, total - 1), total - 1, lambda v: None)
    estado = {"sel": 0, "clic": None}

    def poner_controles():
        for (nombre, _), v in zip(CONTROLES, a_controles(umbrales[ARTICULACIONES[estado["sel"]]])):
            cv2.setTrackbarPos(nombre, VENTANA, int(v))

    def raton(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            estado["clic"] = (x, y)

    cv2.setMouseCallback(VENTANA, raton)
    poner_controles()
    fotograma, actual, reproducir = leer(args.cuadro), args.cuadro, False
    while True:
        if total:
            pos = cv2.getTrackbarPos("Cuadro", VENTANA)
            if reproducir:
                pos = (actual + 1) % total
                cv2.setTrackbarPos("Cuadro", VENTANA, pos)
            if pos != actual:
                fotograma, actual = leer(pos), pos
        else:
            fotograma = leer()
        art = ARTICULACIONES[estado["sel"]]
        if estado["clic"]:
            x, y = estado["clic"]
            estado["clic"] = None
            if x < fotograma.shape[1] * args.escala:
                hsv = cv2.cvtColor(fotograma, cv2.COLOR_BGR2HSV)
                umbrales[art] = a_rangos(muestrear(hsv, int(x / args.escala), int(y / args.escala)))
                poner_controles()
        umbrales[art] = a_rangos([cv2.getTrackbarPos(n, VENTANA) for n, _ in CONTROLES])
        cv2.imshow(VENTANA, componer(fotograma, detector, art, args.escala))
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("1"), ord("2"), ord("3")):
            estado["sel"] = tecla - ord("1")
            poner_controles()
        elif tecla == ord(" "):
            reproducir = not reproducir
        elif tecla == ord("s"):
            guardar_umbrales(umbrales, args.salida)
            print(f"Umbrales guardados en {args.salida}")
        elif tecla == ord("r"):
            umbrales.update(copy.deepcopy(UMBRALES_HSV))
            poner_controles()
        elif tecla in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
