"""Análisis de marcha por videografía, con marcadores de color o sin marcadores.

Experimentos 1 y 2 de la práctica: detección en tiempo real, almacenamiento,
filtrado, normalización del ciclo, gráficas y comparación con la marcha ideal.

Uso:
    py analisis_marcha.py ../21.MP4                        # marcadores, vista lateral, ventana en vivo
    py analisis_marcha.py ../21.MP4 --metodo sin-marcadores # sin marcadores (RTMPose)
    py analisis_marcha.py ../23.MP4 --sin-ventana --video-anotado
    py analisis_marcha.py frontal.mp4 --vista frontal       # las dos piernas, de frente
    py analisis_marcha.py 0 --grabar                        # cámara: graba y almacena a la vez

Teclas en la ventana: q / Esc = terminar, espacio = pausa.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import cinematica as cin
import comparacion
import graficas
import sin_marcadores as sm
import video_alturas
from deteccion import ARTICULACIONES, PIERNAS, DetectorMarcadores, abrir_fuente, cargar_umbrales, leer_fotogramas
from interfaz import COLOR_PIERNA_BGR, Interfaz, dibujar_pierna


def _laterales(frente):
    """Signo en x del lado externo de cada pierna (vista frontal).

    De frente a la cámara la pierna derecha queda a la izquierda de la imagen,
    así que su lado externo es -x.
    """
    s = -1 if frente == "camara" else 1
    return {"derecha": s, "izquierda": -s}


def _fila(indice, t, det, por_pierna, tercero):
    """Una fila de detecciones crudas: x, y y `tercero` (área del marcador o confianza del modelo)."""
    fila = {"fotograma": indice, "tiempo_s": t}
    grupos = det.items() if por_pierna else [(None, det)]
    for pierna, d in grupos:
        for art, v in d.items():
            x, y, c = v if v is not None else (np.nan, np.nan, np.nan)
            pre = f"{pierna}_{art}" if pierna else art
            fila.update({f"{pre}_x": x, f"{pre}_y": y, f"{pre}_{tercero}": c})
    return fila


def _valores_vivos(det, vista, direccion, laterales, por_pierna):
    """Ángulos instantáneos (sin filtrar) que se muestran en la interfaz."""
    def completo(d):
        return all(d.get(a) is not None for a in ARTICULACIONES)

    def xy(d, puntos=ARTICULACIONES):
        return [np.array([d[a][:2]]) for a in puntos]

    valores = {}
    if vista == "frontal":
        for pierna, d in det.items():
            if completo(d):
                ang = cin.angulos_frontales(*xy(d), laterales[pierna])
                valores.update({f"{pierna}_{a}": float(v[0]) for a, v in ang.items()})
    elif por_pierna:      # lateral sin marcadores: las dos piernas, con el pie si se ve
        for pierna, d in det.items():
            if completo(d):
                pie = xy(d, ("talon", "punta")) if d.get("talon") and d.get("punta") else (None, None)
                ang = cin.angulos_sagitales(*xy(d), direccion, *pie)
                valores.update({f"{pierna}_{a}": float(v[0]) for a, v in ang.items()})
    elif completo(det):
        valores = {a: float(v[0]) for a, v in cin.angulos_sagitales(*xy(det), direccion).items()}
    return valores


def _estado(det, por_pierna):
    marca = lambda d: "OK" if d is not None else "--"  # noqa: E731
    if por_pierna:
        return " | ".join(f"{p[:3].capitalize()} " + " ".join(f"{a[0].upper()}:{marca(det[p][a])}" for a in ARTICULACIONES)
                          for p in PIERNAS)
    return "  ".join(f"{a.capitalize()}:{marca(det[a])}" for a in ARTICULACIONES)


def adquirir(args):
    """Experimento 1: detecta las articulaciones cuadro por cuadro y las muestra en vivo."""
    cap, fps, grados, es_camara = abrir_fuente(args.fuente, args.rotacion)
    pose = args.metodo == "sin-marcadores"
    if pose:
        print(f"Cargando el modelo {sm.NOMBRE_MODELO[args.modelo]}...", flush=True)
        detector = sm.DetectorPose(args.modelo, cada=args.detectar_cada)
    else:
        umbrales = cargar_umbrales(args.umbrales) if args.umbrales else None
        detector = DetectorMarcadores(modo=args.modo, escala=args.escala_proceso, umbrales=umbrales,
                                      vista=args.vista, frente=args.frente)
    por_pierna = pose or args.vista == "frontal"
    ui = Interfaz(args.vista, fps, dos_piernas=por_pierna) if (not args.sin_ventana or args.video_anotado) else None
    laterales = _laterales(args.frente)
    direccion = {"auto": 1, "derecha": 1, "izquierda": -1}[args.direccion]
    col_cadera, col_tobillo = ("derecha_cadera_x", "derecha_tobillo_x") if pose else ("cadera_x", "tobillo_x")
    total = None if es_camara else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - args.inicio
    if total is not None and args.fin is not None:
        total = min(total, args.fin - args.inicio)
    if total is not None and total <= 0:      # algunos formatos no informan el número de cuadros
        total = None
    filas, escritor, grabador = [], None, None
    ultimo_visto, pausa = None, False
    for indice, t, fotograma in leer_fotogramas(cap, grados, fps, es_camara, args.inicio, args.fin):
        if args.grabar:   # grabación y almacenamiento empiezan juntos
            if grabador is None:
                grabador = cv2.VideoWriter(str(args.salida / "captura.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps, fotograma.shape[1::-1])
            grabador.write(fotograma)
        det = detector.detectar(fotograma, t) if pose else detector.detectar(fotograma)
        filas.append(_fila(indice, t, det, por_pierna, "conf" if pose else "area"))
        if total and len(filas) % 100 == 0:
            print(f"Procesando: cuadro {len(filas)} de {total} ({100 * len(filas) / total:.0f} %)", flush=True)

        todos = [d for p in det.values() for d in p.values()] if por_pierna else list(det.values())
        if any(d is not None for d in todos):
            ultimo_visto = t
        elif args.auto_detener and ultimo_visto is not None and t - ultimo_visto > args.auto_detener:
            print(f"Sin detecciones durante {args.auto_detener} s: fin de la trayectoria.")
            break
        if args.duracion and t >= args.duracion:
            break

        if args.vista == "lateral" and args.direccion == "auto" and len(filas) >= 60 and len(filas) % 30 == 0:
            xs = pd.DataFrame(filas)[[col_cadera, col_tobillo]].interpolate().dropna()
            if len(xs) > 30:
                direccion = cin.detectar_direccion(xs[col_cadera].values, xs[col_tobillo].values)

        if ui is None:
            continue
        valores = _valores_vivos(det, args.vista, direccion, laterales, por_pierna)
        ui.agregar(valores)
        if por_pierna:
            for pierna, d in det.items():
                dibujar_pierna(fotograma, d, COLOR_PIERNA_BGR[pierna])
        else:
            dibujar_pierna(fotograma, det)
        vista = cv2.resize(fotograma, None, fx=args.escala_vista, fy=args.escala_vista, interpolation=cv2.INTER_AREA)
        vista = ui.componer(vista, t, indice, valores, _estado(det, por_pierna), grabando=args.grabar)
        if args.video_anotado:
            if escritor is None:
                escritor = cv2.VideoWriter(str(args.salida / "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps, vista.shape[1::-1])
            escritor.write(vista)
        if not args.sin_ventana:
            cv2.imshow("Analisis de marcha - q para terminar", vista)
            tecla = cv2.waitKey(0 if pausa else 1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                pausa = not pausa
    cap.release()
    for w in (escritor, grabador):
        if w is not None:
            w.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(filas), fps


def _preparar(crudo, fps, prefijos, fc, opcionales=()):
    """Recorta extremos sin detección, lleva a malla de tiempo uniforme, rellena huecos y filtra.

    Un punto de `opcionales` (p. ej. el talón) que casi no se detectó queda en NaN en
    lugar de detener el análisis.
    """
    vistos = crudo[[f"{p}_x" for p in prefijos]].notna().any(axis=1).to_numpy()
    if not vistos.any():
        raise RuntimeError("No se detectó ninguna articulación; con marcadores, revise los umbrales con "
                           "calibrar_hsv.py; sin marcadores, que el cuerpo completo se vea en el video.")
    i0, i1 = np.flatnonzero(vistos)[[0, -1]]
    crudo = crudo.iloc[i0:i1 + 1].reset_index(drop=True)
    t = crudo["tiempo_s"].to_numpy()
    if t.size > 2 and np.ptp(np.diff(t)) > 1e-6:          # cámara: tiempos irregulares
        fps = 1.0 / np.median(np.diff(t))
        t_unif = np.arange(t[0], t[-1], 1.0 / fps)
        crudo = pd.DataFrame({c: np.interp(t_unif, t, crudo[c]) if c != "tiempo_s" else t_unif for c in crudo.columns})
        t = t_unif
    pos_crudo, pos = {}, {}
    for p in prefijos:
        try:
            xy = np.column_stack([cin.rellenar_huecos(crudo[f"{p}_x"]), cin.rellenar_huecos(crudo[f"{p}_y"])])
        except ValueError:
            if p not in opcionales:
                raise RuntimeError(f"Muy pocos cuadros con '{p}' detectado.") from None
            xy = np.full((len(t), 2), np.nan)
        pos_crudo[p] = xy
        pos[p] = xy if np.isnan(xy).all() else np.column_stack([cin.filtrar(xy[:, k], fps, fc) for k in range(2)])
    return crudo, t, fps, pos_crudo, pos


def _escala(crudo, pos_crudo, prefijo, args):
    """Factor y unidad para las alturas: cm si se conoce el marcador o la estatura, si no px."""
    cols = [f"{prefijo}{a}_area" for a in ARTICULACIONES if f"{prefijo}{a}_area" in crudo]
    diametro_px = None
    if cols:
        areas = np.concatenate([crudo[c].dropna().to_numpy() for c in cols])
        diametro_px = float(np.median(2.0 * np.sqrt(areas / np.pi)))
    diametro_cm = args.diametro_marcador_cm
    if diametro_cm and diametro_px is None:
        print("Aviso: sin marcadores no se usa --diametro-marcador-cm; use --estatura-m para obtener cm.")
        diametro_cm = None
    pierna_px = float(np.median(np.linalg.norm(pos_crudo[f"{prefijo}tobillo"] - pos_crudo[f"{prefijo}rodilla"], axis=1)))
    m_por_px, metodo = cin.escala_m_por_px(diametro_px, pierna_px, diametro_cm, args.estatura_m)
    escala = {"m_por_px": m_por_px and round(m_por_px, 7), "metodo": metodo,
              "diametro_marcador_px": diametro_px and round(diametro_px, 1), "pierna_px": round(pierna_px, 1)}
    return (100.0 * m_por_px, "cm", escala) if m_por_px else (1.0, "px", escala)


def _ciclos_o_tramo(contactos, despegues, t, nombre):
    ciclos = cin.armar_ciclos(contactos, despegues)
    if ciclos:
        return ciclos, False
    # Tramo corto (p. ej. 292:327 de pp3.m): se toma completo como un ciclo, como hacía MATLAB.
    print(f"Aviso ({nombre}): no se detectaron dos contactos iniciales; se toma todo el tramo como un ciclo.")
    apoyo = 100.0 * (despegues[0] - t[0]) / (t[-1] - t[0]) if despegues.size else np.nan
    return [(t[0], t[-1], apoyo)], True


def _tabla_normalizada(norm):
    tabla = {"porcentaje_ciclo": graficas.PCT}
    for pierna, por_tipo in norm.items():
        for tipo, por_art in por_tipo.items():
            for art, m in por_art.items():
                tabla[f"{pierna}_{tipo}_{art}_media"] = np.nanmean(m, axis=0)
                tabla[f"{pierna}_{tipo}_{art}_de"] = np.nanstd(m, axis=0)
    return pd.DataFrame(tabla)


def _datos(t, crudo, series, unidad, tipos_angulo):
    """Estructura de almacenamiento de la práctica: tiempo, x/y, alturas y ángulos de ambas piernas."""
    datos = {"tiempo_s": t, "fotograma": crudo["fotograma"].round().astype(int).to_numpy()}
    for pierna in PIERNAS:
        for art in ARTICULACIONES + ("talon", "punta"):      # talón y punta: solo sin marcadores
            if f"{art}_x" in series[pierna]:
                datos[f"{pierna}_{art}_x"] = series[pierna][f"{art}_x"]
                datos[f"{pierna}_{art}_y"] = series[pierna][f"{art}_y"]
    for pierna in PIERNAS:
        for art in ARTICULACIONES:
            datos[f"{pierna}_{art}_altura_{unidad}"] = series[pierna][f"altura_{art}"]
    for tipo in tipos_angulo:
        for pierna in PIERNAS:
            for art in ARTICULACIONES:
                if f"{tipo}_{art}" in series[pierna]:
                    datos[f"{pierna}_{tipo}_{art}"] = series[pierna][f"{tipo}_{art}"]
    return pd.DataFrame(datos)


def procesar_lateral(crudo, fps, args):
    """Experimento 2 con video lateral: ángulos sagitales y alturas.

    Se mide la pierna del lado de la cámara (con marcadores, la que los lleva). Sin
    marcadores el modelo también estima la pierna lejana, pero queda tapada por la
    cercana en buena parte del ciclo; por omisión se usa el caso especial de la
    práctica (la otra pierna es la medida desplazada medio ciclo) y con
    --pierna-lejana medir se usa la estimación del modelo.
    """
    pose = args.metodo == "sin-marcadores"
    corregidos = 0
    if pose:
        crudo, corregidos = sm.corregir_intercambios(crudo)
        prefijos = [f"{p}_{a}" for p in PIERNAS for a in sm.PUNTOS]
        opcionales = [f"{p}_{a}" for p in PIERNAS for a in ("hombro", "talon", "punta")]
    else:
        prefijos, opcionales = list(ARTICULACIONES), []
    crudo, t, fps, pos_crudo, pos = _preparar(crudo, fps, prefijos, args.fc, opcionales)

    if pose:
        xy = {p: {a: pos[f"{p}_{a}"] for a in sm.PUNTOS} for p in PIERNAS}
        xy_crudo = {p: {a: pos_crudo[f"{p}_{a}"] for a in sm.PUNTOS} for p in PIERNAS}
        pelvis_x = 0.5 * (xy["derecha"]["cadera"][:, 0] + xy["izquierda"]["cadera"][:, 0])
        ref_x = xy["derecha"]["tobillo"][:, 0]
    else:
        pelvis_x, ref_x = pos["cadera"][:, 0], pos["tobillo"][:, 0]
    if args.direccion == "auto":
        direccion = cin.detectar_direccion(pelvis_x, ref_x)
    else:
        direccion = 1 if args.direccion == "derecha" else -1
    # Si el sujeto camina hacia la derecha de la imagen, su lado derecho mira a la cámara.
    cercana = args.pierna if args.pierna != "auto" else ("derecha" if direccion > 0 else "izquierda")
    otra = PIERNAS[1 - PIERNAS.index(cercana)]
    if not pose:
        xy = {cercana: {a: pos[a] for a in ARTICULACIONES}}
        xy_crudo = {cercana: {a: pos_crudo[a] for a in ARTICULACIONES}}
    medidas = [cercana, otra] if (pose and args.pierna_lejana == "medir") else [cercana]
    con_pie = pose and all(np.isfinite(xy[p][a]).all() for p in medidas for a in ("talon", "punta"))

    prefijo = f"{cercana}_" if pose else ""
    factor, unidad, escala = _escala(crudo, pos_crudo, prefijo, args)
    y_suelo = max(xy[p]["tobillo"][:, 1].max() for p in medidas)   # punto más bajo del tobillo = altura 0

    series, norm, ciclos_por_pierna, info, angulos, supuestos = {}, {}, {}, {}, {}, {}
    for p in medidas:
        P = xy[p]
        pie = (P["talon"], P["punta"]) if con_pie else (None, None)
        ang = angulos[p] = cin.angulos_sagitales(P["cadera"], P["rodilla"], P["tobillo"], direccion, *pie)
        # Eventos (Zeni et al., 2008) respecto a la pelvis: contacto inicial = talón (o tobillo)
        # lo más adelantado; despegue = punta del pie (o tobillo) lo más atrasada.
        x_pelvis = pelvis_x if pose else P["cadera"][:, 0]
        x_contacto = P["talon"][:, 0] if con_pie else P["tobillo"][:, 0]
        contactos, despegues, _ = cin.detectar_eventos(t, x_pelvis, x_contacto, direccion, fps)
        if con_pie:
            despegues = cin.detectar_eventos(t, x_pelvis, P["punta"][:, 0], direccion, fps)[1]
        ciclos, supuestos[p] = _ciclos_o_tramo(contactos, despegues, t, f"pierna {p}")
        s = {}
        for art in ARTICULACIONES:
            s[f"{art}_x"], s[f"{art}_y"] = P[art][:, 0], P[art][:, 1]
            s[f"altura_{art}"] = (y_suelo - P[art][:, 1]) * factor
            s[f"angulo_{art}"] = ang[art]
        if con_pie:
            for a in ("talon", "punta"):
                s[f"{a}_x"], s[f"{a}_y"] = P[a][:, 0], P[a][:, 1]
        series[p], ciclos_por_pierna[p] = s, ciclos
        norm[p] = {tipo: {art: cin.normalizar_ciclos(t, s[f"{tipo}_{art}"], ciclos) for art in ARTICULACIONES}
                   for tipo in ("altura", "angulo")}
        apoyo_p = np.array([c[2] for c in ciclos])
        info[p] = {"ciclos": len(ciclos), "apoyo_pct": float(np.nanmean(apoyo_p)), "origen": "medida"}

    ciclos = ciclos_por_pierna[cercana]
    duraciones = np.array([c[1] - c[0] for c in ciclos])
    periodo = float(np.median(duraciones))
    if otra not in medidas:
        # Caso especial de la práctica: solo se grabó una pierna -> la otra es la
        # misma señal desplazada medio ciclo.
        series[otra] = {k: cin.desplazar(t, v, periodo / 2.0, periodo) for k, v in series[cercana].items()}
        ciclos_por_pierna[otra] = [(a + periodo / 2.0, b + periodo / 2.0, ap) for a, b, ap in ciclos]
        # Normalizada a su propio contacto inicial, la pierna desplazada medio ciclo
        # da exactamente los mismos ciclos que la grabada.
        norm[otra] = norm[cercana]
        info[otra] = {**info[cercana], "origen": "sintetica"}

    apoyo = np.array([c[2] for c in ciclos])
    detectado = {pre: round(float(crudo[f"{pre}_x"].notna().mean() * 100), 1) for pre in prefijos}
    Pc = xy_crudo[cercana]
    orden_ok = float(np.mean((Pc["cadera"][:, 1] < Pc["rodilla"][:, 1])
                             & (Pc["rodilla"][:, 1] < Pc["tobillo"][:, 1])) * 100)
    resumen = {
        "fuente": str(args.fuente),
        "ruta_video": _ruta_video(args.fuente),
        "vista": "lateral",
        "metodo": f"sin marcadores ({sm.NOMBRE_MODELO[args.modelo]})" if pose else "marcadores de color",
        "fps": round(fps, 3),
        "cuadros_analizados": int(len(t)),
        "duracion_s": round(float(t[-1] - t[0]), 2),
        "segmentacion": None if pose else args.modo,
        "deteccion_pct": detectado,
        "orden_anatomico_correcto_pct": round(orden_ok, 1),
        "pierna_grabada": cercana,
        "direccion_marcha": "derecha" if direccion > 0 else "izquierda",
        "ciclos_validos": len(ciclos),
        "tramo_tomado_como_un_ciclo": supuestos[cercana],
        "duracion_ciclo_s": [round(float(duraciones.mean()), 3), round(float(duraciones.std()), 3)],
        "cadencia_pasos_min": round(float(120.0 / duraciones.mean()), 1),
        "fase_apoyo_pct": [round(float(np.nanmean(apoyo)), 1), round(float(np.nanstd(apoyo)), 1)],
        "tobillo_con_pie": con_pie,
        "unidad_altura": unidad,
        "escala": escala,
        "rango_articular_deg": {
            a: {"min": round(float(np.nanmean(norm[cercana]["angulo"][a], axis=0).min()), 1),
                "max": round(float(np.nanmean(norm[cercana]["angulo"][a], axis=0).max()), 1)}
            for a in ARTICULACIONES
        },
        "por_pierna": info,
    }
    if pose:
        resumen["pierna_lejana"] = "medida" if otra in medidas else "generada"
        resumen["intercambios_der_izq_corregidos"] = corregidos
    nombre = Path(str(args.fuente)).name
    pie_crudo = (Pc["talon"], Pc["punta"]) if con_pie else (None, None)
    ang_crudo = cin.angulos_sagitales(Pc["cadera"], Pc["rodilla"], Pc["tobillo"], direccion, *pie_crudo)
    paneles = []
    for a in ARTICULACIONES:
        lineas = [("Sin filtrar", ang_crudo[a], graficas.EJE, 1),
                  (f"Filtrado (Butterworth {args.fc:g} Hz), pierna {cercana}", angulos[cercana][a],
                   graficas.COLOR_PIERNA[cercana], 2)]
        if otra in medidas:
            lineas.append((f"Filtrado, pierna {otra} (lejana)", angulos[otra][a], graficas.COLOR_PIERNA[otra], 1.5))
        paneles.append((f"{a.capitalize()} [°]", lineas))
    return {
        "datos": _datos(t, crudo, series, unidad, ("angulo",)), "tabla": _tabla_normalizada(norm),
        "ciclos_por_pierna": ciclos_por_pierna, "resumen": resumen, "crudo": crudo,
        "figuras": [
            ("graficas_12.png", graficas.FILAS_12, ("altura", "angulo"), True,
             f"Análisis de marcha · {nombre} · media ± DE de {len(ciclos)} ciclos"),
        ],
        "serie": (t, paneles, [c[0] for c in ciclos],
                  f"Ángulos articulares contra el tiempo · {nombre} · pierna {cercana}"),
    }


SEGMENTOS = (("cadera", "rodilla"), ("rodilla", "tobillo"))


def _ciclos_frontal(contactos, periodo, t, nombre):
    """Ciclos entre contactos. Con un solo contacto (video corto), un ciclo de la duración
    estimada que empiece o termine en él, si se sale del video menos del 10 % (esa parte
    toma el primer o último cuadro); si no, todo el tramo."""
    ciclos = cin.armar_ciclos(contactos, np.array([]))
    if ciclos:
        return ciclos, False
    if periodo and contactos.size:
        fuera = lambda t0: max(0.0, t[0] - t0) + max(0.0, t0 + periodo - t[-1])  # noqa: E731
        t0 = min((c for tc in contactos for c in (tc, tc - periodo)), key=fuera)
        if fuera(t0) <= 0.1 * periodo:
            print(f"Aviso ({nombre}): un solo contacto inicial; el ciclo se completa con la duración "
                  f"estimada ({periodo:.2f} s). Grabe al menos dos ciclos completos.")
            return [(t0, t0 + periodo, np.nan)], True
    return _ciclos_o_tramo(contactos, np.array([]), t, nombre)


def _escala_frontal(pos, t, fps, periodo, reposo, estatura_m):
    """Metros por píxel en cada cuadro (la distancia a la cámara cambia durante la caminata).

    El tamaño aparente del sujeto se sigue con el tronco (hombros a caderas, sin
    marcadores) o con el ancho entre los marcadores de cadera, promediado sobre un
    ciclo para quitar la oscilación de la marcha. La escala absoluta sale de la
    estatura: de pie, cadera-tobillo = 0.491 H (Winter, 2009), medida en el reposo
    inicial o, si no lo hay, con la mediana de la pierna más extendida de cada cuadro.
    """
    hombros = [pos.get(f"{p}_hombro") for p in PIERNAS]
    if all(h is not None and not np.isnan(h).all() for h in hombros):
        tam = 0.5 * (pos["derecha_cadera"][:, 1] + pos["izquierda_cadera"][:, 1]
                     - hombros[0][:, 1] - hombros[1][:, 1])
        referencia = "tronco (hombros a caderas)"
    else:
        tam = np.abs(pos["derecha_cadera"][:, 0] - pos["izquierda_cadera"][:, 0])
        referencia = "ancho entre los marcadores de cadera"
    tam = cin.suavizar_lento(tam, fps, periodo or 1.0)
    pierna = {p: np.linalg.norm(pos[f"{p}_tobillo"] - pos[f"{p}_cadera"], axis=1) for p in PIERNAS}
    if reposo is not None:
        proporcion = float(np.median(0.5 * (pierna["derecha"] + pierna["izquierda"])[reposo] / tam[reposo]))
        calibracion = "pierna de pie en el reposo inicial = 0.491 H"
    else:
        proporcion = float(np.median(np.maximum(pierna["derecha"], pierna["izquierda"]) / tam))
        calibracion = "pierna más extendida de cada cuadro = 0.491 H (sin reposo inicial)"
    m_por_px = cin.PIERNA_H * estatura_m / (proporcion * tam)
    info = {
        "metodo": f"estatura {estatura_m:g} m; {calibracion}; tamaño seguido con el {referencia}",
        "m_por_px_inicio": round(float(m_por_px[0]), 7),
        "m_por_px_fin": round(float(m_por_px[-1]), 7),
        "cambio_tamano_pct": round(float(100 * (m_por_px[0] / m_por_px[-1] - 1)), 1),
    }
    return m_por_px, info


def procesar_frontal(crudo, fps, args):
    """Experimento 2 con video frontal: alturas, distancias entre articulaciones y ángulos frontales.

    La cámara puede moverse con el sujeto, así que nada se mide contra un punto fijo de
    la imagen: en cada cuadro las alturas se toman sobre el tobillo más bajo (el pie
    apoyado) y se escalan a metros con la estatura (escala por cuadro). Cada altura se
    expresa como cambio respecto a la postura de pie del inicio del video.
    """
    pose = args.metodo == "sin-marcadores"
    prefijos = [f"{p}_{a}" for p in PIERNAS for a in ARTICULACIONES]
    opcionales = [f"{p}_hombro" for p in PIERNAS] if pose and "derecha_hombro_x" in crudo else []
    crudo, t, fps, pos_crudo, pos = _preparar(crudo, fps, prefijos + opcionales, args.fc, opcionales)
    laterales = _laterales(args.frente)
    H = args.estatura_m
    y = {f"{p}_{a}": pos[f"{p}_{a}"][:, 1] for p in PIERNAS for a in ARTICULACIONES}

    pierna_px = np.median([np.linalg.norm(pos[f"{p}_tobillo"] - pos[f"{p}_cadera"], axis=1) for p in PIERNAS])
    reposo = cin.reposo_inicial((y["derecha_tobillo"] - y["izquierda_tobillo"]) / pierna_px, fps)
    caminando = slice(reposo.stop if reposo is not None else 0, None)
    periodo = cin.periodo_frontal(t[caminando], y["derecha_tobillo"][caminando], y["izquierda_tobillo"][caminando],
                                  fps, args.frente)
    m_por_px, escala = _escala_frontal(pos, t, fps, periodo, reposo, H)
    y_cadera = 0.5 * (y["derecha_cadera"] + y["izquierda_cadera"])
    contactos = {p: cin.detectar_contactos_frontal(t[caminando], y[f"{p}_tobillo"][caminando], y_cadera[caminando],
                                                   m_por_px[caminando], fps, periodo, args.frente) for p in PIERNAS}
    y_suelo = np.maximum(y["derecha_tobillo"], y["izquierda_tobillo"])     # pie apoyado
    if reposo is not None:
        referencia = (f"postura de pie medida al inicio del video ({t[reposo][0]:.2f}–{t[reposo][-1]:.2f} s)")
    else:
        referencia = "postura de pie estimada con la estatura (Winter, 2009): el video no empieza con la persona quieta"
        print("Aviso: el video no empieza con la persona de pie y quieta; la postura de pie se estima con la "
              "estatura. Para medirla, grabe 2 s de pie antes de caminar.")

    series, norm, ciclos_por_pierna, por_pierna, supuestos, de_pie = {}, {}, {}, {}, {}, {}
    for pierna in PIERNAS:
        xy = {a: pos[f"{pierna}_{a}"] for a in ARTICULACIONES}
        s, de_pie[pierna] = {}, {}
        for art in ARTICULACIONES:
            s[f"{art}_x"], s[f"{art}_y"] = xy[art][:, 0], xy[art][:, 1]
            s[f"altura_suelo_{art}"] = (y_suelo - xy[art][:, 1]) * m_por_px
            ref = (float(np.median(s[f"altura_suelo_{art}"][reposo])) if reposo is not None
                   else (cin.ALTURA_DE_PIE_H[art] - cin.ALTURA_DE_PIE_H["tobillo"]) * H)
            s[f"altura_{art}"] = s[f"altura_suelo_{art}"] - ref
            de_pie[pierna][f"altura_suelo_{art}"] = round(ref, 4)
        for a, b in SEGMENTOS:
            s[f"segmento_{a}_{b}"] = (xy[b][:, 1] - xy[a][:, 1]) * m_por_px
            ref = (float(np.median(s[f"segmento_{a}_{b}"][reposo])) if reposo is not None
                   else (cin.ALTURA_DE_PIE_H[a] - cin.ALTURA_DE_PIE_H[b]) * H)
            de_pie[pierna][f"segmento_{a}_{b}"] = round(ref, 4)
        for art, v in cin.angulos_frontales(xy["cadera"], xy["rodilla"], xy["tobillo"], laterales[pierna]).items():
            s[f"frontal_{art}"] = v
        ciclos, supuestos[pierna] = _ciclos_frontal(contactos[pierna], periodo, t[caminando], f"pierna {pierna}")
        series[pierna], ciclos_por_pierna[pierna] = s, ciclos
        normalizar = lambda senal: cin.normalizar_ciclos(t, senal, ciclos)  # noqa: E731
        norm[pierna] = {"altura": {}, "segmento": {}, "frontal": {}}
        for art in ARTICULACIONES:
            norm[pierna]["altura"][art] = normalizar(s[f"altura_{art}"])
        for a, b in SEGMENTOS:
            norm[pierna]["segmento"][f"{a}_{b}"] = normalizar(s[f"segmento_{a}_{b}"])
        for art in ("cadera", "rodilla"):
            norm[pierna]["frontal"][art] = normalizar(s[f"frontal_{art}"])
        por_pierna[pierna] = {"ciclos": len(ciclos), "apoyo_pct": None, "origen": "medida"}
    unidad = "m"

    duraciones = np.array([c[1] - c[0] for p in PIERNAS for c in ciclos_por_pierna[p]])
    resumen = {
        "fuente": str(args.fuente),
        "ruta_video": _ruta_video(args.fuente),
        "vista": "frontal",
        "metodo": (f"sin marcadores ({sm.NOMBRE_MODELO[args.modelo]})" if args.metodo == "sin-marcadores"
                   else "marcadores de color"),
        "frente": args.frente,
        "fps": round(fps, 3),
        "cuadros_analizados": int(len(t)),
        "duracion_s": round(float(t[-1] - t[0]), 2),
        "segmentacion": None if args.metodo == "sin-marcadores" else args.modo,
        "deteccion_pct": {f"{p}_{a}": round(float(crudo[f"{p}_{a}_x"].notna().mean() * 100), 1)
                          for p in PIERNAS for a in ARTICULACIONES},
        "ciclos_validos": {p: len(ciclos_por_pierna[p]) for p in PIERNAS},
        "tramo_tomado_como_un_ciclo": supuestos,
        "duracion_ciclo_s": [round(float(duraciones.mean()), 3), round(float(duraciones.std()), 3)],
        "cadencia_pasos_min": round(float(120.0 / duraciones.mean()), 1),
        "estatura_m": H,
        "unidad_altura": unidad,
        "altura_relativa": True,
        "referencia_altura": referencia,
        "reposo_s": [round(float(t[reposo][0]), 3), round(float(t[reposo][-1]), 3)] if reposo is not None else None,
        "rotacion": getattr(args, "rotacion", "auto"),
        "de_pie_m": de_pie,
        "escala": escala,
        "por_pierna": por_pierna,
    }
    datos = _datos(t, crudo, series, unidad, ("frontal",))
    datos = datos.rename(columns={f"{p}_{a}_altura_m": f"{p}_{a}_cambio_altura_m" for p in PIERNAS for a in ARTICULACIONES})
    for p in PIERNAS:
        for a in ARTICULACIONES:
            datos[f"{p}_{a}_altura_sobre_pie_apoyado_m"] = series[p][f"altura_suelo_{a}"]
        for a, b in SEGMENTOS:
            datos[f"{p}_distancia_vertical_{a}_{b}_m"] = series[p][f"segmento_{a}_{b}"]
    datos["escala_m_por_px"] = m_por_px

    nombre = Path(str(args.fuente)).name
    paneles = [(f"Cambio de altura\n{a} [m]", [(f"Pierna {p}", series[p][f"altura_{a}"], graficas.COLOR_PIERNA[p], 2)
                                               for p in PIERNAS]) for a in ARTICULACIONES]
    paneles.append(("Escala [mm/px]\n(distancia a la cámara)", [("Escala", 1000 * m_por_px, graficas.TENUE, 1.5)]))
    reposo_s = (t[reposo][0], t[reposo][-1]) if reposo is not None else None
    return {
        "datos": datos, "tabla": _tabla_normalizada(norm),
        "ciclos_por_pierna": ciclos_por_pierna, "resumen": resumen, "crudo": crudo,
        "figuras": [
            ("graficas_1_6.png", graficas.FILAS_12[:3], ("altura",), True,
             f"Gráficas 1 a 6 · cambio de altura respecto a la postura de pie · {nombre} · media ± DE"),
            ("graficas_frontal.png", graficas.FILAS_FRONTAL, ("altura", "frontal"), False,
             f"Plano frontal · {nombre} · media ± DE por pierna"),
            ("distancias_articulaciones.png", graficas.FILAS_SEGMENTOS, ("segmento",), False,
             f"Distancia vertical entre articulaciones · {nombre} · media ± DE por pierna"),
        ],
        "serie": (t, paneles, [c[0] for c in ciclos_por_pierna["derecha"]],
                  f"Alturas contra el tiempo · {nombre} · contactos de la pierna derecha", reposo_s),
    }


def _ruta_video(fuente):
    ruta = Path(str(fuente))
    return str(ruta.resolve()) if ruta.is_file() else None


def _sin_nan(obj):
    """JSON estándar: NaN -> null."""
    if isinstance(obj, dict):
        return {k: _sin_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sin_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def guardar(res, args):
    s = args.salida
    res["crudo"].to_csv(s / "detecciones_crudas.csv", index=False, float_format="%.3f")
    res["datos"].to_csv(s / "datos_marcha.csv", index=False, float_format="%.4f")
    pd.DataFrame([
        {"pierna": p, "origen": res["resumen"]["por_pierna"][p]["origen"], "ciclo": k + 1,
         "contacto_inicial_s": c[0], "contacto_siguiente_s": c[1], "duracion_s": c[1] - c[0], "fase_apoyo_pct": c[2]}
        for p, cs in res["ciclos_por_pierna"].items() for k, c in enumerate(cs)
    ]).to_csv(s / "ciclos.csv", index=False, float_format="%.4f")
    res["tabla"].to_csv(s / "ciclos_normalizados.csv", index=False, float_format="%.4f")
    res["resumen"] = _sin_nan(res["resumen"])
    with open(s / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(res["resumen"], f, ensure_ascii=False, indent=2)

    normal = comparacion.normativa.cargar(args.referencia)
    for archivo, filas, tipos, sueltas, titulo in res["figuras"]:
        curvas = graficas.curvas_desde_tabla(res["tabla"], res["resumen"], tipos)
        graficas.figura_ciclos(filas, curvas, titulo, s / archivo, s / "graficas" if sueltas else None,
                               numerar=sueltas, normal=normal)
    t, paneles, contactos, titulo, *reposo = res["serie"]
    graficas.serie_temporal(t, paneles, contactos, titulo, s / "serie_temporal.png", reposo[0] if reposo else None)
    if res["resumen"]["vista"] == "lateral":
        ruta, _ = comparacion.generar(s, args.referencia)
        print(f"Comparación con la marcha ideal: {ruta}")
    else:
        print("Generando el video que muestra cómo se miden las alturas...", flush=True)
        try:
            ruta = video_alturas.generar(s)
            if ruta:
                print(f"Video de alturas: {ruta}")
        except Exception as e:          # el análisis ya está guardado; el video es solo explicativo
            print(f"Aviso: no se pudo generar video_alturas.mp4 ({e}).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fuente", help="ruta del video o índice de cámara (0, 1, ...)")
    p.add_argument("--metodo", choices=("marcadores", "sin-marcadores"), default="marcadores",
                   help="marcadores de color o estimación de pose sin marcadores")
    p.add_argument("--modelo", choices=sm.MODELOS, default="rtmpose",
                   help="sin marcadores: rtmpose (más preciso) o mediapipe (más rápido)")
    p.add_argument("--detectar-cada", type=int, default=5,
                   help="sin marcadores (rtmpose): cuadros entre detecciones de la persona")
    p.add_argument("--vista", choices=("lateral", "frontal"), default="lateral")
    p.add_argument("--pierna", choices=("auto",) + PIERNAS, default="auto",
                   help="vista lateral: pierna del lado de la cámara (con marcadores, la que los lleva); "
                        "auto = según hacia dónde camina el sujeto")
    p.add_argument("--pierna-lejana", choices=("generar", "medir"), default="generar",
                   help="lateral sin marcadores: generar la pierna lejana desplazando medio ciclo la cercana "
                        "o usar la estimación del modelo (parcialmente tapada)")
    p.add_argument("--referencia", type=Path, default=None,
                   help="carpeta de resultados de una marcha de referencia para comparar "
                        "(por omisión, el patrón normal de adultos sanos)")
    p.add_argument("--direccion", choices=("auto", "derecha", "izquierda"), default="auto",
                   help="vista lateral: hacia dónde camina el sujeto en la imagen")
    p.add_argument("--frente", choices=("camara", "espalda"), default="camara",
                   help="vista frontal: el sujeto camina hacia la cámara o de espaldas a ella")
    p.add_argument("--modo", choices=("hsv", "rgb"), default="hsv",
                   help="hsv = umbrales recalibrados; rgb = umbrales originales de pp3.m")
    p.add_argument("--umbrales", type=Path, default=None, help="JSON de umbrales HSV (calibrar_hsv.py)")
    p.add_argument("--inicio", type=int, default=0, help="primer cuadro (base 0)")
    p.add_argument("--fin", type=int, default=None, help="cuadro final, exclusivo (base 0)")
    p.add_argument("--rotacion", default="auto", choices=("auto", "0", "90", "180", "270"))
    p.add_argument("--fc", type=float, default=6.0, help="frecuencia de corte del filtro [Hz]")
    p.add_argument("--diametro-marcador-cm", type=float, default=None,
                   help="diámetro real de los marcadores para expresar las alturas en cm")
    p.add_argument("--estatura-m", type=float, default=None,
                   help="estatura del sujeto en metros; obligatoria en vista frontal (escala de las alturas); "
                        "en lateral, opcional para expresar las alturas en cm")
    p.add_argument("--grabar", action="store_true", help="guardar captura.mp4 mientras se almacenan los datos")
    p.add_argument("--auto-detener", type=float, default=2.0,
                   help="terminar si no se detecta ninguna articulación durante estos segundos (0 = nunca)")
    p.add_argument("--duracion", type=float, default=None, help="terminar a los N segundos")
    p.add_argument("--escala-proceso", type=float, default=0.5, help="reducción de la imagen al segmentar")
    p.add_argument("--escala-vista", type=float, default=0.45, help="tamaño de la ventana / video anotado")
    p.add_argument("--sin-ventana", action="store_true", help="no mostrar el video mientras se procesa")
    p.add_argument("--video-anotado", action="store_true", help="guardar video_anotado.mp4")
    p.add_argument("--salida", type=Path, default=None, help="carpeta de resultados")
    args = p.parse_args()
    if args.vista == "frontal" and not args.estatura_m:
        p.error("la vista frontal necesita la estatura del sujeto: --estatura-m (p. ej. 1.70)")
    if args.estatura_m is not None and not 0.5 <= args.estatura_m <= 2.5:
        p.error("--estatura-m va en metros, p. ej. 1.70")
    if args.salida is None:
        nombre = "camara" if str(args.fuente).isdigit() else Path(args.fuente).stem
        if args.metodo == "sin-marcadores":
            nombre += "_sin_marcadores"
        args.salida = Path(__file__).resolve().parent / "resultados" / nombre
    args.salida.mkdir(parents=True, exist_ok=True)

    crudo, fps = adquirir(args)
    print(f"Cuadros leídos: {len(crudo)}")
    res = (procesar_frontal if args.vista == "frontal" else procesar_lateral)(crudo, fps, args)
    guardar(res, args)
    print(json.dumps(res["resumen"], ensure_ascii=False, indent=2))
    print(f"Resultados en: {args.salida}")


if __name__ == "__main__":
    main()
