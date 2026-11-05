"""Cálculo de ángulos articulares, filtrado y normalización del ciclo de marcha.

Convención de coordenadas de imagen: x crece a la derecha, y crece hacia abajo.
`direccion` = +1 si el sujeto camina (mira) hacia la derecha del video, -1 si
hacia la izquierda; así "adelante" siempre tiene signo positivo.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

# El patrón normal (media ± DE) está en normativa.py / datos/normativa_uci.csv.

# Alturas de pie como fracción de la estatura H (Winter, 2009): trocánter mayor,
# rodilla (epicóndilo femoral) y maléolo externo, los puntos de los marcadores.
ALTURA_DE_PIE_H = {"cadera": 0.530, "rodilla": 0.285, "tobillo": 0.039}
PIERNA_H = ALTURA_DE_PIE_H["cadera"] - ALTURA_DE_PIE_H["tobillo"]      # cadera-tobillo = 0.491 H


def angulos_sagitales(cadera, rodilla, tobillo, direccion, talon=None, punta=None):
    """Ángulos en grados a partir de posiciones (N, 2) en px.

    - cadera : muslo (cadera->rodilla) contra la vertical. + = flexión.
    - rodilla: ángulo entre los vectores muslo y pierna (rodilla->tobillo). + = flexión.
    - tobillo: pierna contra la horizontal, expresado como 90° - ese ángulo,
               + = dorsiflexión, - = plantiflexión. Sin talón ni punta (marcadores)
               se supone el pie horizontal; con ellos (sin marcadores) se suma la
               inclinación del pie talón->punta, + = punta arriba.
    """
    muslo = rodilla - cadera
    pierna = tobillo - rodilla
    ang_muslo = np.degrees(np.arctan2(direccion * muslo[:, 0], muslo[:, 1]))
    ang_pierna = np.degrees(np.arctan2(direccion * pierna[:, 0], pierna[:, 1]))
    ang_pie = 0.0
    if talon is not None and punta is not None:
        pie = punta - talon
        ang_pie = np.degrees(np.arctan2(-pie[:, 1], direccion * pie[:, 0]))
    return {
        "cadera": ang_muslo,
        "rodilla": ang_muslo - ang_pierna,
        "tobillo": ang_pie - ang_pierna,
    }


def angulos_frontales(cadera, rodilla, tobillo, lateral):
    """Ángulos en el plano frontal (grados) a partir de posiciones (N, 2) en px.

    lateral = +1 si el lado externo de esa pierna está hacia +x en la imagen.
    - cadera : muslo contra la vertical. + = abducción, - = aducción.
    - rodilla: muslo menos pierna.       + = varo,      - = valgo.
    """
    muslo = rodilla - cadera
    pierna = tobillo - rodilla
    ang_muslo = np.degrees(np.arctan2(lateral * muslo[:, 0], muslo[:, 1]))
    ang_pierna = np.degrees(np.arctan2(lateral * pierna[:, 0], pierna[:, 1]))
    return {"cadera": ang_muslo, "rodilla": ang_muslo - ang_pierna}


def escala_m_por_px(diametro_px, pierna_px, diametro_marcador_cm=None, estatura_m=None):
    """Metros por píxel a partir del diámetro real del marcador o de la estatura.

    Con la estatura se usa la proporción antropométrica de Winter (2009):
    pierna (rodilla-tobillo) = 0.246 x estatura.
    """
    if diametro_marcador_cm:
        return diametro_marcador_cm / 100.0 / diametro_px, "diámetro del marcador"
    if estatura_m:
        return 0.246 * estatura_m / pierna_px, "estatura (pierna = 0.246 H)"
    return None, None


def rellenar_huecos(senal):
    """Interpola linealmente los cuadros donde no se detectó el marcador (NaN)."""
    senal = np.asarray(senal, float).copy()
    validos = np.isfinite(senal)
    if validos.sum() < 2:
        raise ValueError("Muy pocos cuadros con el marcador detectado")
    idx = np.arange(senal.size)
    senal[~validos] = np.interp(idx[~validos], idx[validos], senal[validos])
    return senal


def filtrar(senal, fs, fc=6.0, orden=4):
    """Butterworth pasa-bajas de fase cero (filtfilt), fc típica de 6 Hz para marcha."""
    b, a = butter(orden, min(fc / (fs / 2.0), 0.99))
    return filtfilt(b, a, senal, padlen=min(3 * max(len(a), len(b)), senal.size - 1))


def detectar_direccion(x_cadera, x_tobillo):
    """El pie avanza rápido en el balanceo (~40 %) y retrocede lento en el apoyo (~60 %)."""
    v = np.diff(x_tobillo - x_cadera)
    return 1 if np.mean(v > 0) < 0.5 else -1


def suavizar_lento(senal, fs, ventana_s):
    """Media móvil de `ventana_s` segundos (p. ej. un ciclo): deja solo los cambios lentos."""
    return uniform_filter1d(np.asarray(senal, float), max(3, int(round(ventana_s * fs))), mode="nearest")


def reposo_inicial(senal, fs, umbral=0.03, minimo_s=0.5, margen_s=0.2):
    """Cuadros del inicio en que la persona está de pie y quieta, antes de echar a andar.

    senal: diferencia de altura entre los tobillos dividida entre la longitud de la
    pierna; de pie, con los pies juntos, es ~0 y no cambia. El reposo termina cuando
    se aparta más de `umbral` de su valor inicial. Regresa un slice, o None si el
    video no empieza con la persona quieta (al menos `minimo_s`).
    """
    senal = np.asarray(senal, float)
    base = np.median(senal[:max(3, int(0.3 * fs))])
    if abs(base) > 2 * umbral:                      # no empieza con los pies juntos
        return None
    fuera = np.flatnonzero(np.abs(senal - base) > umbral)
    fin = (fuera[0] if fuera.size else senal.size) - int(margen_s * fs)
    return slice(0, fin) if fin >= minimo_s * fs else None


def _estimar_periodo(senal, fs, t_min=0.6, t_max=2.5):
    s = senal - senal.mean()
    ac = np.correlate(s, s, mode="full")[s.size - 1:]
    lo, hi = int(t_min * fs), min(int(t_max * fs), s.size - 1)
    if hi <= lo:
        return None
    return (lo + int(np.argmax(ac[lo:hi]))) / fs


def _refinar_pico(y, i):
    """Interpolación parabólica para ubicar el pico entre cuadros."""
    if 0 < i < y.size - 1:
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        if den != 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / den
    return float(i)


def detectar_eventos(t, x_cadera, x_tobillo, direccion, fs):
    """Contacto inicial y despegue por el método de Zeni et al. (2008).

    Contacto inicial = máximo de la posición del tobillo por delante de la
    cadera; despegue = mínimo (tobillo lo más atrás posible).
    Regresa (tiempos de contacto, tiempos de despegue, periodo estimado).
    """
    rel = direccion * (x_tobillo - x_cadera)
    periodo = _estimar_periodo(rel, fs)
    maximos, minimos = _maximos_y_minimos(t, rel, fs, periodo)
    return maximos, minimos, periodo


def _maximos_y_minimos(t, senal, fs, periodo):
    """Tiempos (entre cuadros) de los máximos y mínimos de una señal casi periódica."""
    distancia = max(1, int(0.6 * periodo * fs)) if periodo else 1
    prominencia = 0.3 * (np.percentile(senal, 95) - np.percentile(senal, 5))
    idx = np.arange(t.size)
    tiempos = []
    for s in (senal, -senal):
        picos, _ = find_peaks(s, distance=distancia, prominence=prominencia)
        tiempos.append(np.interp([_refinar_pico(s, i) for i in picos], idx, t))
    return tiempos[0], tiempos[1]


def periodo_frontal(t, y_tobillo_der, y_tobillo_izq, fs, frente="camara"):
    """Duración del ciclo en la vista frontal (s) o None.

    Por la perspectiva, el pie adelantado se ve más abajo en la imagen si el sujeto
    camina hacia la cámara, así que y_der - y_izq oscila una vez por ciclo; al ser una
    diferencia entre tobillos, no la afecta que la cámara se mueva. Se mide entre sus
    máximos (o mínimos) y, si solo hay uno de cada, como el doble del paso.
    """
    g = (1 if frente == "camara" else -1) * (np.asarray(y_tobillo_der) - np.asarray(y_tobillo_izq))
    periodo = _estimar_periodo(g, fs, t_max=4.0)
    maximos, minimos = _maximos_y_minimos(t, g, fs, periodo)
    entre = np.concatenate([np.diff(maximos), np.diff(minimos)])
    if entre.size:
        return float(np.median(entre))
    if maximos.size and minimos.size:
        return 2.0 * abs(float(minimos[0] - maximos[0]))
    return periodo


def detectar_contactos_frontal(t, y_tobillo, y_cadera, m_por_px, fs, periodo, frente="camara"):
    """Contactos iniciales de una pierna en la vista frontal (tiempos en s).

    Se usa la altura del tobillo respecto a la cadera (en m): de frente a la cámara,
    el pie que avanza en el balanceo baja en la imagen hasta el contacto y, ya
    apoyado, sube conforme el cuerpo pasa sobre él; el contacto es el máximo. A
    diferencia de la separación entre los pies, no la retrasa el talón de la otra
    pierna al despegar. En el video Marcha_frontal_ejemplo.mp4 coincide, revisando
    los cuadros, con el pie recién apoyado (±0.1 s).
    """
    h = (1 if frente == "camara" else -1) * (np.asarray(y_tobillo) - np.asarray(y_cadera)) * m_por_px
    maximos, _ = _maximos_y_minimos(t, h, fs, periodo)
    return maximos


def armar_ciclos(contactos, despegues, tolerancia=0.25):
    """Ciclos [contacto_k, contacto_k+1] con su despegue; descarta los atípicos."""
    ciclos = []
    for t0, t1 in zip(contactos[:-1], contactos[1:]):
        dentro = despegues[(despegues > t0) & (despegues < t1)]
        apoyo = 100.0 * (dentro[0] - t0) / (t1 - t0) if dentro.size else np.nan
        ciclos.append((t0, t1, apoyo))
    if not ciclos:
        return []
    duracion = np.array([c[1] - c[0] for c in ciclos])
    med = np.median(duracion)
    return [c for c, d in zip(ciclos, duracion) if abs(d - med) <= tolerancia * med]


def normalizar_ciclos(t, senal, ciclos, n=101):
    """Remuestrea cada ciclo a n puntos (0-100 % del ciclo). Regresa (n_ciclos, n).

    Si un ciclo se sale un poco del video, esa parte toma el primer o último cuadro.
    """
    salida = []
    for t0, t1, _ in ciclos:
        salida.append(np.interp(np.linspace(t0, t1, n), t, senal))
    return np.array(salida).reshape(len(salida), n)


def desplazar(t, senal, retraso, periodo):
    """Pierna contralateral: misma señal retrasada `retraso` s (medio ciclo).

    El primer medio ciclo, que no tiene datos previos, se toma un ciclo después
    (se supone marcha periódica).
    """
    tq = t - retraso
    tq = np.where(tq < t[0], tq + periodo, tq)
    return np.interp(tq, t, senal, left=np.nan, right=np.nan)
