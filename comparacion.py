"""Comparación de la marcha de la prueba contra la marcha ideal (formato de reporte clínico).

    py comparacion.py resultados/21                               # contra el patrón normal
    py comparacion.py resultados/21 --referencia resultados/sano  # contra otra prueba

Por cada articulación (cadera, rodilla, tobillo) y pierna: banda gris de la marcha
ideal (media ± 1 DE), curva de la prueba (media ± 1 DE de sus ciclos), línea del
despegue de la prueba y, arriba, las fases del ciclo. Genera en la carpeta de la
prueba comparacion_marcha.png y comparacion_marcha.csv (rangos, diferencia RMS,
correlación y % del ciclo fuera de la banda normal).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.transforms import blended_transform_factory

import graficas
import normativa
from graficas import FONDO, PCT, PIERNAS, TENUE, TINTA, TINTA_2

plt = graficas.plt

ARTICULACIONES = ("cadera", "rodilla", "tobillo")
COLOR_ART = {"cadera": "#008300", "rodilla": "#e34948", "tobillo": "#2a78d6"}
BANDA_NORMAL = "#d6d5cd"
AMBAR = "#eda100"
TITULO_ART = {"cadera": "Flexión/Extensión Cadera", "rodilla": "Flexión/Extensión Rodilla",
              "tobillo": "Dorsiflexión/Plantiflexión Tobillo"}
SIGNOS = {"cadera": ("flexión", "extensión"), "rodilla": ("flexión", "extensión"),
          "tobillo": ("dorsiflexión", "plantiflexión")}
FASES = (0, 12, 31, 50, None, 75, 87, 100)    # None = despegue de la prueba (resaltado)

# Proporciones del cuerpo (fracción de la estatura, Winter 2009) para los monigotes.
_MUSLO, _PIERNA, _PIE, _TOBILLO_H = 0.245, 0.246, 0.152, 0.039
_TRONCO, _CABEZA, _BRAZO, _ANTEBRAZO = 0.288, 0.06, 0.186, 0.25


def _pierna(cadera, ang_cadera, ang_rodilla, ang_tobillo):
    """Puntos de una pierna mirando hacia +x: cadera, rodilla, tobillo, talón, punta."""
    th = np.radians(ang_cadera)
    ts = np.radians(ang_cadera - ang_rodilla)
    rodilla = cadera + _MUSLO * np.array([np.sin(th), -np.cos(th)])
    tobillo = rodilla + _PIERNA * np.array([np.sin(ts), -np.cos(ts)])
    phi = ts + np.radians(ang_tobillo)            # dirección del pie respecto a la horizontal
    u = np.array([np.cos(phi), np.sin(phi)])
    n = np.array([np.sin(phi), -np.cos(phi)])     # perpendicular hacia la planta
    talon = tobillo - 0.22 * _PIE * u + _TOBILLO_H * n
    punta = tobillo + 0.78 * _PIE * u + _TOBILLO_H * n
    return cadera, rodilla, tobillo, talon, punta


def _monigote(ax, pct, normal, resaltar):
    """Postura del ciclo normal en `pct` %: pierna de la gráfica oscura, la otra tenue."""
    ang = {lado: {a: np.interp((pct + desfase) % 100, PCT, normal["curvas"][a][0]) for a in ARTICULACIONES}
           for lado, desfase in (("propia", 0.0), ("otra", 50.0))}
    cadera = np.array([0.0, 0.0])
    piernas = {lado: _pierna(cadera, ang[lado]["cadera"], ang[lado]["rodilla"], ang[lado]["tobillo"])
               for lado in ang}
    suelo = min(p[1] for pts in piernas.values() for p in pts[3:])
    desplazar = np.array([0.0, -suelo])
    color = AMBAR if resaltar else TINTA
    tenue = AMBAR if resaltar else TENUE
    alfa_otra = 0.45 if resaltar else 1.0
    for lado, c, alfa, z in (("otra", tenue, alfa_otra, 1), ("propia", color, 1.0, 3)):
        cad, rod, tob, tal, pun = (p + desplazar for p in piernas[lado])
        ax.plot(*np.array([cad, rod, tob]).T, color=c, alpha=alfa, lw=1.5, solid_capstyle="round", zorder=z)
        ax.plot(*np.array([tob, tal, pun, tob]).T, color=c, alpha=alfa, lw=1.0, zorder=z)
    hombro = cadera + desplazar + _TRONCO * np.array([np.sin(np.radians(4)), np.cos(np.radians(4))])
    ax.plot(*np.array([cadera + desplazar, hombro]).T, color=color, lw=1.5, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle(hombro + np.array([0.012, 0.05 + _CABEZA]), _CABEZA, facecolor=color, edgecolor="none",
                        zorder=2))
    for lado, c, alfa in (("otra", tenue, alfa_otra), ("propia", color, 1.0)):
        # El brazo se balancea al revés que el muslo del mismo lado.
        a = np.radians(-0.6 * (ang[lado]["cadera"] - 9.0))
        codo = hombro + _BRAZO * np.array([np.sin(a), -np.cos(a)])
        b = a + np.radians(20)
        mano = codo + 0.6 * _ANTEBRAZO * np.array([np.sin(b), -np.cos(b)])
        ax.plot(*np.array([hombro, codo, mano]).T, color=c, alpha=alfa, lw=1.1, solid_capstyle="round", zorder=1)
    ax.set_xlim(-0.42, 0.42)
    ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _tira_fases(ax, normal, despegue):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    mezcla = blended_transform_factory(ax.transData, ax.transAxes)
    for pct in FASES:
        resaltar = pct is None
        if not resaltar and abs(pct - despegue) < 9:       # no encimar con el despegue
            continue
        x = despegue if resaltar else pct
        inset = ax.inset_axes([x - 6, 0.0, 12, 1.0], transform=mezcla)
        _monigote(inset, x, normal, resaltar)


def _icono(ax, art):
    """Esquema de la articulación con el sentido positivo (+) y negativo (−) del ángulo."""
    color = COLOR_ART[art]
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    hueso = dict(color=TINTA_2, lw=3.2, solid_capstyle="round")
    fantasma = dict(color=TENUE, lw=2.2, alpha=0.6, solid_capstyle="round")
    flecha = dict(arrowstyle="<|-|>", mutation_scale=11, color=color, lw=1.8)
    if art == "cadera":
        c = np.array([0.0, 0.9])
        ax.add_patch(Circle(c, 0.07, color=TINTA_2))
        ax.plot([0, 0], [0.9, 0.1], **hueso)
        a = np.radians(35)
        ax.plot([0, 0.8 * np.sin(a)], [0.9, 0.9 - 0.8 * np.cos(a)], **fantasma)
        p_mas = c + 0.55 * np.array([np.sin(np.radians(40)), -np.cos(np.radians(40))])
        p_menos = c + 0.55 * np.array([np.sin(np.radians(-25)), -np.cos(np.radians(-25))])
    elif art == "rodilla":
        k = np.array([0.0, 0.5])
        ax.plot([0, 0], [1.05, 0.5], **hueso)
        ax.plot([0, 0], [0.5, -0.05], **hueso)
        a = np.radians(-40)
        ax.plot([0, 0.55 * np.sin(a)], [0.5, 0.5 - 0.55 * np.cos(a)], **fantasma)
        ax.add_patch(Circle(k, 0.06, color=TINTA_2))
        p_mas = k + 0.4 * np.array([np.sin(np.radians(-45)), -np.cos(np.radians(-45))])
        p_menos = k + 0.4 * np.array([np.sin(np.radians(20)), -np.cos(np.radians(20))])
    else:
        t = np.array([0.0, 0.3])
        ax.plot([0, 0], [1.05, 0.3], **hueso)
        ax.plot([-0.1, 0.42], [0.18, 0.18], **hueso)
        ax.plot([0, 0.42], [0.3, 0.18], **hueso)
        a = np.radians(22)
        ax.plot([-0.1, -0.1 + 0.52 * np.cos(a)], [0.18, 0.18 + 0.52 * np.sin(a)], **fantasma)
        ax.add_patch(Circle(t, 0.06, color=TINTA_2))
        p_mas = np.array([0.45, 0.5])
        p_menos = np.array([0.45, -0.08])
    ax.add_patch(FancyArrowPatch(p_menos, p_mas, connectionstyle="arc3,rad=0.45", **flecha))
    centro = {"cadera": np.array([0.0, 0.9]), "rodilla": np.array([0.0, 0.5]), "tobillo": np.array([0.0, 0.2])}[art]
    for p, signo in ((p_mas, "+"), (p_menos, "−")):
        u = (p - centro) / np.linalg.norm(p - centro)
        ax.text(*(p + 0.13 * u), signo, color=TINTA, fontsize=12, ha="center", va="center", fontweight="bold")
    mas, menos = SIGNOS[art]
    ax.text(0, -0.13, f"+ {mas}\n− {menos}", color=TINTA_2, fontsize=7.5, ha="center", va="top")


def curvas_prueba(tabla, resumen):
    """{(art, pierna): {media, de, n, apoyo, origen}} de los ángulos sagitales."""
    return {(art, pierna): c for (tipo, art, pierna), c in
            graficas.curvas_desde_tabla(tabla, resumen, ("angulo",)).items()}


def figura(curvas, normal, titulo, subtitulo, ruta):
    fig = plt.figure(figsize=(12.6, 14.6))
    rejilla = fig.add_gridspec(3, 3, width_ratios=[0.52, 2, 2], left=0.015, right=0.985, top=0.895, bottom=0.06,
                               hspace=0.36, wspace=0.2)
    filas = []
    for fila, art in enumerate(ARTICULACIONES):
        ax_icono = fig.add_subplot(rejilla[fila, 0])
        ax_icono.axis("off")
        ax_icono.text(0.5, 0.97, art.upper(), transform=ax_icono.transAxes, ha="center", va="top",
                      fontsize=11, fontweight="bold", color=TINTA)
        _icono(ax_icono.inset_axes([0.0, 0.08, 1.0, 0.75]), art)
        ejes = []
        for col, pierna in enumerate(PIERNAS):
            sub = rejilla[fila, col + 1].subgridspec(2, 1, height_ratios=[0.3, 1], hspace=0.03)
            ax_fases, ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            ejes.append((ax_fases, ax, pierna))
        filas.append((art, ejes))

    metricas = []
    for art, ejes in filas:
        nm, nd = normal["curvas"][art]
        limites = [np.min(nm - nd), np.max(nm + nd)]
        for ax_fases, ax, pierna in ejes:
            c = curvas.get((art, pierna))
            despegue = c["apoyo"] if c and c["apoyo"] is not None and np.isfinite(c["apoyo"]) else 62.0
            _tira_fases(ax_fases, normal, despegue)
            generada = c is not None and c["origen"] == "sintetica"
            ax_fases.set_title(f"{TITULO_ART[art]} · pierna {pierna}" + (" (generada)" if generada else ""),
                               fontsize=10, fontweight="bold", color=TINTA, pad=4)
            ax.fill_between(PCT, nm - nd, nm + nd, color=BANDA_NORMAL, linewidth=0, zorder=1)
            ax.axhline(0, color=TINTA_2, lw=0.9, zorder=2)
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 10))
            ax.set_ylabel("Ángulo [°]")
            ax.set_xlabel("Ciclo [%]")
            if c is None:
                ax.text(0.5, 0.5, "sin datos", transform=ax.transAxes, ha="center", color=TENUE)
                continue
            m, de = c["media"], c["de"]
            limites += [np.min(m - de), np.max(m + de)]
            color = COLOR_ART[art]
            ax.fill_between(PCT, m - de, m + de, color=color, alpha=0.22, linewidth=0, zorder=3)
            ax.plot(PCT, m, color=color, lw=2.2, solid_capstyle="round", zorder=4)
            if c["apoyo"] is not None and np.isfinite(c["apoyo"]):
                ax.axvline(c["apoyo"], color=AMBAR, lw=1.8, zorder=5)
                ax.annotate(f"despegue {c['apoyo']:.0f} %", (c["apoyo"], 1), xycoords=("data", "axes fraction"),
                            xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=TINTA_2)
            met = normativa.metricas(m, de, nm, nd)
            ax.text(0.0, -0.25, f"Dif. RMS {met['dif_rms']:.1f}° · r = {met['correlacion']:.2f} · "
                                f"rango {met['rango']:.0f}° (normal {met['normal_rango']:.0f}°) · "
                                f"fuera de la banda {met['fuera_banda_pct']:.0f} % del ciclo",
                    transform=ax.transAxes, fontsize=7.5, color=TINTA_2, va="top")
            metricas.append({"articulacion": art, "pierna": pierna, "origen": c["origen"], "ciclos": c["n"],
                             **{k: round(v, 2) for k, v in met.items()}})
        margen = 0.08 * (max(limites) - min(limites))
        for _, ax, _ in ejes:
            ax.set_ylim(min(limites) - margen, max(limites) + margen)

    prueba = tuple(Line2D([], [], color=COLOR_ART[a], lw=2.2) for a in ARTICULACIONES)
    manejadores = [Patch(color=BANDA_NORMAL, label=f"{normal['nombre']} · media ± 1 DE"),
                   prueba, Line2D([], [], color=AMBAR, lw=1.8)]
    etiquetas = [manejadores[0].get_label(), "Prueba · media ± 1 DE de sus ciclos", "Despegue de la prueba"]
    fig.legend(manejadores, etiquetas, loc="upper left", bbox_to_anchor=(0.012, 0.938), ncol=3, fontsize=9,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, handlelength=4)
    fig.text(0.015, 0.987, titulo, fontsize=14, color=TINTA, va="top")
    fig.text(0.015, 0.966, subtitulo, fontsize=8.5, color=TINTA_2, va="top", linespacing=1.5)
    fig.savefig(ruta, dpi=130, facecolor=FONDO)
    plt.close(fig)
    return pd.DataFrame(metricas)


def generar(carpeta, referencia=None, salida=None):
    """Lee los resultados de analisis_marcha.py y genera la comparación. Regresa la ruta del PNG."""
    carpeta = Path(carpeta)
    salida = Path(salida) if salida else carpeta
    salida.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(carpeta / "ciclos_normalizados.csv")
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    if resumen.get("vista") != "lateral":
        raise ValueError("La comparación de ángulos sagitales necesita un análisis lateral.")
    normal = normativa.cargar(referencia)
    curvas = curvas_prueba(tabla, resumen)
    metodo = resumen.get("metodo", "marcadores de color")
    titulo = f"Marcha de la prueba contra la marcha ideal · {Path(resumen['fuente']).name}"
    notas = []
    if not resumen.get("tobillo_con_pie", False):
        notas.append("tobillo sin marcador en el pie: se supone el pie horizontal")
    generadas = [p for p in PIERNAS if resumen["por_pierna"][p]["origen"] == "sintetica"]
    if generadas:
        notas.append(f"pierna {generadas[0]} generada desplazando medio ciclo la otra")
    subtitulo = f"Detección: {metodo} · Marcha ideal: {normal['detalle']}"
    if notas:
        subtitulo += "\n" + "Nota: " + " · ".join(notas)
    ruta = salida / "comparacion_marcha.png"
    tabla_metricas = figura(curvas, normal, titulo, subtitulo, ruta)
    tabla_metricas.insert(0, "marcha_ideal", normal["nombre"])
    tabla_metricas.to_csv(salida / "comparacion_marcha.csv", index=False)
    return ruta, tabla_metricas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", type=Path, help="carpeta de resultados de la prueba (análisis lateral)")
    p.add_argument("--referencia", type=Path, default=None,
                   help="carpeta de resultados de una marcha de referencia (por omisión, el patrón normal)")
    p.add_argument("--salida", type=Path, default=None, help="carpeta de salida (por omisión, la de la prueba)")
    args = p.parse_args()
    ruta, tabla = generar(args.carpeta, args.referencia, args.salida)
    print(tabla.to_string(index=False))
    print(f"Comparación: {ruta}")


if __name__ == "__main__":
    main()
