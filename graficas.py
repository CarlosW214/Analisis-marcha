"""Gráficas del análisis de marcha (las 12 obligatorias, la vista frontal y las series temporales)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import normativa  # noqa: E402

FONDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
COLOR_PIERNA = {"derecha": "#2a78d6", "izquierda": "#eb6834"}
BANDA_NORMAL = "#d6d5cd"
PIERNAS = ("derecha", "izquierda")
PCT = np.linspace(0, 100, 101)

ETIQUETA_ANGULO = {
    "cadera": "Flexión (+) / extensión (−) [°]",
    "rodilla": "Flexión [°]",
    "tobillo": "Dorsiflexión (+) / plantiflexión (−) [°]",
}
ETIQUETA_FRONTAL = {
    "cadera": "Abducción (+) / aducción (−) [°]",
    "rodilla": "Varo (+) / valgo (−) [°]",
}

# (tipo, articulación, título, plano); la numeración sigue la tabla de la práctica.
FILAS_12 = [
    ("altura", "cadera", "Posición Y (altura) cadera", "Frontal"),
    ("altura", "rodilla", "Posición Y (altura) rodilla", "Frontal"),
    ("altura", "tobillo", "Posición Y (altura) tobillo", "Frontal"),
    ("angulo", "cadera", "Ángulo cadera", "Sagital"),
    ("angulo", "rodilla", "Ángulo rodilla", "Sagital"),
    ("angulo", "tobillo", "Ángulo tobillo", "Sagital"),
]
FILAS_FRONTAL = [
    ("altura", "cadera", "Posición Y (altura) cadera", "Frontal"),
    ("altura", "rodilla", "Posición Y (altura) rodilla", "Frontal"),
    ("altura", "tobillo", "Posición Y (altura) tobillo", "Frontal"),
    ("frontal", "cadera", "Ángulo cadera", "Frontal"),
    ("frontal", "rodilla", "Ángulo rodilla", "Frontal"),
]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.facecolor": FONDO,
    "figure.facecolor": FONDO,
    "savefig.facecolor": FONDO,
    "axes.edgecolor": EJE,
    "axes.labelcolor": TINTA_2,
    "axes.titlecolor": TINTA,
    "xtick.color": TENUE,
    "ytick.color": TENUE,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.labelcolor": TINTA_2,
})


def curvas_desde_tabla(tabla, resumen, tipos):
    """{(tipo, art, pierna): curva} a partir de ciclos_normalizados.csv y resumen.json."""
    curvas = {}
    for pierna in PIERNAS:
        info = resumen["por_pierna"][pierna]
        for tipo in tipos:
            for art in ("cadera", "rodilla", "tobillo"):
                col = f"{pierna}_{tipo}_{art}_media"
                if col not in tabla:
                    continue
                curvas[(tipo, art, pierna)] = {
                    "media": np.asarray(tabla[col], float),
                    "de": np.asarray(tabla[f"{pierna}_{tipo}_{art}_de"], float),
                    "n": info["ciclos"],
                    "apoyo": info["apoyo_pct"],
                    "origen": info["origen"],
                    "unidad": resumen["unidad_altura"],
                    "fuente": resumen["fuente"],
                }
    return curvas


def _panel(ax, curva, pierna, tipo, art, numero, titulo, plano, normal):
    color = COLOR_PIERNA[pierna]
    media, de = curva["media"], curva["de"]
    if tipo == "angulo":
        nm, nd = normal["curvas"][art]
        ax.fill_between(PCT, nm - nd, nm + nd, color=BANDA_NORMAL, linewidth=0, zorder=1,
                        label=f"{normal['nombre']} (media ± 1 DE)")
    ax.fill_between(PCT, media - de, media + de, color=color, alpha=0.14, linewidth=0, label="± 1 DE")
    origen = "generada a medio ciclo" if curva["origen"] == "sintetica" else "medida"
    ax.plot(PCT, media, color=color, linewidth=2, solid_capstyle="round",
            label=f"Sujeto, pierna {pierna} ({origen}, n = {curva['n']} ciclos)")
    if tipo == "angulo":
        ax.set_ylabel(ETIQUETA_ANGULO[art])
    elif tipo == "frontal":
        ax.set_ylabel(ETIQUETA_FRONTAL[art])
    else:
        ax.set_ylabel(f"Altura [{curva['unidad']}]")
    apoyo = curva["apoyo"]
    if apoyo is not None and np.isfinite(apoyo):
        ax.axvline(apoyo, color=TENUE, linewidth=1)
        ax.annotate(f"despegue {apoyo:.0f} %", (apoyo, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -10), textcoords="offset points", ha="right", color=TENUE, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    prefijo = f"Gráfica {numero} · " if numero else ""
    sufijo = " (generada)" if curva["origen"] == "sintetica" else ""
    ax.set_title(f"{prefijo}{titulo} · {plano} · pierna {pierna}{sufijo}", loc="left", fontsize=9.5)


def figura_ciclos(filas, curvas, titulo, ruta, carpeta_sueltas=None, numerar=True, normal=None):
    """Figura de filas x 2 piernas con media ± DE por % de ciclo; opcionalmente cada panel suelto.

    normal: marcha ideal de normativa.cargar() para los ángulos (por omisión, el patrón normal).
    """
    normal = normal or normativa.cargar()
    fig, ejes = plt.subplots(len(filas), 2, figsize=(12, 3.15 * len(filas) + 0.6), sharex=True, squeeze=False)
    if carpeta_sueltas:
        Path(carpeta_sueltas).mkdir(parents=True, exist_ok=True)
    for fila, (tipo, art, nombre, plano) in enumerate(filas):
        for col, pierna in enumerate(PIERNAS):
            numero = 2 * fila + col + 1 if numerar else None
            ax = ejes[fila, col]
            curva = curvas.get((tipo, art, pierna))
            if curva is None:
                ax.text(0.5, 0.5, "sin datos", transform=ax.transAxes, ha="center", color=TENUE)
                ax.set_title(f"{nombre} · {plano} · pierna {pierna}", loc="left", fontsize=9.5)
                continue
            _panel(ax, curva, pierna, tipo, art, numero, nombre, plano, normal)
            if carpeta_sueltas:
                f1, a1 = plt.subplots(figsize=(6.4, 3.6))
                _panel(a1, curva, pierna, tipo, art, numero, nombre, plano, normal)
                a1.set_xlabel("% del ciclo de marcha")
                a1.legend(loc="best", fontsize=8)
                f1.tight_layout()
                f1.savefig(Path(carpeta_sueltas) / f"g{numero:02d}_{tipo}_{art}_{pierna}.png", dpi=150)
                plt.close(f1)
    for ax in ejes[-1]:
        ax.set_xlabel("% del ciclo de marcha")
    manejadores = {}
    for ax in ejes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            manejadores.setdefault(l.split(" (")[0], h)
    fig.legend(manejadores.values(), manejadores.keys(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1 - 0.45 / fig.get_figheight()), fontsize=9)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def serie_temporal(t, paneles, contactos, titulo, ruta):
    """paneles: [(etiqueta_y, [(leyenda, valores, color, grosor), ...]), ...] contra el tiempo."""
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(12, 2.6 * len(paneles) + 0.4), sharex=True, squeeze=False)
    ejes = ejes[:, 0]
    for ax, (etiqueta, series) in zip(ejes, paneles):
        for leyenda, valores, color, grosor in series:
            ax.plot(t, valores, color=color, linewidth=grosor, label=leyenda)
        for k, tc in enumerate(contactos):
            ax.axvline(tc, color=TENUE, linewidth=0.8, label="Contacto inicial" if k == 0 else None)
        ax.set_ylabel(etiqueta)
    ejes[0].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=4, fontsize=8)
    ejes[-1].set_xlabel("Tiempo [s]")
    ejes[-1].set_xlim(t[0], t[-1])
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta
