"""Combina los resultados de varios videos en las 12 gráficas obligatorias de la práctica.

    py reporte_12.py --lateral resultados/21
    py reporte_12.py --lateral resultados/21 --frontal resultados/frontal
    py reporte_12.py --lateral resultados/der --lateral resultados/izq --frontal resultados/frontal

Gráficas 7-12 (ángulos sagitales): de los videos laterales. Si una pierna no se grabó
se usa la generada con medio ciclo de desfase (caso especial de la práctica).
Gráficas 1-6 (altura, plano frontal): del video frontal si se da; si no, de los laterales.
En cada gráfica se prefiere la pierna medida sobre la generada.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import graficas


def cargar(carpeta):
    carpeta = Path(carpeta)
    tabla = pd.read_csv(carpeta / "ciclos_normalizados.csv")
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    return tabla, resumen


def elegir(fuentes, tipo, art, pierna):
    """Curva de la primera fuente donde esa pierna se midió; si no hay, la primera disponible."""
    respaldo = None
    for tabla, resumen in fuentes:
        curva = graficas.curvas_desde_tabla(tabla, resumen, (tipo,)).get((tipo, art, pierna))
        if curva is None:
            continue
        if curva["origen"] == "medida":
            return curva
        respaldo = respaldo or curva
    return respaldo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lateral", action="append", required=True, type=Path, help="carpeta de un análisis lateral")
    p.add_argument("--frontal", type=Path, default=None, help="carpeta de un análisis frontal")
    p.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados" / "reporte")
    args = p.parse_args()

    laterales = [cargar(c) for c in args.lateral]
    frontal = [cargar(args.frontal)] if args.frontal else []
    curvas, fuentes = {}, []
    for fila, (tipo, art, nombre, plano) in enumerate(graficas.FILAS_12):
        for col, pierna in enumerate(graficas.PIERNAS):
            curva = elegir(frontal + laterales if tipo == "altura" else laterales, tipo, art, pierna)
            if curva is None:
                continue
            curvas[(tipo, art, pierna)] = curva
            fuentes.append({"grafica": 2 * fila + col + 1, "plano": plano, "variable": nombre, "pierna": pierna,
                            "video": Path(curva["fuente"]).name, "origen": curva["origen"], "ciclos": curva["n"]})

    args.salida.mkdir(parents=True, exist_ok=True)
    sagital = ", ".join(Path(r["fuente"]).name for _, r in laterales)
    titulo = f"Las 12 gráficas · sagital: {sagital}"
    if frontal:
        titulo += f" · frontal: {Path(frontal[0][1]['fuente']).name}"
    graficas.figura_ciclos(graficas.FILAS_12, curvas, titulo, args.salida / "graficas_12.png", args.salida / "graficas")
    tabla = pd.DataFrame(fuentes)
    tabla.to_csv(args.salida / "fuentes_graficas.csv", index=False)
    print(tabla.to_string(index=False))
    print(f"Resultados en: {args.salida}")


if __name__ == "__main__":
    main()
