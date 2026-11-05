# Análisis de marcha por videografía (Python + Blender)

Práctica 4 de Bioinstrumentación IV. Versión en Python del código MATLAB del proyecto,
ampliada para cubrir los tres experimentos de la práctica: detección en tiempo real
(con marcadores de color o sin marcadores), almacenamiento, 12 gráficas, comparación con
la marcha ideal y animación 3D en Blender.

## Uso sin terminal: `Analisis_Marcha.bat`

Doble clic en **`Analisis_Marcha.bat`** (o arrastrar un video sobre él). Se abre una ventana
donde se elige el video o la cámara, el método (sin marcadores o con marcadores), la vista,
la marcha ideal contra la que se compara, y se presiona **Analizar**. Al terminar se abre la
gráfica de comparación; los botones abren la carpeta de resultados o generan el modelo 3D.

La primera vez el .bat prepara todo solo (necesita internet, de 5 a 15 minutos):

1. Busca Python ≥ 3.10 de 64 bits; si no hay, ofrece instalar Python 3.12 con `winget`
   o abre la página de descarga.
2. Crea un entorno propio en `%LOCALAPPDATA%\AnalisisMarcha\venv`, fuera de Google Drive
   (son miles de archivos que Drive intentaría sincronizar).
3. Instala `requirements.txt` y `rtmlib` y descarga los modelos sin marcadores (~100 MB).

Las siguientes veces abre la ventana directamente. Si `requirements.txt` cambia, se
actualiza solo. `Analisis_Marcha.bat /reinstalar` vuelve a instalar las bibliotecas.

## Instalación manual

```
py -m pip install -r requirements.txt
py -m pip install --no-deps rtmlib==0.0.16
```

`rtmlib` va aparte y sin dependencias porque pide `opencv-python`, que choca con
`opencv-contrib-python` (el que necesita MediaPipe). Si ya tenía instalado `opencv-python`
de la versión anterior, conviene usar un entorno nuevo (lo que hace el .bat).

El experimento 3 necesita Blender (probado con 5.1). Se busca en el PATH, en
`C:\Program Files\Blender Foundation` y en la carpeta de Steam. Si está en otro lugar, el
lanzador pide elegir `blender.exe` (o use `--blender RUTA`); la ruta se recuerda en
`%LOCALAPPDATA%\AnalisisMarcha\blender.txt`.

## Flujo de trabajo

| Paso | Comando | Experimento |
|---|---|---|
| 0. Calibrar colores (opcional, con marcadores) | `py calibrar_hsv.py ../21.MP4` → guarda `umbrales_hsv.json` con `s` | 1 |
| 1. Analizar el video lateral | `py analisis_marcha.py ../21.MP4` (marcadores) o `py analisis_marcha.py ../21.MP4 --metodo sin-marcadores` | 1 y 2 |
| 2. Analizar un video frontal (opcional) | `py analisis_marcha.py frontal.mp4 --vista frontal --estatura-m 1.60` (estatura obligatoria; admite `--metodo sin-marcadores`) | 1 y 2 |
| 3. Combinar en las 12 gráficas (opcional) | `py reporte_12.py --lateral resultados/21 --frontal resultados/frontal` | 2 |
| 4. Comparar con otra marcha ideal (opcional) | `py comparacion.py resultados/prueba --referencia resultados/sano` | 2 |
| 5. Modelo 3D, validación y comparación | `py experimento3.py resultados/21` | 3 |

El paso 1 ya genera `comparacion_marcha.png` contra el patrón normal; el paso 4 sirve para
rehacerla contra otra prueba (por ejemplo, un sujeto sano grabado con la misma cámara).

Con cámara en tiempo real: `py analisis_marcha.py 0 --grabar` graba `captura.mp4` y almacena los
datos al mismo tiempo; termina con `q` o sola cuando deja de ver los marcadores 2 s (fin de la trayectoria).
En la ventana: `q` termina y la barra espaciadora pausa. Para usar umbrales calibrados agregue
`--umbrales umbrales_hsv.json`.

Otras opciones de `analisis_marcha.py`: `--pierna izquierda` (pierna del lado de la cámara; por
omisión se deduce de hacia dónde camina el sujeto), `--frente espalda` (vista frontal desde atrás),
`--diametro-marcador-cm 10` o `--estatura-m 1.70` (alturas laterales en cm), `--modo rgb` (umbrales originales
de `pp3.m`), `--inicio/--fin` (cuadros, desde 0: el 292 de MATLAB es el 291 aquí), `--fc` (corte del
filtro), `--sin-ventana`, `--video-anotado`, `--referencia CARPETA` (marcha ideal para comparar).
Sin marcadores: `--modelo mediapipe`, `--pierna-lejana medir`, `--detectar-cada N`.

## Vista frontal: alturas

Se pensó para videos donde la cámara se mueve con el sujeto (o el sujeto camina en su lugar),
de frente a la cámara. Como la cámara se mueve, nada se mide contra un punto fijo de la imagen:

1. **Suelo:** en cada cuadro, el tobillo más bajo (el pie apoyado).
2. **Escala:** se calcula en cada cuadro, porque la distancia a la cámara cambia (en
   `Marcha_frontal_ejemplo.mp4` la persona se ve 47 % más grande al final). La da la estatura
   (cadera–tobillo de pie = 0.491 × estatura, Winter 2009) y sigue el tamaño del tronco (sin
   marcadores) o el ancho entre los marcadores de cadera.
3. **Punto inicial:** la postura de pie del inicio del video (se detecta sola). Cada altura se
   grafica como **cambio respecto a cómo estaba parada la persona**, en metros, contra el % del ciclo.
   Si el video no empieza con la persona quieta, la postura de pie se estima con la estatura (aviso).
4. **Distancias entre articulaciones:** cadera → rodilla y rodilla → tobillo (verticales, en m),
   con la línea de la postura de pie.
5. **Contacto inicial:** máximo de la altura del tobillo respecto a la cadera (el pie que avanza baja
   en la imagen hasta apoyarse). Revisado contra los cuadros del video: ±0.1 s.

**`video_alturas.mp4`** muestra cómo se mide cada cuadro: la línea amarilla es el suelo de ese
cuadro (tobillo apoyado), las líneas verticales van del suelo a cada articulación con su cambio de
altura en m, la marca blanca es dónde estaba esa articulación de pie y la línea gris del tronco es
la que sigue la escala; a la derecha, la tabla de valores. Se genera al terminar el análisis frontal
(el lanzador lo abre solo) o con `py video_alturas.py resultados/<carpeta>`.

Archivos: `graficas_1_6.png` y `graficas/g01…g06` (gráficas 1 a 6 de la práctica),
`graficas_frontal.png` (alturas y ángulos frontales), `distancias_articulaciones.png`,
`serie_temporal.png` (alturas contra el tiempo, tramo de pie sombreado y escala). En
`datos_marcha.csv`: `*_cambio_altura_m`, `*_altura_sobre_pie_apoyado_m`, `*_distancia_vertical_*_m`
y `escala_m_por_px`; en `resumen.json`, las medidas de pie (`de_pie_m`) y cómo se calculó la escala.

### Cómo grabar el video frontal

- **Al empezar, 2 s de pie y quieta**, con los pies juntos, antes de caminar: es el punto inicial.
- La persona completa en el cuadro, de la cabeza a los pies, con algo de margen abajo, todo el tiempo.
- Cámara a la altura de la cadera (≈ 1 m), **nivelada** (sin inclinarla hacia abajo) y a distancia
  constante; de preferencia con estabilizador o tripié con ruedas.
- Al menos **3 ciclos** (6 pasos) a velocidad normal. Con un solo ciclo se completa con la duración
  estimada (aviso).
- Vertical u horizontal funcionan igual; en vertical la persona ocupa más píxeles (mejor).
- 30 fps o más, buena luz, ropa que contraste con el fondo y pantalón que deje ver los tobillos.

## Sin marcadores (`--metodo sin-marcadores`)

`sin_marcadores.py` estima la pose con un modelo preentrenado (opción 1 de la práctica) y entrega,
por cuadro y por pierna, hombro, cadera, rodilla, tobillo, talón y punta del pie, con el mismo
formato que los marcadores; el resto del análisis es el mismo. Los resultados van a
`resultados/<video>_sin_marcadores/`.

| Modelo | | Velocidad en esta PC (CPU) |
|---|---|---|
| `rtmpose` (por omisión) | RTMPose-m Halpe26 (OpenMMLab) con ONNX Runtime vía `rtmlib`. El detector de personas (YOLOX) corre cada 5 cuadros y su caja se desplaza con la cadera | ~7 cuadros/s (32 s de video en ~2.5 min) |
| `mediapipe` | MediaPipe Pose Landmarker heavy (Google) | ~12 cuadros/s |

Diferencias con los marcadores:

- **Tobillo con el pie.** El ángulo suma la inclinación talón→punta, así que ya no supone el pie
  horizontal (con marcadores llega a +55° en el balanceo).
- **Eventos.** Contacto inicial con el talón y despegue con la punta del pie (Zeni et al., 2008); con
  marcadores ambos salen del tobillo. El despegue cae ~5 % del ciclo más tarde (68 % frente a 62 % en
  21.MP4), en concordancia con el mínimo del tobillo y el pico de la rodilla.
- **Pierna lejana.** El modelo la estima, pero la cercana la tapa en buena parte del ciclo. En 21.MP4
  su pico de rodilla sale 9° más bajo que el de la cercana. Por eso, por omisión, se genera desplazando
  medio ciclo la cercana (caso especial de la práctica); `--pierna-lejana medir` usa la estimación.
- **Intercambios derecha/izquierda.** Si el modelo cambia las etiquetas de las piernas en un cuadro,
  se corrigen por continuidad (`intercambios_der_izq_corregidos` en `resumen.json`).

Validación contra los marcadores (misma pierna derecha, mismos ciclos, curva media de 0 a 100 %):

| | 21.MP4 | 23.MP4 |
|---|---|---|
| Ciclos / cadencia | 28 / 110.9 pasos/min (igual) | 24 / 94.9 pasos/min (igual) |
| Cadera: diferencia RMS, sesgo, r | 2.4°, +1.0°, 0.998 | 2.0°, +0.8°, 0.997 |
| Rodilla: diferencia RMS, sesgo, r | 2.3°, −0.7°, 0.997 | 2.2°, −0.5°, 0.999 |
| Intercambios der/izq corregidos | 0 | 0 |

El tobillo no se compara porque con marcadores supone el pie horizontal. El modelo 3D
(`experimento3.py`) funciona igual con los resultados sin marcadores.

En la prueba con 300 cuadros, MediaPipe confundió la rodilla cercana con la lejana cuando se cruzan
(~30 % del ciclo: rodilla de 30° contra 11° de los marcadores; r = 0.94). Por eso RTMPose es el
modelo por omisión.

## Comparación con la marcha ideal (`comparacion_marcha.png`)

Para cada articulación (filas) y pierna (columnas), con el formato de los reportes clínicos:

- **Banda gris:** marcha ideal, media ± 1 DE. Por omisión es `datos/normativa_uci.csv`: 10 adultos
  sanos en caminadora, 10 ciclos por pierna, media y DE entre sujetos (Helwig & Hsiao-Wecksler, 2016,
  *Multivariate Gait Data*, UCI Machine Learning Repository, https://doi.org/10.24432/C5861T,
  CC BY 4.0). Con `--referencia CARPETA` es otra prueba ya analizada (la banda es la DE entre sus
  ciclos).
- **Curva de color:** la prueba, media ± 1 DE de sus ciclos (cadera verde, rodilla roja, tobillo azul).
- **Línea ámbar:** despegue de la prueba. Arriba, las fases del ciclo con la del despegue resaltada.
- **Debajo de cada gráfica:** diferencia RMS contra la media normal, correlación, rango de movimiento
  contra el normal y % del ciclo fuera de la banda. Los mismos valores, con mínimos y máximos, están en
  `comparacion_marcha.csv`.

Las 12 gráficas (`graficas_12.png`) también muestran la banda normal en los ángulos.

## Qué cubre de la práctica

| Práctica | Implementación |
|---|---|
| **Exp. 1** Detección en tiempo real de cadera, rodilla y tobillo, plano lateral y frontal | `analisis_marcha.py` con video o cámara; `--vista lateral` (un marcador por color) o `--vista frontal` (dos por color, asignados a cada pierna con seguimiento) |
| Opción 1: modelos preentrenados sin marcadores (MediaPipe/OpenPose) | `sin_marcadores.py`: RTMPose o MediaPipe, `--metodo sin-marcadores` |
| Opción 2: marcadores de color, umbral HSV, seguimiento por proximidad | `deteccion.py` |
| Ángulos de rodilla (muslo–pierna), cadera (muslo–vertical), tobillo (pierna–horizontal) | `cinematica.angulos_sagitales`; en frontal, abducción/aducción de cadera y varo/valgo de rodilla |
| Coordenadas (x, y) para el análisis de altura | columnas `*_x`, `*_y` y `*_altura_*` de `datos_marcha.csv` |
| Puntos sobre el video, ángulos en la interfaz, ajuste de parámetros | `interfaz.py` (panel con valores y gráfica de los últimos 4 s), `calibrar_hsv.py` |
| **Exp. 2** Estructura de almacenamiento | `datos_marcha.csv`: tiempo, x/y de las 3 articulaciones de ambas piernas, alturas y ángulos de ambas piernas |
| Grabar y almacenar a la vez; al menos 1 ciclo; detener automática o manualmente | `--grabar`, `--auto-detener`, tecla `q` |
| Filtrado y normalización 0–100 % | Butterworth de 4.º orden, 6 Hz, fase cero; ciclos de contacto a contacto remuestreados a 101 puntos |
| 12 gráficas contra patrones normales | `graficas_12.png` y `graficas/g01…g12`; `reporte_12.py` toma las alturas del video frontal si existe |
| Comparación de la marcha de la prueba con la ideal | `comparacion_marcha.png` / `.csv` (`comparacion.py`, `normativa.py`) |
| Caso especial: una sola pierna grabada | la otra pierna es la misma señal desplazada medio ciclo (se marca "generada") |
| **Exp. 3** Exportar a BVH/FBX con rotaciones y posiciones | `marcha.bvh` (raíz con la trayectoria medida de la cadera + flexiones de cadera, rodilla y tobillo) y `marcha_3d.fbx` |
| Importar en Blender, ajustar escala y ejes | `blender_marcha.py`: BVH en metros con Y arriba → Blender con Z arriba |
| Modelo humanoide con el movimiento aplicado al esqueleto | maniquí unido a los huesos, con los marcadores de color: `marcha_3d.blend` |
| Comparar con el video y verificar los ángulos del modelo | `comparacion_3d.mp4` / `.png` y `validacion_modelo.csv` / `.json` / `.png` |

## Resultados con los videos del proyecto

| | 21.MP4 | 23.MP4 |
|---|---|---|
| Detección de los 3 marcadores | 100 % | 100 % |
| Ciclos completos | 28 | 24 |
| Duración del ciclo | 1.08 ± 0.01 s | 1.27 ± 0.02 s |
| Cadencia | 111 pasos/min | 95 pasos/min |
| Fase de apoyo (normal ≈ 60 %) | 62 % | 64.5 % |
| Rodilla (normal 0–60°) | 0° a 61.5° | 0° a 60.7° |
| Cadera (normal −10° a 30°) | −12.5° a 19° | −8.9° a 17.4° |
| Error angular del modelo 3D contra lo calculado | < 0.0001° | < 0.0001° |
| Error de posición del modelo (rodilla / tobillo, RMS) | 0.8 / 1.1 cm | 0.6 / 0.9 cm |

## Archivos de resultados (`resultados/<video>/`)

| Archivo | Contenido |
|---|---|
| `datos_marcha.csv` | estructura de la práctica (filtrada); sin marcadores agrega talón y punta |
| `detecciones_crudas.csv` | por cuadro y sin filtrar: centroides y áreas (marcadores) o puntos y confianza (sin marcadores) |
| `comparacion_marcha.png`, `.csv` | prueba contra la marcha ideal y sus métricas (vista lateral) |
| `ciclos.csv` | contacto inicial, duración y % de apoyo de cada ciclo |
| `ciclos_normalizados.csv` | media y DE de cada variable de 0 a 100 % del ciclo |
| `graficas_12.png`, `graficas/` | las 12 gráficas (conjuntas y sueltas); en frontal, `graficas_frontal.png` |
| `serie_temporal.png` | ángulos (o alturas) contra el tiempo con los contactos iniciales |
| `resumen.json` | cadencia, ciclo, apoyo, rangos articulares, escala |
| `video_anotado.mp4` | con `--video-anotado`: video con puntos, segmentos y panel de ángulos |
| `video_alturas.mp4` | vista frontal: cómo se mide cada altura, cuadro por cuadro |
| `marcha.bvh`, `marcha_bvh.json` | animación exportada y su escala / longitudes |
| `marcha_3d.blend`, `marcha_3d.fbx` | escena de Blender con el modelo animado |
| `render_lateral.mp4`, `render_perspectiva.mp4` | render del modelo |
| `comparacion_3d.mp4`, `comparacion_3d.png` | video original contra el modelo (todo el video y 7 fases de un ciclo) |
| `validacion_modelo.*` | ángulos y posiciones medidos en el modelo contra los calculados |

## Convenciones

- Imagen: x a la derecha, y hacia abajo. "Adelante" es la dirección de la marcha (se detecta sola).
- Cadera: muslo contra la vertical, + flexión. Rodilla: muslo menos pierna, + flexión.
  Tobillo: 90° menos el ángulo pierna–horizontal, + dorsiflexión; con marcadores supone el pie
  horizontal, sin marcadores suma la inclinación talón→punta (+ punta arriba).
- "Pierna del lado de la cámara": si el sujeto camina hacia la derecha de la imagen se ve su lado
  derecho, y al revés.
- Frontal: cadera + abducción, rodilla + varo.
- Contacto inicial, lateral: tobillo lo más adelantado respecto a la cadera (Zeni et al., 2008).
  Frontal: máximo de la altura del tobillo respecto a la cadera (ver "Vista frontal: alturas").
- Lateral: alturas en px salvo que se dé `--diametro-marcador-cm` o `--estatura-m`. El BVH usa esa
  escala o, si no hay, una estatura de 1.70 m (pierna = 0.246 × estatura, Winter 2009).
  Frontal: cambio de altura en m respecto a la postura de pie (estatura obligatoria).

## Equivalencia con el código MATLAB

| MATLAB | Python |
|---|---|
| `R.m` (leer video, ver un cuadro) | `deteccion.abrir_fuente`, `leer_fotogramas` |
| `hy.m`, `t12.m`, `proyecto1.m` (umbral del rojo) | `deteccion.segmentar_rgb_matlab` |
| `proyecto2.m` (umbral de los 3 colores) | `deteccion.segmentar_rgb_matlab` / `segmentar_hsv` |
| `prueba1.m` (`bwlabel` + `regionprops`) | `deteccion.DetectorMarcadores` |
| `pp3.m` (centroides y ángulos `thc`, `tht`) | `cinematica.angulos_sagitales`, `analisis_marcha.py` |

Cambios respecto a `pp3.m`:

- Los acumuladores de coordenadas (`x_rodilla`, `nr`, ...) no se reiniciaban en cada cuadro, así
  que desde el segundo cuadro los centroides eran incorrectos.
- `x_*` guardaba la fila y `y_*` la columna (nombres invertidos), y `tht`, rotulado "rodilla",
  era la inclinación de la pierna. Ahora: rodilla = muslo − pierna.
- En `proyecto2.m`, el ciclo `imgP=imgP(y,x)` reducía la imagen a un escalar.
- El recorrido píxel por píxel se reemplazó por operaciones vectorizadas (≈ 20 cuadros/s a 1080×1920).
- Los umbrales RGB no funcionan igual en OpenCV (otra conversión YUV→RGB): el marcador rojo solo
  se detecta en 56 % de los cuadros. Por eso el modo por omisión es HSV recalibrado.

## Limitaciones

- Los videos 21 y 23 son laterales de la pierna derecha: la izquierda se genera con medio ciclo de
  desfase y las alturas Y se miden en la vista lateral. Para la vista frontal con marcadores, estos
  deben verse de frente (cara anterior de muslo, rodilla y tobillo); los de la cara lateral se ven de canto.
- Tobillo: sin marcador en el pie se supone el pie horizontal. Solo vale en el apoyo medio (en el
  balanceo llega a ~55°) y en el modelo 3D el pie queda siempre horizontal.
- Cadera: es el muslo contra la vertical, sin la inclinación de la pelvis. El patrón normal mide la
  cadera respecto a la pelvis, así que la prueba sale ≈ 5° menos flexionada en promedio (21 y 23)
  aunque sea normal. Una referencia grabada con este sistema (`--referencia`) no tiene esta diferencia.
- El patrón normal es de adultos en caminadora a velocidad cómoda (despegue ≈ 65 %); para marcha en
  piso, niños o adultos mayores conviene una referencia propia.
- Sin marcadores: la pierna lejana es poco confiable en vista lateral (ver arriba); el ángulo del
  tobillo no tiene una toma estática de referencia (el 0° es el pie talón→punta perpendicular a la
  pierna). La primera ejecución descarga los modelos (necesita internet).
- El tronco y los brazos del modelo 3D tienen una postura fija (sujetando los barandales).
- El modo con cámara no se probó con una cámara real; usa el mismo flujo que con un video.
- Vista frontal: por la perspectiva, el pie que va adelante se ve más abajo en la imagen, así que la
  "altura" del tobillo mezcla cuánto se levanta el pie con cuánto avanza o retrocede, y la de la
  cadera sube unos centímetros cuando el pie apoyado está adelante. Es una limitación de medir alturas
  desde el frente con una sola cámara; se reduce con la cámara nivelada a la altura de la cadera.
  Se probó sin marcadores con `Marcha_frontal_ejemplo.mp4` y su versión vertical; con marcadores
  usa el ancho de cadera como referencia de tamaño, sin video de prueba.
