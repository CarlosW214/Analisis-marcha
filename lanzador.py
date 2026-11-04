"""Ventana para usar el análisis de marcha sin escribir comandos (la abre Analisis_Marcha.bat).

    pythonw lanzador.py [video]
"""
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

AQUI = Path(__file__).resolve().parent
RESULTADOS = AQUI / "resultados"
UMBRALES = AQUI / "umbrales_hsv.json"
# Con pythonw no hay consola; los análisis se ejecutan con python.exe sin abrir ventana negra.
PYTHON = str(Path(sys.executable).with_name("python.exe")) if Path(sys.executable).with_name("python.exe").is_file() \
    else sys.executable
SIN_CONSOLA = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RUIDO = re.compile(r"^(W0000|I0000|WARNING: All log|INFO: Created TensorFlow|load .*onnxruntime)|it/s\]|B/s\]")
MODELOS = {"RTMPose (más preciso, recomendado)": "rtmpose", "MediaPipe (más rápido)": "mediapipe"}
PIERNAS = {"Automática (según hacia dónde camina)": "auto", "Derecha": "derecha", "Izquierda": "izquierda"}


class Lanzador(tk.Tk):
    def __init__(self, video_inicial=None):
        super().__init__()
        self.title("Análisis de marcha")
        self.minsize(720, 640)
        self.proceso = None
        self.carpeta_resultados = None
        self.cola = queue.Queue()

        self.fuente = tk.StringVar(value="archivo")
        self.video = tk.StringVar(value=video_inicial or "")
        self.camara = tk.StringVar(value="0")
        self.metodo = tk.StringVar(value="sin-marcadores")
        self.modelo = tk.StringVar(value=next(iter(MODELOS)))
        self.vista = tk.StringVar(value="lateral")
        self.pierna = tk.StringVar(value=next(iter(PIERNAS)))
        self.medir_lejana = tk.BooleanVar(value=False)
        self.estatura = tk.StringVar(value="")
        self.ideal = tk.StringVar(value="normal")
        self.referencia = tk.StringVar(value="")
        self.mostrar = tk.BooleanVar(value=True)
        self.anotado = tk.BooleanVar(value=False)

        self._construir()
        self._actualizar()
        self.after(100, self._leer_cola)

    # ---------------------------------------------------------------- interfaz
    def _construir(self):
        estilo = ttk.Style(self)
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 15, "bold"))
        estilo.configure("Accion.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        raiz = ttk.Frame(self, padding=14)
        raiz.pack(fill="both", expand=True)
        ttk.Label(raiz, text="Análisis de marcha por video", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(raiz, text="Elija el video, cómo detectar las articulaciones y presione Analizar.",
                  foreground="#52514e").pack(anchor="w", pady=(0, 8))

        f1 = ttk.LabelFrame(raiz, text=" 1. Video ", padding=10)
        f1.pack(fill="x", pady=4)
        f1.columnconfigure(1, weight=1)
        ttk.Radiobutton(f1, text="Archivo de video", variable=self.fuente, value="archivo",
                        command=self._actualizar).grid(row=0, column=0, sticky="w")
        self.e_video = ttk.Entry(f1, textvariable=self.video)
        self.e_video.grid(row=0, column=1, sticky="ew", padx=6)
        self.b_video = ttk.Button(f1, text="Buscar…", command=self._elegir_video)
        self.b_video.grid(row=0, column=2)
        ttk.Radiobutton(f1, text="Cámara web (en vivo)", variable=self.fuente, value="camara",
                        command=self._actualizar).grid(row=1, column=0, sticky="w", pady=(6, 0))
        fila_cam = ttk.Frame(f1)
        fila_cam.grid(row=1, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(fila_cam, text="Número de cámara:").pack(side="left")
        self.e_camara = ttk.Spinbox(fila_cam, from_=0, to=9, width=4, textvariable=self.camara)
        self.e_camara.pack(side="left", padx=4)
        ttk.Label(fila_cam, text="(se graba captura.mp4; termine con la tecla q)", foreground="#898781").pack(
            side="left", padx=6)

        f2 = ttk.LabelFrame(raiz, text=" 2. Detección de las articulaciones ", padding=10)
        f2.pack(fill="x", pady=4)
        f2.columnconfigure(1, weight=1)
        ttk.Radiobutton(f2, text="Sin marcadores", variable=self.metodo, value="sin-marcadores",
                        command=self._actualizar).grid(row=0, column=0, sticky="w")
        self.c_modelo = ttk.Combobox(f2, textvariable=self.modelo, values=list(MODELOS), state="readonly", width=36)
        self.c_modelo.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Radiobutton(f2, text="Marcadores de color", variable=self.metodo, value="marcadores",
                        command=self._actualizar).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.b_calibrar = ttk.Button(f2, text="Calibrar colores…", command=self._calibrar)
        self.b_calibrar.grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Separator(f2).grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        fila_vista = ttk.Frame(f2)
        fila_vista.grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(fila_vista, text="Cámara:").pack(side="left")
        ttk.Radiobutton(fila_vista, text="De lado (lateral)", variable=self.vista, value="lateral",
                        command=self._actualizar).pack(side="left", padx=6)
        ttk.Radiobutton(fila_vista, text="De frente (frontal)", variable=self.vista, value="frontal",
                        command=self._actualizar).pack(side="left", padx=6)
        ttk.Label(f2, text="Pierna del lado de la cámara:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.c_pierna = ttk.Combobox(f2, textvariable=self.pierna, values=list(PIERNAS), state="readonly", width=36)
        self.c_pierna.grid(row=4, column=1, sticky="w", padx=6, pady=(6, 0))
        self.k_lejana = ttk.Checkbutton(f2, text="Medir también la pierna lejana (queda tapada; menos precisa). "
                                                 "Si no, se genera desplazando medio ciclo la cercana.",
                                        variable=self.medir_lejana)
        self.k_lejana.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        fila_estatura = ttk.Frame(f2)
        fila_estatura.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(fila_estatura, text="Estatura del sujeto en metros (opcional, para alturas en cm):").pack(side="left")
        ttk.Entry(fila_estatura, textvariable=self.estatura, width=8).pack(side="left", padx=6)
        ttk.Label(fila_estatura, text="ej. 1.70", foreground="#898781").pack(side="left")

        f3 = ttk.LabelFrame(raiz, text=" 3. Marcha ideal para comparar ", padding=10)
        f3.pack(fill="x", pady=4)
        f3.columnconfigure(1, weight=1)
        ttk.Radiobutton(f3, text="Patrón normal (10 adultos sanos)", variable=self.ideal, value="normal",
                        command=self._actualizar).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(f3, text="Otra prueba ya analizada:", variable=self.ideal, value="carpeta",
                        command=self._actualizar).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.e_ref = ttk.Entry(f3, textvariable=self.referencia)
        self.e_ref.grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        self.b_ref = ttk.Button(f3, text="Elegir carpeta…", command=self._elegir_referencia)
        self.b_ref.grid(row=1, column=2, pady=(4, 0))

        opciones = ttk.Frame(raiz)
        opciones.pack(fill="x", pady=(6, 2))
        self.k_mostrar = ttk.Checkbutton(opciones, text="Mostrar el video mientras se analiza", variable=self.mostrar)
        self.k_mostrar.pack(side="left")
        ttk.Checkbutton(opciones, text="Guardar video con las articulaciones marcadas",
                        variable=self.anotado).pack(side="left", padx=14)

        botones = ttk.Frame(raiz)
        botones.pack(fill="x", pady=6)
        self.b_analizar = ttk.Button(botones, text="Analizar", style="Accion.TButton", command=self._analizar)
        self.b_analizar.pack(side="left")
        self.b_detener = ttk.Button(botones, text="Detener", command=self._detener, state="disabled")
        self.b_detener.pack(side="left", padx=6)
        self.b_carpeta = ttk.Button(botones, text="Abrir carpeta de resultados", command=self._abrir_carpeta)
        self.b_carpeta.pack(side="right")
        self.b_comparacion = ttk.Button(botones, text="Ver comparación", command=self._ver_comparacion,
                                        state="disabled")
        self.b_comparacion.pack(side="right", padx=6)
        self.b_3d = ttk.Button(botones, text="Modelo 3D (Blender)", command=self._modelo_3d, state="disabled")
        self.b_3d.pack(side="right")

        self.progreso = ttk.Progressbar(raiz, mode="determinate", maximum=100)
        self.progreso.pack(fill="x", pady=(2, 4))
        marco = ttk.LabelFrame(raiz, text=" Mensajes ", padding=4)
        marco.pack(fill="both", expand=True)
        self.texto = tk.Text(marco, height=10, wrap="word", font=("Consolas", 9), relief="flat",
                             background="#fcfcfb", foreground="#0b0b0b")
        barra = ttk.Scrollbar(marco, command=self.texto.yview)
        self.texto.configure(yscrollcommand=barra.set, state="disabled")
        barra.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)

    def _actualizar(self):
        archivo = self.fuente.get() == "archivo"
        sin = self.metodo.get() == "sin-marcadores"
        lateral = self.vista.get() == "lateral"
        self.e_video.configure(state="normal" if archivo else "disabled")
        self.b_video.configure(state="normal" if archivo else "disabled")
        self.e_camara.configure(state="disabled" if archivo else "normal")
        self.c_modelo.configure(state="readonly" if sin else "disabled")
        self.b_calibrar.configure(state="disabled" if sin else "normal")
        self.c_pierna.configure(state="readonly" if lateral else "disabled")
        self.k_lejana.configure(state="normal" if (sin and lateral) else "disabled")
        ref = self.ideal.get() == "carpeta"
        self.e_ref.configure(state="normal" if ref else "disabled")
        self.b_ref.configure(state="normal" if ref else "disabled")
        if not archivo:
            self.mostrar.set(True)          # con cámara la ventana es la forma de terminar (tecla q)
        self.k_mostrar.configure(state="normal" if archivo else "disabled")

    def _escribir(self, linea):
        self.texto.configure(state="normal")
        self.texto.insert("end", linea + "\n")
        self.texto.see("end")
        self.texto.configure(state="disabled")

    # ---------------------------------------------------------------- acciones
    def _elegir_video(self):
        ruta = filedialog.askopenfilename(title="Elegir video",
                                          filetypes=[("Videos", "*.mp4 *.MP4 *.avi *.mov *.MOV *.mkv"),
                                                     ("Todos", "*.*")])
        if ruta:
            self.video.set(ruta)

    def _elegir_referencia(self):
        ruta = filedialog.askdirectory(title="Carpeta de resultados de la marcha de referencia",
                                       initialdir=RESULTADOS if RESULTADOS.is_dir() else AQUI)
        if ruta:
            if not (Path(ruta) / "ciclos_normalizados.csv").is_file():
                messagebox.showerror("Carpeta no válida", "Esa carpeta no tiene resultados de un análisis "
                                                          "(falta ciclos_normalizados.csv).")
                return
            self.referencia.set(ruta)

    def _comando(self):
        """Argumentos de analisis_marcha.py; None si falta algo."""
        if self.fuente.get() == "archivo":
            video = self.video.get().strip().strip('"')
            if not video or not Path(video).is_file():
                messagebox.showwarning("Falta el video", "Elija un archivo de video con el botón Buscar…")
                return None
            args = [video]
        else:
            args = [self.camara.get().strip() or "0", "--grabar"]
        args += ["--metodo", self.metodo.get(), "--vista", self.vista.get()]
        if self.metodo.get() == "sin-marcadores":
            args += ["--modelo", MODELOS[self.modelo.get()]]
            if self.vista.get() == "lateral" and self.medir_lejana.get():
                args += ["--pierna-lejana", "medir"]
        elif UMBRALES.is_file():
            args += ["--umbrales", str(UMBRALES)]
        if self.vista.get() == "lateral":
            args += ["--pierna", PIERNAS[self.pierna.get()]]
        estatura = self.estatura.get().strip().replace(",", ".")
        if estatura:
            try:
                valor = float(estatura)
                assert 0.5 <= valor <= 2.5
            except (ValueError, AssertionError):
                messagebox.showwarning("Estatura", "Escriba la estatura en metros, por ejemplo 1.70, o déjela vacía.")
                return None
            args += ["--estatura-m", f"{valor:g}"]
        if self.ideal.get() == "carpeta":
            if not self.referencia.get():
                messagebox.showwarning("Marcha ideal", "Elija la carpeta de la prueba de referencia.")
                return None
            args += ["--referencia", self.referencia.get()]
        if not self.mostrar.get():
            args.append("--sin-ventana")
        if self.anotado.get():
            args.append("--video-anotado")
        return args

    def _ejecutar(self, script, args, al_terminar):
        if self.proceso is not None:
            return
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.configure(state="disabled")
        self.progreso["value"] = 0
        self._escribir(f"> {script} " + " ".join(f'"{a}"' if " " in a else a for a in args))
        entorno = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proceso = subprocess.Popen([PYTHON, "-u", str(AQUI / script), *args], cwd=AQUI, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env=entorno, creationflags=SIN_CONSOLA)
        self._al_terminar = al_terminar
        for b in (self.b_analizar, self.b_3d, self.b_calibrar):
            b.configure(state="disabled")
        self.b_detener.configure(state="normal")
        threading.Thread(target=self._leer_salida, args=(self.proceso,), daemon=True).start()

    def _leer_salida(self, proceso):
        for linea in proceso.stdout:
            self.cola.put(("linea", linea.rstrip()))
        self.cola.put(("fin", proceso.wait()))

    def _leer_cola(self):
        try:
            while True:
                tipo, valor = self.cola.get_nowait()
                if tipo == "linea":
                    self._procesar_linea(valor)
                else:
                    self._terminado(valor)
        except queue.Empty:
            pass
        self.after(100, self._leer_cola)

    def _procesar_linea(self, linea):
        if not linea or RUIDO.search(linea):
            return
        m = re.search(r"\((\d+) %\)", linea)
        if m and linea.startswith("Procesando"):
            self.progreso["value"] = int(m.group(1))
        m = re.match(r"Resultados en: (.+)$", linea)
        if m:
            self.carpeta_resultados = Path(m.group(1).strip())
        self._escribir(linea)

    def _hay_comparacion(self):
        return self.carpeta_resultados is not None and (self.carpeta_resultados / "comparacion_marcha.png").is_file()

    def _terminado(self, codigo):
        self.proceso = None
        self.progreso.stop()
        self.progreso.configure(mode="determinate")
        self.b_detener.configure(state="disabled")
        self.b_analizar.configure(state="normal")
        self._actualizar()
        estado = "normal" if self._hay_comparacion() else "disabled"
        self.b_comparacion.configure(state=estado)
        self.b_3d.configure(state=estado)
        accion, self._al_terminar = self._al_terminar, None
        if accion:
            accion(codigo)

    def _analizar(self):
        args = self._comando()
        if args is not None:
            self.carpeta_resultados = None
            self._ejecutar("analisis_marcha.py", args, self._fin_analisis)
            if self.fuente.get() == "camara":       # en vivo no hay total de cuadros
                self.progreso.configure(mode="indeterminate")
                self.progreso.start(15)

    def _fin_analisis(self, codigo):
        hay = self.carpeta_resultados is not None and self.carpeta_resultados.is_dir()
        lateral = self._hay_comparacion()
        if codigo == 0 and hay:
            self.progreso["value"] = 100
            self._escribir("\nListo.")
            if lateral:
                self._ver_comparacion()
            else:
                self._abrir_carpeta()
        elif codigo != 0:
            messagebox.showerror("El análisis no terminó",
                                 "Revise los mensajes de la parte inferior para ver la causa.")

    def _detener(self):
        if self.proceso is not None:
            self.proceso.terminate()
            self._escribir("Detenido por el usuario.")

    def _calibrar(self):
        if self.fuente.get() == "archivo":
            video = self.video.get().strip().strip('"')
            if not video or not Path(video).is_file():
                messagebox.showwarning("Falta el video", "Elija primero el video para calibrar los colores.")
                return
        else:
            video = self.camara.get().strip() or "0"
        messagebox.showinfo("Calibrar colores",
                            "Teclas 1, 2 y 3 eligen cadera, rodilla y tobillo; haga clic sobre el marcador para "
                            "tomar su color; espacio reproduce o pausa; s guarda; q sale.\n\n"
                            "Los colores guardados se usan solos en los siguientes análisis con marcadores.")
        args = [video, "--salida", str(UMBRALES), "--vista", self.vista.get()]
        if UMBRALES.is_file():
            args += ["--umbrales", str(UMBRALES)]
        self._ejecutar("calibrar_hsv.py", args, None)

    def _modelo_3d(self):
        if self.carpeta_resultados is None:
            return
        if not messagebox.askyesno("Modelo 3D", "Se generará el modelo 3D en Blender y los videos de comparación. "
                                                "Necesita Blender instalado y puede tardar varios minutos. "
                                                "¿Continuar?"):
            return
        self._ejecutar("experimento3.py", [str(self.carpeta_resultados)],
                       lambda codigo: self._abrir_carpeta() if codigo == 0 else
                       messagebox.showerror("Modelo 3D", "No terminó; revise los mensajes (¿está instalado Blender?)."))

    def _ver_comparacion(self):
        if self.carpeta_resultados is not None:
            ruta = self.carpeta_resultados / "comparacion_marcha.png"
            if ruta.is_file():
                os.startfile(ruta)

    def _abrir_carpeta(self):
        carpeta = self.carpeta_resultados or RESULTADOS
        carpeta.mkdir(parents=True, exist_ok=True)
        os.startfile(carpeta)


if __name__ == "__main__":
    Lanzador(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
