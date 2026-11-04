"""Experimento 3: modelo humanoide 3D animado con la marcha registrada (se ejecuta dentro de Blender).

    blender --background --factory-startup --python blender_marcha.py -- --carpeta resultados/21 [--render 1:300]

Lo llama experimento3.py. Pasos:
  1. Importa marcha.bvh (Y arriba del BVH -> Z arriba de Blender; 1 unidad = 1 m).
  2. Arma un maniquí (modelo humanoide base) con cada segmento unido a su hueso.
  3. Coloca la banda de la caminadora, una cámara lateral (como el video) y otra 3/4.
  4. Mide en el modelo los ángulos y posiciones de cada cuadro y los compara con los
     calculados en datos_marcha.csv (validacion_modelo.csv / .json).
  5. Guarda marcha_3d.blend y marcha_3d.fbx y, si se pide, renderiza las dos cámaras.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import addon_utils
import bmesh
import bpy
from mathutils import Matrix, Vector

ADELANTE = Vector((0.0, -1.0, 0.0))   # +Z del BVH (hacia donde camina el sujeto)
ARRIBA = Vector((0.0, 0.0, 1.0))
LADOS = {"Right": "derecha", "Left": "izquierda"}


def srgb(hex_color):
    """Color sRGB '#rrggbb' -> RGBA lineal de Blender."""
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c) + (1.0,)


COLORES = {
    "cuerpo": srgb("#c9c7c0"), "Right": srgb("#2a78d6"), "Left": srgb("#eb6834"),
    "marcador_cadera": srgb("#1f5fff"), "marcador_rodilla": srgb("#e0204a"), "marcador_tobillo": srgb("#22b04a"),
    "banda": srgb("#2b2b2b"), "piso": srgb("#e6e5e0"),
}


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", type=Path, required=True)
    p.add_argument("--render", default=None, help="rango de cuadros a renderizar, p. ej. 1:300 o 'todo'")
    p.add_argument("--ancho", type=int, default=540)
    p.add_argument("--alto", type=int, default=960)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def material(nombre, color):
    mat = bpy.data.materials.new(nombre)
    mat.diffuse_color = color
    if mat.node_tree is None:       # Blender < 5 crea los materiales sin nodos
        mat.use_nodes = True
    if "Principled BSDF" in mat.node_tree.nodes:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def objeto_malla(nombre, matriz, mat, forma="esfera"):
    malla = bpy.data.meshes.new(nombre)
    bm = bmesh.new()
    if forma == "esfera":
        bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=16, radius=1.0)
    else:
        bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(malla)
    bm.free()
    if forma == "esfera":
        for poligono in malla.polygons:
            poligono.use_smooth = True
    malla.materials.append(mat)
    obj = bpy.data.objects.new(nombre, malla)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_basis = matriz
    return obj


def matriz_segmento(a, b, rx, ry, extra=1.08):
    """Elipsoide que une los puntos a y b; rx en dirección lateral (X), ry perpendicular."""
    z = b - a
    largo = z.length
    z.normalize()
    lateral = Vector((1.0, 0.0, 0.0))
    x = (lateral - lateral.dot(z) * z).normalized()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed().to_4x4() @ Matrix.Diagonal((rx, ry, largo / 2 * extra, 1.0))
    m.translation = (a + b) / 2
    return m


def matriz_elipsoide(centro, sx, sy, sz):
    m = Matrix.Diagonal((sx, sy, sz, 1.0))
    m.translation = centro
    return m


def emparentar(obj, arm, hueso):
    """Une el objeto al hueso conservando su posición en la postura de reposo."""
    pb = arm.pose.bones[hueso]
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = hueso
    cola = arm.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.bone.length, 0.0))
    obj.matrix_parent_inverse = cola.inverted()


def construir_maniqui(arm, meta):
    """Modelo humanoide base: elipsoides por segmento + marcadores en la pierna medida."""
    k = meta["estatura_m"] / 1.70
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    cabeza = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    cola = {b.name: arm.matrix_world @ b.tail_local for b in arm.data.bones}
    mats = {n: material(n, c) for n, c in COLORES.items()}
    partes = []   # (nombre, hueso, matriz, material)

    raiz = cabeza["Hips"]
    partes.append(("Pelvis", "Hips", matriz_elipsoide(raiz + ARRIBA * 0.02 * k, 0.16 * k, 0.11 * k, 0.12 * k), "cuerpo"))
    partes.append(("Abdomen", "Spine", matriz_segmento(cabeza["Spine"], cabeza["Chest"], 0.14 * k, 0.10 * k), "cuerpo"))
    partes.append(("Torax", "Chest", matriz_segmento(cabeza["Chest"], cabeza["Neck"], 0.17 * k, 0.11 * k), "cuerpo"))
    partes.append(("Cuello", "Neck", matriz_segmento(cabeza["Neck"], cabeza["Head"], 0.05 * k, 0.05 * k), "cuerpo"))
    partes.append(("Cabeza", "Head", matriz_elipsoide(cabeza["Head"] + ARRIBA * 0.10 * k, 0.08 * k, 0.095 * k, 0.115 * k),
                   "cuerpo"))
    for lado in LADOS:
        partes += [
            (f"Brazo_{lado}", f"{lado}Arm", matriz_segmento(cabeza[f"{lado}Arm"], cabeza[f"{lado}ForeArm"], 0.042 * k, 0.042 * k), "cuerpo"),
            (f"Antebrazo_{lado}", f"{lado}ForeArm", matriz_segmento(cabeza[f"{lado}ForeArm"], cabeza[f"{lado}Hand"], 0.035 * k, 0.035 * k), "cuerpo"),
            (f"Mano_{lado}", f"{lado}Hand", matriz_segmento(cabeza[f"{lado}Hand"], cola[f"{lado}Hand"], 0.025 * k, 0.04 * k), "cuerpo"),
            (f"Muslo_{lado}", f"{lado}UpLeg", matriz_segmento(cabeza[f"{lado}UpLeg"], cabeza[f"{lado}Leg"], 0.075 * k, 0.075 * k), lado),
            (f"Pierna_{lado}", f"{lado}Leg", matriz_segmento(cabeza[f"{lado}Leg"], cabeza[f"{lado}Foot"], 0.052 * k, 0.052 * k), lado),
        ]
        tobillo = cabeza[f"{lado}Foot"]
        talon = tobillo - ADELANTE * 0.25 * meta["pie_m"] - ARRIBA * meta["tobillo_h_m"]
        punta = cola[f"{lado}Foot"]
        pie = matriz_segmento(talon + ARRIBA * 0.03 * k, punta + ARRIBA * 0.03 * k, 0.045 * k, 0.035 * k, extra=1.0)
        partes.append((f"Pie_{lado}", f"{lado}Foot", pie, lado))

    # Marcadores de color como en el video (cara externa de la pierna medida).
    lado_m = "Right" if meta["pierna_medida"] == "derecha" else "Left"
    afuera = Vector((-1.0 if lado_m == "Right" else 1.0, 0.0, 0.0)) * 0.085 * k
    r = 0.045 * k
    partes += [
        ("Marcador_cadera", "Hips", matriz_elipsoide(cabeza[f"{lado_m}UpLeg"] + afuera, r, r, r), "marcador_cadera"),
        ("Marcador_rodilla", f"{lado_m}UpLeg", matriz_elipsoide(cabeza[f"{lado_m}Leg"] + afuera, r, r, r), "marcador_rodilla"),
        ("Marcador_tobillo", f"{lado_m}Leg", matriz_elipsoide(cabeza[f"{lado_m}Foot"] + afuera * 0.8, r, r, r),
         "marcador_tobillo"),
    ]
    for nombre, hueso, matriz, mat in partes:
        emparentar(objeto_malla(nombre, matriz, mats[mat]), arm, hueso)
    arm.data.pose_position = "POSE"
    arm.show_in_front = False
    arm.hide_render = True

    # Caminadora: banda (superficie en z = 0) sobre el piso.
    objeto_malla("Banda", matriz_elipsoide(Vector((0, 0, -0.025)), 0.36, 1.1, 0.025), mats["banda"], "cubo")
    objeto_malla("Piso", matriz_elipsoide(Vector((0, 0, -0.06)), 4.0, 4.0, 0.01), mats["piso"], "cubo")


def camara(nombre, ubicacion, objetivo, ortho=None, lente=70.0):
    datos = bpy.data.cameras.new(nombre)
    if ortho:
        datos.type = "ORTHO"
        datos.ortho_scale = ortho
    else:
        datos.lens = lente
    cam = bpy.data.objects.new(nombre, datos)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = ubicacion
    cam.rotation_euler = (Vector(objetivo) - Vector(ubicacion)).to_track_quat("-Z", "Y").to_euler()
    return cam


def _angulo(v):
    """Inclinación respecto a la vertical en el plano sagital, + hacia adelante [°]."""
    return math.degrees(math.atan2(v.dot(ADELANTE), -v.dot(ARRIBA)))


def validar(arm, meta, carpeta):
    """Ángulos y posiciones medidos en el modelo contra los calculados por analisis_marcha.py."""
    with open(carpeta / "datos_marcha.csv", newline="", encoding="utf-8") as f:
        datos = list(csv.DictReader(f))
    escena = bpy.context.scene
    pie_reposo = math.atan2(-meta["tobillo_h_m"], 0.75 * meta["pie_m"])
    medida = meta["pierna_medida"]
    s = meta["m_por_px"]

    def a_blender(x_px, y_px):
        z_bvh = meta["direccion"] * (x_px - meta["x_ref_px"]) * s
        y_bvh = (meta["y_suelo_px"] - y_px) * s + meta["tobillo_h_m"]
        return Vector((-z_bvh, y_bvh))            # (Y, Z) de Blender

    filas = []
    for k, fila_csv in enumerate(datos):
        escena.frame_set(escena.frame_start + k)
        fila = {"cuadro_blender": escena.frame_current, "fotograma_video": int(fila_csv["fotograma"])}
        for lado, pierna in LADOS.items():
            def pos(hueso, en_cola=False):
                pb = arm.pose.bones[hueso]
                return arm.matrix_world @ (pb.tail if en_cola else pb.head)
            cad, rod, tob, punta = pos(f"{lado}UpLeg"), pos(f"{lado}Leg"), pos(f"{lado}Foot"), pos(f"{lado}Foot", True)
            th, ts = _angulo(rod - cad), _angulo(tob - rod)
            pie = punta - tob
            inclinacion_pie = math.atan2(pie.dot(ARRIBA), pie.dot(ADELANTE))
            modelo = {"cadera": th, "rodilla": th - ts, "tobillo": math.degrees(inclinacion_pie - pie_reposo) - ts}
            for art, valor in modelo.items():
                calc = float(fila_csv[f"{pierna}_angulo_{art}"])
                fila[f"{pierna}_modelo_{art}"] = valor
                fila[f"{pierna}_calculado_{art}"] = calc
                fila[f"{pierna}_error_{art}"] = valor - calc
            if pierna == medida:
                for art, p in (("cadera", cad), ("rodilla", rod), ("tobillo", tob)):
                    medido = a_blender(float(fila_csv[f"{pierna}_{art}_x"]), float(fila_csv[f"{pierna}_{art}_y"]))
                    fila[f"{pierna}_error_pos_{art}_cm"] = 100.0 * (Vector((p.y, p.z)) - medido).length
        filas.append(fila)

    with open(carpeta / "validacion_modelo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    resumen = {"cuadros": len(filas), "error_angular_max_deg": {}, "error_angular_rms_deg": {}, "error_posicion_rms_cm": {}}
    for pierna in LADOS.values():
        for art in ("cadera", "rodilla", "tobillo"):
            e = [abs(fl[f"{pierna}_error_{art}"]) for fl in filas]
            resumen["error_angular_max_deg"][f"{pierna}_{art}"] = round(max(e), 6)
            resumen["error_angular_rms_deg"][f"{pierna}_{art}"] = round(math.sqrt(sum(v * v for v in e) / len(e)), 6)
    for art in ("cadera", "rodilla", "tobillo"):
        e = [fl[f"{medida}_error_pos_{art}_cm"] for fl in filas]
        resumen["error_posicion_rms_cm"][f"{medida}_{art}"] = round(math.sqrt(sum(v * v for v in e) / len(e)), 3)
    (carpeta / "validacion_modelo.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    return resumen


def configurar_render(ancho, alto):
    escena = bpy.context.scene
    escena.render.engine = "BLENDER_WORKBENCH"
    sombreado = escena.display.shading
    sombreado.light = "STUDIO"
    sombreado.color_type = "MATERIAL"
    sombreado.show_shadows = True
    sombreado.show_cavity = True
    mundo = bpy.data.worlds.new("Mundo")
    mundo.color = (0.86, 0.86, 0.84)
    escena.world = mundo
    escena.render.resolution_x, escena.render.resolution_y = ancho, alto
    escena.render.resolution_percentage = 100
    escena.render.image_settings.file_format = "PNG"


def main():
    args = argumentos()
    carpeta = args.carpeta.resolve()
    meta = json.loads((carpeta / "marcha_bvh.json").read_text(encoding="utf-8"))
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    addon_utils.enable("io_anim_bvh", default_set=True)
    bpy.ops.import_anim.bvh(filepath=str(carpeta / "marcha.bvh"), axis_forward="-Z", axis_up="Y", global_scale=1.0,
                            frame_start=1, use_fps_scale=False, update_scene_fps=True, update_scene_duration=True,
                            rotate_mode="NATIVE")
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    arm.name = "Esqueleto_marcha"
    construir_maniqui(arm, meta)

    # Cámara lateral del lado de la pierna medida (como el video) y cámara 3/4.
    lado_x = -1.0 if meta["pierna_medida"] == "derecha" else 1.0
    k = meta["estatura_m"] / 1.70
    centro = (0.0, -0.08 * k, 1.05 * k)
    cam_lat = camara("Camara_lateral", (lado_x * 6.0, centro[1], centro[2]), centro, ortho=2.5 * k)
    cam_3d = camara("Camara_3D", (lado_x * 3.4, -3.0, 1.45 * k), (0.0, -0.1, 0.85 * k), lente=62.0)
    configurar_render(args.ancho, args.alto)
    bpy.context.scene.camera = cam_lat

    resumen = validar(arm, meta, carpeta)
    print("VALIDACION", json.dumps(resumen))
    bpy.context.scene.frame_set(1)
    bpy.context.preferences.filepaths.save_version = 0     # sin respaldo .blend1
    bpy.ops.wm.save_as_mainfile(filepath=str(carpeta / "marcha_3d.blend"))
    try:
        addon_utils.enable("io_scene_fbx", default_set=True)
        bpy.ops.export_scene.fbx(filepath=str(carpeta / "marcha_3d.fbx"), object_types={"ARMATURE", "MESH"},
                                 bake_anim=True, add_leaf_bones=False)
    except Exception as e:  # el .blend y el .bvh siguen disponibles
        print("Aviso: no se pudo exportar FBX:", e)

    if args.render:
        escena = bpy.context.scene
        if args.render != "todo":
            a, b = (int(v) for v in args.render.split(":"))
            escena.frame_start, escena.frame_end = a, min(b, escena.frame_end)
        ajustes = escena.render.image_settings
        if hasattr(ajustes, "media_type"):
            ajustes.media_type = "VIDEO"
        ajustes.file_format = "FFMPEG"
        escena.render.ffmpeg.format = "MPEG4"
        escena.render.ffmpeg.codec = "H264"
        escena.render.ffmpeg.constant_rate_factor = "HIGH"
        for cam, nombre in ((cam_lat, "lateral"), (cam_3d, "perspectiva")):
            escena.camera = cam
            destino = carpeta / f"render_{nombre}.mp4"
            for viejo in carpeta.glob(f"render_{nombre}*.mp4"):
                viejo.unlink()
            escena.render.filepath = str(carpeta / f"render_{nombre}_")
            bpy.ops.render.render(animation=True)
            # Blender agrega el rango de cuadros al nombre; se deja uno fijo.
            next(carpeta.glob(f"render_{nombre}_*.mp4")).rename(destino)
    print("LISTO")


main()
