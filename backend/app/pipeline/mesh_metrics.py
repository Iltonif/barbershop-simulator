"""
Medidas de visajismo sobre el gemelo 3D del cliente (`avatar3d.py`).

Del anuncio: "analiza cada rasgo" -- forma del rostro, proporción, simetría,
perfil, ángulo mandibular, cuello. Aquí se calculan sobre la malla, que es
la ventaja de tener el 3D: el perfil y la mandíbula son justo lo que NO se
pudo medir con fotos 2D (ver "Calibración de umbrales" en CLAUDE.md, donde
se dejaron como campos manuales porque el detector no encuentra la cara de
perfil y BiSeNet confundía la barbilla con la oreja).

Cómo funciona, sin dependencias nuevas (solo numpy):

1. `load_glb` lee el .glb a mano (cabecera + JSON + búfer) y saca los
   vértices de todas las mallas ya colocados en el mundo.
2. `orient` encuentra el plano de simetría (el que mejor deja una mitad
   sobre la otra) y hacia dónde mira la cara: la nariz es un saliente
   estrecho y la nuca una superficie ancha, así que se compara lo
   "apretados" que están los vértices más salientes de cada lado.
3. Con la cabeza orientada se construye un mapa de radios (distancia desde
   el centro en cada dirección) y de ahí salen la silueta de perfil, los
   anchos a distintas alturas y la simetría.

AVISO, escrito a propósito: el modelo de Tripo es una RECONSTRUCCIÓN
generativa a partir de las fotos, no un escaneo. Dos fotos distintas de la
misma persona no dan exactamente la misma malla, y la nuca, las orejas y el
pelo son en buena parte inventados. Por eso todo lo que sale de aquí es una
ESTIMACIÓN (`"estimado": True` en la salida) y los umbrales de
clasificación están puestos a ojo y validados solo contra el maniquí de
MakeHuman del repo -- hay que revisarlos con clientes reales antes de
tomárselos al pie de la letra, igual que se hizo con los umbrales 2D.
"""

from __future__ import annotations

import json
import struct

import numpy as np

_FLOAT = 5126
_COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_NUM = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ---------- 1. Leer el .glb ----------
def load_glb(data: bytes) -> np.ndarray:
    """Vértices (N,3) de todas las mallas del .glb, con las transformaciones
    de los nodos ya aplicadas."""
    if len(data) < 12 or data[:4] != b"glTF":
        raise ValueError("Esto no es un .glb")
    gltf, buffers = None, b""
    offset = 12
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            buffers = chunk
        offset += 8 + length + (-length % 4)
    if not gltf:
        raise ValueError("El .glb no trae su parte JSON")

    def accessor(idx: int) -> np.ndarray:
        acc = gltf["accessors"][idx]
        if acc.get("componentType") != _FLOAT:
            raise ValueError("Solo se leen posiciones en float32")
        view = gltf["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        n = acc["count"] * _NUM[acc["type"]]
        stride = view.get("byteStride")
        if stride and stride != 4 * _NUM[acc["type"]]:
            out = np.empty(n, dtype=np.float32)
            per = _NUM[acc["type"]]
            for i in range(acc["count"]):
                out[i * per:(i + 1) * per] = np.frombuffer(
                    buffers, dtype=np.float32, count=per, offset=start + i * stride)
            return out.reshape(acc["count"], per)
        return np.frombuffer(buffers, dtype=np.float32, count=n, offset=start).reshape(acc["count"], per_ := _NUM[acc["type"]])

    def node_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T
        m = np.eye(4)
        if "scale" in node:
            m[:3, :3] = np.diag(node["scale"])
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            r = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            m[:3, :3] = r @ m[:3, :3]
        if "translation" in node:
            m[:3, 3] = node["translation"]
        return m

    chunks: list[np.ndarray] = []

    def walk(node_idx: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][node_idx]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
                pos = prim.get("attributes", {}).get("POSITION")
                if pos is None:
                    continue
                v = accessor(pos)
                chunks.append((world[:3, :3] @ v.T).T + world[:3, 3])
        for child in node.get("children", []):
            walk(child, world)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for root in scene.get("nodes", range(len(gltf.get("nodes", [])))):
        walk(root, np.eye(4))
    if not chunks:
        raise ValueError("El .glb no trae ninguna malla con vértices")
    return np.concatenate(chunks).astype(float)


# ---------- 2. Orientar ----------
def _symmetry_error(pts: np.ndarray, angle: float, grid: int = 48) -> float:
    """Cuánto se parecen las dos mitades si se corta por el plano que
    contiene el eje vertical y forma `angle` con el eje x."""
    c, s = np.cos(angle), np.sin(angle)
    x = pts[:, 0] * c + pts[:, 2] * s          # a lo ancho del corte
    z = -pts[:, 0] * s + pts[:, 2] * c         # profundidad
    y = pts[:, 1]
    ys = np.linspace(y.min(), y.max(), grid)
    err = tot = 0.0
    for lo, hi in zip(ys[:-1], ys[1:]):
        band = (y >= lo) & (y < hi)
        if band.sum() < 20:
            continue
        zb, xb = z[band], x[band]
        zs = np.linspace(zb.min(), zb.max(), 16)
        for zlo, zhi in zip(zs[:-1], zs[1:]):
            cell = (zb >= zlo) & (zb < zhi)
            if cell.sum() < 4:
                continue
            left, right = xb[cell][xb[cell] > 0], xb[cell][xb[cell] < 0]
            if not len(left) or not len(right):
                continue
            err += abs(left.max() + right.min())
            tot += (left.max() - right.min()) / 2
    return err / tot if tot else 1.0


def orient(pts: np.ndarray) -> np.ndarray:
    """Devuelve los vértices centrados y girados a los ejes del proyecto:
    +y arriba, +z hacia la cara, +x hacia la izquierda del cliente.

    Se asume +y arriba (lo estándar en glTF, y lo que devuelve Tripo); el
    giro alrededor de ese eje se busca: primero el plano de simetría, luego
    de qué lado está la cara."""
    pts = pts - np.array([np.median(pts[:, 0]), 0.0, np.median(pts[:, 2])])
    angles = np.linspace(0, np.pi, 36, endpoint=False)
    best = min(angles, key=lambda a: _symmetry_error(pts, a))
    # Refinar alrededor del mejor.
    best = min(np.linspace(best - 0.09, best + 0.09, 13), key=lambda a: _symmetry_error(pts, a))
    c, s = np.cos(best), np.sin(best)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = pts @ rot.T

    # ¿Hacia dónde mira? La nariz es un saliente estrecho; la nuca, ancha.
    head = out[out[:, 1] > np.percentile(out[:, 1], 55)]
    def tightness(sign: int) -> float:
        d = head[:, 2] * sign
        tip = head[d > np.percentile(d, 99)]
        return float(np.abs(tip[:, 0]).mean() + np.abs(tip[:, 1] - np.median(tip[:, 1])).mean())
    if tightness(+1) > tightness(-1):      # el saliente estrecho está en -z
        out[:, [0, 2]] *= -1
    out[:, 1] -= out[:, 1].max()           # 0 = punto más alto de la cabeza
    return out


# ---------- 3. Medir ----------
def _width_at(pts: np.ndarray, y: float, band: float) -> float:
    sel = pts[np.abs(pts[:, 1] - y) < band]
    return float(sel[:, 0].max() - sel[:, 0].min()) if len(sel) > 10 else float("nan")


def _profile_curve(pts: np.ndarray, band: float) -> np.ndarray:
    """Silueta del perfil: por cada altura, el punto que más sobresale hacia
    la cara (dentro de una franja estrecha alrededor del plano de simetría)."""
    mid = pts[np.abs(pts[:, 0]) < band]
    ys = np.linspace(mid[:, 1].min(), 0, 120)
    out = []
    for lo, hi in zip(ys[:-1], ys[1:]):
        sel = mid[(mid[:, 1] >= lo) & (mid[:, 1] < hi)]
        if len(sel) > 3:
            out.append(((lo + hi) / 2, float(sel[:, 2].max())))
    return np.array(out) if out else np.zeros((0, 2))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _width_profile(pts: np.ndarray, steps: int = 80) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(pts[:, 1].min(), 0, steps)
    w = np.array([_width_at(pts, y, (0 - pts[:, 1].min()) / steps * 1.5) for y in ys])
    return ys, w


def _neck_y(ys: np.ndarray, w: np.ndarray) -> float:
    """Altura del cuello: el estrechamiento entre la cabeza y los hombros.

    De arriba abajo el ancho crece (cabeza), baja (cuello) y vuelve a crecer
    (hombros). Se busca el mínimo por debajo de la parte más ancha de la
    cabeza; si la malla es solo cabeza y no hay hombros, se queda con el
    punto más bajo."""
    ok = ~np.isnan(w)
    ys, w = ys[ok], w[ok]
    upper = ys > ys.min() + (0 - ys.min()) * 0.45
    if upper.sum() < 3:
        return float(ys.min())
    widest = ys[upper][int(np.argmax(w[upper]))]
    below = ys < widest
    if below.sum() < 3:
        return float(ys.min())
    return float(ys[below][int(np.argmin(w[below]))])


def measure(pts: np.ndarray) -> dict:
    """Medidas de la cabeza ya orientada (`orient`). Todo en proporciones
    respecto al ancho de la cabeza, nunca en centímetros: el modelo no trae
    escala real (una foto no dice cuánto mide una cabeza)."""
    pts = orient(pts)
    ys, wprof = _width_profile(pts)
    neck = _neck_y(ys, wprof)
    head = pts[pts[:, 1] >= neck]              # cabeza y cuello, sin hombros
    if len(head) < 500:
        raise ValueError("La malla no parece una cabeza")
    W = float(np.nanmax(wprof[ys >= neck]))    # ancho máximo de la cabeza
    band = W * 0.03

    profile = _profile_curve(head, band)
    if len(profile) < 15:
        raise ValueError("La malla no tiene suficiente detalle para medir el perfil")
    nose_i = int(np.argmax(profile[:, 1]))
    nose = profile[nose_i]
    below = profile[profile[:, 0] < nose[0] - W * 0.18]
    chin = below[int(np.argmax(below[:, 1]))] if len(below) > 3 else nose
    above = profile[profile[:, 0] > nose[0] + W * 0.14]
    brow = above[int(np.argmax(above[:, 1]))] if len(above) > 3 else nose

    face_h = abs(chin[0])                       # de la barbilla a lo más alto
    cheek_w = _width_at(head, (nose[0] + brow[0]) / 2, W * 0.05)
    jaw_w = _width_at(head, chin[0] + face_h * 0.10, W * 0.05)
    brow_w = _width_at(head, brow[0], W * 0.05)
    neck_w = _width_at(pts, neck, W * 0.04)

    # Convexidad del perfil (entrecejo -> subnasal -> mentón), el mismo
    # ángulo que ilustra frontend/guia-visagismo.html: 0° = recto,
    # positivo = convexo (frente/nariz salientes respecto al mentón).
    sub_y = nose[0] - W * 0.11                  # base de la nariz, no la punta
    sub = profile[int(np.argmin(np.abs(profile[:, 0] - sub_y)))]
    convexity = 180 - _angle(np.array([brow[1], brow[0]]), np.array([sub[1], sub[0]]),
                             np.array([chin[1], chin[0]]))

    # Definición de la mandíbula. A propósito NO se da en grados: el
    # "ángulo mandibular" de verdad (gonion, ~120-130°) necesita puntos
    # anatómicos que esta malla no marca, y dar un número inventado con
    # dos decimales es peor que no darlo. Se mide lo que sí se puede ver:
    # cuánto se mantiene el ancho de la mandíbula respecto al de los
    # pómulos (1 = tan ancha como los pómulos, cuadrada; 0,8 = se estrecha
    # mucho, mandíbula suave).
    jaw_index = jaw_w / cheek_w if cheek_w else float("nan")

    symmetry = max(0.0, 100.0 * (1 - _symmetry_error(head, 0.0)))

    return {
        "estimado": True,
        "proporcion": round(float(face_h / cheek_w), 2) if cheek_w else None,
        "simetria_pct": round(float(symmetry), 1),
        "perfil_grados": round(float(convexity), 1),
        "mandibula_indice": round(float(jaw_index), 3),
        "ancho_pomulos": round(float(cheek_w / W), 3),
        "ancho_mandibula": round(float(jaw_w / W), 3),
        "ancho_frente": round(float(brow_w / W), 3),
        "ancho_cuello": round(float(neck_w / W), 3),
        "alto_cabeza": round(float(face_h / W), 3),
    }


# Clasificación a partir de las medidas. Umbrales de partida (ver el aviso
# de arriba): proporción y anchos relativos como en las guías de visajismo
# al uso, convexidad con los mismos cortes que frontend/guia-visagismo.html.
def classify(m: dict) -> dict:
    ratio, jaw, cheek, brow = m["proporcion"], m["ancho_mandibula"], m["ancho_pomulos"], m["ancho_frente"]
    if ratio >= 1.55:
        geometry = "rectangular_elongated"
    elif ratio <= 1.25:
        geometry = "round" if jaw < cheek * 0.92 else "square"
    elif jaw >= cheek * 0.97 and brow >= cheek * 0.93:
        geometry = "square"
    elif brow > jaw * 1.12:
        geometry = "heart" if jaw < cheek * 0.85 else "triangle"
    elif cheek > brow * 1.08 and cheek > jaw * 1.08:
        geometry = "diamond"
    else:
        geometry = "oval"

    conv = m["perfil_grados"]
    # Cortes de la convexidad de Legan (8-16° = normal); fuera de ahí,
    # convexo o cóncavo. Ver frontend/guia-visagismo.html.
    profile = "convex_prominent_nose" if conv >= 16 else "concave" if conv <= 0 else "straight"
    ji = m.get("mandibula_indice")
    jaw_def = None if ji is None else ("defined" if ji >= 0.88 else "soft")
    neck = "short_thick" if m["ancho_cuello"] >= 0.72 else "long_thin" if m["ancho_cuello"] <= 0.55 else None
    out = {"facial_geometry": geometry, "profile_type": profile,
           "face_symmetry_percent": m["simetria_pct"]}
    if jaw_def:
        out["jawline_definition"] = jaw_def
    if neck:
        out["neck_proportions"] = neck
    return out


def analyze(glb: bytes) -> dict:
    """.glb -> {"medidas": ..., "rasgos": ...} para guardar en la ficha."""
    m = measure(load_glb(glb))
    return {"medidas": m, "rasgos": classify(m)}
