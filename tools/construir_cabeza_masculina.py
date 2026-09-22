"""
Construye frontend/assets/head.glb (maniquí masculino con la zona del pelo
pintada) a partir del maniquí neutro v3 y del base mesh + targets de
MakeHuman (todo CC0, ver CLAUDE.md).

Uso (el resultado ya está en el repo; solo hace falta para rehacerlo):

    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/makehumancommunity/makehuman.git /tmp/mh
    cd /tmp/mh && git sparse-checkout set makehuman/data/3dobjs \\
        makehuman/data/targets/macrodetails makehuman/data/targets/head \\
        makehuman/data/targets/chin makehuman/data/targets/eyebrows
    git show 5948b9e:frontend/assets/head.glb > /tmp/head_v3_neutral.glb
    pip install trimesh pygltflib scipy rtree
    python tools/construir_cabeza_masculina.py /tmp/mh /tmp/head_v3_neutral.glb /tmp/v5/head_a.glb
    python tools/mejorar_ojos_cabeza.py /tmp/v5/head_a.glb frontend/assets/head.glb
    rm -rf frontend/assets/rasgos && cp -r /tmp/v5/rasgos frontend/assets/rasgos

(el sparse-checkout necesita además makehuman/data/targets/{ears,eyes,nose,
neck,forehead} para los rasgos del paso 6)

Pasos:
1. Se ajusta (ICP con escala) el base mesh recortado a la piel del maniquí
   neutro, para pasar de coordenadas de MakeHuman a las del .glb.
2. Se aplican al base mesh los targets de hombre joven (media de las tres
   etnias, lo mismo que "género 100 %" en MakeHuman) y unos pocos de rasgos
   (cabeza algo cuadrada, mentón y mandíbula más marcados, cejas más bajas).
3. Cada vértice del maniquí se proyecta sobre el triángulo más cercano del
   base mesh original y se mueve con ese triángulo ya deformado
   (coordenadas baricéntricas). El desplazamiento se suaviza un poco sobre
   la malla: el base mesh es de baja resolución y sin eso se notan facetas.
4. Se realinea para que el cráneo quede donde estaba (error ~2 cm a escala
   real): los mapas de remolinos ya guardados siguen cayendo sobre la
   cabeza, y growth-map.html además los pega a la superficie al cargarlos.
5. Se pintan por vértice las cejas y una sombra de barba suave. El color
   del pelo ya NO se hornea (v5): se guardan la máscara del cuero
   cabelludo (`_SCALP`), la distancia a la línea del pelo (`_HAIRH`), el
   ángulo desde la cara (`_HAIRTH`) y las orejas (`_EARS`), y
   growth-map.html pinta el color del cliente y mueve la línea del pelo
   (entradas, frente alta, pico) al vuelo.
6. Rasgos del cliente ("morph targets", tabla MORPHS): cada uno se guarda
   aparte en `rasgos/<nombre>.bin` (int16, solo los vértices que se mueven)
   con `rasgos/index.json`; la página descarga solo los que necesita ese
   cliente. Se compensa la traslación del cráneo para que los mapas de
   remolinos sigan cayendo encima.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from pygltflib import GLTF2, Accessor, BufferView

MALE = 1.0
EXTRA_TARGETS = {
    "head/head-square": 0.35,
    "chin/chin-width-incr": 0.35,
    "chin/chin-bones-incr": 0.35,
    "chin/chin-prominent-incr": 0.2,
    "eyebrows/eyebrows-trans-down": 0.35,
    "eyebrows/eyebrows-trans-forward": 0.3,
}
# Rasgos del cliente como "morph targets" (se mezclan en growth-map.html
# según su ficha, ver backend/app/pipeline/avatar.py). Cada uno es una
# combinación de targets CC0 de MakeHuman. "l" = lado izquierdo del
# personaje = +x del maniquí = lado izquierdo del cliente.
MORPHS = {
    "face-oval": {"head/head-oval": 1.0},
    "face-round": {"head/head-round": 1.0},
    "face-square": {"head/head-square": 1.0},
    "face-rectangular": {"head/head-rectangular": 1.0},
    "face-diamond": {"head/head-diamond": 1.0},
    "face-triangle": {"head/head-triangular": 1.0},
    "face-heart": {"head/head-invertedtriangular": 1.0},
    "skull-short": {"head/head-back-scale-depth-decr": 1.0},
    "skull-long": {"head/head-back-scale-depth-incr": 1.0},
    "chin-retruded": {"chin/chin-prominent-decr": 1.0, "chin/chin-prognathism-decr": 0.5},
    "chin-prominent": {"chin/chin-prominent-incr": 1.0, "chin/chin-prognathism-incr": 0.5},
    "jaw-soft": {"chin/chin-bones-decr": 1.0, "chin/chin-width-decr": 0.5},
    "jaw-defined": {"chin/chin-bones-incr": 1.0, "chin/chin-width-incr": 0.5},
    "ears-prominent": {"ears/l-ear-wing-incr": 1.0, "ears/r-ear-wing-incr": 1.0},
    "eyes-close": {"eyes/l-eye-trans-in": 1.0, "eyes/r-eye-trans-in": 1.0},
    "eyes-wide": {"eyes/l-eye-trans-out": 1.0, "eyes/r-eye-trans-out": 1.0},
    "eye-l-small": {"eyes/l-eye-height2-decr": 1.0, "eyes/l-eye-height1-decr": 0.5},
    "eye-r-small": {"eyes/r-eye-height2-decr": 1.0, "eyes/r-eye-height1-decr": 0.5},
    "nose-convex": {"nose/nose-hump-incr": 1.0, "nose/nose-scale-depth-incr": 0.5},
    "nose-concave": {"nose/nose-curve-concave": 1.0},
    "neck-short": {"neck/neck-scale-vert-decr": 1.0, "neck/neck-scale-horiz-incr": 0.6},
    "neck-long": {"neck/neck-scale-vert-incr": 1.0, "neck/neck-scale-horiz-decr": 0.5},
    "brow-ridge": {"eyebrows/eyebrows-trans-forward": 1.0, "forehead/forehead-nubian-incr": 0.5},
    "forehead-tall": {"forehead/forehead-scale-vert-incr": 1.0},
    "forehead-short": {"forehead/forehead-scale-vert-decr": 1.0},
}
CROP_Y = 5.5            # recorte del base mesh (cabeza + cuello), unidades MakeHuman
SMOOTH_ITERS = 8
HAIR_RGB = np.array([0.24, 0.19, 0.16])
# Línea del nacimiento del pelo: (ángulo desde la cara en grados, altura
# normalizada: 0 = ojos, 1 = parte más alta). Ajustada a ojo con capturas.
HAIRLINE = np.array([[0, .56], [22, .57], [36, .62], [50, .44], [64, .24], [74, -.05], [80, -.28],
                     [86, -.20], [92, .22], [102, .20], [112, .02], [124, -.42], [145, -.60], [180, -.64]])

CT = {5126: np.float32, 5125: np.uint32, 5121: np.uint8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# --- glTF ---
def read_acc(g, blob, i):
    a = g.accessors[i]
    bv = g.bufferViews[a.bufferView]
    n = NC[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    arr = np.frombuffer(blob, dtype=CT[a.componentType], count=a.count * n, offset=off)
    return arr.reshape(a.count, n).copy() if n > 1 else arr.copy()


def rebuild(g, blob, replace, add=()):
    views = [bytearray(blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]) for bv in g.bufferViews]
    for ai, arr in replace.items():
        a = g.accessors[ai]
        views[a.bufferView] = bytearray(np.ascontiguousarray(arr.astype(CT[a.componentType])).tobytes())
        if a.type == "VEC3":
            a.max, a.min = arr.max(0).tolist(), arr.min(0).tolist()
    ids = []
    for arr, typ, ct in add:
        views.append(bytearray(np.ascontiguousarray(arr.astype(CT[ct])).tobytes()))
        g.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=0))
        acc = Accessor(bufferView=len(g.bufferViews) - 1, componentType=ct, count=len(arr), type=typ)
        if typ == "SCALAR":
            acc.max, acc.min = [float(arr.max())], [float(arr.min())]
        g.accessors.append(acc)
        ids.append(len(g.accessors) - 1)
    out = bytearray()
    for i, v in enumerate(views):
        while len(out) % 4:
            out.append(0)
        g.bufferViews[i].byteOffset, g.bufferViews[i].byteLength = len(out), len(v)
        out += v
    while len(out) % 4:
        out.append(0)
    g.buffers[0].byteLength = len(out)
    g.set_binary_blob(bytes(out))
    return ids


def write_morph(path, dm):
    """Rasgo en binario compacto (lo lee assets/rasgos.js):
    uint32 n_vértices_movidos, float32 escala, n x uint32 índices,
    n x 3 x int16 desplazamiento / escala * 32767."""
    idx = np.where(np.abs(dm).max(1) > 0)[0].astype(np.uint32)
    vals = dm[idx]
    scale = float(np.abs(vals).max()) if len(idx) else 1.0
    q = np.round(vals / scale * 32767).astype(np.int16)
    with open(path, "wb") as f:
        f.write(np.array([len(idx)], np.uint32).tobytes())
        f.write(np.array([scale], np.float32).tobytes())
        f.write(idx.tobytes())
        f.write(q.tobytes())


# --- MakeHuman ---
def load_obj(path):
    V, faces, groups, grp = [], [], [], None
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            grp = line.split()[1]
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
            groups.append(grp)
    return np.array(V), faces, groups


def load_target(path, n):
    d = np.zeros((n, 3))
    for line in open(path):
        if line[0] != "#" and line.strip():
            p = line.split()
            d[int(p[0])] = [float(x) for x in p[1:4]]
    return d


def main(mh_root, neutral_glb, out_glb):
    data = Path(mh_root) / "makehuman" / "data"
    V, F, G = load_obj(data / "3dobjs" / "base.obj")
    n = len(V)
    D = sum(load_target(data / f"targets/macrodetails/{e}-male-young.target", n) for e in ("african", "asian", "caucasian")) / 3 * MALE
    for name, w in EXTRA_TARGETS.items():
        D = D + load_target(data / f"targets/{name}.target", n) * w
    groups = {}
    for fc, gname in zip(F, G):
        groups.setdefault(gname, set()).update(fc)

    g = GLTF2().load(neutral_glb)
    blob = g.binary_blob()
    meshes = {m.name: m.primitives[0] for m in g.meshes}
    skin, lashes, eyes = meshes["geometry_0"], meshes["geometry_1"], meshes["geometry_2"]
    P_skin = read_acc(g, blob, skin.attributes.POSITION).astype(float)

    # 1. MakeHuman -> glb
    body = np.array(sorted(groups["body"]))
    P = V[body[V[body, 1] > CROP_Y]]
    sc = 0.95 / (P[:, 1].max() - P[:, 1].min())
    init = np.eye(4)
    init[:3, :3] *= sc
    init[:3, 3] = -(P.max(0) + P.min(0)) / 2 * sc
    T, _, _ = trimesh.registration.icp(P, P_skin, initial=init, scale=True, max_iterations=60, reflection=False)
    s, t = T[0, 0], T[:3, 3]
    to_glb = lambda q: q * s + t

    # 2-3. Deformación por baricéntricas
    tri = []
    for fc, gname in zip(F, G):
        if gname == "body":
            tri += [[fc[0], fc[1], fc[2]], [fc[0], fc[2], fc[3]]] if len(fc) == 4 else [fc[:3]]
    tri = np.array(tri)
    tri = tri[(V[tri][:, :, 1] > CROP_Y - 0.9).all(1)]
    base = trimesh.Trimesh(to_glb(V), tri, process=False)
    Vd = to_glb(V + D)

    def transfer(pts):
        cp, _, fid = trimesh.proximity.closest_point(base, pts)
        bary = trimesh.triangles.points_to_barycentric(base.triangles[fid], cp)
        return (Vd[tri[fid]] * bary[:, :, None]).sum(1) + (pts - cp)

    cp_s, _, fid_s = trimesh.proximity.closest_point(base, P_skin)
    bary_s = trimesh.triangles.points_to_barycentric(base.triangles[fid_s], cp_s)

    new = {skin.attributes.POSITION: transfer(P_skin),
           lashes.attributes.POSITION: transfer(read_acc(g, blob, lashes.attributes.POSITION).astype(float))}
    PE = read_acc(g, blob, eyes.attributes.POSITION).astype(float)
    le, rei = np.array(sorted(groups["helper-l-eye"])), np.array(sorted(groups["helper-r-eye"]))
    is_l = np.linalg.norm(PE - to_glb(V[le].mean(0)), axis=1) < np.linalg.norm(PE - to_glb(V[rei].mean(0)), axis=1)
    PE[is_l] += D[le].mean(0) * s
    PE[~is_l] += D[rei].mean(0) * s
    new[eyes.attributes.POSITION] = PE

    faces = read_acc(g, blob, skin.indices).reshape(-1, 3)
    edges = trimesh.Trimesh(P_skin, faces, process=False).edges_unique
    A = sp.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(P_skin),) * 2).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(1 / np.maximum(deg, 1)) @ A
    disp = new[skin.attributes.POSITION] - P_skin
    for _ in range(SMOOTH_ITERS):
        disp = 0.5 * disp + 0.5 * (L @ disp)
    new[skin.attributes.POSITION] = P_skin + disp

    # 4. Realinear el cráneo (Umeyama, mismos índices)
    crane = P_skin[:, 1] > 0.22
    Aa, Bb = new[skin.attributes.POSITION][crane], P_skin[crane]
    ma, mb = Aa.mean(0), Bb.mean(0)
    U, S_, Vt = np.linalg.svd((Aa - ma).T @ (Bb - mb))
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    k = (S_ * [1, 1, d]).sum() / ((Aa - ma) ** 2).sum()
    for key in new:
        new[key] = (k * (R @ (new[key] - ma).T)).T + mb
    rebuild(g, blob, new)
    blob = g.binary_blob()

    # 5. Pintura
    S = new[skin.attributes.POSITION]
    col = read_acc(g, blob, skin.attributes.COLOR_0).astype(float) / 255
    E = new[eyes.attributes.POSITION]
    eye_pts = [E[E[:, 0] > 0].mean(0), E[E[:, 0] <= 0].mean(0)]
    eye_y = float((eye_pts[0][1] + eye_pts[1][1]) / 2)
    ipd = float(eye_pts[0][0] - eye_pts[1][0])
    h = (S[:, 1] - eye_y) / (S[:, 1].max() - eye_y)
    th = np.degrees(np.abs(np.arctan2(S[:, 0], S[:, 2] + 0.02)))

    def smooth(e0, e1, x):
        tt = np.clip((x - e0) / (e1 - e0), 0, 1)
        return tt * tt * (3 - 2 * tt)

    hair_h = h - np.interp(th, HAIRLINE[:, 0], HAIRLINE[:, 1])   # >0 = por encima del nacimiento del pelo
    ears = ((th > 70) & (th < 125) & (h > -0.75) & (h < 0.30) & (np.abs(S[:, 0]) > 0.235)).astype(np.float32)
    mask = smooth(-0.035, 0.035, hair_h) * (1 - ears)
    # El color del pelo YA NO va horneado: lo pinta growth-map.html con el
    # color de pelo del cliente y su línea del pelo (entradas, frente alta).
    out = col.copy()
    for ex, ey, ez in eye_pts:
        side = np.sign(ex)
        dx = (S[:, 0] - ex) / (0.36 * ipd)
        dy = (S[:, 1] - (ey + 0.33 * ipd + 0.05 * ipd * (1 - dx ** 2))) / (0.075 * ipd)
        b = np.clip(1 - (dx ** 2 + dy ** 2), 0, 1) ** 0.6 * (S[:, 2] > ez - 0.25 * ipd)
        b = np.clip(b * np.clip(1.15 - 0.35 * dx * side, 0, 1) * 1.1, 0, 0.9)
        out[:, :3] = out[:, :3] * (1 - b[:, None]) + np.array([0.16, 0.12, 0.10]) * b[:, None]
    beard = smooth(-0.30, -0.42, h) * (1 - smooth(-1.05, -1.12, h)) * (1 - smooth(80, 100, th))
    beard = np.maximum(beard, smooth(-0.26, -0.30, h) * (1 - smooth(-0.40, -0.44, h)) * (th < 22)) * 0.22
    out[:, :3] = out[:, :3] * (1 - beard[:, None]) + np.array([0.30, 0.26, 0.25]) * beard[:, None]
    rebuild(g, blob, {skin.attributes.COLOR_0: (np.clip(out, 0, 1) * 255).round()})
    ids = rebuild(g, g.binary_blob(), {}, [(mask.astype(np.float32), "SCALAR", 5126),
                                           (hair_h.astype(np.float32), "SCALAR", 5126),
                                           (th.astype(np.float32), "SCALAR", 5126),
                                           (ears, "SCALAR", 5126)])
    skin.attributes._SCALP, skin.attributes._HAIRH, skin.attributes._HAIRTH, skin.attributes._EARS = ids

    # 6. Rasgos (morph targets). Desplazamiento del base mesh -> maniquí por
    #    las mismas baricéntricas, suavizado igual, girado/escalado como el
    #    realineado, y sin traslación del cráneo (si un rasgo "mueve" la
    #    cabeza entera, p.ej. el cuello, se compensa para que los mapas de
    #    remolinos sigan cayendo encima).
    top = S[:, 1] > 0.33
    out_dir = Path(out_glb).parent / "rasgos"
    out_dir.mkdir(exist_ok=True)
    index = {}
    for name, parts in MORPHS.items():
        Dm = sum(load_target(data / f"targets/{t}.target", n) * w for t, w in parts.items())
        dm = s * (Dm[tri[fid_s]] * bary_s[:, :, None]).sum(1)
        for _ in range(SMOOTH_ITERS):
            dm = 0.5 * dm + 0.5 * (L @ dm)
        dm = (k * (R @ dm.T)).T
        drift = dm[top].mean(0)
        dm -= drift
        # Por debajo de ~0,3 mm reales no se ve: fuera (el archivo baja mucho).
        dm[np.abs(dm).max(1) < 6e-4] = 0
        write_morph(out_dir / f"{name}.bin", dm)
        eyes_d = [((k * (R @ (Dm[side].mean(0) * s))) - drift).round(5).tolist() for side in (le, rei)]
        index[name] = {"eyes": eyes_d}   # [izquierdo, derecho]: los ojos se mueven enteros
        print(f"  rasgo {name}: {int((np.abs(dm).max(1) > 0).sum())} vértices, máx {np.abs(dm).max():.4f}")
    (out_dir / "index.json").write_text(json.dumps({"vertices": len(S), "morphs": index}, indent=1))
    g.save(out_glb)
    print(f"{out_glb}: cuero cabelludo {int((mask > 0.5).sum())} vértices de {len(S)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
