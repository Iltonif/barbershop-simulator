"""
Genera las ilustraciones de perfil de `frontend/guia-visagismo.html` y la
silueta del marco de la cámara guiada de `frontend/visagismo.html`.

Cada perfil se construye a partir de puntos de referencia (glabela,
subnasal, labio inferior, pogonion, mentón, punto cervical...) y se
COLOCAN para que su ángulo sea exactamente el que ilustra, usando las
mismas definiciones que las referencias publicadas:

- Convexidad de Legan: 180° - ángulo(glabela, subnasal, pogonion).
  Normal 8-16°.
- Ángulo labio inferior-mentón: entre la vertical por el labio inferior y
  la recta labio inferior -> pogonion. Positivo = mentón por detrás.
  Aceptable de -5° a 15°.
- Ángulo cervicomental: entre la recta submental y la del cuello. Ideal
  105-120°.

El script comprueba al final que cada ilustración mide lo que dice y
escribe `frontend/assets/guia/perfiles.json` (SVG listo para incrustar).

Uso (desde la raíz del repo):
    python3 frontend/tools/generar_perfiles_guia.py
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "guia" / "perfiles.json"
W, H = 240, 360


def _rot_ccw(v, deg):
    """Gira v en sentido antihorario tal como se ve en pantalla (y hacia abajo)."""
    t = math.radians(deg)
    return (v[0] * math.cos(t) + v[1] * math.sin(t), -v[0] * math.sin(t) + v[1] * math.cos(t))


def _norm(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def _angle(a, b):
    c = (a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def convexity(p):
    g, sn, pog = p["G"], p["Sn"], p["Pog"]
    return 180 - _angle((g[0] - sn[0], g[1] - sn[1]), (pog[0] - sn[0], pog[1] - sn[1]))


def chin_angle(p):
    v = (p["Pog"][0] - p["Li"][0], p["Pog"][1] - p["Li"][1])
    return math.degrees(math.atan2(-v[0], v[1]))


def cervicomental(p):
    c = p["C"]
    return _angle((p["Me"][0] - c[0], p["Me"][1] - c[1]), (p["NeckLow"][0] - c[0], p["NeckLow"][1] - c[1]))


def base_points():
    return {
        "G": (170.0, 118.0), "N": (165.0, 138.0), "Dorsum": (180.0, 160.0), "Prn": (193.0, 178.0),
        "Col": (181.0, 188.0), "Sn": (171.0, 191.0), "Ls": (176.0, 202.0), "Sto": (170.0, 210.0),
        "Li": (174.0, 217.0), "Sm": (166.0, 229.0), "Pog": (170.0, 243.0), "Me": (160.0, 257.0),
    }


def build(conv=None, chin=None, cma=112.0):
    p = base_points()
    li, sn = p["Li"], p["Sn"]

    def move_chin(pog_x):
        dx = pog_x - p["Pog"][0]
        p["Pog"] = (pog_x, p["Pog"][1])
        p["Sm"] = (p["Sm"][0] + 0.55 * dx, p["Sm"][1])
        p["Me"] = (p["Me"][0] + 0.9 * dx, p["Me"][1])

    if chin is not None:
        # Pogonion justo donde da el ángulo pedido respecto al labio inferior.
        move_chin(li[0] - math.tan(math.radians(chin)) * (p["Pog"][1] - li[1]))
    if conv is not None:
        lo, hi = sn[0] - 80, sn[0] + 80
        for _ in range(60):  # bisección sobre la x del pogonion
            mid = (lo + hi) / 2
            trial = dict(p, Pog=(mid, p["Pog"][1]))
            if convexity(trial) > conv:
                lo = mid
            else:
                hi = mid
        move_chin((lo + hi) / 2)

    # Cuello: la recta del cuello baja casi vertical desde el punto
    # cervical; la submental sube hacia el mentón con el ángulo pedido.
    neck_dir = _norm((0.18, 1.0))
    sub_dir = _rot_ccw(neck_dir, cma)
    length = 48.0
    me = p["Me"]
    p["C"] = (me[0] - length * sub_dir[0], me[1] - length * sub_dir[1])
    p["NeckLow"] = (p["C"][0] + 70 * neck_dir[0], p["C"][1] + 70 * neck_dir[1])
    return p


def outline(p):
    """Contorno cerrado de la cabeza (en sentido horario desde la nuca)."""
    c, nl = p["C"], p["NeckLow"]
    front = [p["G"], p["N"], p["Dorsum"], p["Prn"], p["Col"], p["Sn"], p["Ls"], p["Sto"], p["Li"],
             p["Sm"], p["Pog"], p["Me"], ((p["Me"][0] + c[0]) / 2 + 2, (p["Me"][1] + c[1]) / 2 + 1), c,
             ((c[0] + nl[0]) / 2, (c[1] + nl[1]) / 2), (nl[0], H + 6)]
    back = [(58.0, H + 6), (62.0, 300.0), (68.0, 246.0), (56.0, 206.0), (38.0, 150.0), (46.0, 104.0),
            (74.0, 66.0), (118.0, 46.0), (152.0, 58.0), (166.0, 82.0), (171.0, 100.0)]
    return back + front


def hair(p):
    """Pelo corto con volumen arriba, sobre la frente sin taparla."""
    return [(58.0, 214.0), (40.0, 172.0), (36.0, 128.0), (48.0, 88.0), (76.0, 56.0), (118.0, 38.0),
            (156.0, 48.0), (174.0, 72.0), (173.0, 90.0), (160.0, 86.0), (140.0, 82.0), (118.0, 88.0),
            (104.0, 104.0), (98.0, 128.0), (92.0, 150.0), (84.0, 176.0), (78.0, 200.0), (70.0, 218.0)]


def _catmull_rom(points, closed):
    pts = list(points)
    n = len(pts)
    get = (lambda i: pts[i % n]) if closed else (lambda i: pts[max(0, min(n - 1, i))])
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d + (" Z" if closed else "")


def _arc(center, v_from, v_to, r):
    a1 = math.atan2(v_from[1], v_from[0])
    a2 = math.atan2(v_to[1], v_to[0])
    delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    x1, y1 = center[0] + r * math.cos(a1), center[1] + r * math.sin(a1)
    x2, y2 = center[0] + r * math.cos(a1 + delta), center[1] + r * math.sin(a1 + delta)
    sweep = 1 if delta > 0 else 0
    large = 1 if abs(delta) > math.pi else 0
    mid = a1 + delta / 2
    label = (center[0] + (r + 16) * math.cos(mid), center[1] + (r + 16) * math.sin(mid))
    return f"M{x1:.1f},{y1:.1f} A{r},{r} 0 {large} {sweep} {x2:.1f},{y2:.1f}", label


def _line(a, b, cls):
    return f'<line class="{cls}" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>'


def _dot(p):
    return f'<circle class="m-dot" cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2.6"/>'


def overlay(p, kind):
    parts = []
    if kind == "conv":
        g, sn, pog = p["G"], p["Sn"], p["Pog"]
        ext = (sn[0] + (sn[0] - g[0]) * 0.75, sn[1] + (sn[1] - g[1]) * 0.75)
        parts += [_line(g, sn, "m-line"), _line(sn, pog, "m-line"), _line(sn, ext, "m-dash")]
        arc, lab = _arc(sn, (ext[0] - sn[0], ext[1] - sn[1]), (pog[0] - sn[0], pog[1] - sn[1]), 30)
        value = convexity(p)
        center, pts = sn, [g, sn, pog]
        # Con ángulos pequeños el arco no deja sitio: etiqueta a un lado.
        lab = (max(sn[0], pog[0]) + 22, sn[1] + 30)
    elif kind == "chin":
        li, pog = p["Li"], p["Pog"]
        down = (li[0], li[1] + 48)
        parts += [_line(li, down, "m-dash"), _line(li, (li[0] + (pog[0] - li[0]) * 1.5, li[1] + (pog[1] - li[1]) * 1.5), "m-line")]
        arc, lab = _arc(li, (0, 1), (pog[0] - li[0], pog[1] - li[1]), 32)
        value = chin_angle(p)
        center, pts = li, [li, pog]
    else:
        c, me, nl = p["C"], p["Me"], p["NeckLow"]
        parts += [_line(c, (c[0] + (me[0] - c[0]) * 1.2, c[1] + (me[1] - c[1]) * 1.2), "m-line"), _line(c, nl, "m-line")]
        arc, lab = _arc(c, (me[0] - c[0], me[1] - c[1]), (nl[0] - c[0], nl[1] - c[1]), 22)
        value = cervicomental(p)
        center, pts = c, [c]
    parts.append(f'<path class="m-arc" d="{arc}"/>')
    parts += [_dot(q) for q in pts]
    text = f"{value:.0f}°" if kind != "chin" else f"{abs(value):.0f}°"
    parts.append(f'<text class="m-label" x="{lab[0]:.1f}" y="{lab[1] + 4:.1f}" text-anchor="middle">{text}</text>')
    return "".join(parts), value


def features(p):
    """Oreja, ojo y ceja, colocados respecto a la cara."""
    n = p["N"]
    eye = (n[0] - 12, n[1] + 10)
    me = p["Me"]
    return (
        # Borde de la mandíbula (desde debajo de la oreja hasta el mentón):
        # solo una línea suave, para que se lea dónde está el hueso.
        f'<path class="f-jaw" d="M100,206 Q112,{me[1] - 6:.1f} {me[0] - 14:.1f},{me[1] - 1:.1f}"/>'
        f'<path class="f-ear" d="M98,146 C84,146 80,166 84,180 C87,192 96,196 102,190 C106,184 104,176 100,172 '
        f'C104,164 106,150 98,146 Z"/>'
        f'<path class="f-line" d="M{eye[0] - 7:.1f},{eye[1]:.1f} Q{eye[0]:.1f},{eye[1] - 4:.1f} {eye[0] + 6:.1f},{eye[1] + 1:.1f}"/>'
        f'<path class="f-brow" d="M{eye[0] - 12:.1f},{eye[1] - 10:.1f} Q{eye[0] - 2:.1f},{eye[1] - 15:.1f} {eye[0] + 8:.1f},{eye[1] - 11:.1f}"/>'
    )


# Encuadre de cada tipo de tarjeta: el perfil facial se ve entero; para
# mentón y mandíbula se amplía la parte de abajo de la cara, que es donde
# está la medida.
VIEWBOX = {None: f"0 0 {W} {H}", "conv": "96 84 132 196", "chin": "112 176 104 104",
           "cma": "72 200 160 160"}


def svg(p, kind=None, extra=""):
    body = (
        f'<path class="f-skin" d="{_catmull_rom(outline(p), True)}"/>'
        f'<path class="f-hair" d="{_catmull_rom(hair(p), True)}"/>'
        + features(p) + extra
    )
    value = None
    if kind:
        ov, value = overlay(p, kind)
        body += ov
    return f'<svg viewBox="{VIEWBOX[kind]}" class="profile-svg k-{kind or "plain"}" role="img">{body}</svg>', value


CATEGORIES = {
    "perfil": [
        ("concave", "Cóncavo", dict(conv=3), "conv", "Menos de 8°"),
        ("straight", "Recto (normal)", dict(conv=12), "conv", "Entre 8° y 16°"),
        ("convex", "Convexo", dict(conv=23), "conv", "Más de 16°"),
    ],
    "menton": [
        ("prominent", "Prominente", dict(chin=-12), "chin", "Más de 5° por delante"),
        ("balanced", "Equilibrado", dict(chin=4), "chin", "De 5° por delante a 15° por detrás"),
        ("retruded", "Retraído", dict(chin=24), "chin", "Más de 15° por detrás"),
    ],
    "mandibula": [
        ("defined", "Definida", dict(cma=108), "cma", "105-120° (ideal)"),
        ("soft", "Poco definida", dict(cma=140), "cma", "Más de 120°"),
    ],
}

TARGET = {"conv": "conv", "chin": "chin", "cma": "cma"}


def _phone(x, y, angle=0):
    return (f'<g class="p-phone" transform="translate({x},{y}) rotate({angle})">'
            f'<rect x="-9" y="-16" width="18" height="32" rx="4"/><circle cx="0" cy="-10" r="2.2"/></g>')


def _mark(ok):
    if ok:
        return '<g class="p-ok" transform="translate(206,40)"><circle r="16"/><path d="M-7,0 L-2,6 L8,-6"/></g>'
    return '<g class="p-bad" transform="translate(206,40)"><circle r="16"/><path d="M-6,-6 L6,6 M6,-6 L-6,6"/></g>'


def photo_cards():
    """Cómo hacer la foto de perfil: correcta y errores que cambian la medida."""
    base = build(conv=12)
    eye = (base["N"][0] - 12, base["N"][1] + 10)
    cards = []

    def head(p=base, transform="", extra_front="", extra_back=""):
        return (f'<g transform="{transform}">{extra_back}'
                f'<path class="f-skin" d="{_catmull_rom(outline(p), True)}"/>'
                f'<path class="f-hair" d="{_catmull_rom(hair(p), True)}"/>' + features(p) + extra_front + '</g>')

    horizon = f'<line class="p-guide" x1="{eye[0]:.1f}" y1="{eye[1]:.1f}" x2="232" y2="{eye[1]:.1f}"/>'
    cards.append(("ok", "Correcta", "Cámara a la altura de la cara, cliente mirando al horizonte, perfil completo, fondo liso.",
                  head() + horizon + _phone(226, eye[1]) + _mark(True)))
    cards.append(("camara-baja", "Cámara por debajo",
                  "Se ve más la parte de abajo de la barbilla y cambian los ángulos (en las pruebas, la misma persona pasó de 15° a 33° de convexidad).",
                  head() + f'<line class="p-guide bad" x1="{eye[0]:.1f}" y1="{eye[1]:.1f}" x2="228" y2="318"/>' + _phone(228, 318, -30) + _mark(False)))
    cards.append(("cabeza-agachada", "Cabeza agachada o echada atrás",
                  "El mentón se mide respecto a la vertical: con la cabeza inclinada, cambia aunque la cara sea la misma.",
                  head(transform="rotate(14 150 180)") + horizon + _phone(226, eye[1]) + _mark(False)))
    fringe = ('<path class="f-hair" d="M150,60 C176,70 184,96 178,128 C172,134 164,128 160,118 C152,104 140,96 128,94 Z"/>')
    cards.append(("flequillo", "Flequillo sobre la frente",
                  "Tapa la glabela (el punto de la frente entre las cejas), que es uno de los tres puntos del perfil facial.",
                  head(extra_front=fringe) + horizon + _phone(226, eye[1]) + _mark(False)))
    c = base["C"]
    collar = (f'<path class="p-collar" d="M40,366 L50,300 C60,262 90,252 112,254 C{c[0]:.1f},{c[1] - 6:.1f} '
              f'{c[0] + 18:.1f},{c[1] - 4:.1f} {c[0] + 30:.1f},{c[1] + 2:.1f} C{c[0] + 34:.1f},{c[1] + 40:.1f} '
              f'174,330 178,366 Z"/>')
    cards.append(("cuello", "Cuello tapado",
                  "Con cuello alto, capucha o bufanda no se ve el ángulo entre mentón y cuello (la línea mandibular).",
                  head(extra_front=collar) + horizon + _phone(226, eye[1]) + _mark(False)))
    cards.append(("fondo", "Fondo del color de la piel",
                  "Si la pared se parece al tono de piel, el sistema no distingue bien el borde de la cara. Mejor una pared lisa y de otro color.",
                  '<rect class="p-wall" x="0" y="0" width="240" height="360"/>' + head() + _mark(False)))
    # Vista desde arriba: perfil completo frente a 3/4.
    top = ('<g class="p-top">'
           '<g transform="translate(70,150)"><ellipse rx="34" ry="40"/><path d="M0,-40 L6,-54 L12,-38"/></g>'
           '<g transform="translate(70,300)">' + _phone(0, 0, 0)[len('<g class="p-phone" transform="translate(0,0) rotate(0)">'):-4].join(["", ""]) + '</g>'
           '</g>')
    top = ('<g class="p-top">'
           '<text class="p-cap" x="60" y="44" text-anchor="middle">Perfil completo</text>'
           '<ellipse cx="60" cy="140" rx="34" ry="40"/><path class="p-nose" d="M92,128 L112,140 L92,152"/>'
           '<line class="p-guide" x1="60" y1="190" x2="60" y2="276"/>' + _phone(60, 292) +
           '<text class="p-cap" x="180" y="44" text-anchor="middle">3/4 (no vale)</text>'
           '<ellipse cx="180" cy="140" rx="34" ry="40"/><path class="p-nose" d="M198,166 L214,186 L190,182"/>'
           '<line class="p-guide bad" x1="180" y1="190" x2="180" y2="276"/>' + _phone(180, 292) +
           '<text class="p-cap small" x="120" y="340" text-anchor="middle">visto desde arriba</text></g>')
    cards.append(("tres-cuartos", "Perfil completo, no de 3/4",
                  "La nariz tiene que apuntar hacia un lado, no hacia la cámara. Una foto de 3/4 no enseña el contorno del perfil.",
                  top))
    return [dict(clave=k, titulo=t, texto=x, svg=f'<svg viewBox="0 0 {W} {H}" class="photo-svg" role="img">{body}</svg>')
            for k, t, x, body in cards]


def main():
    out = {"categorias": {}, "marco_perfil": None}
    for group, items in CATEGORIES.items():
        out["categorias"][group] = []
        for key, label, params, kind, rule in items:
            p = build(**params)
            markup, value = svg(p, kind)
            target = params.get("conv", params.get("chin", params.get("cma")))
            assert abs(value - target) < 0.5, (key, value, target)
            out["categorias"][group].append(dict(clave=key, nombre=label, regla=rule, valor=round(value, 1), svg=markup))
            print(f"{group:10s} {key:10s} objetivo {target:6.1f}  medido {value:6.1f}")
    # Silueta del marco de la cámara: el perfil normal, solo el contorno.
    out["fotos"] = photo_cards()
    marco = build(conv=12)
    out["marco_perfil"] = {"viewBox": f"0 0 {W} {H}", "d": _catmull_rom(outline(marco), True),
                           "ojo": [marco["N"][0] - 12, marco["N"][1] + 10]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
