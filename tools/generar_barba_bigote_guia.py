"""
Genera las 18 ilustraciones de la guía de barba y bigote
(`frontend/guia-barba-bigote.html`): 13 tipos de bigote + 5 estilos base de
barba, siguiendo el mismo patrón que `generar_perfiles_guia.py` -- un
script en vez de SVG pegado a mano -- que escribe
`frontend/assets/guia/barba-bigote.json`.

Diferencia importante con `generar_perfiles_guia.py`: aquel genera perfiles
a partir de ÁNGULOS con una definición publicada (convexidad de Legan,
ángulo mentón-cuello, ángulo cervicomental) y verifica al final que cada
ilustración MIDE exactamente lo que dice. Un tipo de bigote o un estilo de
barba no es una medida angular -- es una silueta reconocible (el bigote
inglés se reconoce por ser fino y recto hasta las comisuras, el fu manchu
por los mechones que caen de las comisuras hasta pasar la mandíbula, la
barba de candado por trazar solo el borde de la mandíbula sin bigote...),
así que aquí no hay ningún número que comprobar al final: la única
"verificación" posible es visual (se generó `guia-barba-bigote.html` y se
revisó con una captura de pantalla completa antes de dar el resultado por
bueno).

Estilo visual: el mismo lenguaje que los iconos de `cuestionario.html`
(trazo simple, un color, viewBox 100x100) en vez de las siluetas
anatómicas detalladas de `generar_perfiles_guia.py` -- esto es una guía de
referencia rápida de "qué es cada estilo", no una herramienta de medición
de fotos reales.

Fuentes de los 18 estilos y de qué forma de rostro le sienta a cada uno:
ver el docstring de `app/pipeline/beard_mustache_rules.py` (los mismos
vídeos/texto que dio Pedro, peluquero dueño del proyecto, en sept 2026).

Uso (desde la raíz del repo):
    python3 tools/generar_barba_bigote_guia.py
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "assets" / "guia" / "barba-bigote.json"


def _catmull_rom(points, closed=True):
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


def _mirror(d, cls):
    """Un lado (x >= 50) dibujado en `d`, más su reflejo especular respecto
    al eje central de la cara (x=50) para el lado izquierdo -- así los dos
    lados quedan perfectamente simétricos sin repetir la geometría a mano."""
    return (f'<path class="{cls}" d="{d}"/>'
            f'<g transform="translate(100,0) scale(-1,1)"><path class="{cls}" d="{d}"/></g>')


def _sliver(p0, ctrl, p1, w0, w1):
    """Mechon ahusado de `p0` (grosor `w0`) a `p1` (grosor `w1`), curvado a
    traves de `ctrl`. El borde de arriba y el de abajo pasan por el MISMO
    punto de control (desplazado por la normal en cada extremo), asi los
    dos bordes quedan casi paralelos en vez de abrir una bolsa grande entre
    ellos -- ver la nota de `_horn` sobre por que encadena varios de estos
    en vez de una sola curva con una voluta anadida al final."""
    def normal(a, b, w):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1
        return (-dy / length * w, dx / length * w)

    n0 = normal(p0, ctrl, w0)
    n1 = normal(ctrl, p1, w1)
    top0, bot0 = (p0[0] + n0[0], p0[1] + n0[1]), (p0[0] - n0[0], p0[1] - n0[1])
    top1, bot1 = (p1[0] + n1[0], p1[1] + n1[1]), (p1[0] - n1[0], p1[1] - n1[1])
    return (f"M{top0[0]:.1f},{top0[1]:.1f} Q{ctrl[0]:.1f},{ctrl[1]:.1f} {top1[0]:.1f},{top1[1]:.1f} "
            f"L{bot1[0]:.1f},{bot1[1]:.1f} Q{ctrl[0]:.1f},{ctrl[1]:.1f} {bot0[0]:.1f},{bot0[1]:.1f} Z")


def _horn(base, ctrl, tip, thick, hook=None, hook_ctrl=None, tip_thick=None):
    """Mechon ahusado desde `base` (grosor `thick`) hasta `tip` (punta, muy
    fina), pasando por `ctrl`. `hook`/`hook_ctrl`, si se dan, encadenan un
    SEGUNDO mechon mas fino desde `tip` hasta `hook` (la voluta de las
    puntas enroscadas, p.ej. Dali o Imperial).

    Nota: la primera version de esta funcion probaba a anadir la voluta
    como un tramo mas dentro de UNA sola curva cerrada (borde de ida hasta
    `hook`, borde de vuelta directo a `base`); el resultado eran manchas
    grandes en vez de una punta fina, porque los dos bordes dejaban de ser
    paralelos en cuanto la voluta cambiaba de direccion. Encadenar dos
    mechones independientes, cada uno con su propio `_sliver` (bordes
    paralelos de extremo a extremo), evita el problema."""
    end_thick = tip_thick if tip_thick is not None else thick * (0.35 if hook else 0.15)
    d = _sliver(base, ctrl, tip, thick, end_thick)
    if hook:
        d += " " + _sliver(tip, hook_ctrl or tip, hook, end_thick, end_thick * 0.25)
    return d


# ---------------------------------------------------------------------
# Cara base: mismo óvalo y rasgos mínimos en todas las tarjetas, para que
# el bigote/la barba de cada una se lea por contraste con la misma cara.
# ---------------------------------------------------------------------

def face(extra_before_features="", extra_after_features=""):
    features = (
        '<path class="bb-line" d="M40 44 h4 M56 44 h4"/>'
        '<path class="bb-line" d="M36 34 q4 -4 8 -1 M56 33 q4 -3 8 1"/>'
        '<path class="bb-line" d="M50 46 L48 59 Q50 61 52 59"/>'
        '<path class="bb-line bb-mouth" d="M43 66 q7 4 14 0"/>'
    )
    return (
        f'<svg viewBox="0 0 100 100" class="bb-art" role="img">'
        f'<ellipse class="bb-face" cx="50" cy="52" rx="27" ry="36"/>'
        f'{extra_before_features}{features}{extra_after_features}</svg>'
    )


# ---------------------------------------------------------------------
# Bigotes (13 tipos). Todos se dibujan ANTES de los rasgos (`face(mustache
# ,"")`) para quedar por debajo del trazo de la nariz/boca, salvo cuando el
# estilo baja por debajo de la boca (fu manchu, herradura): esos añaden esa
# parte también antes, ya que no tapan ojos/cejas/nariz.
# ---------------------------------------------------------------------

def mustache_chevron():
    d = _catmull_rom([(34, 60), (38, 56), (50, 55), (62, 56), (66, 60), (66, 64), (50, 65), (34, 64)])
    return f'<path class="bb-fill" d="{d}"/>'


def mustache_dali():
    horn = _horn((58, 60), (70, 48), (80, 34), 2.0, tip_thick=1.0, hook_ctrl=(84, 26), hook=(74, 18))
    return _mirror(horn, "bb-fill")


def mustache_ingles():
    horn = _horn((58, 61), (69, 60), (81, 60), 1.3)
    return _mirror(horn, "bb-fill")


def mustache_hungaro():
    horn = _horn((58, 61), (72, 54), (85, 48), 4.2, tip_thick=2.2)
    return _mirror(horn, "bb-fill")


def mustache_fu_manchu():
    top = _horn((57, 61), (61, 63), (65, 61), 1.6)
    drop = _horn((60, 63), (65, 80), (63, 97), 1.1)
    return _mirror(top, "bb-fill") + _mirror(drop, "bb-fill")


def mustache_horizontal_lapiz():
    horn = _horn((54, 61), (58, 61), (62, 61), 0.8)
    return _mirror(horn, "bb-fill")


def mustache_herradura():
    top = _horn((58, 60), (64, 59), (68, 59), 2.0)
    drop = _horn((66, 60), (71, 71), (69, 83), 1.8)
    return _mirror(top, "bb-fill") + _mirror(drop, "bb-fill")


def mustache_imperial():
    block = _catmull_rom([(38, 62), (42, 57), (50, 56), (58, 57), (62, 62), (58, 65), (50, 66), (42, 65)])
    horn = _horn((59, 59), (70, 50), (80, 38), 2.2, tip_thick=1.1, hook_ctrl=(84, 30), hook=(76, 22))
    return f'<path class="bb-fill" d="{block}"/>' + _mirror(horn, "bb-fill")


def mustache_piramidal():
    d = _catmull_rom([(35, 65), (42, 58), (50, 54), (58, 58), (65, 65), (58, 67), (50, 68), (42, 67)])
    return f'<path class="bb-fill" d="{d}"/>'


def mustache_morsa():
    block = _catmull_rom([(36, 61), (42, 55), (50, 54), (58, 55), (64, 61), (64, 68), (50, 71), (36, 68)])
    horn = _horn((62, 63), (73, 70), (77, 82), 3.6)
    return f'<path class="bb-fill" d="{block}"/>' + _mirror(horn, "bb-fill")


def mustache_mosquetero():
    top = mustache_corto()
    patch = '<circle class="bb-fill" cx="50" cy="75" r="3.4"/>'
    return top + patch


def mustache_revolucionario():
    block = _catmull_rom([(34, 61), (40, 54), (50, 52), (60, 54), (66, 61), (66, 67), (50, 70), (34, 67)])
    horn = _horn((64, 64), (80, 71), (88, 84), 4.6)
    return f'<path class="bb-fill" d="{block}"/>' + _mirror(horn, "bb-fill")


def mustache_corto():
    d = _catmull_rom([(42, 62), (46, 58), (50, 57), (54, 58), (58, 62), (54, 65), (50, 66), (46, 65)])
    return f'<path class="bb-fill" d="{d}"/>'


MUSTACHES = [
    ("chevron", "Chevron", mustache_chevron,
     "Grueso y ancho, cubre todo el labio superior de comisura a comisura sin dejar hueco en el centro."),
    ("dali", "Dalí", mustache_dali,
     "Muy fino, con las puntas larguísimas y enroscadas casi en vertical."),
    ("ingles", "Inglés", mustache_ingles,
     "Fino y recto, sin rizar, con las puntas llegando justo hasta las comisuras."),
    ("hungaro", "Húngaro", mustache_hungaro,
     "Poblado y ancho, con las puntas abiertas hacia los lados sin llegar a rizarse."),
    ("fu_manchu", "Fu Manchu", mustache_fu_manchu,
     "Fino en el centro, con dos mechones que caen desde las comisuras hasta pasar la barbilla."),
    ("horizontal_lapiz", "Horizontal / lápiz", mustache_horizontal_lapiz,
     "Una línea finísima pegada al labio, corta y muy recortada."),
    ("herradura", "Herradura", mustache_herradura,
     "El bigote baja por ambos lados de la boca hasta la mandíbula, como una herradura invertida."),
    ("imperial", "Imperial", mustache_imperial,
     "Bigote con volumen y puntas largas que se enroscan hacia arriba, muy marcado."),
    ("piramidal", "Piramidal", mustache_piramidal,
     "Forma triangular: más estrecho junto a la nariz y más ancho sobre el labio."),
    ("morsa", "Morsa (walrus)", mustache_morsa,
     "Muy poblado y caído, tapa el labio superior por completo y cae sobre las comisuras."),
    ("mosquetero", "Mosquetero", mustache_mosquetero,
     "Bigote corto y cuidado, más una perilla pequeña justo bajo el labio inferior, sin conectar con la barbilla."),
    ("revolucionario", "Revolucionario", mustache_revolucionario,
     "Muy ancho, poblado y caído hacia la mandíbula, más grande y voluminoso que la morsa."),
    ("corto", "Corto", mustache_corto,
     "Pequeño, discreto y muy bien perfilado, cubre solo el ancho del labio."),
]


# ---------------------------------------------------------------------
# Barbas (5 estilos base). Se dibujan ANTES que los rasgos de la cara para
# que ojos/cejas/nariz/boca queden por encima del relleno.
# ---------------------------------------------------------------------

_JAW_OUTER = [(21, 50), (24, 64), (30, 78), (40, 88), (50, 91), (60, 88), (70, 78), (76, 64), (79, 50)]
_JAW_INNER = [(26, 45), (28, 58), (34, 70), (42, 80), (50, 83), (58, 80), (66, 70), (72, 58), (74, 45)]
# y=48-54: por debajo de los ojos (y=44) y las cejas (y=33-34), para que la
# barba completa/media sombra no tape ningún rasgo de la cara.
_CHEEK_TOP = [(24, 54), (35, 50), (50, 48), (65, 50), (76, 54)]


def beard_en_collar():
    ring = _JAW_OUTER + list(reversed(_JAW_INNER))
    d = _catmull_rom(ring)
    return f'<path class="bb-fill bb-fill-soft" d="{d}"/>'


def beard_completa():
    outline = _CHEEK_TOP + _JAW_OUTER[1:-1][::-1]
    d = _catmull_rom(outline)
    return f'<path class="bb-fill" d="{d}"/>'


def beard_perilla():
    chin = _catmull_rom([(40, 72), (38, 84), (44, 90), (50, 92), (56, 90), (62, 84), (60, 72)])
    return f'<path class="bb-fill" d="{chin}"/>' + mustache_corto()


def beard_chiva():
    chin = _catmull_rom([(44, 76), (43, 86), (47, 92), (50, 94), (53, 92), (57, 86), (56, 76)])
    return f'<path class="bb-fill" d="{chin}"/>'


def beard_media_sombra():
    outline = _CHEEK_TOP + _JAW_OUTER[1:-1][::-1]
    d = _catmull_rom(outline)
    return f'<path class="bb-fill bb-fill-stubble" d="{d}"/>'


BEARDS = [
    ("en_collar", "En collar", beard_en_collar,
     "Traza solo el borde de la mandíbula, de patilla a patilla pasando por la barbilla, sin cubrir mejillas ni bigote."),
    ("completa_clasica", "Completa / clásica", beard_completa,
     "Cubre mejillas, mandíbula y barbilla por completo, normalmente conectada con el bigote."),
    ("perilla", "Perilla", beard_perilla,
     "Barbilla cubierta y bigote, sin llegar a las mejillas ni a los laterales de la mandíbula."),
    ("chiva_chivita", "Chiva / chivita", beard_chiva,
     "Solo la barbilla, más estrecha y apurada que la perilla, sin bigote."),
    ("varios_dias_media_sombra", "De varios días / media sombra", beard_media_sombra,
     "Sombra uniforme y corta por toda la cara, sin perfilar ninguna línea."),
]


# ---------------------------------------------------------------------
# Metodología de proporciones (referencia educativa, PDF/vídeo de Pedro:
# "VISAGISMO MASCULINO, tipos de rostros"). Es una técnica de dibujo y
# clasificación manual -- se documenta tal cual, sin convertirla en una
# medición automática sobre fotos (mismo criterio que el resto de rasgos
# de visagismo, ver `app/pipeline/face_analysis.py`).
# ---------------------------------------------------------------------

def proportions_diagram():
    verticals = [18, 34, 50, 66, 82]
    horizontals = [14, 33, 52, 71, 90]
    lines = "".join(f'<line class="bb-guide-v" x1="{x}" y1="8" x2="{x}" y2="96"/>' for x in verticals)
    lines += "".join(f'<line class="bb-guide-h" x1="4" y1="{y}" x2="96" y2="{y}"/>' for y in horizontals)
    return (
        '<svg viewBox="0 0 100 100" class="bb-art bb-art-guide" role="img">'
        f'<ellipse class="bb-face" cx="50" cy="52" rx="27" ry="36"/>{lines}'
        '<path class="bb-line" d="M40 44 h4 M56 44 h4"/>'
        '<path class="bb-line" d="M50 46 L48 59 Q50 61 52 59"/>'
        '<path class="bb-line bb-mouth" d="M43 66 q7 4 14 0"/></svg>'
    )


def main():
    out = {"bigotes": [], "barbas": [], "proporciones": None}
    for key, label, builder, detail in MUSTACHES:
        out["bigotes"].append(dict(clave=key, nombre=label, detalle=detail, svg=face(builder())))
        print("bigote  ", key)
    for key, label, builder, detail in BEARDS:
        out["barbas"].append(dict(clave=key, nombre=label, detalle=detail, svg=face(builder())))
        print("barba   ", key)
    out["proporciones"] = dict(
        svg=proportions_diagram(),
        texto=(
            "Técnica clásica de dibujo para clasificar la forma del rostro (de la guía de Pedro): 5 líneas "
            "verticales (central, una delante de cada oreja y una detrás de cada una) dividen la cara en "
            "\"dos unidades y media\"; 5 líneas horizontales (barbilla, base de la nariz, contorno del "
            "nacimiento del pelo y arranque del cráneo) la dividen en \"tres unidades y media\". Un rostro "
            "equilibrado en ambas proporciones es ovalado; si las horizontales predominan sobre las "
            "verticales y la mandíbula/barbilla no se marcan, es redondo. Es una técnica de dibujo y "
            "clasificación manual, no una medición automática -- en esta app la forma de rostro se sigue "
            "marcando a mano (ficha o cuestionario), nunca detectada sola de una foto."
        ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
