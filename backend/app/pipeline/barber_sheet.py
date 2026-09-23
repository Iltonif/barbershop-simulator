"""
"Ficha para tu barbero": el corte recomendado, escrito como se pide en la
silla (del anuncio de ILTONIF: "Y cómo pedirlo").

Convierte lo que ya sabe la app -- el corte del catálogo (largos en mm y
degradado), el mapa de crecimiento del cliente y su tipo de pelo -- en las
seis líneas que un peluquero necesita: laterales, parte superior, nuca,
peinado, producto y cada cuánto volver.

No hay nada "inteligente" aquí: es una traducción de datos que ya están en
la ficha a frases cortas, para no hacerle leer al cliente un párrafo. Si un
dato no está (p.ej. no hay mapa de crecimiento), esa línea simplemente no
sale, en vez de inventarla.
"""

from __future__ import annotations

from app.pipeline import maintenance
from app.pipeline.style_catalog import HaircutStyle

_FADE = {
    "skin": "Fade a piel",
    "alto": "Fade alto",
    "medio": "Fade medio",
    "bajo": "Fade bajo",
    "ninguno": "A tijera",
}
_LADO = {"izquierda": "a su izquierda", "derecha": "a su derecha"}


def _cm(mm: int) -> str:
    if mm < 10:
        return f"{mm} mm"
    return f"{mm / 10:.0f}-{mm / 10 + 1:.0f} cm" if mm >= 30 else f"{mm / 10:.1f} cm".replace(".", ",")


def _sides(style: HaircutStyle) -> str:
    fade = _FADE.get(style.fade_type, "A tijera")
    if style.fade_type in ("ninguno", None):
        return f"{fade} · {_cm(style.length_sides_mm)}"
    desde = "0" if style.fade_type == "skin" else "0,5"
    return f"{fade} · {desde} → {_cm(style.length_sides_mm)}"


def _top(style: HaircutStyle, texture: str | None) -> str:
    acabado = {"rizado": "respetando el rizo", "afro": "con peine, forma redonda",
               "ondulado": "desfilado, que caiga la onda"}.get(texture or "", "textura con tijera")
    if style.length_top_mm < 15:
        acabado = "a máquina, sin textura"
    return f"{_cm(style.length_top_mm)} · {acabado}"


def _nape(style: HaircutStyle) -> str:
    texto = f"{style.name} {style.description}".lower()
    if "recta" in texto or "cuadrada" in texto:
        return "Línea recta"
    if style.fade_type in ("alto", "medio", "bajo", "skin"):
        return "Cónica, degradada"
    if style.length_back_mm >= 60:
        return "Natural, sin marcar línea"
    return "Cónica natural"


def _styling(style: HaircutStyle, growth) -> str:
    if growth is not None:
        if growth.front_growth == "delante":
            return "Hacia delante, con volumen"
        if growth.front_growth == "atras":
            return "Hacia atrás, a favor del crecimiento"
        if growth.natural_part in _LADO:
            return f"Raya {_LADO[growth.natural_part]}"
    if style.length_top_mm < 20:
        return "Sin peinar, tal cual cae"
    return "Con volumen arriba"


def _product(style: HaircutStyle, texture: str | None) -> str | None:
    if style.length_top_mm < 15:
        return None                       # a esa longitud no hace falta
    if texture in ("rizado", "afro"):
        return "Crema de rizos, fijación ligera"
    if style.length_top_mm >= 120:
        return "Aceite ligero en puntas"
    if texture == "ondulado":
        return "Cera mate, fijación media"
    return "Pasta mate, fijación media"


def build_sheet(style: HaircutStyle, texture: str | None = None, growth=None) -> list[dict]:
    """Devuelve las líneas de la ficha: [{"campo", "valor"}]."""
    semanas = maintenance.weeks_for_style(style)
    filas = [
        ("Laterales", _sides(style)),
        ("Parte superior", _top(style, texture)),
        ("Nuca", _nape(style)),
        ("Peinado", _styling(style, growth)),
        ("Producto", _product(style, texture)),
        ("Mantenimiento", f"Cada {semanas[0]} semanas" if semanas[0] == semanas[1]
         else f"Cada {semanas[0]}-{semanas[1]} semanas"),
    ]
    return [{"campo": c, "valor": v} for c, v in filas if v]
