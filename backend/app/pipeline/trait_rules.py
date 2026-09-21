"""
Reglas de recomendación a partir de los rasgos faciales de la ficha
(`visagismo_profile.anatomical_metrics.facial_features_profile`): orejas,
mentón, línea mandibular, perfil y cuello. Se añadieron en sept 2026 porque
esos rasgos se guardaban (a mano o con el análisis de fotos de
`visagismo.html`) pero ninguna regla los leía.

Igual que el resto de reglas (ver `visagismo_rules.py`), nada descarta un
corte: cada regla sube (razón a favor) o baja (aviso) cortes con una
explicación. Las reglas salen de estas fuentes, las mismas de la guía de
`frontend/guia-visagismo.html` más una específica de orejas:

- Luc Vincent, "Morphological problems and their corrections"
  (lucvincent.com): cuello corto -> nuca despejada, nada de pelo largo;
  cuello largo -> media melena que lo abrigue; mentón retraído -> evitar
  pelo hacia atrás y coletas ("echan la cara hacia atrás"), algo de
  volumen en la nuca; mentón prominente -> flequillo; nariz grande ->
  volumen arriba y en los laterales; perfil cóncavo -> flequillo suave,
  evitar peinados planos; perfil convexo -> volumen detrás.
- Book of Barbering, "Men's hairstyles for big ears": evitar fades altos
  y a piel ("dejan la oreja a la vista en su punto más ancho"), no bajar
  de un #2 (~6 mm) en los laterales, y cortes medios con textura arriba.
- Castlebeard (mentón retraído) y Beard Resource (mandíbula poco
  definida): barba, ver `beard_advice`. Beard Resource además: laterales
  cortos para alargar la cara.

Cómo se leen los cortes del catálogo (solo hay largos en mm, degradado,
familia y texto): ver las funciones `_es_*` de abajo. Son aproximaciones:
p.ej. "peinado hacia atrás" se detecta por el nombre/descripción.
"""

from __future__ import annotations

import re

from app.pipeline.rule_effects import AVISO_FUERTE, AVISO_SUAVE, BOOST_SUAVE, Effect
from app.pipeline.style_catalog import HaircutStyle

_HACIA_ATRAS = re.compile(r"hacia atr[aá]s|hacia detr[aá]s|engominad|gomina|slick", re.I)
_FLEQUILLO = re.compile(r"flequillo|fringe|french crop|tazón|bowl", re.I)
_VOLUMEN = re.compile(r"volumen|tup[eé]|pompadour|quiff", re.I)
_FAMILIAS_VOLUMEN = {"tupe_pompadour_clasico", "fade_undercut_textura", "corte_texturizado_general",
                     "afro_rizado_voluminoso"}
_FAMILIAS_FLEQUILLO = {"corte_tazon_bowl_liso", "corte_tazon_bowl_rizado"}


def _texto(style: HaircutStyle) -> str:
    return f"{style.name} {style.description or ''}"


def _es_hacia_atras(style: HaircutStyle) -> bool:
    return bool(_HACIA_ATRAS.search(_texto(style))) or style.style_family == "recogido_mono_coleta"


def _tiene_flequillo(style: HaircutStyle) -> bool:
    return style.style_family in _FAMILIAS_FLEQUILLO or bool(_FLEQUILLO.search(_texto(style)))


def _tiene_volumen_arriba(style: HaircutStyle) -> bool:
    if style.length_top_mm < 40 or _es_hacia_atras(style) and not _VOLUMEN.search(_texto(style)):
        return False
    return style.style_family in _FAMILIAS_VOLUMEN or bool(_VOLUMEN.search(_texto(style)))


def _deja_orejas_a_la_vista(style: HaircutStyle) -> bool:
    # Fade alto/a piel, o laterales por debajo de un #2 (~6 mm).
    return style.fade_type in ("alto", "skin") or style.length_sides_mm < 6


def _equilibra_orejas(style: HaircutStyle) -> bool:
    # Laterales con algo de largo, o degradado bajo con volumen arriba
    # (más peso visual encima y delante de la oreja).
    if _deja_orejas_a_la_vista(style):
        return False
    return style.length_sides_mm >= 15 or (style.fade_type in ("bajo", "medio") and style.length_top_mm >= 50)


def _features(profile: dict | None) -> dict:
    if not isinstance(profile, dict):
        return {}
    anat = profile.get("anatomical_metrics") or {}
    return anat.get("facial_features_profile") or {}


def _orejas(style: HaircutStyle, f: dict) -> list[Effect]:
    if f.get("ears_projection") != "prominent_protruding":
        return []
    if _deja_orejas_a_la_vista(style):
        return [Effect(AVISO_FUERTE, "Deja las orejas a la vista",
                       "Con orejas prominentes, un fade alto o a piel (o laterales muy rapados) deja la "
                       "oreja expuesta en su punto más ancho y se convierte en lo primero que se ve.")]
    if _equilibra_orejas(style):
        return [Effect(BOOST_SUAVE, "Disimula las orejas",
                       "Los laterales con algo de largo, o un degradado bajo con volumen arriba, ponen más "
                       "peso visual encima y delante de la oreja, y el conjunto se ve equilibrado.")]
    return []


def _menton_y_mandibula(style: HaircutStyle, f: dict) -> list[Effect]:
    out: list[Effect] = []
    chin = f.get("chin_projection")
    jaw = f.get("jawline_definition")
    if chin == "retruded":
        if _es_hacia_atras(style):
            out.append(Effect(AVISO_SUAVE, "Resalta el mentón retraído",
                              "El pelo peinado hacia atrás o recogido lleva la mirada hacia atrás y hace que "
                              "el mentón parezca aún más retraído."))
        elif 30 <= style.length_back_mm <= 120:
            out.append(Effect(BOOST_SUAVE, "Equilibra el mentón",
                              "Algo de largo en la nuca enmarca la parte baja de la cara y compensa un mentón "
                              "retraído."))
    elif chin == "prominent" and _tiene_flequillo(style):
        out.append(Effect(BOOST_SUAVE, "Compensa el mentón",
                          "Un flequillo lleva peso a la parte alta de la cara y suaviza un mentón prominente."))
    if jaw == "soft" and style.length_sides_mm <= 15 and style.length_top_mm >= 30:
        out.append(Effect(BOOST_SUAVE, "Afina la mandíbula",
                          "Laterales cortos con algo de largo arriba alargan la cara y hacen que la "
                          "mandíbula parezca más definida (mejor aún con barba, ver consejo de barba)."))
    return out


def _perfil(style: HaircutStyle, f: dict) -> list[Effect]:
    profile = f.get("profile_type")
    if profile == "convex_prominent_nose":
        if _es_hacia_atras(style):
            return [Effect(AVISO_SUAVE, "Resalta la nariz",
                           "Con perfil convexo o nariz prominente, el pelo pegado hacia atrás deja la nariz "
                           "como protagonista del perfil.")]
        if _tiene_volumen_arriba(style) and not _deja_orejas_a_la_vista(style):
            return [Effect(BOOST_SUAVE, "Equilibra la nariz",
                           "El volumen arriba y algo de largo en los laterales hacen que la nariz se vea "
                           "más proporcionada con el resto de la cabeza.")]
    elif profile == "concave":
        if _tiene_flequillo(style):
            return [Effect(BOOST_SUAVE, "Suaviza el perfil",
                           "Con perfil cóncavo, un flequillo suave con algo de volumen compensa el perfil.")]
        if _es_hacia_atras(style):
            return [Effect(AVISO_SUAVE, "Acentúa el perfil cóncavo",
                           "Los peinados planos o hacia atrás acentúan el perfil cóncavo.")]
    return []


def _cuello(style: HaircutStyle, f: dict) -> list[Effect]:
    neck = f.get("neck_proportions")
    if neck == "short_thick":
        if style.length_back_mm >= 60:
            return [Effect(AVISO_SUAVE, "Acorta el cuello",
                           "Con cuello corto, el pelo largo por la nuca lo tapa y lo acorta todavía más.")]
        if style.length_back_mm <= 15:
            return [Effect(BOOST_SUAVE, "Alarga el cuello",
                           "Una nuca corta que deja ver la piel del cuello lo hace parecer más largo.")]
    elif neck == "long_thin" and 30 <= style.length_back_mm <= 150:
        return [Effect(BOOST_SUAVE, "Abriga el cuello",
                       "Con cuello largo, una media melena que lo envuelve lo equilibra.")]
    return []


def evaluate_traits(style: HaircutStyle, profile: dict | None) -> list[Effect]:
    f = _features(profile)
    if not f:
        return []
    return _orejas(style, f) + _menton_y_mandibula(style, f) + _perfil(style, f) + _cuello(style, f)


def beard_advice(profile: dict | None) -> list[dict]:
    """Consejo de barba (no depende del corte). Vacío si no hay rasgos que
    lo justifiquen. Si el cliente ha dicho que va afeitado, se da igual
    pero como sugerencia."""
    f = _features(profile)
    lifestyle = (profile or {}).get("lifestyle_and_preferences") or {} if isinstance(profile, dict) else {}
    shaved = lifestyle.get("beard_preference") == "clean_shaven"
    prefix = "Si quiere probar barba: " if shaved else ""
    out = []
    if f.get("chin_projection") == "retruded":
        out.append({
            "label": "Barba para el mentón",
            "detail": prefix + "barba completa o perilla cerrada (con bigote), dejando unos 2-3 cm en la "
                      "barbilla para alargarla y darle cuerpo. Evitar la barba de pocos días muy clara, "
                      "que lo hace parecer más débil.",
        })
    if f.get("jawline_definition") == "soft":
        out.append({
            "label": "Barba para marcar la mandíbula",
            "detail": prefix + "barba con largo y líneas rectas en la mandíbula, y el perfilado del cuello "
                      "bajo (no más de un dedo por encima de la nuez) para no dejar a la vista la zona "
                      "blanda de debajo.",
        })
    return out
