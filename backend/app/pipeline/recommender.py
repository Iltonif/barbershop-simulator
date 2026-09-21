"""
Recomendador de cortes de pelo a partir del perfil del cliente.

Cruza el tipo de cabello (fijado a mano por el barbero, ver
`ClientProfile.hair_texture_override`) con el catálogo de cortes
(`style_catalog.load_catalog`), y tiene en cuenta varias señales
adicionales del perfil, todas opcionales:

- El mapa de crecimiento/remolinos dibujado en `frontend/growth-map.html`:
  avisa cuando un corte muy corto y uniforme (buzz) puede no ser buena idea
  si hay remolinos marcados, porque con tan poco largo arriba se notan
  mucho más y son más difíciles de disimular.
- La forma de cara (`ClientProfile.face_shape_override`, fijada a mano por
  el barbero — ver la nota en `clients_routes.get_recommendations` sobre
  por qué no se usa la detección automática sin confirmar): avisa cuando un
  corte no encaja bien con las recomendaciones habituales de barbería para
  esa forma de cara.
- El perfil de visagismo (`ClientProfile.visagismo_profile`, ver
  `app/pipeline/visagismo_rules.py`): morfología craneal/facial, forma de
  nacimiento del pelo y estilo de vida (mantenimiento diario, frecuencia
  de visitas). Igual que las dos señales anteriores, es un matiz sobre el
  orden, no un filtro — ver el docstring de `visagismo_rules.py` para el
  porqué y para el mapeo concreto de cada regla sobre este catálogo.

Ninguna señal descarta cortes: todas los avisan/priorizan y los mueven
arriba o abajo en la lista, con una nota explicando el motivo, para que
decida el barbero — son heurísticas orientativas, no reglas estrictas.

Las reglas de forma de cara están basadas en varias fuentes de peluquería/
barbería consultadas en septiembre de 2026 (Llongueras, fabricbarberia.com,
manofmany.com, entre otras — ver la respuesta al usuario donde se citan).
Solo se aplican reglas de "evitar" cuando varias fuentes independientes
coinciden; por eso "ovalada" y "cuadrada" no tienen ninguna restricción
aquí (las fuentes las describen como versátiles y a veces se contradicen
entre sí en los detalles), mientras que "redonda" y "alargada" sí, porque
ahí todas las fuentes consultadas coinciden en la misma dirección.
"""

from dataclasses import dataclass, field

from app.pipeline import trait_rules, visagismo_rules
from app.pipeline.rule_effects import AVISO_SUAVE, BOOST_SUAVE, Effect
from app.pipeline.style_catalog import HaircutStyle, load_catalog

# Un remolino es más difícil de disimular cuanto más corto es el largo
# arriba: por eso el umbral mira solo `length_top_mm` y no el fade_type de
# los laterales — en este catálogo un fade "alto"/"skin" en los laterales
# convive con largo normal arriba (p.ej. undercuts), así que no es un buen
# indicador por sí solo. Lo realmente problemático es un corte tipo buzz,
# muy corto y uniforme en toda la cabeza.
_UMBRAL_LARGO_PROBLEMATICO_MM = 10

# Familias del catálogo (ver style_catalog.py / el script que asignó las
# fotos) que refuerzan visualmente una cara redonda: el tazón por su
# silueta redondeada, y el afro/rizado voluminoso por su forma esférica.
_FAMILIAS_REDONDEADAS = {
    "corte_tazon_bowl_liso",
    "corte_tazon_bowl_rizado",
    "afro_rizado_voluminoso",
}
_FAMILIA_TUPE_POMPADOUR = "tupe_pompadour_clasico"


@dataclass
class StyleRecommendation:
    style: HaircutStyle
    note: str | None  # todos los avisos juntos (compatibilidad con la API anterior)
    # Por qué encaja (razones a favor) y qué no encaja (avisos), cada uno
    # con etiqueta corta + explicación. Ver rule_effects.py.
    reasons: list[Effect] = field(default_factory=list)
    warnings: list[Effect] = field(default_factory=list)


def _es_potencialmente_problematico_con_remolinos(style: HaircutStyle) -> bool:
    return style.length_top_mm <= _UMBRAL_LARGO_PROBLEMATICO_MM


def _efecto_remolinos(style: HaircutStyle, whorls: list[dict]) -> Effect | None:
    nota = _nota_remolinos(style, whorls)
    if nota:
        return Effect(AVISO_SUAVE, "Deja ver los remolinos", nota)
    # A favor: con remolinos, algo de largo arriba ayuda a dominarlos.
    if whorls and style.length_top_mm >= 40:
        return Effect(BOOST_SUAVE, "Disimula los remolinos",
                      "Con algo de largo arriba el peso del pelo ayuda a dominar los remolinos marcados.")
    return None


def _efecto_forma_cara(style: HaircutStyle, face_shape: str | None) -> Effect | None:
    nota = _nota_forma_cara(style, face_shape)
    if nota:
        label = "Redondea más la cara" if face_shape == "redonda" else "Alarga más la cara"
        return Effect(AVISO_SUAVE, label, nota)
    # A favor, con las mismas fuentes: altura arriba para la cara redonda,
    # anchura/flequillo para la alargada.
    # "Altura" = largo arriba claramente mayor que en los laterales; una
    # melena larga por igual no la da.
    if (face_shape == "redonda" and 50 <= style.length_top_mm <= 150
            and style.length_sides_mm <= style.length_top_mm / 2
            and style.style_family not in _FAMILIAS_REDONDEADAS):
        return Effect(BOOST_SUAVE, "Alarga la cara",
                      "Con cara redonda, dar altura arriba alarga visualmente la cara.")
    if face_shape == "alargada" and (trait_rules._tiene_flequillo(style)
                                     or style.style_family == "clasico_raya_lateral"):
        return Effect(BOOST_SUAVE, "Acorta la cara",
                      "Con cara alargada, un flequillo o una raya lateral suman anchura y acortan la cara.")
    return None


def _nota_remolinos(style: HaircutStyle, whorls: list[dict]) -> str | None:
    if not whorls or not _es_potencialmente_problematico_con_remolinos(style):
        return None
    if len(whorls) >= 2:
        return (
            "Este cliente tiene varios remolinos marcados: un corte tan corto y "
            "uniforme puede dejarlos muy a la vista. Puede valer la pena dejar "
            "algo más de largo arriba para disimularlos."
        )
    return (
        "Hay un remolino marcado en el mapa de crecimiento: con tan poco largo "
        "arriba puede notarse. Valorar dejar algo más de longitud para "
        "disimularlo."
    )


def _nota_forma_cara(style: HaircutStyle, face_shape: str | None) -> str | None:
    """Aviso (no descarte) cuando el corte va a contracorriente de lo que
    recomiendan varias fuentes de barbería para esta forma de cara. Solo
    cubre "redonda" y "alargada" — ver el docstring del módulo."""
    if not face_shape:
        return None

    is_bowl_or_voluminous = style.style_family in _FAMILIAS_REDONDEADAS
    is_buzz_muy_corto = style.fade_type == "ninguno" and style.length_top_mm <= 10

    if face_shape == "redonda" and (is_bowl_or_voluminous or is_buzz_muy_corto):
        return (
            "Con cara redonda, varias guías de barbería recomiendan dar volumen "
            "arriba en vez de una silueta redondeada o muy uniforme — puede "
            "acentuar la redondez de la cara."
        )

    is_tupe_alto = style.style_family == _FAMILIA_TUPE_POMPADOUR
    is_fade_muy_alto_con_top_largo = style.fade_type in ("alto", "skin") and style.length_top_mm >= 90

    if face_shape == "alargada" and (is_tupe_alto or is_fade_muy_alto_con_top_largo):
        return (
            "Con cara alargada, sumar más altura arriba (tipo tupé/pompadour o "
            "un fade muy marcado con mucho largo arriba) tiende a alargarla "
            "más — suele funcionar mejor un corte que sume anchura, como uno "
            "con flequillo o raya lateral."
        )

    return None


def recommend_styles(
    hair_texture: str,
    whorls: list[dict] | None = None,
    face_shape: str | None = None,
    visagismo_profile: dict | None = None,
) -> list[StyleRecommendation]:
    """Devuelve los cortes del catálogo compatibles con `hair_texture`,
    ordenados de más a menos recomendados. `whorls` es la lista tal cual se
    guarda en `ClientProfile.custom_growth_map["whorls"]`, `face_shape` es
    `ClientProfile.face_shape_override`, y `visagismo_profile` es
    `ClientProfile.visagismo_profile` (ver `visagismo_rules.py`) — los tres
    opcionales (`None`/vacío si el barbero no los ha rellenado; en ese caso
    simplemente no se aplica esa señal, no es un error)."""

    whorls = whorls or []

    compatibles = [
        style for style in load_catalog() if hair_texture in style.suitable_hair_types
    ]

    recomendaciones = []
    for style in compatibles:
        effects = [e for e in (_efecto_remolinos(style, whorls), _efecto_forma_cara(style, face_shape)) if e]
        effects += trait_rules.evaluate_traits(style, visagismo_profile)
        effects += visagismo_rules.evaluate_profile(style, visagismo_profile).effects

        reasons = [e for e in effects if e.is_reason]
        warnings = [e for e in effects if not e.is_reason]
        note = " ".join(e.detail for e in warnings) or None
        score = sum(e.score for e in effects)
        recomendaciones.append((score, StyleRecommendation(style=style, note=note, reasons=reasons,
                                                          warnings=warnings)))

    recomendaciones.sort(key=lambda par: par[0])
    return [rec for _score, rec in recomendaciones]
