"""
Recomendador de cortes de pelo a partir del perfil del cliente.

Cruza el tipo de cabello (fijado a mano por el barbero, ver
`ClientProfile.hair_texture_override`) con el catálogo de cortes
(`style_catalog.load_catalog`), y tiene en cuenta varias señales
adicionales del perfil, todas opcionales:

- El mapa de crecimiento dibujado en `frontend/growth-map.html` (flechas
  de dirección y remolinos): ver `growth_rules.py` (remolino de coronilla,
  frente y nuca, peinar a favor del crecimiento, raya natural).
- Cada cuánto hay que retocar el corte (`maintenance.py`): solo como
  desempate, primero los de retoque más frecuente. La frecuencia de visitas
  del cliente no se usa para recomendar.
- La forma de cara (`ClientProfile.face_shape_override`, fijada a mano por
  el barbero — ver la nota en `clients_routes.get_recommendations` sobre
  por qué no se usa la detección automática sin confirmar): avisa cuando un
  corte no encaja bien con las recomendaciones habituales de barbería para
  esa forma de cara.
- El perfil de visagismo (`ClientProfile.visagismo_profile`, ver
  `app/pipeline/visagismo_rules.py`): morfología craneal/facial, forma de
  nacimiento del pelo y estilo de vida (mantenimiento diario). Igual que las dos señales anteriores, es un matiz sobre el
  orden, no un filtro — ver el docstring de `visagismo_rules.py` para el
  porqué y para el mapeo concreto de cada regla sobre este catálogo.
- Cruces entre el mapa de crecimiento y la forma de cara/visagismo (p.ej.
  cara redonda + raya natural, o entradas + remolino en la frente): ver
  `combined_rules.py` para el porqué de un módulo aparte y qué combinaciones
  concretas cubre.
- La recomendación de forma de cara que trae la propia ficha de un corte
  CONCRETO (`HaircutStyle.recommended_face_shapes`, ver
  `_efecto_forma_cara_especifica`): a diferencia de la regla general de
  forma de cara de arriba (por atributos del corte), esta es la
  recomendación explícita de un corte en particular según su fuente de
  origen (sept 2026: documento aportado por Pedro con 5 cortes con nombre
  propio -- fade clásico, corte texturizado con flequillo, buzz cut,
  pompadour moderno, crop francés -- cada uno con su recomendación de
  forma de cara). Puede convivir con un aviso general de la regla de
  arriba para el mismo corte (ver la nota en el propio código sobre
  "Pompadour moderno" + cara alargada): no se resuelve a mano, el barbero
  ve las dos señales.

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

from app.pipeline import combined_rules, growth_analysis, growth_rules, maintenance, trait_rules, visagismo_rules
from app.pipeline.rule_effects import AVISO_SUAVE, BOOST_SUAVE, Effect
from app.pipeline.style_catalog import HaircutStyle, load_full_catalog

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
    maintenance_weeks: tuple[int, int] = maintenance.DEFAULT_WEEKS
    # Suma de las puntuaciones de las reglas (negativo = encaja mejor).
    score: float = 0.0


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


# Formas de cara recomendadas para 5 cortes CONCRETOS del catálogo, tal
# cual las trae su propia ficha de origen (sept 2026: documento aportado
# por Pedro con nombre + descripción + recomendación de forma de cara por
# corte -- ver `HaircutStyle.recommended_face_shapes` y la nota de
# `source` de cada uno en catalog_data/styles.json). A diferencia de
# `_efecto_forma_cara` (una regla GENERAL por atributos -- largo, fade,
# familia -- que se aplica a cualquier corte del catálogo, basada en guías
# de barbería genéricas), esta es la recomendación explícita que trae la
# ficha de ESE corte en particular: no se infiere nada, solo se mira si la
# forma de cara del cliente está en la lista.
#
# OJO, contradicción conocida y deliberada: "Pompadour moderno" trae
# "alargada" en su lista (su ficha dice que "aporta altura y ayuda a
# equilibrar las proporciones" en cara alargada), pero
# `_nota_forma_cara` avisa en contra de dar más altura arriba con cara
# alargada para toda la familia tupé/pompadour (fuentes de barbería
# genéricas). Se deja que las dos señales convivan -- un aviso general y
# una razón a favor específica de este corte -- en vez de resolverlo a
# mano: el barbero ve las dos y decide, mismo criterio que ya se sigue en
# el resto del motor de reglas (ninguna señal descarta, solo prioriza).
_FACE_SHAPE_LABELS_ADJ = {
    "ovalada": "ovalados",
    "redonda": "redondos",
    "cuadrada": "cuadrados",
    "alargada": "alargados",
    "diamante": "diamante",
    "triangular": "triangulares",
    "triangular_invertida": "triangulares invertidos",
}


def _efecto_forma_cara_especifica(style: HaircutStyle, face_shape: str | None) -> Effect | None:
    if not face_shape or not style.recommended_face_shapes:
        return None
    if face_shape not in style.recommended_face_shapes:
        return None
    etiqueta = _FACE_SHAPE_LABELS_ADJ.get(face_shape, face_shape)
    return Effect(
        BOOST_SUAVE, "Recomendado para su rostro",
        f"Este corte está recomendado especialmente para rostros {etiqueta}.",
    )


def _desempate(weeks: tuple[int, int]) -> int:
    """Entre cortes con la misma puntuación, primero los de retoque más
    frecuente (evita que el cliente alargue demasiado las visitas). Cada
    cuánto dice el cliente que viene NO cuenta para recomendar (Pedro lo
    quitó); solo se usa para calcular su próximo corte (maintenance.py)."""
    return weeks[0]


def recommend_styles(
    hair_texture: str | None,
    whorls: list[dict] | None = None,
    face_shape: str | None = None,
    visagismo_profile: dict | None = None,
    growth_map: dict | None = None,
) -> list[StyleRecommendation]:
    """Devuelve los cortes del catálogo compatibles con `hair_texture`,
    ordenados de más a menos recomendados.

    `growth_map` es `ClientProfile.custom_growth_map` (flechas + remolinos
    de growth-map.html, ver growth_analysis.py y growth_rules.py);
    `whorls` se mantiene para quien solo tenga la lista de remolinos.
    `face_shape` es `ClientProfile.face_shape_override` y
    `visagismo_profile` es `ClientProfile.visagismo_profile` (ver
    `visagismo_rules.py`). Todo opcional: lo que falta simplemente no se
    aplica, no es un error."""

    if growth_map is None and whorls:
        growth_map = {"whorls": whorls}
    growth = growth_analysis.summarize(growth_map) if growth_map else None

    # Sin tipo de pelo (cliente nuevo que aún no lo ha dicho ni se lo ha
    # marcado el peluquero) no se filtra: se ordena todo el catálogo con el
    # resto de señales.
    compatibles = [
        style for style in load_full_catalog()
        if hair_texture is None or hair_texture in style.suitable_hair_types
    ]

    recomendaciones = []
    for style in compatibles:
        weeks = maintenance.weeks_for_style(style)
        effects = [e for e in (_efecto_forma_cara(style, face_shape),
                                _efecto_forma_cara_especifica(style, face_shape)) if e]
        effects += growth_rules.evaluate_growth(style, growth)
        effects += trait_rules.evaluate_traits(style, visagismo_profile)
        effects += visagismo_rules.evaluate_profile(style, visagismo_profile).effects
        effects += combined_rules.evaluate_combined(style, growth, face_shape, visagismo_profile)

        reasons = [e for e in effects if e.is_reason]
        warnings = [e for e in effects if not e.is_reason]
        note = " ".join(e.detail for e in warnings) or None
        score = sum(e.score for e in effects)
        recomendaciones.append(((score, _desempate(weeks)), StyleRecommendation(
            style=style, note=note, reasons=reasons, warnings=warnings, maintenance_weeks=weeks,
            score=score)))

    recomendaciones.sort(key=lambda par: par[0])
    return [rec for _score, rec in recomendaciones]


def match_percent(score: float) -> int:
    """Puntuación -> "% de encaje" para enseñárselo al cliente (como en el
    anuncio: "Textured Crop + Mid Fade 97%").

    No es una probabilidad ni sale de ningún modelo: es la misma
    puntuación de las reglas escrita de forma legible. Un corte sin
    señales ni a favor ni en contra queda en 60 %; cada razón a favor sube
    y cada aviso baja. Se mantiene entre 20 y 99 para no dar un 100 % (no
    hay nada perfecto) ni un 0 % (el catálogo no descarta cortes, solo los
    ordena; ver la decisión de diseño en visagismo_rules.py)."""
    return int(max(20, min(99, round(60 - score * 10))))
