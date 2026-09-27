"""
Reglas que cruzan el mapa de crecimiento dibujado a mano
(`growth_analysis.py` / `growth_rules.py`) con el perfil de visagismo
(`visagismo_rules.py`) o la forma de cara (`ClientProfile.face_shape_override`).

Por qué un módulo aparte: `growth_rules.py`, `visagismo_rules.py` y la
regla de forma de cara de `recommender.py` ya avisan o priorizan cada uno
por su cuenta (p.ej. cara redonda -> más altura arriba; remolino en la
frente -> tupé o textura; entradas en M -> texturizado/raya lateral). Pedro
pidió (sept 2026) que, además de sumar esas notas sueltas, el recomendador
reconozca cuándo dos señales de sitios distintos apuntan a la MISMA
conclusión (o, al contrario, tiran en direcciones opuestas) y lo explique
de una vez en una sola nota conjunta, en vez de dejar que el barbero junte
dos avisos sueltos por su cuenta y no sepa si es una coincidencia o algo
que de verdad hay que tener en cuenta.

Solo se implementan aquí combinaciones que:
- No estén ya cubiertas (ni parcialmente) por una regla individual de
  `growth_rules.py`/`visagismo_rules.py`/`recommender.py` -- si no, sería
  puntuar dos veces exactamente lo mismo con otras palabras.
- Tengan una base de barbería razonable y de uso extendido, o sean una
  composición directa de dos reglas que ya están documentadas en otro
  módulo (nunca un hecho nuevo sin más base que "parece razonable").

Las tres reglas de abajo:

1. Cara redonda + raya natural (por remolino de coronilla o por hacia
   dónde crece el pelo de la frente) + corte con raya lateral: ninguna
   regla de forma de cara de aquí toca la raya/asimetría todavía (la de
   `recommender.py` para "redonda" solo mira altura arriba). Que una raya
   lateral rompe la simetría de una cara redonda es una recomendación muy
   repetida en guías de barbería generalistas -- el mismo motivo por el
   que aquí ya se prioriza la raya lateral para cara ALARGADA, pero por
   anchura en vez de por asimetría (ver `recommender._efecto_forma_cara`).
   Si además el pelo ya crece de forma natural hacia ese lado, se suma que
   aguanta sin pelear con el crecimiento.
2. Cara alargada + remolino en la coronilla + corte con volumen arriba, en
   el rango de largo donde `growth_rules._coronilla` ya dice que "gana el
   remolino" (2,5-7,5 cm) y sin textura que lo disimule: aquí no se
   inventa nada nuevo, se compone lo que ya dicen dos reglas por separado
   (alargada -> evitar más altura; ese largo con remolino -> se levanta
   igualmente) para explicar que el motivo pesa el doble, no solo que
   coinciden dos avisos sueltos.
3. Entradas en M (visagismo) + remolino en la frente (growth) + tupé/
   pompadour con mucho largo peinado hacia atrás: aquí las dos reglas
   individuales van en direcciones CONTRARIAS para el mismo corte
   (`growth_rules._frente` anima a aprovechar el impulso del remolino;
   `visagismo_rules._regla_nacimiento_pelo` avisa de que ese mismo peinado
   deja las entradas a la vista). Sin esta regla, el barbero vería un
   "a favor" y un "en contra" sin más contexto y podría parecer una
   contradicción del sistema; aquí se explica que, en este caso concreto,
   pesa más tapar las entradas que aprovechar el remolino.

Igual que el resto del recomendador (ver `recommender.py`): ninguna regla
descarta un corte, solo suma una razón o un aviso con su nota.
"""

from __future__ import annotations

from app.pipeline import growth_rules
from app.pipeline.growth_analysis import GrowthSummary
from app.pipeline.rule_effects import AVISO_SUAVE, BOOST_SUAVE, Effect
from app.pipeline.style_catalog import HaircutStyle
from app.pipeline.trait_rules import _tiene_volumen_arriba
from app.pipeline.visagismo_rules import _get


def _redonda_raya_lateral(style: HaircutStyle, face_shape: str | None, growth: GrowthSummary | None) -> Effect | None:
    if face_shape != "redonda" or not growth or not growth.natural_part:
        return None
    if not growth_rules._raya_lateral(style):
        return None
    return Effect(
        BOOST_SUAVE, "Asimetría a favor del crecimiento",
        f"Con cara redonda, una raya rompe la simetría y suele estilizar más que un peinado uniforme -- y "
        f"como el pelo ya tiende de forma natural a su {growth.natural_part}, esta raya concreta aguanta "
        "sin pelear con el crecimiento.",
    )


def _alargada_remolino_corona(style: HaircutStyle, face_shape: str | None, growth: GrowthSummary | None) -> Effect | None:
    if face_shape != "alargada" or not growth or not growth.crown_whorls:
        return None
    if not (25 <= style.length_top_mm <= 75):
        return None
    if growth_rules._texturizado(style) or not _tiene_volumen_arriba(style):
        return None
    doble = growth.crown_whorls >= 2
    que = "los dos remolinos de la coronilla" if doble else "el remolino de la coronilla"
    return Effect(
        AVISO_SUAVE, "Doble motivo para no dar más altura",
        f"Con cara alargada, más volumen arriba ya alarga la cara -- y a este largo {que} tiende a "
        "levantarse por su cuenta, así que el efecto se nota el doble. Mejor con textura, más corto o "
        "más largo.",
    )


def _entradas_remolino_frente(style: HaircutStyle, growth: GrowthSummary | None, profile: dict | None) -> Effect | None:
    if not growth or not growth.front_whorl:
        return None
    if _get(profile, "hair_physical_metrics", "frontal_hairline_shape") != "m_shaped_receding":
        return None
    if style.style_family != "tupe_pompadour_clasico" or style.length_top_mm < 70:
        return None
    return Effect(
        AVISO_SUAVE, "El impulso del remolino expone las entradas",
        "El remolino de la frente ayuda a levantar un tupé, pero con entradas en M ese mismo "
        "levantamiento deja la línea de nacimiento todavía más a la vista -- aquí pesa más taparlas que "
        "aprovechar el remolino.",
    )


_REGLAS_FORMA_CARA = [_redonda_raya_lateral, _alargada_remolino_corona]


def evaluate_combined(
    style: HaircutStyle,
    growth: GrowthSummary | None,
    face_shape: str | None,
    profile: dict | None,
) -> list[Effect]:
    """Todas las reglas que cruzan crecimiento + (forma de cara o
    visagismo). `growth`/`face_shape`/`profile` pueden faltar -- las
    reglas que dependen de un dato ausente simplemente no se activan."""
    effects = [e for fn in _REGLAS_FORMA_CARA if (e := fn(style, face_shape, growth))]
    e = _entradas_remolino_frente(style, growth, profile)
    if e:
        effects.append(e)
    return effects
