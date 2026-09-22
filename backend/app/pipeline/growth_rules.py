"""
Reglas de recomendación a partir del mapa de crecimiento del pelo
(remolinos y dirección, ver `growth_analysis.py`).

Sustituye a la regla antigua de `recommender.py`, que solo contaba
remolinos y avisaba con cortes de menos de 1 cm arriba. Las guías de
barbería dicen casi lo contrario: el problema está en el tramo medio.

Fuentes (sept 2026):
- Book of Barbering, "Men's Cowlick Haircuts": con remolino en la
  coronilla, entre 2,5 y 7,5 cm arriba "gana el remolino" (tiene palanca
  pero no peso); funciona por debajo de 2,5 cm (máquina) o por encima de
  7,5 cm (el peso lo aplana), o un crop con textura. Remolino en la frente:
  mal con peinados planos hacia delante; bien con tupé, flequillo con
  textura, raya al medio o raya justo en el remolino. Doble remolino:
  undercut o peinados con textura.
- Cadmen Academy: peinar a favor del crecimiento (si el pelo de la frente
  va a la izquierda, barrido a la izquierda o flequillo); en la nuca,
  adaptar la línea al crecimiento en vez de cortar una línea recta; el
  largo alrededor del remolino ayuda a taparlo.
- Fellow Barber / Man For Himself: remolino de la coronilla en sentido
  horario -> raya a la izquierda; antihorario -> a la derecha.

Como el resto de reglas, solo ordenan y explican; no descartan cortes.
"""

from __future__ import annotations

import re

from app.pipeline import trait_rules
from app.pipeline.growth_analysis import GrowthSummary
from app.pipeline.rule_effects import AVISO_SUAVE, BOOST_SUAVE, Effect
from app.pipeline.style_catalog import HaircutStyle

_TEXTURA = re.compile(r"textur|despeinad|desenfadad|efecto nido|messy", re.I)
_RAYA_LATERAL = re.compile(r"raya (lateral|al lado|a un lado|de lado)|raya a la (izquierda|derecha)", re.I)
_RAYA_MEDIO = re.compile(r"raya (en (el )?medio|central|al medio)", re.I)
_NUCA_RECTA = re.compile(r"cuadrad|bloque", re.I)


def _texto(style: HaircutStyle) -> str:
    return f"{style.name} {style.description}"


def _texturizado(style: HaircutStyle) -> bool:
    return style.style_family == "corte_texturizado_general" or bool(_TEXTURA.search(_texto(style)))


def _raya_lateral(style: HaircutStyle) -> bool:
    return style.style_family == "clasico_raya_lateral" or bool(_RAYA_LATERAL.search(_texto(style)))


def _raya_medio(style: HaircutStyle) -> bool:
    return bool(_RAYA_MEDIO.search(_texto(style)))


def _coronilla(style: HaircutStyle, g: GrowthSummary) -> list[Effect]:
    if not g.crown_whorls:
        return []
    top = style.length_top_mm
    doble = g.crown_whorls >= 2
    que = "Los dos remolinos de la coronilla" if doble else "El remolino de la coronilla"
    if top < 25:
        return [Effect(BOOST_SUAVE, "Remolino sin levantar",
                       f"{que} no llega a levantarse con menos de 2,5 cm arriba.")]
    if top > 75:
        return [Effect(BOOST_SUAVE, "El peso aplana el remolino",
                       "Con más de 7,5 cm arriba, el propio peso del pelo aplana "
                       + ("los dos remolinos de la coronilla." if doble else "el remolino de la coronilla."))]
    if _texturizado(style):
        return [Effect(BOOST_SUAVE, "La textura disimula el remolino",
                       f"Entre 2,5 y 7,5 cm arriba {que.lower()} tiende a levantarse, pero con textura el movimiento queda buscado.")]
    if doble and style.fade_type in ("alto", "skin") and top >= 40:
        return [Effect(BOOST_SUAVE, "Undercut para el doble remolino",
                       "Con doble remolino, rapar alto los lados y dejar largo arriba separa las dos direcciones.")]
    return [Effect(AVISO_SUAVE, "Se levanta en la coronilla",
                   f"Entre 2,5 y 7,5 cm arriba {que.lower()} tiene palanca pero no peso: suele levantarse. "
                   "Mejor más corto, más largo o con textura.")]


def _frente(style: HaircutStyle, g: GrowthSummary) -> list[Effect]:
    out = []
    flequillo = trait_rules._tiene_flequillo(style)
    atras = trait_rules._es_hacia_atras(style)
    volumen = trait_rules._tiene_volumen_arriba(style)
    if g.front_whorl:
        if flequillo and not _texturizado(style):
            out.append(Effect(AVISO_SUAVE, "El remolino abre el flequillo",
                              "Con un remolino en la frente, un flequillo liso hacia delante se abre enseguida."))
        elif flequillo:
            out.append(Effect(BOOST_SUAVE, "Flequillo con textura",
                              "Con remolino en la frente, un flequillo con textura disimula el giro."))
        if volumen and style.length_top_mm >= 60:
            out.append(Effect(BOOST_SUAVE, "El tupé aprovecha el remolino",
                              "Un tupé usa el impulso del remolino de la frente en vez de pelearse con él."))
        if _raya_medio(style) or _raya_lateral(style):
            out.append(Effect(BOOST_SUAVE, "Raya en el remolino",
                              "Hacer la raya justo donde está el remolino de la frente reparte el pelo a favor del giro."))
        return out
    growth = g.front_growth
    if growth == "delante":
        if flequillo:
            out.append(Effect(BOOST_SUAVE, "A favor del crecimiento",
                              "El pelo de la frente ya crece hacia delante: el flequillo cae solo."))
        elif atras:
            out.append(Effect(AVISO_SUAVE, "Contra el crecimiento",
                              "El pelo de la frente crece hacia delante; peinarlo hacia atrás pide producto cada día."))
    elif growth == "atras":
        if atras or volumen:
            out.append(Effect(BOOST_SUAVE, "A favor del crecimiento",
                              "El pelo de la frente crece hacia atrás: los peinados hacia atrás o con tupé se sostienen solos."))
        elif flequillo:
            out.append(Effect(AVISO_SUAVE, "Contra el crecimiento",
                              "El pelo de la frente crece hacia atrás; el flequillo tiende a levantarse."))
    return out


def _raya(style: HaircutStyle, g: GrowthSummary) -> list[Effect]:
    if not g.natural_part or not _raya_lateral(style):
        return []
    por = ("el remolino de la coronilla" if g.natural_part_source == "remolino"
           else "hacia dónde va el pelo de la frente")
    return [Effect(BOOST_SUAVE, f"Raya a su {g.natural_part}",
                   f"Por {por}, la raya aguanta mejor a su {g.natural_part}: el pelo cae a favor.")]


def _nuca(style: HaircutStyle, g: GrowthSummary) -> list[Effect]:
    if not (g.nape_whorl or g.nape_growth_irregular):
        return []
    que = "el remolino de la nuca" if g.nape_whorl else "el pelo de la nuca, que crece de lado"
    if _NUCA_RECTA.search(_texto(style)) and style.fade_type == "ninguno":
        return [Effect(AVISO_SUAVE, "Nuca recta, difícil",
                       f"Una línea recta en la nuca deja muy a la vista {que}; mejor adaptarla al crecimiento.")]
    if style.length_back_mm >= 50:
        return [Effect(BOOST_SUAVE, "El largo tapa la nuca", f"Con largo en la nuca, {que} no se nota.")]
    if style.fade_type != "ninguno":
        return [Effect(BOOST_SUAVE, "Nuca degradada",
                       f"Con degradado no hace falta una línea en la nuca, así que {que} no la deforma.")]
    if 10 <= style.length_back_mm < 50:
        return [Effect(AVISO_SUAVE, "Se nota en la nuca",
                       f"Con un largo medio en la nuca, {que} se nota: mejor degradado o dejar más largo.")]
    return []


def evaluate_growth(style: HaircutStyle, growth: GrowthSummary | None) -> list[Effect]:
    if growth is None:
        return []
    return _coronilla(style, growth) + _frente(style, growth) + _raya(style, growth) + _nuca(style, growth)
