"""Efecto de una regla de recomendación sobre un corte.

Cada regla (remolinos, forma de cara, visagismo, rasgos faciales) devuelve
uno o varios `Effect`: una puntuación para ordenar (negativa = sube en la
lista, positiva = baja) y el porqué, en dos longitudes: `label` es la
etiqueta corta que se ve en la tarjeta ("Disimula las orejas") y `detail`
la explicación completa que se abre con el ⓘ. Con puntuación negativa es
una razón a favor; con positiva, un aviso. Ninguna regla descarta cortes.
"""

from dataclasses import dataclass

BOOST_SUAVE = -1
BOOST_FUERTE = -2
AVISO_SUAVE = 1
AVISO_FUERTE = 2


@dataclass(frozen=True)
class Effect:
    score: int
    label: str
    detail: str

    @property
    def is_reason(self) -> bool:
        return self.score < 0
