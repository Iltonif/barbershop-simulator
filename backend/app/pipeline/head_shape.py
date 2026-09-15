"""
Ajuste de proporciones del maniquí 3D de `frontend/growth-map.html` a
partir de la foto del cliente.

Esto NO es una reconstrucción 3D real de la cabeza (ver la nota de
coordenadas en `head_mesh.py`): desde una única foto frontal no hay forma
de medir la profundidad real de la cabeza (cuánto sobresale la nuca, por
ejemplo, o lo prominente que es la frente), así que solo se ajustan los
DOS ejes que sí son medibles en una foto de frente -- ancho y alto de cara
-- y se deja la profundidad (Z) del maniquí sin tocar. Es una aproximación
honesta: acerca la silueta frontal del maniquí a la del cliente real, no
es una copia 3D de su cabeza (esa reconstrucción real -- a partir de varias
fotos o un sensor de profundidad -- sigue siendo trabajo futuro, igual que
ya se anotaba en head_mesh.py).
"""

from dataclasses import dataclass

import numpy as np

# Proporción media ancho/alto de cara (mandíbula a mandíbula / entrecejo a
# barbilla) usada como referencia neutra: una cara con esta proporción
# exacta no cambia el maniquí genérico (scale_x = scale_y = 1.0).
_AVERAGE_WIDTH_TO_HEIGHT = 0.78

# Límite de la corrección para que un fallo puntual de detección de
# landmarks (foto de mal ángulo, etc.) no pueda producir un maniquí
# absurdamente ancho o estrecho.
_MAX_RELATIVE_ADJUST = 1.4
_MIN_RELATIVE_ADJUST = 1 / _MAX_RELATIVE_ADJUST


@dataclass
class HeadShapeParams:
    scale_x: float  # factor de ancho (lateral) a aplicar sobre el maniquí genérico
    scale_y: float  # factor de alto (vertical) a aplicar sobre el maniquí genérico
    width_to_height: float  # proporción ancho/alto medida, sin más (informativo/depuración)


def derive_head_shape(landmarks: np.ndarray) -> HeadShapeParams:
    """`landmarks` usa el mismo esquema de 68 puntos (dlib/iBUG) que
    `head_mesh.build_default_growth_map`: jaw 0-16, entrecejo 27, barbilla 8.

    El ancho se mide entre los dos extremos de la mandíbula (0 y 16) y el
    alto entre el entrecejo (27) y la barbilla (8) -- las mismas
    referencias que ya usa `head_mesh.py` para el mapa de crecimiento por
    defecto, para no introducir una tercera forma distinta de medir la cara.
    """
    face_width = float(np.linalg.norm(landmarks[0] - landmarks[16]))
    face_height = float(np.linalg.norm(landmarks[27] - landmarks[8]))
    ratio = face_width / face_height if face_height > 0 else _AVERAGE_WIDTH_TO_HEIGHT

    # Redistribuye el ancho/alto del maniquí para acercarse a esta
    # proporción manteniendo aproximadamente su "volumen" (si se ensancha,
    # se acorta un poco, y viceversa) en vez de simplemente estirar un eje
    # y dejar el otro fijo, que se nota mucho más como una deformación.
    relative = ratio / _AVERAGE_WIDTH_TO_HEIGHT
    relative = float(np.clip(relative, _MIN_RELATIVE_ADJUST, _MAX_RELATIVE_ADJUST))
    scale_x = relative**0.5
    scale_y = relative**-0.5

    return HeadShapeParams(scale_x=scale_x, scale_y=scale_y, width_to_height=ratio)
