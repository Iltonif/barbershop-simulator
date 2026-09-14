"""
Etapa 3 del pipeline: tipo de pelo del cliente.

Esto es lo MÁS IMPORTANTE para que la simulación no se vea falsa: el mismo
corte se comporta de forma completamente distinta según la textura del
pelo. Clasificar bien esto pesa más en el resultado final que el propio
corte de referencia elegido.

Usamos como referencia la escala de Andre Walker (1A-4C), agrupada aquí en
4 categorías para simplificar el condicionamiento del generador:
    "liso", "ondulado", "rizado", "afro"

⚠️ PLACEHOLDER: heurística basada en la orientación de los bordes dentro
de la máscara de pelo (pelo liso = bordes muy alineados en pocas
direcciones, como mechones paralelos; pelo rizado = bordes dispersos en
muchas direcciones). Es un criterio más específico que medir solo
contraste/varianza de intensidad (versión anterior de este archivo), pero
SIGUE SIENDO una heurística sin calibrar con fotos reales etiquetadas, no
un clasificador entrenado. Sustituir por un modelo entrenado antes de
confiar en el resultado de cara al cliente (ver Fase 3 del roadmap en
CLAUDE.md).
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class HairTexture(str, Enum):
    STRAIGHT = "liso"
    WAVY = "ondulado"
    CURLY = "rizado"
    COILY = "afro"


@dataclass
class HairTypeResult:
    texture: HairTexture
    density: str  # "baja" | "media" | "alta"
    thickness: str  # "fino" | "medio" | "grueso"
    confidence: float  # 0.0 - 1.0, baja a propósito mientras sea heurística


def classify_hair_type(image_bgr: np.ndarray, hair_mask: np.ndarray) -> HairTypeResult:
    """Clasifica el tipo de pelo a partir de la imagen y su máscara de pelo.

    TODO(Fase 3 del roadmap): sustituir por un modelo entrenado. Mientras
    tanto, esta heurística sirve para no bloquear el resto del pipeline,
    pero NO debe usarse para decisiones de cara al cliente sin supervisión.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hair_pixels = gray[hair_mask > 0]

    if hair_pixels.size == 0:
        return HairTypeResult(
            texture=HairTexture.STRAIGHT,
            density="media",
            thickness="medio",
            confidence=0.0,
        )

    dispersion = _orientation_dispersion(gray, hair_mask)
    print(f"[debug] hair_type: dispersión de orientación = {dispersion:.4f}")

    # Umbrales recalibrados tras ver un dato real (pelo con rizo suelto
    # marcado, no afro de rizo apretado, dio dispersión ~0.815 — con el
    # umbral anterior de 0.75 caía ya en "afro"). Con una sola foto de
    # referencia esto sigue siendo una calibración muy débil, no una
    # validación real: a esta escala de análisis (gradiente pixel a pixel),
    # el rizo suelto y el afro apretado producen bordes igual de dispersos
    # localmente, así que esta heurística probablemente seguirá confundiendo
    # "rizado" con "afro" en más casos. Ver Fase 3 del roadmap en CLAUDE.md.
    if dispersion < 0.35:
        texture = HairTexture.STRAIGHT
    elif dispersion < 0.55:
        texture = HairTexture.WAVY
    elif dispersion < 0.88:
        texture = HairTexture.CURLY
    else:
        texture = HairTexture.COILY

    coverage = float(np.count_nonzero(hair_mask)) / hair_mask.size
    density = "alta" if coverage > 0.25 else "media" if coverage > 0.12 else "baja"

    return HairTypeResult(
        texture=texture,
        density=density,
        thickness="medio",  # TODO: estimar grosor real requiere mejor resolución/modelo
        confidence=0.3,  # deliberadamente baja: es una heurística, no un modelo entrenado
    )


def _orientation_dispersion(gray: np.ndarray, mask: np.ndarray) -> float:
    """Mide cuánto varían las direcciones de los bordes dentro de la
    máscara de pelo, ponderando por la fuerza de cada borde.

    Devuelve un valor 0.0 (bordes todos alineados en la misma dirección,
    típico de mechones lisos) a 1.0 (direcciones completamente dispersas,
    típico de rizo). Usa estadística circular sobre el ángulo doblado
    (2*theta) porque la orientación de un borde es periódica cada 180°,
    no 360° (un borde y su opuesto son la misma dirección).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    mask_bool = mask > 0
    mag = magnitude[mask_bool]
    ang = angle[mask_bool]

    if mag.size == 0:
        return 0.0

    # Ignorar píxeles con borde muy débil (zonas casi planas / ruido) para
    # que no aporten una dirección arbitraria y ensucien la medida.
    threshold = mag.mean() * 0.5
    strong = mag > threshold
    if int(np.count_nonzero(strong)) < 20:
        return 0.0

    mag = mag[strong]
    ang = ang[strong]

    sin_sum = np.sum(mag * np.sin(2 * ang))
    cos_sum = np.sum(mag * np.cos(2 * ang))
    resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / np.sum(mag)

    return float(1.0 - resultant_length)
