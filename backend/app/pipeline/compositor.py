"""
Etapa 8 del pipeline: compositing final.

Une la imagen generada con la foto original respetando oclusiones (orejas,
gafas, cuello de la ropa) y evitando el efecto "pegado" — bordes duros,
iluminación/grano inconsistente entre el pelo nuevo y el resto de la foto.
"""

import cv2
import numpy as np


def blend_with_original(
    original_bgr: np.ndarray, generated_bgr: np.ndarray, hair_mask: np.ndarray, feather_px: int = 15
) -> np.ndarray:
    """Combina la imagen generada con la original usando una máscara con
    bordes suavizados (feathering), para que la transición no se note.

    TODO: esto asume que `generated_bgr` ya tiene la resolución/alineación
    exacta de `original_bgr`. Si el generador trabaja a una resolución
    distinta, alinear/reescalar antes de llamar a esta función.
    """
    feathered_mask = cv2.GaussianBlur(
        hair_mask.astype(np.float32), (0, 0), sigmaX=feather_px
    )
    feathered_mask = np.clip(feathered_mask / 255.0, 0, 1)[..., None]

    blended = (
        generated_bgr.astype(np.float32) * feathered_mask
        + original_bgr.astype(np.float32) * (1 - feathered_mask)
    )
    return blended.astype(np.uint8)


def match_grain_and_light(reference_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Iguala aproximadamente el nivel de ruido/grano y el brillo global
    entre la foto original y la región generada, para que no se note un
    salto de calidad/iluminación entre ambas.

    TODO(Fase 3): esto es una igualación global simple. Un enfoque más
    fino trabajaría por regiones locales, no con un único factor global.
    """
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    ref_brightness = float(np.mean(ref_gray))
    target_brightness = float(np.mean(target_gray))
    if target_brightness < 1e-6:
        return target_bgr

    factor = ref_brightness / target_brightness
    adjusted = np.clip(target_bgr.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)
