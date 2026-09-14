"""
Etapa 6 del pipeline: color de pelo.

Dos casos distintos:
1. Mantener el color real del cliente (extraerlo de la foto, corrigiendo
   el balance de blancos de la iluminación ambiente).
2. Aplicar un color nuevo (tinte), respetando el tono de piel para que no
   se vea como un parche de color plano.
"""

import cv2
import numpy as np


def extract_dominant_hair_color(image_bgr: np.ndarray, hair_mask: np.ndarray) -> tuple[int, int, int]:
    """Color dominante del pelo actual, en BGR, usando la máscara de segmentación."""
    hair_pixels = image_bgr[hair_mask > 0]
    if hair_pixels.size == 0:
        return (30, 30, 30)  # fallback: castaño oscuro

    median_color = np.median(hair_pixels.reshape(-1, 3), axis=0)
    return tuple(int(c) for c in median_color)


def apply_new_hair_color(
    image_bgr: np.ndarray, hair_mask: np.ndarray, target_bgr: tuple[int, int, int], strength: float = 0.75
) -> np.ndarray:
    """Aplica un color nuevo sobre la región de pelo, preservando luces y
    sombras (trabaja en espacio LAB para no aplastar el volumen del pelo).

    `strength` controla cuánto se acerca al color objetivo (0 = sin cambio,
    1 = color plano).

    TODO: esto es un recolor básico. Para tonos muy claros (rubio decolorado,
    canas) o efectos tipo balayage, este enfoque simple no basta — habrá que
    trabajar por mechones/zonas en vez de con una máscara única.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(
        np.uint8([[list(target_bgr)]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)

    mask_bool = hair_mask > 0
    # Mantiene L (luminosidad = sombras/luces del pelo real) y mezcla solo
    # los canales de color A/B hacia el color objetivo.
    lab[mask_bool, 1] = (
        lab[mask_bool, 1] * (1 - strength) + target_lab[1] * strength
    )
    lab[mask_bool, 2] = (
        lab[mask_bool, 2] * (1 - strength) + target_lab[2] * strength
    )

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result
