"""
Etapa 7 del pipeline: generación de la imagen final.

Aquí es donde vive el "hiperrealismo". Enfoque elegido para el MVP:
difusión (Stable Diffusion) + ControlNet, condicionado por:
    - la máscara de pelo (dónde puede pintar el modelo)
    - los landmarks/pose de la cara (para no desalinear el resultado)
    - los parámetros del corte elegido + el tipo de pelo del cliente
      (para construir el prompt de texto)

⚠️ ESQUELETO, NO EJECUTADO TODAVÍA: requiere GPU para ir a una velocidad
razonable y requiere descargar pesos de varios GB la primera vez. No lo
actives hasta llegar a la Fase 2 del roadmap (ver CLAUDE.md). Antes de
eso, `generate_haircut_preview` puede sustituirse por un mock que devuelva
la imagen original sin cambios, para poder probar el resto del pipeline.
"""

from dataclasses import dataclass

import numpy as np

from app.config import CONTROLNET_MODEL_ID, DIFFUSION_MODEL_ID
from app.pipeline.hair_type import HairTypeResult
from app.pipeline.style_catalog import HaircutStyle

_pipeline_cache = None  # se inicializa perezosamente, es costoso de cargar


@dataclass
class GenerationRequest:
    image_bgr: np.ndarray
    hair_mask: np.ndarray
    style: HaircutStyle
    hair_type: HairTypeResult


def _load_pipeline():
    """Carga perezosa del pipeline de diffusers + ControlNet.

    TODO: elegir el tipo de condicionamiento de ControlNet más adecuado.
    "canny" es el punto de partida más simple, pero un mapa de normales o
    de profundidad derivado de `head_mesh.py` puede dar mejores resultados
    para respetar la geometría de la cabeza.
    """
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
    import torch

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        DIFFUSION_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    _pipeline_cache = pipe
    return pipe


def build_prompt(style: HaircutStyle, hair_type: HairTypeResult) -> str:
    """Construye el prompt de texto a partir del corte y el tipo de pelo.

    TODO: iterar mucho sobre este prompt en la Fase 2 — es el punto que más
    se ajusta a base de prueba y error con fotos reales.
    """
    return (
        f"professional barbershop haircut photo, {style.name}, "
        f"{hair_type.texture.value} hair texture, {hair_type.density} density, "
        f"fade type {style.fade_type}, photorealistic, natural lighting, "
        f"sharp focus, same person, same face, same skin tone"
    )


def generate_haircut_preview(request: GenerationRequest) -> np.ndarray:
    """Genera la imagen final con el corte simulado aplicado.

    TODO(Fase 2): implementar la llamada real al pipeline (`_load_pipeline`)
    usando `request.hair_mask` como máscara de inpainting y un control_image
    derivado de los landmarks/geometría de cabeza. Mientras tanto, para no
    bloquear el resto del pipeline, se puede sustituir esta función por un
    mock que devuelva `request.image_bgr` sin modificar.
    """
    raise NotImplementedError(
        "generate_haircut_preview: implementar en la Fase 2 del roadmap "
        "(ver CLAUDE.md). Usa un mock temporal si necesitas probar el resto "
        "del pipeline mientras tanto."
    )
