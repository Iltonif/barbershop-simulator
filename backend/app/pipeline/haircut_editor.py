"""
Simulación del corte con un modelo externo de edición de imagen (Fase 2).

Sustituye, para la generación, al esqueleto de difusión + ControlNet de
`generator.py`, que necesitaba GPU propia (Railway no la tiene). En vez de
eso se manda la foto del cliente a un modelo de edición de imagen por API
con una instrucción construida a partir del corte del catálogo:

- **gemini**: Google Gemini (modelo de imagen "Nano Banana"). Recibe la
  foto del cliente Y la foto de referencia del corte del catálogo, así que
  puede copiar el peinado de la foto además de leerlo del texto.
- **flux**: FLUX.1 Kontext [pro] (Black Forest Labs) a través de fal.ai.
  Por defecto solo recibe la foto del cliente + el texto (el endpoint con
  varias imágenes es el [max], el doble de caro); `FAL_USE_REFERENCE=1`
  lo activa.

Se dejan los dos para compararlos con fotos reales (decisión de Pedro,
sept 2026) y quedarse con el que mejor salga. Ninguno se usa si no está su
clave configurada (`GEMINI_API_KEY`, `FAL_KEY`).

RGPD: esto SÍ envía la foto del cliente a un tercero (a diferencia del
informe de IA de `visagismo_ai_advisor.py`, que nunca manda fotos). Por
eso `POST /api/simulate` exige un consentimiento propio por simulación
(`consent_external_photo`), y aquí:
- nunca se guarda la foto ni el resultado en disco;
- con fal se pide `sync_mode` para que el resultado vuelva en la propia
  respuesta y no quede en su historial de peticiones;
- con Gemini hay que usar una clave con facturación activada: en el nivel
  gratuito Google puede usar lo que se envía para mejorar sus productos.
"""

from __future__ import annotations

import base64
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from app import config
from app.pipeline.style_catalog import HaircutStyle


class HaircutEditorNotConfigured(RuntimeError):
    """No hay clave para ese proveedor (o falta su paquete)."""


class HaircutEditorError(RuntimeError):
    """El proveedor ha fallado o no ha devuelto ninguna imagen."""


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    company: str  # a quién se envía la foto (se enseña en el consentimiento)


PROVIDERS = {
    "gemini": Provider("gemini", "Gemini", "Google"),
    "flux": Provider("flux", "FLUX", "Black Forest Labs (vía fal.ai)"),
}

_FADE_TEXT = {
    "ninguno": "no fade: the sides are cut with scissors, same length all around",
    "bajo": "low fade on the sides and back (starts just above the ears and neckline)",
    "medio": "mid fade on the sides and back (starts around the temples)",
    "alto": "high fade on the sides and back (starts high, near the top of the head)",
    "skin": "skin fade: sides and back taken down to the skin at the bottom",
}
_TEXTURE_TEXT = {
    "liso": "straight",
    "ondulado": "wavy",
    "rizado": "curly",
    "afro": "afro-textured, tightly coiled",
}
# Relaciones de aspecto que acepta Gemini; se elige la más cercana a la foto.
_GEMINI_ASPECTS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3,
                   "4:5": 0.8, "5:4": 1.25, "9:16": 9 / 16, "16:9": 16 / 9}
_MAX_SIDE_PX = 1536


def available_providers() -> list[Provider]:
    out = []
    if config.GEMINI_API_KEY:
        out.append(PROVIDERS["gemini"])
    if config.FAL_KEY:
        out.append(PROVIDERS["flux"])
    return out


def _length_text(mm: int) -> str:
    if mm <= 3:
        return f"buzzed very short (~{mm} mm)"
    if mm < 10:
        return f"very short (~{mm} mm)"
    cm = mm / 10
    return f"about {cm:.0f} cm long" if cm >= 1.5 else f"about {mm} mm long"


def build_edit_prompt(style: HaircutStyle, hair_texture: str | None,
                      target_color_hex: str | None = None, with_reference: bool = False) -> str:
    """Instrucción de edición en inglés (los modelos siguen mejor el inglés).
    Combina lo que dice el catálogo (nombre, descripción, largos en mm,
    degradado) con el tipo de pelo real del cliente, e insiste en no tocar
    la cara: es lo que más se nota si falla."""
    texture = _TEXTURE_TEXT.get(hair_texture or "", None)
    lines = [
        "Edit this photo of a real barbershop client: change ONLY the haircut.",
        f"New haircut: \"{style.name}\". {style.description}".strip(),
        f"- Top: {_length_text(style.length_top_mm)}.",
        f"- Sides: {_length_text(style.length_sides_mm)}; back: {_length_text(style.length_back_mm)}.",
        f"- {_FADE_TEXT.get(style.fade_type, style.fade_type)}.",
    ]
    if texture:
        lines.append(f"- Keep the client's natural {texture} hair texture.")
    if target_color_hex:
        lines.append(f"- Dye the hair to this colour: {target_color_hex}.")
    else:
        lines.append("- Keep the client's current hair colour.")
    if with_reference:
        lines.append(
            "The SECOND image is only a reference for the haircut shape. Do not copy anything "
            "else from it: not the face, skin, beard, clothes, pose or background."
        )
    lines.append(
        "Keep everything else exactly the same: same person and identity, face, facial "
        "features, skin tone, expression, beard and eyebrows, ears, pose, head angle, "
        "clothing, background, lighting and camera framing. Photorealistic, like a real "
        "photo taken right after the haircut, with a clean professional finish."
    )
    return "\n".join(lines)


def _to_jpeg(image_bgr: np.ndarray) -> bytes:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, _MAX_SIDE_PX / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HaircutEditorError("No se pudo preparar la foto para enviarla")
    return buf.tobytes()


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HaircutEditorError("El proveedor devolvió una imagen que no se puede leer")
    return image


def _closest_aspect(image_bgr: np.ndarray) -> str:
    h, w = image_bgr.shape[:2]
    ratio = w / h
    return min(_GEMINI_ASPECTS, key=lambda k: abs(_GEMINI_ASPECTS[k] - ratio))


def _run_gemini(image_bgr: np.ndarray, prompt: str, reference: bytes | None) -> bytes:
    if not config.GEMINI_API_KEY:
        raise HaircutEditorNotConfigured("Falta GEMINI_API_KEY")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise HaircutEditorNotConfigured("Falta el paquete google-genai") from exc

    parts = [prompt, types.Part.from_bytes(data=_to_jpeg(image_bgr), mime_type="image/jpeg")]
    if reference:
        parts.append(types.Part.from_bytes(data=reference, mime_type="image/jpeg"))
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_IMAGE_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=_closest_aspect(image_bgr)),
            ),
        )
    except Exception as exc:
        raise HaircutEditorError(f"Gemini: {exc}") from exc

    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    reason = ""
    if response.candidates and response.candidates[0].finish_reason:
        reason = f" (motivo: {response.candidates[0].finish_reason})"
    raise HaircutEditorError(f"Gemini no devolvió ninguna imagen{reason}")


def _fetch(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (URL de fal)
        return resp.read()


def _run_flux(image_bgr: np.ndarray, prompt: str, reference: bytes | None) -> bytes:
    if not config.FAL_KEY:
        raise HaircutEditorNotConfigured("Falta FAL_KEY")
    try:
        import fal_client
    except ImportError as exc:  # pragma: no cover
        raise HaircutEditorNotConfigured("Falta el paquete fal-client") from exc

    image_uri = fal_client.encode(_to_jpeg(image_bgr), "image/jpeg")
    args = {"prompt": prompt, "output_format": "jpeg", "safety_tolerance": "2",
            "num_images": 1, "sync_mode": True}
    if reference:
        app = config.FAL_MULTI_MODEL
        args["image_urls"] = [image_uri, fal_client.encode(reference, "image/jpeg")]
    else:
        app = config.FAL_MODEL
        args["image_url"] = image_uri
    try:
        result = fal_client.SyncClient(key=config.FAL_KEY).subscribe(app, arguments=args)
        images = (result or {}).get("images") or []
        if not images:
            raise HaircutEditorError("FLUX no devolvió ninguna imagen")
        return _fetch(images[0]["url"])
    except HaircutEditorError:
        raise
    except Exception as exc:
        raise HaircutEditorError(f"FLUX: {exc}") from exc


def edit_haircut(image_bgr: np.ndarray, style: HaircutStyle, hair_texture: str | None,
                 provider: str, target_color_hex: str | None = None,
                 reference: bytes | None = None) -> np.ndarray:
    """Devuelve la foto con el corte aplicado (BGR). `reference` es la foto
    del corte del catálogo en JPEG, o None."""
    if provider not in PROVIDERS:
        raise HaircutEditorNotConfigured(f"Proveedor desconocido: {provider}")
    if provider == "flux" and not config.FAL_USE_REFERENCE:
        reference = None
    prompt = build_edit_prompt(style, hair_texture, target_color_hex, with_reference=reference is not None)
    runner = _run_gemini if provider == "gemini" else _run_flux
    return _decode(runner(image_bgr, prompt, reference))


def load_reference_photo(style: HaircutStyle) -> bytes | None:
    """Foto del corte del catálogo (vive en frontend/, ver `reference_image`)."""
    if not style.reference_image:
        return None
    path = (config.FRONTEND_DIR / style.reference_image).resolve()
    if config.FRONTEND_DIR.resolve() not in path.parents or not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return _to_jpeg(image) if image is not None else None
