"""Endpoints de la API: catálogo de cortes y simulación."""

import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import HeadShapeOut, SimulationResponse, StyleOut
from app.config import CLIENT_PHOTOS_DIR
from app.db import repository
from app.pipeline import (
    color_transfer,
    compositor,
    face_analysis,
    hair_segmentation,
    hair_type,
    head_mesh,
    head_shape,
)
from app.pipeline.generator import GenerationRequest, generate_haircut_preview
from app.pipeline.style_catalog import get_style_by_id, load_catalog

router = APIRouter()

# Carpeta de depuración: aquí se guarda la última máscara de pelo detectada
# (y una superposición sobre la foto) en cada petición, para poder inspeccionar
# visualmente qué está segmentando BiSeNet. Solo para desarrollo, no pensado
# para producción — se sobrescribe en cada llamada a /api/simulate.
_DEBUG_DIR = Path(__file__).resolve().parent.parent.parent / "debug_output"


@router.get("/styles", response_model=list[StyleOut])
def list_styles():
    return [StyleOut(**style.__dict__) for style in load_catalog()]


@router.post("/growth-map/head-shape", response_model=HeadShapeOut)
async def growth_map_head_shape(photo: UploadFile = File(...)):
    """Mide el ancho/alto de cara en la foto para que
    `frontend/growth-map.html` pueda ajustar la silueta del maniquí 3D a
    este cliente en concreto (ver `pipeline/head_shape.py`).

    A propósito NO guarda la foto ni ningún dato en ningún sitio, ni
    acepta/usa `client_id`: es un cálculo al vuelo y stateless, coherente
    con que el resto del pipeline evite guardar datos biométricos por
    defecto (ver la nota RGPD al principio de `clients_routes.py`) --
    aquí ni siquiera hay opción de guardar nada, no hace falta ningún
    consentimiento para usar esta herramienta.
    """
    contents = await photo.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada")

    face_result = face_analysis.analyze_face(image_bgr)
    if face_result is None:
        raise HTTPException(status_code=422, detail="No se detectó ninguna cara en la foto")

    shape = head_shape.derive_head_shape(face_result.landmarks)
    return HeadShapeOut(
        scale_x=shape.scale_x,
        scale_y=shape.scale_y,
        width_to_height=shape.width_to_height,
    )


@router.post("/simulate", response_model=SimulationResponse)
async def simulate(
    photo: UploadFile = File(...),
    style_id: str = Form(...),
    target_hair_color_hex: str | None = Form(None),
    client_id: str | None = Form(None),
    manual_hair_texture: str | None = Form(None),
):
    style = get_style_by_id(style_id)
    if style is None:
        raise HTTPException(status_code=404, detail=f"Corte '{style_id}' no encontrado en el catálogo")

    contents = await photo.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada")

    warnings: list[str] = []

    # Etapa 1: análisis facial
    face_result = face_analysis.analyze_face(image_bgr)
    if face_result is None:
        raise HTTPException(status_code=422, detail="No se detectó ninguna cara en la foto")
    if not face_result.is_symmetric_enough:
        warnings.append("La foto tiene un ángulo pronunciado; el resultado puede ser menos preciso.")

    # Etapa 2: segmentación de pelo (BiSeNet real, ver hair_segmentation.py)
    hair_mask = hair_segmentation.segment_hair(image_bgr)
    _save_debug_hair_mask(image_bgr, hair_mask)
    if hair_segmentation.mask_coverage_ratio(hair_mask) < 0.02:
        warnings.append("La segmentación de pelo detectó muy poca cobertura; revisar iluminación/ángulo de la foto.")

    # Etapa 3: tipo de pelo (heurística automática — ver limitaciones conocidas
    # en CLAUDE.md). Se guarda aparte el valor "en crudo" antes de aplicar
    # cualquier corrección manual, porque es justo la comparación entre este
    # valor automático y la corrección real del peluquero lo que en el futuro
    # podría servir como dato para entrenar un clasificador (ver historial de
    # visitas más abajo).
    hair_type_result = hair_type.classify_hair_type(image_bgr, hair_mask)
    raw_detected_hair_texture = hair_type_result.texture.value

    # Etapa 4: mapa de crecimiento / geometría de cabeza
    growth_map = head_mesh.build_default_growth_map(face_result.landmarks, image_bgr.shape)

    # Etapa 4.5: perfil de cliente (opcional). Si se pasa client_id y ese
    # cliente tiene correcciones manuales guardadas por el barbero (tipo de
    # pelo, forma de cara, remolinos), se aplican aquí por encima de lo que
    # detectó el pipeline automáticamente — el barbero conoce al cliente
    # mejor que la heurística. Ver app/db/ y app/api/clients_routes.py.
    client = None
    if client_id:
        client = repository.get_client(client_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Cliente '{client_id}' no encontrado")

        if client.hair_texture_override:
            try:
                hair_type_result.texture = hair_type.HairTexture(client.hair_texture_override)
            except ValueError:
                warnings.append(
                    "El perfil del cliente tiene guardada una corrección de tipo de "
                    f"pelo no válida ('{client.hair_texture_override}'); se ignora."
                )

        if client.face_shape_override:
            face_result.face_shape = client.face_shape_override

        if client.custom_growth_map:
            growth_map = head_mesh.apply_custom_growth_map(
                growth_map,
                strokes=client.custom_growth_map.get("strokes", []),
                whorls=client.custom_growth_map.get("whorls", []),
            )

    # Etapa 4.6: el peluquero indica el tipo de pelo a mano, en el momento.
    # Tiene prioridad sobre TODO lo anterior (heurística automática Y
    # corrección guardada del perfil): el peluquero mirando al cliente en
    # persona ahora mismo es más fiable que cualquiera de las dos. La
    # heurística automática ha demostrado confundir pelo despeinado/con
    # frizz con rizado o afro (ver CLAUDE.md) — por eso esto no es solo un
    # "override" más, es la forma recomendada de usar la app mientras no
    # exista un clasificador entrenado.
    if manual_hair_texture:
        try:
            hair_type_result.texture = hair_type.HairTexture(manual_hair_texture)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Tipo de pelo '{manual_hair_texture}' no válido. "
                    "Usa uno de: liso, ondulado, rizado, afro."
                ),
            )
        # Si el cliente tiene perfil, esta elección queda guardada como su
        # corrección — así no hay que repetirla en la próxima visita, y
        # sigue pudiendo cambiarse a mano cualquier día que haga falta.
        if client is not None:
            client = repository.update_hair_type_override(client.id, manual_hair_texture)

    # Etapa 6: color (opcional, solo si el cliente pide cambiar de color)
    working_image = image_bgr
    if target_hair_color_hex:
        target_bgr = _hex_to_bgr(target_hair_color_hex)
        working_image = color_transfer.apply_new_hair_color(working_image, hair_mask, target_bgr)

    # Etapa 7: generación (placeholder — ver generator.py, Fase 2 del roadmap)
    try:
        generated = generate_haircut_preview(
            GenerationRequest(
                image_bgr=working_image,
                hair_mask=hair_mask,
                style=style,
                hair_type=hair_type_result,
            )
        )
    except NotImplementedError:
        warnings.append(
            "Generación aún no implementada (Fase 2 del roadmap): se devuelve la foto original sin el corte aplicado."
        )
        generated = working_image

    # Etapa 8: compositing final
    final_image = compositor.blend_with_original(image_bgr, generated, hair_mask)

    _, buffer = cv2.imencode(".jpg", final_image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Etapa 9: registrar la visita en el historial del cliente, si tiene
    # perfil. `client.consent_history` es siempre True si el cliente existe
    # (se exige al crear el perfil), se comprueba igualmente por claridad y
    # por si en el futuro se permite revocar el consentimiento sin borrar
    # el perfil entero.
    if client is not None and client.consent_history:
        photo_path_str: str | None = None
        if client.consent_save_photo:
            client_dir = CLIENT_PHOTOS_DIR / client.id
            client_dir.mkdir(parents=True, exist_ok=True)
            photo_path = client_dir / f"{uuid.uuid4()}.jpg"
            cv2.imwrite(str(photo_path), image_bgr)
            photo_path_str = str(photo_path)

        repository.add_visit(
            client_id=client.id,
            style_id=style.id,
            detected_hair_texture=raw_detected_hair_texture,
            detected_face_shape=face_result.face_shape,
            used_hair_texture=hair_type_result.texture.value,
            warnings=warnings,
            photo_path=photo_path_str,
        )

    return SimulationResponse(
        style_id=style.id,
        detected_hair_texture=hair_type_result.texture.value,
        detected_face_shape=face_result.face_shape,
        warnings=warnings,
        image_base64=image_base64,
    )


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _save_debug_hair_mask(image_bgr: np.ndarray, hair_mask: np.ndarray) -> None:
    """Guarda la máscara de pelo detectada y una superposición en rojo sobre
    la foto original, en `backend/debug_output/`. Sirve para ver de verdad
    qué está segmentando BiSeNet como "pelo" en una foto concreta, sin tener
    que adivinarlo a partir del resultado final. Se sobrescribe en cada
    petición a /api/simulate — solo para depurar en desarrollo.
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(_DEBUG_DIR / "hair_mask.png"), hair_mask)

        overlay = image_bgr.copy()
        mask_bool = hair_mask > 0
        red = np.array([0, 0, 255], dtype=np.float32)
        overlay[mask_bool] = (
            0.5 * overlay[mask_bool].astype(np.float32) + 0.5 * red
        ).astype(np.uint8)
        cv2.imwrite(str(_DEBUG_DIR / "hair_mask_overlay.png"), overlay)
    except Exception as exc:  # nunca debe romper la petición real por esto
        print(f"[debug] no se pudo guardar la máscara de depuración: {exc}")
