"""
Flujo "cliente esperando en el sillón" (sept 2026).

El cliente suele esperar mientras el peluquero termina con otro. La web se
organiza alrededor de eso:

1. **Primera visita**: el cliente, en la tablet de la barbería o en su móvil
   (QR), se da de alta con nombre + teléfono + consentimientos
   (`POST /api/me/register`), responde "Mi perfil" y mira catálogo y
   recomendaciones. Queda apuntado en la lista de espera de hoy.
2. El **peluquero** (entra con PIN) ve la lista de espera
   (`GET /api/waiting`), abre la ficha del cliente y completa lo técnico:
   tipo de pelo, remolinos, visagismo, y la foto para simular
   (`POST /api/clients/{id}/simulation-photo`, se guarda con su permiso).
3. **Siguientes visitas**: el cliente entra con su teléfono y ya tiene
   recomendaciones completas, favoritos y simulación de cortes sobre su
   foto (`POST /api/me/simulate`, con tope diario).

Rutas de cliente: `/api/me/*` (solo su propia ficha, cookie
`client_session`). Rutas de peluquero: `/api/barber/*`, `/api/waiting*` y
`/api/clients/{id}/simulation-photo|consents|phone|simulate` (cookie
`barber_session`). Ver `auth.py`.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.api import auth, client_service
from app.api.schemas import (
    BarberLoginIn,
    ClientLoginIn,
    ClientOut,
    ClientRegisterIn,
    ConsentsIn,
    HaircutOut,
    HaircutRequestIn,
    LikesIn,
    NextVisitOut,
    QuestionnaireIn,
    RecommendationsOut,
    ReturnDueOut,
    SimulationResponse,
    StoredPhotoSimulationIn,
    WaitingOut,
    WaitingStatusIn,
)
from app.db import repository
from app.pipeline.style_catalog import get_style_by_id, load_catalog

router = APIRouter()

_MAX_PHOTO_SIDE = 2000
_VALID_STATUS = {"waiting", "in_service", "done"}


def _client_out(client) -> ClientOut:
    return ClientOut(**client.__dict__)


def _validate_phone(phone: str) -> str:
    normalized = repository.normalize_phone(phone)
    if len(normalized) < 9:
        raise HTTPException(status_code=422, detail="Escribe un teléfono válido (9 cifras).")
    return normalized


# ---------------------------------------------------------------- peluquero

@router.get("/barber/session")
def barber_session(request: Request):
    return {"logged_in": auth.is_barber(request), "pin_configured": bool(config.BARBER_PIN)}


@router.post("/barber/login")
def barber_login(payload: BarberLoginIn, request: Request, response: Response):
    if not config.BARBER_PIN:
        raise HTTPException(status_code=503, detail="Falta configurar BARBER_PIN en el servidor.")
    if not auth.check_pin(payload.pin):
        raise HTTPException(status_code=401, detail="PIN incorrecto")
    auth.set_cookie(response, request, auth.BARBER_COOKIE, auth.make_token("barber", "barber"), auth.barber_max_age())
    # En la tablet compartida: al entrar el peluquero, se cierra la sesión
    # del cliente que la estuviera usando.
    response.delete_cookie(auth.CLIENT_COOKIE, path="/")
    return {"ok": True}


@router.post("/barber/logout")
def barber_logout(response: Response):
    response.delete_cookie(auth.BARBER_COOKIE, path="/")
    return {"ok": True}


def _waiting_out(entry) -> WaitingOut | None:
    client = repository.get_client(entry.client_id)
    if client is None:
        return None
    life = ((client.visagismo_profile or {}).get("lifestyle_and_preferences") or {})
    hair = ((client.visagismo_profile or {}).get("hair_physical_metrics") or {})
    questionnaire_done = bool(client.face_shape_override or hair.get("hair_pattern_shape")
                              or life.get("daily_maintenance_commitment") or life.get("beard_preference"))
    requested = None
    if entry.requested_history_id:
        record = repository.get_haircut(client.id, entry.requested_history_id)
        requested = _haircut_out(record) if record else None
    return WaitingOut(
        id=entry.id, status=entry.status, created_at=entry.created_at, client=_client_out(client),
        is_new=not client.hair_texture_override and not client.simulation_photo_path,
        questionnaire_done=questionnaire_done,
        has_hair_texture=bool(client.hair_texture_override),
        requested=requested,
    )


@router.get("/waiting", response_model=list[WaitingOut], dependencies=[Depends(auth.require_barber)])
def list_waiting(include_done: bool = False):
    return [w for w in (_waiting_out(e) for e in repository.list_waiting_today(include_done)) if w]


@router.patch("/waiting/{entry_id}", response_model=WaitingOut, dependencies=[Depends(auth.require_barber)])
def set_waiting_status(entry_id: str, payload: WaitingStatusIn):
    if payload.status not in _VALID_STATUS:
        raise HTTPException(status_code=422, detail="Estado no válido")
    entry = repository.update_waiting_status(entry_id, payload.status)
    if entry is None:
        raise HTTPException(status_code=404, detail="No está en la lista de espera")
    return _waiting_out(entry)


@router.post("/clients/{client_id}/check-in", response_model=WaitingOut, dependencies=[Depends(auth.require_barber)])
def barber_check_in(client_id: str):
    """El peluquero apunta a mano a un cliente que no ha usado la tablet."""
    if repository.get_client(client_id) is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return _waiting_out(repository.check_in(client_id))


def _get_client_or_404(client_id: str):
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client


@router.patch("/clients/{client_id}/consents", response_model=ClientOut, dependencies=[Depends(auth.require_barber)])
def barber_set_consents(client_id: str, payload: ConsentsIn):
    """El peluquero marca un consentimiento que el cliente le da en ese
    momento (p.ej. para guardar la foto en la primera visita). Si se retira
    el de guardar fotos, se borran todas las suyas."""
    _get_client_or_404(client_id)
    updated = repository.update_consents(client_id, payload.consent_save_photo, payload.consent_simulation)
    if payload.consent_save_photo is False:
        _delete_all_photos(updated)
    return _client_out(repository.get_client(client_id))


@router.put("/clients/{client_id}/simulation-photo", response_model=ClientOut, dependencies=[Depends(auth.require_barber)])
async def upload_simulation_photo(client_id: str, photo: UploadFile = File(...)):
    """Foto de frente que el peluquero hace en la primera visita y se
    guarda (en el Volume de Railway, junto a clients.db) para que el
    cliente pueda simular cortes en las siguientes visitas. Solo con
    `consent_save_photo`. Sustituye a la anterior."""
    client = _get_client_or_404(client_id)
    if not client.consent_save_photo:
        raise HTTPException(status_code=422, detail="El cliente no ha dado permiso para guardar su foto.")
    data = await photo.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la foto")
    h, w = image.shape[:2]
    scale = min(1.0, _MAX_PHOTO_SIDE / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    folder = config.CLIENT_PHOTOS_DIR / client.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "simulation.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return _client_out(repository.set_simulation_photo_path(client.id, str(path)))


def _photo_response(client) -> FileResponse:
    if not client.simulation_photo_path:
        raise HTTPException(status_code=404, detail="Sin foto")
    return FileResponse(client.simulation_photo_path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@router.get("/clients/{client_id}/simulation-photo", dependencies=[Depends(auth.require_barber)])
def barber_get_photo(client_id: str):
    return _photo_response(_get_client_or_404(client_id))


def _unlink(path: str | None) -> None:
    if path:
        try:
            from pathlib import Path
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _delete_all_photos(client) -> None:
    _delete_photo(client)
    for path in repository.clear_haircut_photos(client.id):
        _unlink(path)


def _delete_photo(client) -> None:
    if client.simulation_photo_path:
        try:
            from pathlib import Path
            Path(client.simulation_photo_path).unlink(missing_ok=True)
        except OSError:
            pass
    repository.set_simulation_photo_path(client.id, None)


@router.delete("/clients/{client_id}/simulation-photo", response_model=ClientOut, dependencies=[Depends(auth.require_barber)])
def barber_delete_photo(client_id: str):
    client = _get_client_or_404(client_id)
    _delete_photo(client)
    return _client_out(repository.get_client(client_id))


@router.post("/clients/{client_id}/simulate", response_model=SimulationResponse, dependencies=[Depends(auth.require_barber)])
def barber_simulate(client_id: str, payload: StoredPhotoSimulationIn):
    return client_service.simulate_with_stored_photo(_get_client_or_404(client_id), payload.style_id,
                                                     payload.provider, requested_by="peluquero")


@router.get("/qr.svg")
def qr_code(request: Request, path: str = "/cliente.html?qr=1"):
    """QR con la dirección de la parte del cliente, para imprimirlo o
    enseñarlo en la sala de espera (frontend/sala.html)."""
    import segno

    if not path.startswith("/"):
        raise HTTPException(status_code=422, detail="Ruta no válida")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    buf = io.BytesIO()
    segno.make(f"{proto}://{host}{path}", error="m").save(buf, kind="svg", scale=8, border=2, dark="#05060f", light="#ffffff")
    return Response(buf.getvalue(), media_type="image/svg+xml")


# ------------------------------------------------------------------ cliente

def _login_client(request: Request, response: Response, client) -> None:
    auth.set_cookie(response, request, auth.CLIENT_COOKIE, auth.make_token("client", client.id), auth.client_max_age())
    # En la tablet: si el peluquero se la ha dejado abierta, se cierra.
    response.delete_cookie(auth.BARBER_COOKIE, path="/")
    repository.check_in(client.id)


@router.post("/me/register", response_model=ClientOut)
def client_register(payload: ClientRegisterIn, request: Request, response: Response):
    if not payload.consent_history:
        raise HTTPException(status_code=422, detail="Para crear tu perfil hace falta que aceptes guardar tus datos.")
    if not payload.display_name.strip():
        raise HTTPException(status_code=422, detail="Escribe tu nombre.")
    phone = _validate_phone(payload.phone)
    if repository.get_client_by_phone(phone):
        raise HTTPException(status_code=409, detail="Ya hay un perfil con ese teléfono: entra con él.")
    client = repository.create_client(display_name=payload.display_name.strip(), consent_history=True,
                                      consent_save_photo=payload.consent_save_photo)
    repository.set_phone(client.id, phone)
    client = repository.update_consents(client.id, consent_simulation=payload.consent_simulation)
    _login_client(request, response, client)
    return _client_out(client)


@router.post("/me/login", response_model=ClientOut)
def client_login(payload: ClientLoginIn, request: Request, response: Response):
    client = repository.get_client_by_phone(_validate_phone(payload.phone))
    if client is None:
        raise HTTPException(status_code=404, detail="No hay ningún perfil con ese teléfono.")
    _login_client(request, response, client)
    return _client_out(client)


@router.post("/me/logout")
def client_logout(response: Response):
    response.delete_cookie(auth.CLIENT_COOKIE, path="/")
    return {"ok": True}


def _me(request: Request):
    client = repository.get_client(auth.require_client(request))
    if client is None:
        raise HTTPException(status_code=401, detail="Entra con tu teléfono.")
    return client


@router.get("/me", response_model=ClientOut)
def me(request: Request):
    return _client_out(_me(request))


@router.get("/me/recommendations", response_model=RecommendationsOut)
def my_recommendations(request: Request):
    return client_service.build_recommendations(_me(request))


@router.put("/me/likes", response_model=ClientOut)
def my_likes(payload: LikesIn, request: Request):
    client = _me(request)
    known = {s.id for s in load_catalog()}
    ids = list(dict.fromkeys(i for i in payload.style_ids if i in known))[:50]
    return _client_out(repository.set_liked_styles(client.id, ids))


@router.patch("/me/consents", response_model=ClientOut)
def my_consents(payload: ConsentsIn, request: Request):
    client = _me(request)
    updated = repository.update_consents(client.id, payload.consent_save_photo, payload.consent_simulation)
    # Si retira el permiso de guardar fotos, se borran todas las suyas
    # (la de simular y las del historial de cortes).
    if payload.consent_save_photo is False:
        _delete_all_photos(updated)
        updated = repository.get_client(client.id)
    return _client_out(updated)


@router.patch("/me/questionnaire", response_model=ClientOut)
def my_questionnaire(payload: QuestionnaireIn, request: Request):
    """Guarda las respuestas de "Mi perfil" sobre lo que ya hubiera en la
    ficha (no borra los datos técnicos del peluquero)."""
    client = _me(request)
    vp = dict(client.visagismo_profile or {})
    anat = dict(vp.get("anatomical_metrics") or {})
    hair = dict(vp.get("hair_physical_metrics") or {})
    life = dict(vp.get("lifestyle_and_preferences") or {})
    geometry = {"ovalada": "oval", "redonda": "round", "cuadrada": "square", "alargada": "rectangular_elongated"}
    if payload.hair_pattern_shape:
        hair["hair_pattern_shape"] = payload.hair_pattern_shape
    if payload.frontal_hairline_shape:
        hair["frontal_hairline_shape"] = payload.frontal_hairline_shape
    if payload.face_shape in geometry:
        anat["facial_geometry"] = geometry[payload.face_shape]
        repository.update_face_shape_override(client.id, payload.face_shape)
    if payload.daily_maintenance_commitment:
        life["daily_maintenance_commitment"] = payload.daily_maintenance_commitment
    if payload.barbershop_visit_frequency_days:
        life["barbershop_visit_frequency_days"] = payload.barbershop_visit_frequency_days
    if payload.beard_preference:
        life["beard_preference"] = payload.beard_preference
    vp.update(anatomical_metrics=anat, hair_physical_metrics=hair, lifestyle_and_preferences=life)
    return _client_out(repository.update_visagismo_profile(client.id, vp))


@router.get("/me/simulation-photo")
def my_photo(request: Request):
    return _photo_response(_me(request))


@router.delete("/me/simulation-photo", response_model=ClientOut)
def my_delete_photo(request: Request):
    client = _me(request)
    _delete_photo(client)
    return _client_out(repository.get_client(client.id))


@router.post("/me/simulate", response_model=SimulationResponse)
def my_simulate(payload: StoredPhotoSimulationIn, request: Request):
    return client_service.simulate_with_stored_photo(_me(request), payload.style_id, payload.provider,
                                                     requested_by="cliente")


@router.get("/me/simulations-left")
def my_simulations_left(request: Request):
    client = _me(request)
    used = repository.count_simulations_today(client.id, "cliente")
    return {"left": max(0, config.MAX_CLIENT_SIMULATIONS_PER_DAY - used),
            "max": config.MAX_CLIENT_SIMULATIONS_PER_DAY}


# ---------------------------------------------------------- historial de cortes

_HISTORY_PHOTO_SIDE = 1400


def _haircut_out(record) -> HaircutOut:
    style = get_style_by_id(record.style_id) if record.style_id else None
    return HaircutOut(id=record.id, created_at=record.created_at, style_id=record.style_id,
                      style_name=record.style_name or (style.name if style else None), notes=record.notes,
                      reference_image=style.reference_image if style else None, photo_path=record.photo_path)


def _history(client) -> list[HaircutOut]:
    return [_haircut_out(r) for r in repository.list_haircuts(client.id)]


@router.get("/clients/{client_id}/history", response_model=list[HaircutOut], dependencies=[Depends(auth.require_barber)])
def barber_history(client_id: str):
    return _history(_get_client_or_404(client_id))


@router.post("/clients/{client_id}/history", response_model=list[HaircutOut], dependencies=[Depends(auth.require_barber)])
async def barber_add_haircut(client_id: str, style_id: str | None = Form(None), style_name: str | None = Form(None),
                             notes: str | None = Form(None), photo: UploadFile | None = File(None)):
    """El peluquero registra el corte que acaba de hacer: del catálogo
    (`style_id`) o uno libre (`style_name`), notas técnicas y, si el cliente
    dio permiso para guardar fotos, la foto del resultado. Al pasar de
    `MAX_HAIRCUT_HISTORY` se borran los más antiguos con su foto."""
    client = _get_client_or_404(client_id)
    if style_id and get_style_by_id(style_id) is None:
        raise HTTPException(status_code=404, detail="Corte no encontrado en el catálogo")
    if not style_id and not (style_name or "").strip():
        raise HTTPException(status_code=422, detail="Elige el corte del catálogo o escribe cuál ha sido.")
    photo_path = None
    if photo is not None and photo.filename:
        if not client.consent_save_photo:
            raise HTTPException(status_code=422, detail="El cliente no ha dado permiso para guardar fotos suyas.")
        image = cv2.imdecode(np.frombuffer(await photo.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la foto")
        h, w = image.shape[:2]
        scale = min(1.0, _HISTORY_PHOTO_SIDE / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        folder = config.CLIENT_PHOTOS_DIR / client.id / "history"
        folder.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid
        path = folder / f"{_uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_path = str(path)
    _, removed = repository.add_haircut(client.id, style_id, (style_name or "").strip() or None,
                                        (notes or "").strip()[:500] or None, photo_path, config.MAX_HAIRCUT_HISTORY)
    for r in removed:
        _unlink(r.photo_path)
    return _history(client)


def _history_photo(client, record_id: str) -> FileResponse:
    record = repository.get_haircut(client.id, record_id)
    if record is None or not record.photo_path:
        raise HTTPException(status_code=404, detail="Sin foto")
    return FileResponse(record.photo_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/clients/{client_id}/history/{record_id}/photo", dependencies=[Depends(auth.require_barber)])
def barber_history_photo(client_id: str, record_id: str):
    return _history_photo(_get_client_or_404(client_id), record_id)


@router.delete("/clients/{client_id}/history/{record_id}", response_model=list[HaircutOut], dependencies=[Depends(auth.require_barber)])
def barber_delete_haircut(client_id: str, record_id: str):
    client = _get_client_or_404(client_id)
    record = repository.delete_haircut(client.id, record_id)
    if record:
        _unlink(record.photo_path)
    return _history(client)


@router.get("/me/history", response_model=list[HaircutOut])
def my_history(request: Request):
    return _history(_me(request))


@router.get("/me/history/{record_id}/photo")
def my_history_photo(record_id: str, request: Request):
    return _history_photo(_me(request), record_id)


@router.delete("/me/history/{record_id}", response_model=list[HaircutOut])
def my_delete_haircut(record_id: str, request: Request):
    client = _me(request)
    record = repository.delete_haircut(client.id, record_id)
    if record:
        _unlink(record.photo_path)
    return _history(client)


@router.get("/me/request")
def my_request(request: Request):
    entry = repository.get_waiting_entry_today(_me(request).id)
    return {"in_waiting": bool(entry), "history_id": entry.requested_history_id if entry else None}


@router.put("/me/request")
def my_set_request(payload: HaircutRequestIn, request: Request):
    """"Quiero repetir este": el cliente elige un corte de su historial y el
    peluquero lo ve en la sala de espera y en su ficha."""
    client = _me(request)
    if payload.history_id and repository.get_haircut(client.id, payload.history_id) is None:
        raise HTTPException(status_code=404, detail="Ese corte no está en tu historial")
    entry = repository.get_waiting_entry_today(client.id) or repository.check_in(client.id)
    repository.set_requested_history(entry.id, payload.history_id)
    return {"in_waiting": True, "history_id": payload.history_id}


# --- Cuándo toca volver (ver pipeline/maintenance.py) ---

@router.get("/return-due", response_model=list[ReturnDueOut], dependencies=[Depends(auth.require_barber)])
def return_due(days_ahead: int = 2, max_overdue_days: int = 120):
    """Clientes a los que ya les toca (o les tocará en `days_ahead` días)
    y no están hoy en la sala: para avisarles. Primero los que menos se
    han pasado, que son los que más fácil vuelven."""
    waiting_today = {w.client_id for w in repository.list_waiting_today(include_done=True)}
    last = repository.last_visits()
    out = []
    for client in repository.list_clients():
        if client.id in waiting_today or client.id not in last:
            continue
        nv = client_service.next_visit_for(client, last[client.id])
        if nv and -max_overdue_days <= nv["days_left"] <= days_ahead:
            out.append(ReturnDueOut(client=_client_out(client), next_visit=NextVisitOut(**nv)))
    out.sort(key=lambda r: -r.next_visit.days_left)
    return out[:60]


@router.get("/me/next-visit", response_model=NextVisitOut | None)
def my_next_visit(request: Request):
    client = _me(request)
    nv = client_service.next_visit_for(client, repository.last_visits().get(client.id))
    return NextVisitOut(**nv) if nv else None
