"""
Acceso a datos de clientes y su historial de visitas.

Todas las escrituras aquí implican datos personales — y, en el caso de
tipo de pelo/forma de cara/remolinos, datos biométricos — de una persona
identificable. No llamar a `create_client` sin haber comprobado antes que
el cliente ha dado su consentimiento explícito: la función lo exige
(`consent_history=True`), pero la decisión de PEDIR ese consentimiento de
forma clara es responsabilidad de quien use esta API (ver sección RGPD de
CLAUDE.md). `consent_model_improvement`, `consent_save_photo` y
`consent_ai_analysis` son consentimientos separados a propósito: guardar
el historial para atender mejor al cliente es una finalidad distinta de
usar sus datos para mejorar el sistema, de guardar directamente su foto,
o de enviar su perfil de visagismo a un servicio externo de pago (API de
Claude, ver `app/pipeline/visagismo_ai_advisor.py`) — el RGPD exige que
cada finalidad tenga su propio consentimiento, no vale uno genérico.
"""

import json
import uuid
from datetime import datetime, timezone

from app.db.database import get_connection
from app.db.models import ClientProfile, Visit, WaitingEntry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_client(row) -> ClientProfile:
    return ClientProfile(
        id=row["id"],
        created_at=row["created_at"],
        display_name=row["display_name"],
        consent_history=bool(row["consent_history"]),
        consent_model_improvement=bool(row["consent_model_improvement"]),
        consent_save_photo=bool(row["consent_save_photo"]),
        consent_ai_analysis=bool(row["consent_ai_analysis"]),
        hair_texture_override=row["hair_texture_override"],
        face_shape_override=row["face_shape_override"],
        custom_growth_map=json.loads(row["custom_growth_map"]) if row["custom_growth_map"] else None,
        visagismo_profile=json.loads(row["visagismo_profile"]) if row["visagismo_profile"] else None,
        notes=row["notes"],
        phone=row["phone"],
        consent_simulation=bool(row["consent_simulation"]),
        simulation_photo_path=row["simulation_photo_path"],
        liked_styles=json.loads(row["liked_styles"]) if row["liked_styles"] else [],
    )


def _row_to_visit(row) -> Visit:
    return Visit(
        id=row["id"],
        client_id=row["client_id"],
        created_at=row["created_at"],
        style_id=row["style_id"],
        detected_hair_texture=row["detected_hair_texture"],
        detected_face_shape=row["detected_face_shape"],
        used_hair_texture=row["used_hair_texture"],
        warnings=json.loads(row["warnings"] or "[]"),
        photo_path=row["photo_path"],
    )


def create_client(
    display_name: str | None,
    consent_history: bool,
    consent_model_improvement: bool = False,
    consent_save_photo: bool = False,
    consent_ai_analysis: bool = False,
    notes: str | None = None,
) -> ClientProfile:
    if not consent_history:
        raise ValueError(
            "No se puede crear un perfil de cliente sin consent_history=True: "
            "guardar tipo de pelo/forma de cara/remolinos es dato biométrico, "
            "requiere consentimiento explícito (ver sección RGPD de CLAUDE.md)."
        )

    client_id = str(uuid.uuid4())
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO clients (
                id, created_at, display_name,
                consent_history, consent_history_at,
                consent_model_improvement, consent_model_improvement_at,
                consent_save_photo, consent_save_photo_at,
                consent_ai_analysis, consent_ai_analysis_at,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                client_id,
                now,
                display_name,
                1,
                now,
                1 if consent_model_improvement else 0,
                now if consent_model_improvement else None,
                1 if consent_save_photo else 0,
                now if consent_save_photo else None,
                1 if consent_ai_analysis else 0,
                now if consent_ai_analysis else None,
                notes,
            ),
        )

    client = get_client(client_id)
    assert client is not None  # se acaba de insertar
    return client


def get_client(client_id: str) -> ClientProfile | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    return _row_to_client(row) if row else None


def list_clients() -> list[ClientProfile]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM clients ORDER BY created_at DESC").fetchall()
    return [_row_to_client(row) for row in rows]


def update_hair_type_override(client_id: str, texture: str) -> ClientProfile | None:
    if get_client(client_id) is None:
        return None
    with get_connection() as conn:
        conn.execute(
            "UPDATE clients SET hair_texture_override = ? WHERE id = ?",
            (texture, client_id),
        )
    return get_client(client_id)


def update_face_shape_override(client_id: str, face_shape: str) -> ClientProfile | None:
    if get_client(client_id) is None:
        return None
    with get_connection() as conn:
        conn.execute(
            "UPDATE clients SET face_shape_override = ? WHERE id = ?",
            (face_shape, client_id),
        )
    return get_client(client_id)


def update_custom_growth_map(
    client_id: str, strokes: list[dict], whorls: list[dict]
) -> ClientProfile | None:
    """Sustituye por completo el mapa de crecimiento dibujado a mano del
    cliente (no hace merge con lo anterior: la herramienta de dibujo en
    `frontend/growth-map.html` siempre manda el estado completo del lienzo,
    así que un trazo borrado allí debe desaparecer aquí también)."""
    if get_client(client_id) is None:
        return None
    custom_growth_map = {"strokes": strokes, "whorls": whorls}
    with get_connection() as conn:
        conn.execute(
            "UPDATE clients SET custom_growth_map = ? WHERE id = ?",
            (json.dumps(custom_growth_map), client_id),
        )
    return get_client(client_id)


def update_visagismo_profile(client_id: str, profile: dict) -> ClientProfile | None:
    """Sustituye por completo el perfil de visagismo del cliente (igual
    que `update_custom_growth_map`: el frontend/API manda siempre el
    perfil completo, no un delta, así que un campo borrado en la ficha
    debe desaparecer aquí también)."""
    if get_client(client_id) is None:
        return None
    with get_connection() as conn:
        conn.execute(
            "UPDATE clients SET visagismo_profile = ? WHERE id = ?",
            (json.dumps(profile), client_id),
        )
    return get_client(client_id)


def add_visit(
    client_id: str,
    style_id: str | None,
    detected_hair_texture: str | None,
    detected_face_shape: str | None,
    used_hair_texture: str | None,
    warnings: list[str],
    photo_path: str | None = None,
) -> Visit:
    visit_id = str(uuid.uuid4())
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO visits (
                id, client_id, created_at, style_id,
                detected_hair_texture, detected_face_shape, used_hair_texture,
                warnings, photo_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                visit_id,
                client_id,
                now,
                style_id,
                detected_hair_texture,
                detected_face_shape,
                used_hair_texture,
                json.dumps(warnings),
                photo_path,
            ),
        )
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
    return _row_to_visit(row)


def list_visits(client_id: str) -> list[Visit]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM visits WHERE client_id = ? ORDER BY created_at DESC",
            (client_id,),
        ).fetchall()
    return [_row_to_visit(row) for row in rows]


# ---------------------------------------------------------------------------
# Flujo "cliente esperando en el sillón" (ver app/api/session_routes.py)
# ---------------------------------------------------------------------------

def normalize_phone(phone: str) -> str:
    """Solo dígitos, sin prefijo +34/0034 (para que "612 34 56 78" y
    "+34612345678" sean la misma ficha)."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if digits.startswith("0034"):
        digits = digits[4:]
    elif digits.startswith("34") and len(digits) == 11:
        digits = digits[2:]
    return digits


def get_client_by_phone(phone: str) -> ClientProfile | None:
    normalized = normalize_phone(phone)
    if not normalized:
        return None
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM clients WHERE phone = ?", (normalized,)).fetchone()
    return _row_to_client(row) if row else None


def set_phone(client_id: str, phone: str | None) -> ClientProfile | None:
    with get_connection() as conn:
        conn.execute("UPDATE clients SET phone = ? WHERE id = ?",
                     (normalize_phone(phone) if phone else None, client_id))
    return get_client(client_id)


def update_consents(client_id: str, consent_save_photo: bool | None = None,
                    consent_simulation: bool | None = None) -> ClientProfile | None:
    now = _now()
    with get_connection() as conn:
        if consent_save_photo is not None:
            conn.execute("UPDATE clients SET consent_save_photo = ?, consent_save_photo_at = ? WHERE id = ?",
                         (1 if consent_save_photo else 0, now if consent_save_photo else None, client_id))
        if consent_simulation is not None:
            conn.execute("UPDATE clients SET consent_simulation = ?, consent_simulation_at = ? WHERE id = ?",
                         (1 if consent_simulation else 0, now if consent_simulation else None, client_id))
    return get_client(client_id)


def set_liked_styles(client_id: str, style_ids: list[str]) -> ClientProfile | None:
    with get_connection() as conn:
        conn.execute("UPDATE clients SET liked_styles = ? WHERE id = ?", (json.dumps(style_ids), client_id))
    return get_client(client_id)


def set_simulation_photo_path(client_id: str, path: str | None) -> ClientProfile | None:
    with get_connection() as conn:
        conn.execute("UPDATE clients SET simulation_photo_path = ? WHERE id = ?", (path, client_id))
    return get_client(client_id)


def _today_prefix() -> str:
    return _now()[:10]


def check_in(client_id: str) -> WaitingEntry:
    """Apunta al cliente en la lista de espera de hoy (si ya está esperando
    o siendo atendido hoy, devuelve esa misma entrada)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM waiting WHERE client_id = ? AND created_at LIKE ? AND status != 'done' "
            "ORDER BY created_at DESC LIMIT 1",
            (client_id, _today_prefix() + "%"),
        ).fetchone()
        if row:
            return WaitingEntry(**dict(row))
        entry = WaitingEntry(str(uuid.uuid4()), client_id, _now(), _now(), "waiting")
        conn.execute("INSERT INTO waiting (id, client_id, created_at, updated_at, status) VALUES (?, ?, ?, ?, ?)",
                     (entry.id, entry.client_id, entry.created_at, entry.updated_at, entry.status))
    return entry


def list_waiting_today(include_done: bool = False) -> list[WaitingEntry]:
    query = "SELECT * FROM waiting WHERE created_at LIKE ?" + ("" if include_done else " AND status != 'done'")
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY created_at", (_today_prefix() + "%",)).fetchall()
    return [WaitingEntry(**dict(r)) for r in rows]


def update_waiting_status(entry_id: str, status: str) -> WaitingEntry | None:
    with get_connection() as conn:
        conn.execute("UPDATE waiting SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), entry_id))
        row = conn.execute("SELECT * FROM waiting WHERE id = ?", (entry_id,)).fetchone()
    return WaitingEntry(**dict(row)) if row else None


def log_simulation(client_id: str, requested_by: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO simulation_log (id, client_id, created_at, requested_by) VALUES (?, ?, ?, ?)",
                     (str(uuid.uuid4()), client_id, _now(), requested_by))


def count_simulations_today(client_id: str, requested_by: str | None = None) -> int:
    query = "SELECT COUNT(*) FROM simulation_log WHERE client_id = ? AND created_at LIKE ?"
    params: list = [client_id, _today_prefix() + "%"]
    if requested_by:
        query += " AND requested_by = ?"
        params.append(requested_by)
    with get_connection() as conn:
        return conn.execute(query, params).fetchone()[0]
