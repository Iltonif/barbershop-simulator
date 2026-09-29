"""
Conexión a la base de datos SQLite de perfiles de cliente + historial.

Se usa SQLite (no un servidor de base de datos aparte) porque para el
volumen de una barbería (cientos o miles de clientes, no millones) es más
que suficiente y no añade infraestructura que mantener. Si el proyecto
crece mucho, migrar a Postgres es sencillo porque toda la lógica de acceso
a datos está aislada en `repository.py`, no desperdigada por las rutas de
la API.

Guardar cualquier dato aquí implica tratar datos biométricos de una
persona identificable (tipo de pelo, forma de cara, remolinos) de forma
persistente — algo que el resto del pipeline evita por defecto (ver
sección RGPD de CLAUDE.md). Ver `repository.create_client`, que exige
`consent_history=True` explícito antes de guardar nada.
"""

import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    display_name TEXT,
    consent_history INTEGER NOT NULL DEFAULT 0,
    consent_history_at TEXT,
    consent_model_improvement INTEGER NOT NULL DEFAULT 0,
    consent_model_improvement_at TEXT,
    consent_save_photo INTEGER NOT NULL DEFAULT 0,
    consent_save_photo_at TEXT,
    consent_ai_analysis INTEGER NOT NULL DEFAULT 0,
    consent_ai_analysis_at TEXT,
    consent_3d_scan INTEGER NOT NULL DEFAULT 0,
    consent_3d_scan_at TEXT,
    avatar3d_path TEXT,
    avatar3d_at TEXT,
    avatar3d_metrics TEXT,
    hair_texture_override TEXT,
    face_shape_override TEXT,
    custom_growth_map TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS visits (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    created_at TEXT NOT NULL,
    style_id TEXT,
    detected_hair_texture TEXT,
    detected_face_shape TEXT,
    used_hair_texture TEXT,
    warnings TEXT,
    photo_path TEXT
);

-- Lista de espera del día (ver app/api/session_routes.py): el cliente se
-- apunta al entrar en la tablet o con el QR, y el peluquero la ve en
-- frontend/sala.html. status: "waiting" | "in_service" | "done".
CREATE TABLE IF NOT EXISTS waiting (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting'
);

-- Historial de cortes del cliente (ver session_routes.py): el peluquero
-- registra al terminar el corte hecho y, con permiso, una foto. Máximo
-- MAX_HAIRCUT_HISTORY por cliente (se borran los más antiguos).
CREATE TABLE IF NOT EXISTS haircut_history (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    created_at TEXT NOT NULL,
    style_id TEXT,
    style_name TEXT,
    notes TEXT,
    photo_path TEXT
);

-- Una fila por simulación con modelo externo, para el tope diario por
-- cliente (cada una cuesta dinero, ver haircut_editor.py).
CREATE TABLE IF NOT EXISTS simulation_log (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    requested_by TEXT NOT NULL
);

-- Cortes que el peluquero añade desde el propio selector de cortes cuando
-- no encuentra el que busca (ver frontend/assets/style-picker.js, sept
-- 2026). Van aquí -- en el Volume, igual que clients.db -- y NO en
-- app/pipeline/catalog_data/styles.json, que vive fuera del Volume a
-- propósito y con cada despliegue se sustituye por la copia de git (ver
-- STYLES_CATALOG_PATH en app/config.py): un corte guardado ahí en
-- producción se perdería en el siguiente despliegue. `style_catalog.
-- load_full_catalog()` combina esta tabla con el catálogo base para que
-- estos cortes aparezcan igual que cualquier otro (catálogo, simulador,
-- recomendaciones, historial).
CREATE TABLE IF NOT EXISTS custom_styles (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    length_top_mm INTEGER NOT NULL,
    length_sides_mm INTEGER NOT NULL,
    length_back_mm INTEGER NOT NULL,
    fade_type TEXT NOT NULL,
    suitable_hair_types TEXT NOT NULL
);
"""

# Columnas añadidas después de la primera versión del esquema. `CREATE
# TABLE IF NOT EXISTS` no las añade a una base de datos ya creada con el
# esquema antiguo, así que `init_db()` las agrega a mano si faltan (ver
# `_ensure_column`). Añadir aquí cualquier columna nueva que se incorpore
# más adelante, en vez de solo cambiar `_SCHEMA`.
_MIGRATIONS = [
    ("clients", "custom_growth_map", "TEXT"),
    ("clients", "visagismo_profile", "TEXT"),
    # Consentimiento separado para el informe de visagismo por IA (API de
    # Claude, ver `app/pipeline/visagismo_ai_advisor.py`) -- una base de
    # datos ya creada con un esquema anterior no tiene estas columnas.
    ("clients", "consent_ai_analysis", "INTEGER NOT NULL DEFAULT 0"),
    ("clients", "consent_ai_analysis_at", "TEXT"),
    # Flujo "cliente esperando en el sillón" (sept 2026): el cliente se da
    # de alta solo con su teléfono, guarda favoritos, y el peluquero le hace
    # una foto que se guarda (con su consentimiento) para simular cortes.
    ("clients", "phone", "TEXT"),
    ("clients", "consent_simulation", "INTEGER NOT NULL DEFAULT 0"),
    ("clients", "consent_simulation_at", "TEXT"),
    ("clients", "simulation_photo_path", "TEXT"),
    ("clients", "liked_styles", "TEXT"),
    # "Quiero repetir este corte": el cliente elige uno de su historial
    # mientras espera y el peluquero lo ve en la sala y en la ficha.
    ("waiting", "requested_history_id", "TEXT"),
    # Maniquí del cliente (growth-map.html): color de pelo y largo actual
    # puesto a mano por el peluquero ({"top","sides","back"} en mm) con su
    # fecha, para seguir sumando lo que crece desde entonces.
    ("clients", "hair_color", "TEXT"),
    ("clients", "current_length", "TEXT"),
    ("clients", "current_length_at", "TEXT"),
    # Gemelo digital 3D (app/pipeline/avatar3d.py): consentimiento propio
    # (la foto va a un tercero nuevo y el modelo 3D de su cara SÍ se guarda)
    # y ruta del .glb dentro del Volume, con la fecha en que se generó.
    ("clients", "consent_3d_scan", "INTEGER NOT NULL DEFAULT 0"),
    ("clients", "consent_3d_scan_at", "TEXT"),
    ("clients", "avatar3d_path", "TEXT"),
    ("clients", "avatar3d_at", "TEXT"),
    # Medidas calculadas sobre esa malla (ver mesh_metrics.py).
    ("clients", "avatar3d_metrics", "TEXT"),
    # "Quiero este" desde el catálogo/recomendaciones (sept 2026, item 8 de
    # Pedro): el cliente pide un estilo mientras espera SIN que esté aún en
    # su historial -- a diferencia de `requested_history_id`, que repite un
    # corte ya hecho. Ver `repository.set_requested_style`.
    ("waiting", "requested_style_id", "TEXT"),
]


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    """Crea las tablas si no existen todavía, y añade a mano cualquier
    columna nueva que falte en una base de datos creada con una versión
    anterior del esquema (ver `_MIGRATIONS`). Se llama una vez al arrancar
    la app (ver `main.py`)."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        for table, column, coltype in _MIGRATIONS:
            _ensure_column(conn, table, column, coltype)
        # Un teléfono = una ficha (NULL permitido para fichas creadas por el
        # peluquero sin teléfono).
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_phone ON clients(phone)")
