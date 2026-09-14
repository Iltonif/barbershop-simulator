"""Configuración global del proyecto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STYLES_CATALOG_PATH = BASE_DIR / "data" / "styles" / "styles.json"

# Si se activa, el pipeline puede guardar imágenes intermedias en disco (debug).
# Por defecto en False: no persistir fotos de clientes (ver sección RGPD en CLAUDE.md).
PERSIST_UPLOADED_IMAGES = False

# Perfil de cliente + historial (opt-in, ver app/db/ y app/api/clients_routes.py).
# No confundir con PERSIST_UPLOADED_IMAGES de arriba: esto es una base de
# datos NUEVA y separada que solo se rellena cuando una petición a
# /api/simulate incluye un client_id de un cliente que dio su
# consentimiento explícito (consent_history=True al crear el perfil).
DB_PATH = BASE_DIR / "data" / "clients.db"

# Las fotos de un cliente con perfil SOLO se guardan aquí si además dio un
# consentimiento específico y separado para ello (consent_save_photo=True) —
# tener un perfil con historial no implica automáticamente guardar fotos.
CLIENT_PHOTOS_DIR = BASE_DIR / "data" / "client_photos"

# Modelo de generación (Fase 2 del roadmap). Se deja como constante para poder
# cambiarlo sin tocar generator.py.
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # placeholder, revisar alternativas
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"  # placeholder
