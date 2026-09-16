"""Configuración global del proyecto."""

import os
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


# Informe de visagismo por IA (app/pipeline/visagismo_ai_advisor.py):
# primera llamada del proyecto a un servicio EXTERNO de pago (API de
# Claude). Sin ANTHROPIC_API_KEY configurada, ese endpoint devuelve un
# error claro en vez de intentar la llamada -- el resto de la app
# funciona igual sin esta variable, no es obligatoria para arrancar.
# En Railway: Settings -> Variables -> añadir ANTHROPIC_API_KEY con una
# clave de https://console.anthropic.com/ (cuenta de pago del propio
# usuario, cada informe generado tiene coste real de tokens).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
# Configurable por si cambia el modelo disponible en el futuro sin tener
# que tocar código -- ver "Modelos" en la documentación de la API de
# Claude para el identificador vigente.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
