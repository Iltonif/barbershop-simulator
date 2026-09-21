"""Configuración global del proyecto."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# INCIDENTE (producción, sept 2026): este fichero vivía antes en
# `data/styles/styles.json`, es decir, dentro de la MISMA carpeta `data/`
# donde vive `clients.db` (ver DB_PATH más abajo). En Railway hay un volumen
# persistente montado sobre esa carpeta `data/` entera para que la base de
# datos de clientes sobreviva a cada despliegue -- pero eso significa que
# CUALQUIER fichero dentro de `data/`, incluido este catálogo (que SÍ está
# versionado en git y se actualiza con cada commit), queda "tapado" por la
# instantánea que Railway guardó la primera vez que se creó ese volumen. El
# resultado: `recommend_styles`/`GET .../recommendations` fallaba con un 500
# en producción (mientras que en local, sin volumen de por medio, funcionaba
# perfecto) porque el catálogo que se leía de verdad era una copia antigua
# desde antes de que existiera el campo `style_family`, con una forma
# distinta a la que espera hoy `HaircutStyle` -- ver `load_catalog()` en
# `style_catalog.py`. Se corrigió moviendo el catálogo FUERA de `data/`, a
# `app/pipeline/catalog_data/`, para que nunca vuelva a quedar bajo un
# volumen pensado solo para persistir datos de clientes. `load_catalog()`
# además ahora ignora claves desconocidas del JSON en vez de reventar, como
# segunda red de seguridad ante este mismo tipo de desajuste en el futuro.
STYLES_CATALOG_PATH = Path(__file__).resolve().parent / "pipeline" / "catalog_data" / "styles.json"

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


# Simulación del corte con un modelo externo de edición de imagen
# (app/pipeline/haircut_editor.py). Igual que ANTHROPIC_API_KEY: sin clave,
# ese proveedor no aparece y la app sigue funcionando (la simulación
# devuelve la foto sin cambios, como antes). Se pueden poner las dos para
# comparar. En Railway: Settings -> Variables.
#   GEMINI_API_KEY: clave de https://aistudio.google.com/ CON facturación
#     activada (en el nivel gratuito Google puede usar lo enviado para
#     mejorar sus productos -- no vale para fotos de clientes).
#   FAL_KEY: clave de https://fal.ai/ (FLUX.1 Kontext).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
FAL_KEY = os.environ.get("FAL_KEY")
FAL_MODEL = os.environ.get("FAL_MODEL", "fal-ai/flux-pro/kontext")
FAL_MULTI_MODEL = os.environ.get("FAL_MULTI_MODEL", "fal-ai/flux-pro/kontext/max/multi")
# Con 1, FLUX recibe también la foto del corte del catálogo (usa el modelo
# [max] multi, ~0,08 $ en vez de ~0,04 $ por imagen).
FAL_USE_REFERENCE = os.environ.get("FAL_USE_REFERENCE", "0") == "1"

# Carpeta del frontend (de aquí salen las fotos de referencia del catálogo).
FRONTEND_DIR = BASE_DIR.parent / "frontend"


# Flujo "cliente esperando en el sillón" (app/api/session_routes.py).
# BARBER_PIN: PIN numérico para entrar en la parte del peluquero (sala de
#   espera, fichas, simulación). OBLIGATORIO en Railway: sin él, la parte
#   del peluquero no deja entrar a nadie (así no quedan las fichas abiertas
#   por olvido). La parte del cliente funciona igual.
BARBER_PIN = os.environ.get("BARBER_PIN")
# Secreto para firmar las cookies de sesión. Si no se configura, se genera
# uno y se guarda en data/ (que en Railway es el Volume), para que las
# sesiones sobrevivan a los reinicios.
APP_SECRET = os.environ.get("APP_SECRET")
# El peluquero se desconecta solo tras este tiempo sin usar la web, porque
# en la tablet compartida el siguiente cliente podría entrar en su parte.
BARBER_IDLE_MINUTES = int(os.environ.get("BARBER_IDLE_MINUTES", "15"))
# Sesión del cliente en su propio móvil (QR). En la tablet la web le saca
# sola a los pocos minutos sin tocarla (ver frontend/assets/session.js).
CLIENT_SESSION_DAYS = int(os.environ.get("CLIENT_SESSION_DAYS", "30"))
# Tope de simulaciones con modelo externo que puede lanzar un cliente él
# solo al día (cada una cuesta ~0,04-0,05 $). El peluquero no tiene tope.
MAX_CLIENT_SIMULATIONS_PER_DAY = int(os.environ.get("MAX_CLIENT_SIMULATIONS_PER_DAY", "6"))
