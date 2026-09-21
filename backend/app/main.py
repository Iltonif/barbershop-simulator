"""Punto de entrada de la app FastAPI.

Arrancar con: uvicorn backend.app.main:app --reload
"""

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import require_barber
from app.api.clients_routes import router as clients_router
from app.api.session_routes import router as session_router
from app.api.routes import router as api_router
from app.db.database import init_db

app = FastAPI(
    title="Simulador de Cortes de Pelo/Barba",
    description="API del pipeline de simulación hiperrealista de cortes de pelo/barba.",
    version="0.1.0",
)

# CORS abierto para el MVP local. Restringir en producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
# Las fichas (/api/clients/*) solo para el peluquero (ver app/api/auth.py);
# el cliente accede a la suya por /api/me/* (session_routes.py).
app.include_router(clients_router, prefix="/api", dependencies=[Depends(require_barber)])
app.include_router(session_router, prefix="/api")


@app.middleware("http")
async def _no_stale_cache(request, call_next):
    """Evita servir versiones viejas de la web tras cada despliegue.

    StaticFiles no manda Cache-Control por defecto, así que el navegador
    aplica cacheo heurístico basado en Last-Modified y puede reutilizar
    una copia vieja de growth-map.html/head.glb sin volver a preguntar
    al servidor, incluso con Ctrl+Shift+R en algunos casos (el
    service-worker de frontend/sw.js reenvía la petición con
    fetch(event.request), que no siempre hereda el "ignora caché" de la
    recarga forzada). Con "no-cache" el navegador sigue guardando una
    copia local, pero SIEMPRE revalida con el servidor (If-None-Match)
    antes de usarla, así que un despliegue nuevo se ve de inmediato sin
    perder los 304 baratos cuando no ha cambiado nada.

    No se toca /api/* ni /health: esas respuestas ya gestionan su propio
    cacheo (o ninguno) y no queremos interferir.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def _on_startup() -> None:
    # Crea las tablas de clients/visits si no existen. No falla si ya
    # existen (ver init_db en app/db/database.py).
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- Servir la web estática (frontend/) desde el mismo backend --------
#
# Antes de esto, FastAPI solo exponía /api/* y /health: cualquier URL
# como http://localhost:8000/inicio.html devolvía 404, así que los
# enlaces entre páginas (inicio.html -> growth-map.html, catalogo.html,
# etc.) nunca cargaban nada aunque los archivos existieran en disco.
# Montamos frontend/ como estático para que toda la web viva en un único
# origen (http://localhost:8000) y las rutas relativas entre páginas
# funcionen tal cual están escritas.
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.get("/")
def _root() -> RedirectResponse:
    # Punto de entrada único de la web: siempre aterriza en la portada
    # con el selector "soy peluquero / soy cliente".
    return RedirectResponse(url="/inicio.html")


if FRONTEND_DIR.is_dir():
    # Registrado el último a propósito: Starlette prueba las rutas en el
    # orden en que se registran, así que /api/* y /health siguen
    # resolviéndose primero y este mount solo captura el resto (las
    # páginas .html y sus assets).
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
