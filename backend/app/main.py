"""Punto de entrada de la app FastAPI.

Arrancar con: uvicorn backend.app.main:app --reload
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.clients_routes import router as clients_router
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
app.include_router(clients_router, prefix="/api")


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
