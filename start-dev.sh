#!/bin/bash
# Arranca el servidor local de desarrollo con un solo comando, en vez de
# escribir a mano "cd backend", "source .venv/bin/activate" y el "uvicorn
# ..." cada vez (ver "Puesta en marcha rápida" en README.md). Solo hace
# falta ejecutarlo una vez por sesión de trabajo: --reload ya detecta solo
# los cambios en los archivos mientras el servidor sigue corriendo.
#
# Uso: ./start-dev.sh   (desde cualquier carpeta; el script se sitúa solo)
set -euo pipefail

# Se sitúa en la carpeta donde está este script (la raíz del repo), pase lo
# que pase con la carpeta desde la que se haya lanzado.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d ".venv" ]; then
  echo "No encuentro la carpeta .venv aquí (${PWD}). Créala primero con:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
cd backend
echo "Arrancando en http://localhost:8000/inicio.html (Ctrl+C para parar)..."
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
