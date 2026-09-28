#!/bin/bash
# Arranca el servidor local de desarrollo con un solo comando, en vez de
# escribir a mano "cd backend", "source .venv/bin/activate", "BARBER_PIN=..."
# y el "uvicorn ..." cada vez (ver "Puesta en marcha rápida" en README.md).
# Solo hace falta ejecutarlo una vez por sesión de trabajo: --reload ya
# detecta solo los cambios en los archivos mientras el servidor sigue
# corriendo.
#
# Uso: ./start-dev.sh   (desde cualquier carpeta; el script se sitúa solo)
set -euo pipefail

# 1) Se sitúa en la carpeta donde está este script (la raíz del repo), pase
#    lo que pase con la carpeta desde la que se haya lanzado.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2) Comprueba que el entorno virtual existe antes de seguir.
if [ ! -d ".venv" ]; then
  echo "No encuentro la carpeta .venv aquí (${PWD}). Créala primero con:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# 3) BARBER_PIN (y cualquier otra clave real: TRIPO_API_KEY, etc.) se lee de
#    un archivo ".env" local si existe -- ese archivo NUNCA se sube a git
#    (ya está en .gitignore), así que aquí no se deja escrito ningún PIN o
#    clave real. Si no existe ".env", se usa 1234 solo como PIN de
#    conveniencia para desarrollo en local (no vale para Railway).
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi
export BARBER_PIN="${BARBER_PIN:-1234}"

# 4) Activa el entorno virtual y entra en backend/.
source .venv/bin/activate
cd backend

echo "Arrancando en http://localhost:8000/inicio.html (PIN de peluquero: ${BARBER_PIN}) -- Ctrl+C para parar..."
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
