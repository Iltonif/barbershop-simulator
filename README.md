# Simulador Hiperrealista de Cortes de Pelo/Barba

MVP de arquitectura para una app de barbería que, a partir de una foto del cliente, genera una simulación personalizada de un corte de pelo/barba.

Ver `CLAUDE.md` para el contexto completo del proyecto, las decisiones de arquitectura y el roadmap — está pensado para que Claude Code lo lea automáticamente al abrir este repo.

## Puesta en marcha rápida

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd backend
python -m app.pipeline.download_weights      # pesos de BiSeNet (una sola vez)
python -m app.pipeline.download_landmark_model  # modelo de landmarks faciales (una sola vez)
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

El `--host 0.0.0.0` (en vez del `127.0.0.1` por defecto) es lo que permite
abrir la web también desde un móvil o tablet en la misma red WiFi, no solo
desde este ordenador — la web está pensada para usarse en pantalla táctil
(ver más abajo).

Luego abre http://localhost:8000/ en el navegador de este ordenador — te
lleva a la portada (`frontend/inicio.html`), donde eliges "Soy peluquero/a"
o "Soy cliente" y desde ahí accedes a todas las páginas (simulador, mapa de
remolinos, catálogo, recomendaciones). El backend sirve toda la carpeta
`frontend/` como web estática, así que no hace falta abrir los archivos
.html sueltos a mano.

### Abrir desde un móvil o tablet

`localhost` solo funciona en el mismo ordenador donde corre el backend.
Para abrir la web desde otro dispositivo (móvil, tablet) conectado a la
misma red WiFi:

1. Arranca el backend con `--host 0.0.0.0` (como arriba).
2. Averigua la IP local de este ordenador en la red WiFi, por ejemplo con
   `ipconfig getifaddr en0` (Mac) o `ipconfig` (Windows, busca "Dirección
   IPv4").
3. En el navegador del móvil/tablet, abre `http://<esa-ip>:8000/`
   (ej. `http://192.168.1.23:8000/`).

## Despliegue en la nube (para que una peluquería lo use en su tablet)

Para uso interno rápido en el mismo local, `--host 0.0.0.0` + la IP local
(sección de arriba) funciona, pero depende de que el ordenador de la
peluquería esté siempre encendido, de la misma WiFi para todos los
dispositivos, y de que no haya firewall/aislamiento de red por medio — en
la práctica da bastantes problemas. Para una instalación real en una
tablet de peluquería, lo robusto es desplegar el backend en la nube con
HTTPS: la tablet simplemente abre una URL fija (`https://...`) desde
cualquier WiFi o datos móviles, sin tocar IPs ni firewalls.

Este repo ya está preparado para desplegarse en [Railway](https://railway.app)
sin Docker:

- `requirements-prod.txt`: dependencias reales en producción (sin
  `diffusers`/`transformers`/`accelerate`/`controlnet-aux`/`gdown`, que
  solo hacen falta para la Fase 2 del generador — todavía no implementada,
  ver `generator.py` — sin esto el despliegue sería mucho más pesado y
  lento sin ningún beneficio real ahora mismo).
- `railway.json`: le dice a Railway cómo instalar (`pip install -r
  requirements-prod.txt`) y arrancar (`cd backend && uvicorn app.main:app
  --host 0.0.0.0 --port $PORT`) la app, y usa `/health` como healthcheck.
- Los pesos de BiSeNet (`79999_iter.pth`) van incluidos directamente en el
  repo en vez de descargarse con `gdown` en cada despliegue (Google Drive
  no es fiable desde IPs de datacenter).
- `frontend/manifest.webmanifest` + `frontend/sw.js` + `frontend/assets/icons/`:
  hacen que la web se pueda "Añadir a pantalla de inicio" en la tablet
  (iOS y Android) y se abra como una app con icono propio, sin la barra
  del navegador — para esto hace falta que la web esté servida por HTTPS,
  cosa que Railway (y cualquier hosting serio) da automáticamente.

Pasos para desplegar:

1. Sube este repo a GitHub (repositorio privado si prefieres que el
   código no sea público).
2. Entra en [railway.app](https://railway.app) y crea una cuenta
   (puedes registrarte con la propia cuenta de GitHub).
3. "New Project" → "Deploy from GitHub repo" → elige este repositorio.
   Railway detecta `railway.json` solo y usa esos comandos de instalación
   y arranque.
4. En la configuración del servicio, añade un **Volume** (disco
   persistente) montado en `/app/backend/data` — así la base de datos de
   clientes (`clients.db`) y las fotos guardadas (si algún cliente dio su
   consentimiento) sobreviven a los redespliegues en vez de borrarse cada
   vez que subas un cambio.
5. En "Settings" → "Networking", genera un dominio público (algo como
   `https://tu-proyecto.up.railway.app`).
6. Abre esa URL en la tablet de la peluquería. En Safari (iPad): botón de
   compartir → "Añadir a pantalla de inicio". En Chrome (Android): menú →
   "Instalar app" o "Añadir a pantalla de inicio". Queda un icono propio
   (las tijeras) como cualquier otra app.

El primer despliegue tarda unos minutos (instala OpenCV + PyTorch), y el
coste depende del plan de Railway vigente en el momento — conviene mirar
su página de precios antes de confirmar, porque puede cambiar.

## Nota sobre requisitos

- El análisis facial (OpenCV/LBF) y la segmentación de pelo con BiSeNet (usando la GPU de Apple Silicon vía MPS si está disponible) funcionan bien en un Mac normal.
- `diffusers` + ControlNet (la parte de generación, Fase 2 del roadmap) va muy lenta en CPU; para eso conviene una máquina con GPU (local o en la nube).
- La descarga de pesos de BiSeNet (`download_weights.py`) usa Google Drive, que a veces bloquea descargas automáticas por cuota. Si falla, el propio mensaje de error te da el enlace para descargarlo a mano.

## Estructura

```
backend/app/
  main.py              # arranque de FastAPI
  api/routes.py         # endpoints (POST /api/simulate, GET /api/styles)
  api/schemas.py         # modelos de request/response
  pipeline/
    face_analysis.py     # landmarks, pose, forma de cara
    hair_segmentation.py  # máscara de pelo — BiSeNet real, ver bisenet/
    bisenet/                # modelo vendorizado (MIT) + pesos descargados
    download_weights.py      # descarga los pesos de BiSeNet desde Google Drive
    hair_type.py           # textura/densidad de pelo (placeholder)
    head_mesh.py            # geometría 3D + dirección de crecimiento/remolinos (placeholder)
    style_catalog.py         # catálogo de cortes
    color_transfer.py         # color de pelo
    generator.py                # generación con difusión + ControlNet
    compositor.py                # blending final
backend/data/styles/styles.json  # catálogo de ejemplo
frontend/index.html               # UI mínima de prueba
```
