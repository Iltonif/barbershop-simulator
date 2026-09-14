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
