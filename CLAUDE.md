# Simulador Hiperrealista de Cortes de Pelo/Barba — Contexto del Proyecto

Este archivo es la memoria persistente del proyecto para Claude Code. Léelo al empezar cualquier sesión en este repo.

## Objetivo

Construir una web app para barberías: el barbero (o el cliente) sube una foto, elige un corte de un catálogo, y la app genera una simulación hiperrealista y personalizada de cómo quedaría ese corte sobre ESA persona concreta — respetando su tipo de pelo, forma de cabeza, color, iluminación de la foto y rasgos faciales.

## Por qué está montado así (decisiones ya tomadas)

- **Backend Python + FastAPI**, no Node, porque todo el pipeline de visión/generación es Python (mediapipe, diffusers, torch).
- **Pipeline en etapas separadas** (`backend/app/pipeline/`) en vez de un único script, porque cada etapa se puede mejorar/sustituir de forma independiente (p.ej. cambiar el modelo de segmentación sin tocar el resto).
- **Generación por difusión + ControlNet** (no GAN, no render 3D de hebras) como punto de partida: es el enfoque con mejor relación realismo/coste de desarrollo para un MVP. Si el realismo no es suficiente más adelante, valorar un motor de hebras 3D (mucho más caro de construir).
- **Catálogo de cortes paramétrico** (`data/styles/styles.json`), no solo imágenes de referencia: cada corte se describe por longitud por zona (arriba/laterales/nuca), tipo de degradado y tipos de pelo para los que funciona bien. Esto permite condicionar la generación en vez de solo "pegar" una imagen de referencia.
- **Sin persistencia de fotos por defecto**: las imágenes de clientes se procesan en memoria y no se guardan en disco salvo que se implemente explícitamente un flujo de consentimiento (ver sección RGPD). Esto es intencional, no un olvido.

## Pipeline (7 etapas, ver `backend/app/pipeline/`)

1. **`face_analysis.py`** — Landmarks faciales, pose de la cabeza (yaw/pitch/roll), forma de cara y simetría. YA FUNCIONAL (esqueleto real, no placeholder). Usa Haar Cascade + Facemark LBF de OpenCV (esquema de 68 puntos dlib/iBUG), NO mediapipe. Se probó mediapipe (primero la API legacy `mp.solutions.face_mesh`, luego la Tasks API `FaceLandmarker`) y ambas fallan en macOS: la legacy fue eliminada en mediapipe 1.0 (agosto 2026), y la Tasks API revienta con un crash nativo (`Check failed: service_ Service is unavailable` en DrishtiMetalHelper) al detectar la cara, incluso forzando `delegate=CPU` — parece un bug real de esa versión tan reciente. NO reintroducir mediapipe aquí sin verificar antes que ese bug se ha resuelto en una versión más nueva. Limitación conocida: el esquema de 68 puntos no tiene puntos de frente, así que `hairline_points` es una extrapolación aproximada desde las cejas, no una detección real (ver TODO en `_estimate_hairline`). Requiere descargar el modelo una vez con `python -m app.pipeline.download_landmark_model`.
2. **`hair_segmentation.py`** — Máscara de segmentación del pelo actual del cliente (separar pelo de cara/fondo/ropa). YA IMPLEMENTADO con un modelo real: BiSeNet (ResNet-18) entrenado sobre CelebAMask-HQ, vendorizado en `bisenet/` desde https://github.com/zllrunning/face-parsing.PyTorch (licencia MIT, permite uso comercial — se descartó `jonathandinu/face-parsing` de Hugging Face por tener licencia no comercial, y `Allison/segformer-hair-segmentation-10k-steps` por no tener model card ni métricas documentadas). Requiere descargar los pesos una vez con `python -m app.pipeline.download_weights` (están en Google Drive, no en el repo). De propina, este modelo también da máscaras de orejas/gafas/cuello/ropa (`segment_face_parts()`), útiles para `compositor.py`.
3. **`hair_type.py`** — Clasifica textura (liso/ondulado/rizado/afro, escala Andre Walker), densidad y grosor. PLACEHOLDER, y con un techo estructural comprobado (no solo falta de calibración): probé erosionar la máscara (para descartar ruido del borde) y difuminar la imagen (para descartar sensibilidad a la resolución/distancia de la foto) sobre un caso real que fallaba, y ninguna de las dos hipótesis explicaba el error — el pelo despeinado/con mechones sueltos en varias direcciones produce la misma dispersión de bordes alta que un rizo apretado, porque la métrica mide "caos direccional", no tamaño/periodicidad de rizo. Conclusión: no seguir ajustando umbrales a ciegas, es una limitación real de usar una sola característica hecha a mano. **Por eso `POST /api/simulate` acepta `manual_hair_texture`, con prioridad sobre la heurística y sobre cualquier corrección guardada — es la forma recomendada de usar la app**: el peluquero mira al cliente y lo indica él mismo, en vez de confiar en la detección automática. La heurística se sigue calculando siempre igualmente (log `[debug] hair_type: dispersión de orientación = ...`), porque el valor "en crudo" queda guardado en el historial de cada visita junto con el valor realmente usado — es la comparación entre ambos, acumulada con el tiempo, la que podría servir de dato de verdad para entrenar un clasificador (ver sección de dataset propio más abajo).
4. **`head_mesh.py`** — Reconstrucción aproximada de la geometría 2D de la cabeza + mapa de dirección de crecimiento (trazos + remolinos/"whorls"). PLACEHOLDER: `build_default_growth_map()` genera un patrón estándar (corona/flequillo/laterales/nuca) a partir de los landmarks; el barbero puede sustituirlo por completo dibujando el suyo a mano en `frontend/growth-map.html`, que llama a `apply_custom_growth_map()`. Coordenadas siempre normalizadas 0.0-1.0 (no píxeles), para que un mapa dibujado una vez valga independientemente de la resolución/encuadre de cada foto futura. No hay reconstrucción 3D real todavía, es un mapa 2D sobre la imagen.
5. **`style_catalog.py`** — Carga y consulta el catálogo de cortes (`data/styles/styles.json`).
   **Catálogo ampliado con el ranking de Esquire (2023)**: además de los 4 estilos de ejemplo originales, `data/styles/styles.json` incluye 100 cortes importados del artículo de Esquire España "100 peinados de hombre modernos y actuales para 2023" (el usuario pegó el texto completo en el chat porque la web está bloqueada para las herramientas de scraping; ver nota de licencia/proceso más abajo). Se importaron con `python -m app.pipeline.import_esquire_styles` (idempotente, no duplica si se re-ejecuta) — el script (`backend/app/pipeline/import_esquire_styles.py`) documenta en su docstring exactamente qué es dato real del artículo (nombre/descripción) y qué es estimación propia (longitud en mm, `fade_type`, tipos de pelo cuando el texto no lo menciona explícitamente). Solo se guardó texto/clasificación, NUNCA imágenes: el artículo solo da créditos de foto tipo "© Zara"/"Getty Images", no URLs descargables, y este proyecto ya había decidido antes (caso Figaro-1k) no incorporar fotos de personas identificables sin licencia confirmada. Dos campos nuevos en `HaircutStyle` para esto: `length_category` ("corto"/"medio"/"largo"/"extra_largo", la clasificación por longitud que pidió el usuario) y `source` (de dónde sale cada corte, para poder auditar/corregir clasificaciones concretas). Pendiente real: la clasificación de tipo de cabello de estos 100 es una aproximación de partida hecha leyendo cada descripción (la mayoría no menciona textura de pelo) — conviene que un barbero real las revise/corrija según vaya usando el catálogo, no tomarlas como definitivas.
6. **`color_transfer.py`** — Extrae el color de pelo real del cliente o aplica un color nuevo respetando el tono de piel.
7. **`generator.py`** — Genera la imagen final: difusión + ControlNet condicionado por (máscara de pelo + landmarks + parámetros del corte elegido). Aquí es donde vive el "hiperrealismo".
8. **`compositor.py`** — Blending final: bordes, oclusiones (orejas, gafas, cuello), igualar grano/iluminación con la foto original.

La API (`backend/app/api/routes.py`) orquesta estas etapas en `POST /api/simulate`.

## Perfil de cliente + historial (`app/db/`, `app/api/clients_routes.py`)

Sistema OPCIONAL (opt-in) para que un cliente que vuelve no tenga que empezar de cero cada vez, y para ir acumulando datos propios reales que en el futuro puedan servir para entrenar un clasificador de tipo de pelo (ver conversación sobre por qué NO usar datasets scrapeados de internet: problemas de licencia/consentimiento sobre personas identificables — Figaro-1k, por ejemplo, no se puede usar comercialmente).

- Base de datos: SQLite en `backend/data/clients.db` (se crea sola al arrancar la app, `init_db()` en `main.py`). Suficiente para el volumen de una barbería; toda la lógica de acceso a datos vive en `app/db/repository.py`, así que migrar a Postgres más adelante no debería tocar las rutas de la API.
- Tablas: `clients` (perfil + consentimientos + correcciones manuales del barbero) y `visits` (historial de simulaciones de ese cliente).
- Consentimientos, deliberadamente separados (ver sección RGPD arriba): `consent_history` (obligatorio para crear el perfil), `consent_model_improvement` (opcional), `consent_save_photo` (opcional, por defecto NO se guardan fotos ni con perfil creado).
- Endpoints (`POST/GET /api/clients`, `GET /api/clients/{id}`, `PATCH /api/clients/{id}/hair-type`, `.../face-shape`, `.../growth-map`): el barbero corrige aquí el resultado del pipeline cuando se equivoca (p.ej. el caso real de rizado detectado como afro). Esas correcciones quedan guardadas en el perfil. `.../growth-map` recibe SIEMPRE el estado completo de la escena 3D de `frontend/growth-map.html` (lista de trazos + remolinos, cada uno con x,y,z reales sobre la cabeza genérica), no un delta — sustituye lo que hubiera antes.
- **Nota de licencia (RESUELTO)**: se buscó primero usar un modelo 3D de cabeza descargado de Sketchfab, pero no se pudo confirmar que ninguno de esos candidatos tuviera licencia clara para uso comercial (el "Standard License" por defecto de Sketchfab normalmente no lo permite, y su badge de licencia no es verificable automáticamente — WebFetch da error ROBOTS_DISALLOWED contra su API y respuesta vacía contra la página del modelo). Se probó primero con geometría propia (esfera+cuello+orejas) sin ninguna duda legal pero poco realista. Después se encontró una fuente con licencia CC0 confirmada de forma verificable (el propio archivo fuente lo declara en texto plano, no solo una página web): el **base mesh oficial de MakeHuman** (`makehuman/data/3dobjs/base.obj`, https://github.com/makehumancommunity/makehuman), cuya cabecera dice literalmente: *"This asset was explicitly released as CC0 in september 2020"* (Copyright (C) 2020 Data Collection AB / Joel Palmius / Jonas Hauquier). Confirmado además por la comunidad de MakeHuman (FAQ oficial: "the asset license is CC0... no restriction on commercial use, modification or redistribution"). Se descargó ese `base.obj` (malla completa de cuerpo, en pose neutra), se recortó solo la región de cabeza+cuello (grupo `body` + `helper-l-eye`/`helper-r-eye` del OBJ, filtrando por altura para excluir hombros/torso y los grupos `joint-*`/otros `helper-*` que son ayudas de rigging, no geometría visible), se recompusieron normales y se cerró el hueco inferior del cuello con `trimesh` (Python), y se exportó a `frontend/assets/head.glb`. `frontend/growth-map.html` la carga con `THREE.GLTFLoader` en vez de la geometría procedural anterior; el raycasting para dibujar flechas/remolinos usa ahora las mallas reales del modelo cargado (`headGroup.traverse` → `raycastTargets`). No requiere atribución (CC0), pero se documenta aquí la procedencia exacta por trazabilidad, igual que se hizo con BiSeNet.

  **Iteración de realismo (v2, tras el primer recorte)**: la primera versión (~156 KB, solo la malla `body` recortada, sin ojos) se veía correcta en forma pero bastante poligonal/plana y con las cuencas de los ojos vacías. Se mejoró en `head.glb` (ahora ~620 KB, sigue siendo el mismo base mesh CC0 de MakeHuman, sin texturas nuevas ni geometría añadida de fuera):
  - **Suavizado**: subdivisión Loop (1 iteración, `trimesh.remesh.subdivide_loop`) + suavizado de Taubin (`trimesh.smoothing.filter_taubin`, preserva volumen a diferencia de un suavizado Laplaciano simple) sobre la malla de piel, para quitar el aspecto poligonal/bajo-poli del base mesh original.
  - **Cuello cerrado**: el hueco inferior del cuello (borde abierto tras el recorte por altura) se tapa con un abanico de triángulos hacia el centroide del borde, calculado a mano (el `fill_holes()` automático de trimesh no cerraba bien este borde concreto) — la malla de piel es ahora watertight.
  - **Ojos como pieza separada**: se exportan `helper-l-eye`/`helper-r-eye` del OBJ original como dos mallas independientes (`eye_l`, `eye_r`) en vez de fusionarlas con la piel, para poder pintarlas con un material distinto (más claro y con algo de brillo) en vez del mismo tono de piel plano — antes las cuencas de los ojos no se distinguían de la piel.
  - **`frontend/growth-map.html`** distingue el material por nombre de malla (`child.name.startsWith("eye")` → `eyeMaterial`, si no → `skinMaterial`) y usa iluminación de 3 puntos (`HemisphereLight` de base + luz principal + relleno + contraluz tenue) en vez de una sola luz ambiental plana, para que se noten mejor los rasgos de la cara.
  - Si se quiere ir más allá (textura de piel con poros, cejas/pestañas, iris con color), se puede repetir este mismo proceso con más piezas/target de MakeHuman — pero cuidado con el peso: cada mejora de geometría (sin texturas) ya cuesta varios cientos de KB embebidos como base64 en el HTML.

  **Iteración de realismo (v3)**: hecho justo lo que apuntaba el punto anterior, con dos cambios de fondo más:
  - **Ya no va embebido como base64**: `growth-map.html` ahora carga `assets/head.glb` con `fetch` normal (`GLTFLoader().load("assets/head.glb", ...)`), no como data URI dentro del HTML. La razón de usar data URI (evitar que el navegador bloquee `fetch`/XHR a archivos locales al abrir el HTML con `file://`) ya no aplica: la app se sirve siempre por HTTP/HTTPS de verdad desde que existe el despliegue en la nube (ver sección más abajo). Esto también quita la presión de peso que limitaba cuánto mejorar la malla antes.
  - **Pestañas reales**: los grupos `helper-l-eyelashes-*`/`helper-r-eyelashes-*` que ya traía `base.obj` (parte del mismo base mesh CC0, sin ningún archivo nuevo) se exportan como una malla `eyelashes` aparte, con un material plano oscuro (no hace falta textura).
  - **Iris con color de verdad**: se usa `makehuman/data/eyes/high-poly/high-poly.obj` + su textura `brown_eye.png` (mismo repo CC0) en vez del material plano `eyeMaterial` de antes. Ese `.obj` trae en realidad DOS esferas casi concéntricas por ojo: la de dentro con la textura real (esclerótica + iris) y una "cáscara" de fuera con el UV puesto a propósito sobre un círculo azul liso de la esquina de esa textura -- es la córnea transparente del shader propio de MakeHuman (litsphere + alpha, ver `brown.mhmat`: `transparent True`), que sin ese shader sale opaca y tapa el iris por completo. Se descartan esas caras (las que tienen las tres coordenadas UV en la esquina `u>0.8, v<0.2`) al exportar; un ojo sin refracción de córnea es más que suficiente aquí.
  - **Suavizado**: 1 iteración de subdivisión Loop + Taubin sobre la piel (igual que v2, no se sube a 2 aunque ya no haya presión de base64: con 2 la malla ronda 140k triángulos, demasiado para una tablet de gama media/baja -- el límite real ahora es fluidez en el navegador, no peso de descarga).
  - **Color de piel**: sigue sin haber ninguna textura de piel reutilizable (el material `DefaultSkin` de MakeHuman usa su propio shader -- litsphere + auto-blend de tono de piel, no una imagen simple -- así que no hay ningún PNG de piel con licencia clara para reusar, ver `default.mhmat`). En su lugar se calculan colores por vértice (`COLOR_0`) a partir de la posición (sonrojo sutil en pómulos, sombra bajo la mandíbula) directamente en el `.glb`, para que cualquier visor lo vea igual sin depender de JS.
  - Script de construcción: no quedó guardado en el repo (se generó en una sesión de trabajo puntual); si hace falta rehacerlo, el proceso es: `git clone` (sparse-checkout) de `makehuman/data/{3dobjs,eyes}`, recorte por altura + `trimesh` igual que v1/v2, filtrar las caras de córnea del ojo por su UV, y exportar con `pygltflib` (necesario para poner manualmente el material de la piel con vertex colors -- `trimesh` no deja asignar un `PBRMaterial` normal a una malla que ya lleva `COLOR_0`, se queda con el material por defecto del glTF, que sale metálico).
- `POST /api/simulate` acepta ahora un `client_id` opcional: si se pasa, antes de generar aplica las correcciones guardadas del cliente (tipo de pelo, forma de cara, remolinos) por encima de lo que detecte el pipeline esa vez, y al final registra la visita en su historial (con foto solo si `consent_save_photo=true`).
- También acepta `manual_hair_texture` opcional: el peluquero elige el tipo de pelo en el momento (selector en `frontend/index.html`), con prioridad sobre la heurística Y sobre la corrección guardada del perfil. Si hay `client_id`, esa elección se guarda automáticamente como la nueva corrección del cliente — no hace falta llamar aparte a `PATCH /api/clients/{id}/hair-type` para el caso normal de uso.
- El historial de cada visita guarda POR SEPARADO `detected_hair_texture` (lo que dijo la heurística automática, sin tocar) y `used_hair_texture` (lo que se usó de verdad, tras aplicar corrección/selección manual) — esa comparación es el dato real que en el futuro podría alimentar un clasificador entrenado.
- Pendiente: no hay UI en el frontend todavía, solo API. Tampoco hay ningún mecanismo que exporte este historial como dataset de entrenamiento — eso es un paso futuro deliberadamente no implementado aún.

## Motor de reglas de visagismo (`app/pipeline/visagismo_rules.py`)

Capa opcional por encima de `recommend_styles` (ver más arriba) que añade
un perfil de visagismo mucho más detallado que `face_shape_override`:
morfología craneal, geometría/proporción facial, forma de nacimiento del
pelo, remolinos (descripción rápida, sin coordenadas -- no confundir con
el mapa de crecimiento 3D de `growth-map.html`) y estilo de vida
(mantenimiento diario, frecuencia de visitas, entorno profesional,
preferencia de barba). Esquema completo en `VisagismoProfileIn`
(`app/api/schemas.py`) y persistencia en `ClientProfile.visagismo_profile`
(columna `visagismo_profile` de `clients`, JSON, migrada igual que
`custom_growth_map` -- ver `database.py`). Se guarda/lee con
`PATCH /api/clients/{id}/visagismo-profile`, siempre el perfil completo,
igual que `.../growth-map`.

Origen: el usuario pegó un JSON con el esquema de perfil y 5 reglas de
inferencia ya redactadas (morfología craneal, geometría facial, entradas
en M, mantenimiento diario, frecuencia de visitas). Ese JSON usa conceptos
de visagismo bastante abstractos y vocabulario en inglés no directamente
alineado con este catálogo (12 `style_family`, 5 `fade_type`); las 5
reglas se implementaron en `visagismo_rules.py` traduciendo cada concepto
abstracto a la combinación concreta de `style_family`/`fade_type`/
`length_top_mm` más parecida en `data/styles/styles.json`, documentando
en el propio código cada mapeo y, cuando un término del schema (p.ej.
"fringe_straight", "slick_back") no tiene ningún equivalente razonable en
este catálogo de 12 familias, dejando esa mitad de la regla sin
implementar en vez de inventar un mapeo forzado -- mejor ser honesto en
el código sobre el límite del catálogo actual que fingir una cobertura
que no existe.

**Decisión de diseño importante**: el JSON original describe algunas
reglas con lenguaje de exclusión dura ("bypass", "avoid", "restrict"). Se
implementaron TODAS como ajuste de orden (puntuación + nota explicativa),
nunca como descarte real, para mantener la misma filosofía que ya tenía
`recommend_styles` con remolinos/forma de cara: el barbero decide, la app
solo sugiere. Las reglas que el JSON expresa como exclusión dura usan una
puntuación más fuerte (en la práctica casi siempre acaban al final de la
lista) en vez de desaparecer, para no arriesgarse a dejar a un cliente
sin ninguna recomendación visible por un fallo de mapeo en este puente
igualmente heurístico.

Tests: `backend/tests/test_visagismo_rules.py` (stdlib `unittest`, sin
dependencia nueva -- este proyecto no usa pytest). Es el primer test
automatizado del repo; se puede ejecutar con
`cd backend && python3 -m unittest discover -s tests`.

Nota RGPD (ver sección dedicada más abajo): este perfil es un desglose
más fino de datos biométricos que `face_shape_override`/
`hair_texture_override`. Vive en la misma fila de `clients` y por tanto
bajo el mismo `consent_history` que el resto del perfil -- no se añadió
un consentimiento aparte porque es la misma finalidad de tratamiento
(recomendar cortes a ESE cliente), pero conviene tenerlo en cuenta al
decidir cuánto detalle rellenar para un cliente real.

Pendiente: no hay UI en el frontend todavía para rellenar este perfil
(igual que el resto de `clients_routes.py`, ver nota al final de la
sección anterior), solo API.

## Roadmap sugerido (por fases, no lo hagas todo a la vez)

**Fase 1 — Pipeline visible de extremo a extremo (sin generación real todavía)**
- Verificar que `face_analysis.py` detecta landmarks correctamente en fotos reales variadas.
- ~~Sustituir el placeholder de `hair_segmentation.py` por un modelo real~~ HECHO: ver BiSeNet en `bisenet/`. Pendiente: descargar los pesos (`python -m app.pipeline.download_weights`) y probarlo con fotos reales variadas (distintos tonos de piel, tipos de pelo, canas) para detectar fallos antes de dar por buena la máscara.
- Ampliar `data/styles/styles.json` con el catálogo real de la barbería (pedir al usuario las fotos/descripciones de sus cortes).

**Fase 2 — Generación**
- Integrar un modelo de difusión con ControlNet en `generator.py` (empezar con Stable Diffusion + ControlNet de tipo "canny" o "normal map" derivado de la máscara de pelo + landmarks).
- Iterar sobre el prompt/condicionamiento hasta que el pelo generado respete longitud por zona y tipo de pelo del cliente.

**Fase 3 — Realismo y producción**
- Mejorar `compositor.py` (oclusiones, iluminación, grano).
- Clasificador de tipo de pelo entrenado (sustituir heurística) — ver idea de dataset propio más abajo.
- ~~Corrección manual de remolinos en `head_mesh.py` para casos atípicos~~ HECHO: herramienta visual en `frontend/growth-map.html` — cabeza 3D navegable (Three.js + `GLTFLoader`, modelo real `frontend/assets/head.glb` recortado del base mesh CC0 de MakeHuman, ver nota de licencia más abajo) sobre la que el barbero rota la vista y dibuja flechas de dirección + remolinos horario/antihorario, persistido por cliente vía `PATCH /api/clients/{id}/growth-map`. Pendiente real: las coordenadas de este mapa son sobre la cabeza genérica (aunque ahora con forma realista), NO sobre la foto real del cliente — conciliar ambas requeriría reconstrucción 3D real de la cabeza a partir de la foto (ver nota de coordenadas en `head_mesh.py`).
- Frontend pulido para uso real en el mostrador de la barbería — de momento el perfil de cliente solo existe como API, sin UI.

## RGPD / privacidad (obligatorio, no opcional)

Las fotos de clientes son datos sensibles (biométricos) en España/UE. Antes de cualquier despliegue real:
- Consentimiento explícito del cliente antes de subir su foto.
- No almacenar fotos más tiempo del necesario para generar la simulación (por defecto, procesar en memoria y descartar).
- Si se decide guardar fotos (p.ej. para que el cliente vea su historial), documentar política de retención y borrado, y cifrar en reposo.

Desde que existe el perfil de cliente (ver sección siguiente) esto ya no es solo teórico: `POST /api/clients` guarda tipo de pelo/forma de cara/remolinos de forma persistente y por eso EXIGE `consent_history=true` para crear el perfil (la API lo rechaza si no). `consent_model_improvement` (usar los datos para mejorar el sistema) y `consent_save_photo` (guardar la foto en sí, no solo los rasgos derivados) son consentimientos separados y opcionales a propósito: son tres finalidades de tratamiento distintas y el RGPD exige un consentimiento específico por finalidad, no uno genérico que valga para todo. Cómo se recoge ese consentimiento en el mostrador de la barbería (checkbox en tablet, papel firmado, verbal registrado) es una decisión de producto todavía pendiente — la API solo modela que el consentimiento tiene que existir, no cómo se obtiene. No lanzar esto con clientes reales sin que alguien con conocimiento de RGPD revise el flujo completo.

## Despliegue en la nube (Railway)

Preparado para desplegarse sin Docker en Railway (ver sección "Despliegue
en la nube" del README para los pasos completos): `requirements-prod.txt`
(sin `diffusers`/`transformers`/`accelerate`/`controlnet-aux`/`gdown` —
Fase 2 del generador aún no implementada, ver `generate_haircut_preview`
en `generator.py`, así que esas dependencias pesadas no se usan en
producción todavía), `railway.json` (build/start command), y los pesos de
BiSeNet incluidos directamente en el repo (`backend/app/pipeline/bisenet/weights/79999_iter.pth`,
NO gitignored a propósito, a diferencia de otros `.pt`/`.ckpt` — ver
`.gitignore`) en vez de descargarlos con `gdown` en cada despliegue.

⚠️ Importante para RGPD (ver sección de arriba): desplegar en un hosting
externo significa que, si en algún momento se usa con clientes reales
(perfil + historial, `consent_history=true`), esos datos biométricos
viajarían y se guardarían fuera del propio local de la peluquería, en la
infraestructura del proveedor de hosting elegido. Añade un **Volume**
persistente montado en `/app/backend/data` (si no, `clients.db` se borra
en cada redespliegue) y revisa la política de privacidad/retención del
proveedor antes de usarlo con clientes reales, no solo en pruebas.

`frontend/manifest.webmanifest`, `frontend/sw.js` y `frontend/assets/icons/`
hacen la web instalable como PWA ("Añadir a pantalla de inicio") en tablets
iOS/Android — requiere HTTPS, por eso solo funciona bien una vez desplegado
en la nube, no en `http://192.168.x.x:8000` en local.

⚠️ **Caché del navegador tras cada despliegue (detectado al verificar el
despliegue de la v3 del modelo de cabeza)**: `StaticFiles` no manda cabecera
`Cache-Control`, así que el navegador cachea `growth-map.html`/`head.glb`
por heurística (basada en `Last-Modified`) y puede seguir sirviendo una
copia vieja después de un `git push` + redespliegue en Railway, incluso con
recarga forzada (Ctrl+Shift+R / Cmd+Shift+R): el `fetch(event.request)` de
`frontend/sw.js` no siempre hereda el "ignora caché" de esa recarga. Se
arregló con un middleware en `backend/app/main.py` (`_no_stale_cache`) que
añade `Cache-Control: no-cache` a todo lo que no sea `/api/*`: el navegador
sigue guardando copia local pero SIEMPRE revalida con el servidor
(`If-None-Match`) antes de usarla, así que un despliegue nuevo se ve al
momento sin perder los 304 baratos cuando no ha cambiado nada. Importante
en tablets de peluquería que se quedan con la pestaña/PWA abierta días
enteros entre despliegues.

**Iteración de zonas/brújula (tras feedback de uso real)**: la primera
versión del sistema por zonas tenía dos problemas detectados al usarlo:
(1) los marcadores de "flequillo" y "laterales" caían sobre la piel de la
cara (entre las cejas / en la mejilla, respectivamente) en vez de sobre el
cuero cabelludo real, porque sus vectores de dirección (usados para
lanzar un rayo desde el centro de la cabeza) no eran anatómicamente
correctos aunque fueran simétricos; y (2) fijar la dirección arrastrando
el dedo directamente sobre la cabeza 3D era difícil de hacer con
precisión en tablet. Se arreglaron los dos por separado:

- Los vectores de `HEAD_ZONES` (duplicados en `frontend/growth-map.html`
  y `backend/app/pipeline/head_mesh.py`, ver el aviso ya existente sobre
  mantenerlos sincronizados a mano) se reajustaron a ojo comparando
  capturas del maniquí desde varios ángulos hasta que cada marcador cae
  sobre pelo de verdad: coronilla en la parte más alta, flequillo justo
  encima de las cejas (línea de nacimiento del pelo), laterales encima de
  la oreja (sien), nuca en el nacimiento del pelo de la nuca.
- El arrastre sobre la cabeza 3D para fijar la dirección de cada zona se
  sustituyó por una brújula 2D (`#compass-panel` / `#compass-svg`, disco
  con aguja) que aparece siempre en el mismo sitio de la pantalla (encima
  de la cabeza, a la derecha) al armar una zona: es un control de tamaño
  fijo, no depende del ángulo de cámara ni de acertar sobre la piel, y
  tiene imán a los 8 puntos cardinales (cada 45°) para poder apuntar
  "justo hacia delante/atrás/al lado" sin pulso perfecto. El ángulo 0° de
  la brújula siempre corresponde a la dirección "natural" de esa zona
  (el propio `zone.direction` de HEAD_ZONES proyectado sobre el plano
  tangente a la piel en ese punto), así que gira de forma intuitiva por
  zona en vez de usar un eje arbitrario del mundo (ver
  `directionForAngle()`/`angleFromDirection()`/`applyZoneAngle()` en
  `growth-map.html`). El esquema de guardado no cambió: se sigue mandando
  un `start`/`end` en 3D por zona, así que no hace falta tocar el backend
  más allá de los vectores de HEAD_ZONES.

**Refinamiento del arreglo de caché (el mismo día, al comprobar el
despliegue de las zonas/brújula de arriba)**: el middleware
`_no_stale_cache` de `backend/app/main.py` no bastaba por sí solo.
`frontend/sw.js` reenviaba cada petición con `fetch(event.request)` a
secas, y esa llamada puede seguir usando el modo de caché "default" del
navegador aunque la petición original fuera una recarga forzada del
usuario -- así que el navegador podía reutilizar una copia ya cacheada
sin preguntarle nunca al servidor, sin llegar a ver siquiera la cabecera
`Cache-Control: no-cache`. Se arregló reconstruyendo la petición dentro
del service worker con `new Request(event.request, { cache: "no-store" })`
antes de reenviarla, que es lo que de verdad fuerza que cada petición
vaya a la red sin pasar por la caché HTTP en ningún sentido. Un
dispositivo que ya hubiera cacheado una copia vieja ANTES de este
arreglo puede necesitar una recarga forzada (o borrar datos del sitio)
una única vez para des-atascarse; a partir de ahí ya no debería volver a
pasar.

## Cómo trabajar en este repo

- Instala dependencias: `pip install -r requirements.txt` (usa un entorno virtual).
- Arranca el backend: `cd backend && uvicorn app.main:app --reload --host 0.0.0.0` (los imports usan `app.X`, así que hay que ejecutar uvicorn desde dentro de `backend/`, no desde la raíz del repo). El `--host 0.0.0.0` deja el backend accesible desde otros dispositivos de la misma red WiFi (móvil, tablet), no solo desde `localhost` — la web está pensada para usarse también en pantalla táctil (ver `frontend/growth-map.html`: dibujo con Pointer Events, `meta viewport`, botones ≥44px).
- Abre http://localhost:8000/ en el navegador de este ordenador: el propio backend sirve toda la
carpeta `frontend/` como estático (ver el mount de `StaticFiles` al final de
`backend/app/main.py`) y `/` redirige a `frontend/inicio.html`, la portada
desde la que se llega a todas las páginas (simulador, mapa de remolinos,
catálogo, recomendaciones). Desde un móvil/tablet en la misma WiFi, en vez de
`localhost` usa la IP local del ordenador (`ipconfig getifaddr en0` en Mac),
ej. `http://192.168.1.23:8000/`.
- Cada módulo de `pipeline/` tiene una función principal clara y un docstring con su TODO. Antes de sustituir un placeholder por un modelo real, deja el placeholder anterior comentado o en una rama, porque sirve para probar el resto del pipeline sin depender de modelos pesados.
- Prioriza que el pipeline funcione de extremo a extremo con placeholders simples ANTES de invertir tiempo en el modelo de generación final — así se puede validar la arquitectura completa rápido.
