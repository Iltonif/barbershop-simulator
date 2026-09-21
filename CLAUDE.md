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
- Consentimientos, deliberadamente separados (ver sección RGPD arriba): `consent_history` (obligatorio para crear el perfil), `consent_model_improvement` (opcional), `consent_save_photo` (opcional, por defecto NO se guardan fotos ni con perfil creado), `consent_ai_analysis` (opcional, exigido además para poder llamar a `POST /api/clients/{id}/visagismo-ai-report` -- ver sección dedicada más abajo).
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

**Grados de la brújula sin sentido real (tras feedback de uso real)**: se
detectó que el número de grados que muestra la brújula de dirección
(`frontend/growth-map.html`) no correspondía a ninguna dirección física
reconocible -- una flecha señalando claramente hacia la nuca podía leerse
como "346°" en vez de un valor con sentido. Causa: el "0°" de cada zona se
definía como el propio `zone.direction` de `HEAD_ZONES` proyectado sobre
el plano tangente a la piel en ese punto, pero `zone.direction` es
(por construcción) el mismo vector usado para lanzar el rayo que coloca
el marcador, así que en el punto exacto de impacto es casi paralelo a la
normal real de la superficie -- su residuo tangencial, una vez restada la
componente normal, es casi ruido numérico, no una dirección con
significado. Se arregló sustituyendo esa referencia por un eje FIJO y
compartido por las 5 zonas: `WORLD_FRONT` (0,0,1), "hacia la cara" --
el mismo eje que ya usa la cámara para la vista "front". Con una
referencia fija, 0°/90°/180°/270° significan siempre lo mismo (cara/
derecha/nuca/izquierda) en cualquier zona, verificado numéricamente
(un vector sintético "hacia atrás" da exactamente 180° en las 5 zonas)
y visualmente (la flecha dibujada al fijar 180° apunta de verdad hacia
la nuca vista desde perfil/detrás). El texto de la brújula ahora muestra
una etiqueta junto al número en los 4 puntos cardinales (p.ej. "180°
(hacia la nuca)") para que sea evidente que el número tiene un
significado real y no solo relativo a la zona. No hace falta ninguna
migración de datos: los trazos ya guardados (start/end en 3D) no
cambian, solo cambia qué número de grados se les asocia al reabrir esa
zona.

**Remolinos con el mismo bug de oclusión que los marcadores de zona**: al
revisar de nuevo la oclusión de puntos sobre la cabeza 3D (ver el punto
anterior sobre los marcadores de zona) se encontró que el icono de
remolino (`makeWhorlSprite`, el círculo con ↻/↺ que se coloca al tocar la
cabeza en modo remolino) tenía el mismo `depthTest: false` que ya se
había corregido en el marcador de zona, así que un remolino marcado en un
lateral o en la nuca seguía viéndose "flotando" sobre la cara al mirar la
cabeza de frente. Se corrigió de la misma forma (`depthTest: true`) y se
verificó con un test geométrico (raycasting contra la malla de la piel):
un remolino colocado en el lateral derecho queda oculto al ver la cabeza
desde el lado izquierdo, y visible desde el lado en el que se colocó.

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

## Análisis automático de rasgos faciales (`app/pipeline/facial_traits_analysis.py`)

Capa que rellena automáticamente parte de `facial_features_profile`
(dentro de `anatomical_metrics` en `visagismo_profile`, ver sección
anterior) a partir de 3 fotos guiadas del cliente -- frontal, perfil
izquierdo y perfil derecho --, en vez de depender solo de que el barbero
rellene esos campos a mano. **Hoy solo se analiza la foto frontal**: la
detección desde las fotos de perfil se probó con fotos reales y no
funciona con los modelos actuales (ver "Calibración de umbrales" más
abajo). Pedro decidió mantener igualmente las 3 fotos en el flujo, para
no cambiarlo si más adelante se incorpora un modelo que sí funcione de
perfil; el endpoint las recibe pero ni siquiera decodifica las de perfil.

Endpoint: `POST /api/clients/{id}/visagismo-auto-analysis`
(multipart con 3 ficheros: `photo_frontal`, `photo_perfil_izquierdo`,
`photo_perfil_derecho`). Frontend: `frontend/visagismo.html` (enlazado
desde `inicio.html` → "Análisis de visajismo" y desde el nav de
`growth-map.html`).

Origen: petición de Pedro de mejorar el análisis de visajismo con rasgos
de nariz, ojos, orejas, asimetrías ("un ojo más abierto que otro") y
gafas -- resuelta, tras dos rondas de preguntas, como detección
automática (no solo campos manuales) a partir de 3 fotos (sin vídeo),
asimetría expresada como porcentaje, gafas como dato meramente
informativo (no afecta a `visagismo_rules.py`), y RGPD "procesar y
descartar, igual que ahora" (mismo criterio que el resto del perfil, sin
consentimiento nuevo).

**Qué se detecta y cómo, reutilizando SOLO piezas ya existentes del
pipeline (cero dependencias/modelos nuevos)**:
- `face_analysis.analyze_face()` (68 landmarks dlib/iBUG, ya usado en el
  resto del pipeline) da `eye_spacing` (distancia entre lagrimales /
  ancho de cara, ver calibración más abajo).
- `eye_symmetry` + `eye_symmetry_percent`: diferencia porcentual del EAR
  (Eye Aspect Ratio, misma fórmula estándar de apertura ocular) entre
  ambos ojos en la foto frontal -- es el "% de simetría/anomalías" que
  pidió Pedro. Por encima del 20% (calibrado, ver más abajo) se marca
  `asymmetric` y se genera una nota en `detected_anomalies_notes` (p.ej.
  "el ojo derecho está aproximadamente un 24% más abierto que el
  otro"). Si la cabeza sale girada en la foto frontal, no se mide y se
  pide repetir la foto.
- `has_glasses`: **reutiliza el modelo BiSeNet**
  (`hair_segmentation.segment_face_parts`, MIT, ya vendorizado para
  segmentar el pelo), cuya salida de 19 clases (CelebAMask-HQ) ya incluye
  `eye_g` (gafas). Es solo informativo: se guarda y se muestra en el
  informe de IA (ver siguiente sección) pero **no** se usa en
  `visagismo_rules.py`, tal como pidió Pedro.
- **Campos manuales** (se probaron y se descartaron, ver calibración más
  abajo; la página lo indica junto a cada campo): `eyebrow_type` (forma
  de cejas), `profile_type` (perfil de nariz) y `ears_projection`
  (proyección de orejas). Para las orejas Pedro había elegido un
  "detector dedicado"; se usó la segmentación de orejas de BiSeNet en vez
  de un modelo nuevo (no se encontró ninguno maduro con licencia de uso
  comercial), y al probarla con fotos reales resultó no ser fiable.

**Fusión, no sustitución** (`merge_detected_features` en
`facial_traits_analysis.py`): el endpoint solo rellena los campos que el
barbero no hubiera puesto ya a mano en `facial_features_profile` -- una
detección automática nunca pisa una corrección manual previa. Esto es
distinto de `PATCH .../visagismo-profile`, que siempre sustituye el
perfil completo.

⚠️ Bug real corregido (sept 2026, encontrado al probar el endpoint de
extremo a extremo con fotos reales): la versión inicial usaba
`dict.setdefault()`, pero al guardar la ficha con `PATCH
.../visagismo-profile` Pydantic escribe TODOS los campos, con `null` en
los vacíos. Esas claves ya existían, así que `setdefault` no las
rellenaba: en cuanto la ficha de un cliente se había guardado una sola
vez, el análisis automático devolvía 200, sin avisos, y no guardaba
ningún resultado. Ahora `null`/`""` cuentan como vacío. Además
`has_glasses: false` también cuenta como vacío, porque la página lo
guarda con una casilla, que no distingue "no lleva gafas" de "sin
especificar". Y `eye_symmetry_percent` solo se escribe junto con
`eye_symmetry`, para no guardar un porcentaje medido que contradiga una
simetría puesta a mano. Consecuencia de diseño que se mantiene a
propósito: si se repite el análisis con otra foto, no cambia lo que ya
rellenó el anterior (el barbero lo corrige a mano).

**RGPD**: las 3 fotos se procesan en memoria dentro del propio endpoint y
se descartan justo después de extraer las categorías -- nunca se guardan
en disco ni en la base de datos, igual que ya hace `POST /api/simulate`
con la foto de simulación. Como el dato final persistido es el mismo
`visagismo_profile` que ya cubre `consent_history`, no hace falta ningún
consentimiento nuevo (a diferencia de `consent_save_photo`, que sería
para guardar la foto en sí -- cosa que este endpoint no hace -- o
`consent_ai_analysis`, que es solo para la llamada externa a la API de
Claude en `visagismo_ai_advisor.py`, no para este análisis 100% local).

Limitaciones honestas (documentadas también en el docstring del módulo):
heurísticas geométricas 2D con umbrales calibrados con fotos reales (ver
subsección siguiente); si no se detecta cara en la foto frontal o la
cabeza sale girada, se avisa sin romper nada; sin los pesos de BiSeNet
descargados (ver `python -m app.pipeline.download_weights` en la sección
de roadmap), las gafas se omiten con un aviso pero el resto del análisis
sigue funcionando.

Tests: `backend/tests/test_facial_traits_analysis.py` (stdlib
`unittest`, mismo criterio que el resto del repo -- sin dependencia
nueva). Cubre EAR, simetría ocular (incluido un caso del 18% que con el
umbral antiguo salía como asimetría), la guarda de cabeza girada,
separación de ojos, gafas (con mapas de segmentación sintéticos, sin
cargar BiSeNet), que las fotos de perfil ya no rellenan nariz, orejas
ni cejas, y la fusión con una ficha ya guardada (el bug de arriba).

### Calibración de umbrales (sept 2026)

Los umbrales originales se habían puesto "a ojo". Se calibraron los de
la foto frontal ejecutando el pipeline real (Haar + LBF + BiSeNet, el
mismo código que en producción) sobre 61 fotos frontales de un conjunto
público de pruebas (`tests/unit/dataset` del proyecto open source
`serengil/deepface`, solo para medir; no se ha guardado ninguna foto en
este repo). Varias fotos son de la misma persona, lo que permitió
separar el ruido de la medición (misma persona, distinta foto) de la
diferencia real entre personas -- una métrica solo sirve si lo segundo
es claramente mayor que lo primero. Además de mirar los números se
revisaron las fotos a mano, que es lo que destapó el problema de las
cejas.

| Rasgo | Antes | Después | Por qué |
|---|---|---|---|
| Simetría ocular | ≥15% | ≥20%, y no se mide si la cabeza está girada (giro > 0,15) | En caras normales la diferencia llega al 15% solo por expresión, pose y ruido (p95 = 10%). El giro de cabeza era la causa principal (correlación 0,46): el ojo lejano sale escorzado. Con el 15%, 1 de 61 caras normales salía "asimétrica"; ahora 0. |
| Separación de ojos | intercantal / ancho de ojo, <0,9 o >1,5 | intercantal / ancho de cara, <0,215 o >0,300 | La fórmula anterior no podía salir nunca "juntos" (0/61) y daba "separados" al 16% de caras normales. Además apenas distinguía entre personas (el ruido de una misma persona era el 85% de la variación total). La nueva es algo mejor (74%) pero sigue ruidosa, así que solo clasifica valores claramente extremos; lo normal cae en "proporcional". |
| Gafas | píxeles de gafas / imagen entera ≥0,004 | píxeles de gafas / píxeles de piel ≥0,05 | Funcionaba, pero dependía de lo cerca que estuviera la cámara. Normalizado por la piel: con gafas 0,21-0,22, sin gafas 0,003 como mucho. |
| Forma de cejas | curvatura ≥0,08 → arqueada | ya no se detecta | Con 0,08, las 61 caras salían "arqueadas". Y recalibrar no sirve: al revisar las fotos, la métrica va AL REVÉS de lo que ve una persona (cejas depiladas muy arqueadas salían rectas; cejas gruesas y planas salían arqueadas). Los 5 puntos de ceja del modelo LBF siguen una plantilla que no reproduce dónde está el pico real. Campo manual. |

Límite importante de esta calibración: ninguna de las 61 personas tiene
asimetría ocular real ni ojos especialmente juntos/separados, así que lo
que está comprobado es que ya NO hay falsas alarmas, no que los casos
reales se detecten. El 20% de asimetría se eligió además porque
corresponde aproximadamente a la diferencia que ya se ve a simple vista
(~2 mm sobre una apertura de ~10 mm), pero conviene confirmarlo con el
primer cliente real que la tenga.

**Nariz y orejas, probadas con las 3 fotos guiadas de Pedro** (frontal +
dos perfiles a ~90°, sept 2026). Resultado: no funcionan con los modelos
actuales, y no por los umbrales:

- **Perfil de nariz**: Haar + LBF no encuentran ninguna cara en las fotos
  de perfil, así que `profile_type` nunca llegaba a rellenarse. En una
  foto de 3/4 quizá sí la encontrarían, pero el propio giro deforma la
  geometría de la nariz que se quería medir (la desviación de la punta
  respecto a la línea entrecejo-mentón depende tanto del ángulo como de
  la nariz), así que no se ha seguido por ahí.
- **Orejas desde el perfil**: BiSeNet está entrenado con caras
  frontales. En las fotos de perfil etiquetó la barbilla y la boca como
  "oreja", y la oreja real como piel o pelo. El valor que daba el código
  anterior ("prominentes") salía de medir la barbilla.
- **Orejas desde la foto frontal** (se probó como alternativa, porque es
  desde donde se juzga si las orejas sobresalen): BiSeNet sí segmenta
  bien las orejas de frente, pero ninguna medida de "cuánto sobresalen"
  resultó fiable. Con los puntos 0/16 del contorno de la mandíbula como
  referencia, el modelo LBF colocaba esos puntos encima de las propias
  orejas justo en las personas con orejas más separadas (la de Pedro
  salía la más baja de las 62, aunque en la foto se ve que sobresalen).
  Con el borde de la máscara de piel como referencia, las dos orejas de
  una misma cara (casi idénticas en la realidad) daban valores sin
  ninguna relación (correlación izquierda-derecha -0,05, y la diferencia
  entre las dos orejas de una persona era tan grande como la diferencia
  entre personas). Además el pelo tapaba las orejas en 14 de las 61
  fotos.

Por eso `profile_type` y `ears_projection` pasan a ser manuales. Si se
retoma, hace falta otro modelo, no otro umbral: un detector/landmarks
que funcione de perfil (para la nariz) o una segmentación de orejas
entrenada también con vistas laterales.

Foto frontal de Pedro con los umbrales nuevos: simetría 0,5%
(simétrico), separación proporcional, sin gafas -- todo correcto.

Nota encontrada de paso: `bisenet/resnet.py` descarga los pesos ImageNet
de ResNet-18 desde `download.pytorch.org` cada vez que se construye el
modelo en una máquina sin esa caché, aunque luego `79999_iter.pth` los
sobrescribe por completo. En un servidor sin acceso a ese dominio la
segmentación (pelo, gafas) fallaría al arrancar. No se ha tocado.

Pendiente / no cubierto a propósito en `frontend/visagismo.html`: la
página nueva solo cubre `facial_features_profile` (los campos de esta
sección) -- `hair_physical_metrics`, `lifestyle_and_preferences`,
`cranial_morphology` y `facial_geometry` del resto de
`visagismo_profile` siguen sin tener UI dedicada (mismo estado "solo
API" que ya tenían antes de esta feature, ver nota al final de la
sección de reglas de visagismo).

## Guía de visajismo de perfil y cámara con marco (sept 2026)

**`frontend/guia-visagismo.html`** (enlazada desde `inicio.html` y desde el
nav de `visagismo.html`): guía para valorar a mano tres rasgos de perfil
con el mismo criterio -- perfil facial (recto/convexo/cóncavo), mentón
(retraído/equilibrado/prominente) y línea mandibular (definida/poco
definida) --, con el consejo de corte y barba habitual en visajismo para
cada categoría y una sección de cómo hacer las fotos de perfil (correcta
frente a los errores que cambian la medida: cámara por debajo, cabeza
inclinada, flequillo sobre la frente, cuello tapado, fondo del color de
la piel, 3/4 en vez de perfil). Pedro la pidió como "imágenes
artificiales de diferentes perfiles de hombre que sirvan como guía".

- Las ilustraciones NO son fotos ni dibujos a ojo: las genera
  `tools/generar_perfiles_guia.py`, que coloca los puntos de
  cada perfil (glabela, subnasal, labio inferior, pogonion, mentón, punto
  cervical) para que su ángulo sea exactamente el que ilustra, con las
  mismas definiciones que las referencias publicadas (convexidad de Legan
  8-16°, ángulo labio inferior-mentón de -5° a 15°, ángulo cervicomental
  105-120°), y comprueba al generar que cada dibujo mide lo que dice.
  Salida: `frontend/assets/guia/perfiles.json` (SVG ya listo, la página lo
  carga con `fetch`). Si se cambia el dibujo, volver a ejecutar el script.
- Los consejos de corte/barba vienen de fuentes de visajismo y barbería
  enlazadas al final de la propia página (Luc Vincent, Castlebeard, Beard
  Resource, el visagista Igor Cuts); son orientaciones, no reglas, y la
  página lo dice.
- La página deja claro que esos tres rasgos se rellenan a mano: la
  detección automática desde la foto de perfil sigue en pruebas en la
  rama `wip-perfil-automatico` (su CLAUDE.md explica en qué punto está y
  qué falta: calibrar con fotos de perfil de 10-15 personas).

**Cámara con marco en `frontend/visagismo.html`**: cada foto (frontal,
perfil izquierdo, perfil derecho) tiene un botón "Hacer foto con marco"
que abre la cámara a pantalla completa (`getUserMedia`) con la guía
superpuesta: óvalo + línea de ojos para la frontal, y la silueta del
perfil normal de la guía (misma `marco_perfil` del JSON) mirando hacia el
lado que toca. "Perfil izquierdo" = se ve el lado izquierdo de la cara,
así que la nariz apunta a la izquierda de la foto. Con la cámara frontal
la vista previa va en espejo y la silueta también se gira, para que la
foto guardada (sin espejo) quede con la nariz hacia el lado correcto.

- Indicador de nivel con el sensor del móvil (`deviceorientation`): avisa
  si el móvil apunta hacia arriba o hacia abajo o está torcido. Es el
  error que más cambió las medidas en las pruebas (cámara por debajo de
  la cara). En iOS hay que pedir permiso dentro del toque del usuario
  (`DeviceOrientationEvent.requestPermission`); si lo deniega, simplemente
  no hay indicador. El sensor mide la inclinación del móvil, no su altura:
  la altura sigue dependiendo de quien hace la foto.
- Se guarda el fotograma COMPLETO a hasta 2000 px, no solo lo que se ve
  en pantalla: recortar a la pantalla (cámara apaisada, pantalla en
  vertical) perdía resolución, y en la prueba la asimetría ocular medida
  subió de 0,5% a 7,1% solo por eso (con el fotograma completo, 0,8%).
- La foto hecha con la cámara (Blob) tiene prioridad sobre el archivo del
  selector; elegir un archivo la descarta.
- **Necesita HTTPS** (o `localhost`): `getUserMedia` no existe en
  `http://192.168.x.x`. En ese caso, o si se deniega el permiso, se avisa
  y se abre el selector de fotos normal (que en el móvil abre la cámara
  del sistema, sin marco). En la web publicada en Railway funciona.
- Probado con Playwright y una cámara simulada que emite las fotos de
  Pedro: encuadre de los tres tipos, aviso de nivel, captura, subida al
  backend real y formulario relleno con el resultado; y el caso sin
  cámara.

Arreglado de paso en la misma página: el botón "Guardar" reconstruía
`facial_features_profile` solo con los campos del formulario y borraba
los valores medidos sin campo propio (`eye_symmetry_percent`). Ahora parte
de lo ya guardado y sobrescribe solo los campos del formulario.

## Informe de visagismo por IA (`app/pipeline/visagismo_ai_advisor.py`)

Segunda capa opcional sobre el perfil de visagismo (además del motor de
reglas determinista de la sección anterior), pero de una naturaleza muy
distinta: en vez de traducir reglas simples if/then a código, aquí se le
pasa el perfil completo del cliente a un LLM (API de Claude) para que
razone de forma cualitativa sobre muchos rasgos a la vez y genere un
informe en lenguaje natural. Endpoint: `POST /api/clients/{id}/visagismo-ai-report`
(sin payload -- usa los datos ya guardados del cliente). No sustituye a
`visagismo_rules.py`: ese motor sigue matizando `recommend_styles` igual
que antes, este es un informe adicional bajo demanda, no una fuente de
verdad para el catálogo.

Origen: el usuario pegó un system prompt completo ya redactado ("Motor
Experto en Visajismo Masculino"), con un pipeline de razonamiento por
prioridades (restricciones óseas → micro-rasgos faciales → línea capilar
→ estilo de vida), un manual de compensación geométrica rasgo por rasgo
(frente, nariz, orejas, ojos/cejas, mandíbula) y un formato de salida
obligatorio de 5 secciones (diagnóstico morfológico, prescripción técnica
del corte, diseño de barba, guía de estilizado, y un prompt generador
para Midjourney/Stable Diffusion). Ese texto se usa TAL CUAL como
`system` de la llamada a la API (`SYSTEM_PROMPT` en
`visagismo_ai_advisor.py`), sin reescribir su contenido -- solo se le
añadió al final una nota explicando qué campos pueden faltar y que el
modelo no debe inventar medidas que no se le han dado.

**Por qué esto es distinto a todo lo demás en `app/pipeline/`**: es la
PRIMERA llamada del proyecto a un servicio externo de pago. Hasta ahora
todo el pipeline (segmentación, landmarks, catálogo, reglas de
recomendación) corre en local, sin salir del servidor ni tener coste por
petición -- esto rompe esa propiedad. Por eso:

- **Consentimiento separado**: `consent_ai_analysis` en `ClientProfile`
  (columnas `consent_ai_analysis`/`consent_ai_analysis_at` en `clients`,
  migradas igual que `visagismo_profile` -- ver `_MIGRATIONS` en
  `database.py`). Es una finalidad de tratamiento distinta a guardar el
  perfil en el propio servidor: aquí se transfieren datos a un tercero
  (Anthropic). El endpoint devuelve 422 si el cliente no tiene este
  consentimiento, con el mismo criterio que `create_client` exige
  `consent_history=true` -- pero el motivo es más fuerte todavía por la
  transferencia a terceros. Solo se puede fijar al CREAR el cliente (no
  hay un `PATCH` dedicado, igual que `consent_model_improvement` y
  `consent_save_photo` tampoco lo tienen hoy) -- si en la práctica hace
  falta activarlo para un cliente ya existente sin recrear su perfil,
  añadir ese `PATCH` es la extensión natural, deliberadamente no hecha
  todavía para no adelantarse a una necesidad real.
- **Nunca se envía la foto ni datos identificables**: solo los campos
  categóricos ya recogidos en `visagismo_profile`/`hair_texture_override`/
  `face_shape_override`/remolinos (`build_user_message` en
  `visagismo_ai_advisor.py`), igual que ya hace `visagismo_rules.py` con
  esos mismos datos. Nunca el nombre del cliente ni sus notas libres.
- **Configuración** (`app/config.py`): `ANTHROPIC_API_KEY` (obligatoria
  para que el endpoint funcione) y `ANTHROPIC_MODEL` (por defecto
  `claude-sonnet-5`, configurable sin tocar código si cambia el modelo
  disponible más adelante). Sin `ANTHROPIC_API_KEY`, el endpoint devuelve
  503 con un mensaje claro en vez de fallar de forma confusa o exponer un
  500 críptico -- el resto de la app funciona exactamente igual sin esa
  variable, no es obligatoria para arrancar. En Railway: Settings →
  Variables → añadir `ANTHROPIC_API_KEY` con una clave de
  https://console.anthropic.com/ (cuenta de pago del propio usuario).
- **Coste real por llamada**: cada informe generado consume tokens de
  pago de la API de Claude (hasta 2000 tokens de salida por informe, ver
  `max_tokens` en `generate_ai_report`) -- a diferencia de todo lo demás
  en este pipeline, que no tiene coste variable por petición.
- El texto que devuelve el modelo es una interpretación cualitativa de
  estética/peluquería, NO un diagnóstico médico real, a pesar del tono
  "clínico" del prompt original (términos como "evaluación de rasgos
  críticos" son terminología de peluquería, no medicina).

Errores manejados explícitamente (`clients_routes.generate_visagismo_ai_report`):
`AIAdvisorNotConfigured` (falta la API key o el paquete `anthropic`) →
503; `AIAdvisorError` (falla la llamada -- red, cuota, autenticación,
respuesta vacía) → 502. Ninguno de los dos expone la traza original del
SDK al barbero.

Tests: `backend/tests/test_visagismo_ai_advisor.py` (stdlib `unittest` +
`unittest.mock`, sin llamadas reales a la API -- se mockea
`anthropic.Anthropic` por completo, igual filosofía que el resto del
repo de no depender de red/credenciales para los tests).

Pendiente: no se persiste ningún informe generado (cada llamada genera
uno nuevo bajo demanda, sin guardar historial) -- si en el futuro se
quiere que el barbero pueda volver a ver un informe ya generado sin pagar
de nuevo por él, haría falta una tabla nueva (algo como `ai_reports`,
con su propio `created_at`) y decidir cuánto tiempo conservarlo (RGPD:
esto ya no serían solo rasgos categóricos del cliente, sería el texto
completo generado sobre él). Tampoco hay UI en el frontend todavía
(mismo estado que el resto de `clients_routes.py`).

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

Desde que existe el perfil de cliente (ver sección siguiente) esto ya no es solo teórico: `POST /api/clients` guarda tipo de pelo/forma de cara/remolinos de forma persistente y por eso EXIGE `consent_history=true` para crear el perfil (la API lo rechaza si no). `consent_model_improvement` (usar los datos para mejorar el sistema), `consent_save_photo` (guardar la foto en sí, no solo los rasgos derivados) y `consent_ai_analysis` (enviar el perfil de visagismo a un servicio externo de pago para generar un informe con IA, ver sección dedicada más abajo) son consentimientos separados y opcionales a propósito: son cuatro finalidades de tratamiento distintas y el RGPD exige un consentimiento específico por finalidad, no uno genérico que valga para todo. `consent_ai_analysis` es el único de los tres que implica además una TRANSFERENCIA de datos a un tercero (Anthropic), no solo un tratamiento adicional en el propio servidor -- tenerlo en cuenta al redactar el texto de consentimiento real que vea el cliente. Cómo se recoge ese consentimiento en el mostrador de la barbería (checkbox en tablet, papel firmado, verbal registrado) es una decisión de producto todavía pendiente — la API solo modela que el consentimiento tiene que existir, no cómo se obtiene. No lanzar esto con clientes reales sin que alguien con conocimiento de RGPD revise el flujo completo.

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

⚠️ **El Volume de arriba tapa TODO lo que haya dentro de `backend/data/`,
no solo `clients.db` (incidente real en producción, sept 2026)**: el
catálogo de cortes vivía antes en `backend/data/styles/styles.json` --
versionado en git, se actualiza con cada commit -- pero al estar DENTRO
de la misma carpeta que el Volume monta para persistir `clients.db`,
Railway lo "tapaba" con la instantánea que guardó la primera vez que se
creó ese Volume, ignorando cualquier actualización posterior del fichero
por mucho que se subiera a git y se desplegara. Sintoma real: `GET
/api/clients/{id}/recommendations` devolvía 500 en producción (mientras
que en local, sin Volume de por medio, funcionaba perfecto) porque
`load_catalog()` (`style_catalog.py`) intentaba construir `HaircutStyle`
a partir de una copia del catálogo de antes de que existiera el campo
`style_family`, con una forma distinta a la que espera el dataclass hoy.
Se corrigió por dos vías:
1. Se movió el catálogo FUERA de `data/`, a
   `app/pipeline/catalog_data/styles.json` (`STYLES_CATALOG_PATH` en
   `app/config.py`) -- cualquier fichero que deba actualizarse con cada
   despliegue no puede vivir dentro de una carpeta cubierta por un Volume
   persistente, solo lo que debe SOBREVIVIR a los despliegues (aquí,
   únicamente `clients.db` y `client_photos/`) debería estar ahí.
2. `load_catalog()` ahora ignora claves del JSON que no sean un campo
   conocido de `HaircutStyle` en vez de reventar con `TypeError` -- una
   sola entrada con forma inesperada ya no puede tirar abajo la lista de
   recomendaciones para TODOS los clientes; ver
   `backend/tests/test_style_catalog.py`.

Lección para el futuro: antes de guardar cualquier fichero nuevo dentro
de `backend/data/`, pensar primero si ese fichero debe persistir entre
despliegues (va ahí) o si viene de git y se actualiza con cada uno (va
en otro sitio, como `app/pipeline/catalog_data/`).

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

**`API_BASE` fijo a `localhost:8000` roto en producción (sept 2026)**:
`index.html`, `growth-map.html`, `catalogo.html` y `recomendaciones.html`
tenían `const API_BASE = "http://localhost:8000/api";` copiado y pegado en
cada página. Funcionaba en local, pero una vez desplegado en Railway
cualquier `fetch(`${API_BASE}/...`)` intentaba conectar al propio
ordenador del cliente (`localhost`) en vez de al backend real, así que la
web en producción daba "Error de red: Failed to fetch" en todas las
páginas que llaman a la API (corte, catálogo, etc.) aunque el backend
funcionara perfectamente. Se corrigió cambiando las 4 apariciones a
`const API_BASE = "/api";` (ruta relativa): como el propio backend sirve
el frontend como estático desde el mismo origen (ver más abajo), una ruta
relativa apunta siempre al sitio correcto tanto en local
(`http://localhost:8000/api`) como en producción
(`https://<dominio-railway>/api`), sin mantener una URL distinta por
entorno.

**Catálogo de cortes con foto propia, casi 1 foto por corte (sept
2026)**: hasta ahora los 104 cortes de `styles.json` compartían solo 12
fotos (`frontend/assets/style_photos/*.jpg`) agrupadas por
`style_family` en `catalogo.html` — hasta 23 cortes distintos llegaban a
mostrar la misma imagen de "buzz cut". A petición de Pedro ("necesito un
catálogo extenso con los cortes de pelo reales y su correspondiente
foto", cobertura máxima) se sustituyó por una foto real y distinta para
cada uno de los 104 `id`:

- Origen de las fotos: búsqueda por texto en la API interna (sin clave,
  de uso público) de Unsplash, con una consulta en inglés generada a
  partir del `name`/`description` en español de cada corte (reglas de
  traducción de palabras clave: mullet, tazón → bowl cut, rastas →
  dreadlocks, moño → man bun, degradado → fade, flequillo → fringe,
  etc., más el largo en mm para anteponer short/medium/long). Las fotos
  de Unsplash usan la Unsplash License (uso comercial y no comercial
  gratuito, sin atribución obligatoria), así que no hay riesgo de
  derechos de autor.
- Deduplicación global: como consultas distintas a veces devuelven la
  misma foto en primer lugar, se llevó un registro de ids de foto ya
  usadas across TODAS las búsquedas (no solo dentro de cada una), para
  que dos cortes no acaben compartiendo foto salvo que de verdad sea el
  mismo corte repetido.
- Revisión manual: tras descargar las 104 fotos se hizo una pasada
  visual conjunta y se repitió la búsqueda a mano para las pocas que
  salieron mal emparejadas según el texto alternativo de Unsplash
  (describían a una mujer en vez de un hombre, o no se veía claramente
  un corte de pelo).
- `frontend/catalogo.html` (`groupByFamily()`): ahora agrupa primero por
  `reference_image` en vez de por `style_family`. Como cada corte tiene
  hoy su propia foto, esto deja en la práctica un grupo de un único
  corte por tarjeta, con el nombre propio de ese corte como etiqueta
  (`family.label = family.items[0].name`) en vez de la etiqueta
  genérica de familia. El agrupado por `style_family`/`FAMILY_LABELS`
  se mantiene como fallback del código, por si en el futuro se añaden
  cortes que vuelvan a compartir una misma foto a propósito.

**Corrección de 34 fotos mal emparejadas + rediseño de tarjetas y zoom
(sept 2026)**: tras la pasada anterior, Pedro reportó que "la mayoría de
cortes no coinciden con las fotos, además de no poder ampliarse y ser un
catálogo un poco cutre". Se hicieron las dos cosas que pidió:

- **Corrección de fotos**: revisión visual corte por corte de los 104
  contra su nombre/descripción real (no solo comprobar que aparece un
  hombre, sino que el peinado coincide: raya, flequillo, longitud,
  rizado/liso, mechas, etc.). 34 de los 104 no encajaban y se volvieron a
  buscar en Unsplash con consultas más específicas (a veces 2-4 intentos
  por corte hasta encontrar una coincidencia razonable), manteniendo el
  mismo proceso que la pasada anterior (Unsplash License, deduplicación
  global de ids de foto). En un puñado de cortes muy concretos sin foto
  de stock exacta (p.ej. "puntas decoloradas" en una melena) se aceptó la
  aproximación más cercana disponible, con el visto bueno de Pedro.
- **Rediseño de `frontend/catalogo.html`**: se mantiene toda la lógica
  existente (agrupado, filtros, llamada a `/api/styles`) y solo cambia el
  aspecto/interacción:
  - Tarjetas con esquinas más redondeadas, elevación y ligero
    `transform` al pasar el ratón, foto con relación de aspecto fija
    (4:5) y un pequeño icono de lupa que aparece al hacer hover como
    pista de que se puede ampliar.
  - Zoom real: al hacer click en la foto de una tarjeta se abre un
    lightbox a pantalla completa con la imagen ampliada y su
    nombre/descripción debajo; se cierra con el botón de cerrar,
    haciendo click fuera de la imagen o con la tecla Escape.
  - Sigue usando `frontend/assets/theme.css` (tokens de color/tipografía
    compartidos con el resto de la web), así que el resto de páginas no
    se ven afectadas por este cambio.

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
