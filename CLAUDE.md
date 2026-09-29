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
7. **`generator.py`** — Genera la imagen final: difusión + ControlNet condicionado por (máscara de pelo + landmarks + parámetros del corte elegido). Sigue siendo un esqueleto sin usar: **la generación real va por `haircut_editor.py`** (modelos externos de edición de imagen por API, ver la sección "Simulación del corte" más abajo). `generator.py` solo se usa si no hay ninguna clave configurada, y entonces devuelve la foto sin cambios.
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
preferencia de barba). La regla de frecuencia de visitas se quitó en sept
2026 a petición de Pedro: ese dato ya no cambia las recomendaciones. Esquema completo en `VisagismoProfileIn`
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
- **Campos manuales por geometría, pero automáticos por IA con consentimiento**
  (se probaron por geometría clásica y se descartaron, ver calibración más
  abajo; la página lo indica junto a cada campo): `eyebrow_type` (forma
  de cejas), `profile_type` (perfil de nariz), `ears_projection`
  (proyección de orejas), `chin_projection` (mentón), `jawline_definition`
  (mandíbula) y `neck_proportions` (cuello). Para las orejas Pedro había
  elegido un "detector dedicado"; se usó la segmentación de orejas de
  BiSeNet en vez de un modelo nuevo (no se encontró ninguno maduro con
  licencia de uso comercial), y al probarla con fotos reales resultó no
  ser fiable. Ver la sección dedicada más abajo, "Rasgos de perfil
  juzgados por IA con visión (sept 2026)", para cómo se rellenan estos 6
  campos hoy sin geometría: un modelo de IA con visión los juzga
  directamente, si el cliente dio su consentimiento.

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

**Mentón, mandíbula y cuello, probados con el mismo método (sept 2026).**
Pedro pidió detectar también `jawline_definition` (mandíbula), `chin_projection`
(mentón) y `neck_proportions` (cuello) de forma automática. Se repitió
exactamente la misma prueba de arriba (mismas 61 fotos de
`serengil/deepface`, comparando la diferencia entre dos fotos de la MISMA
persona contra la diferencia entre fotos de personas DISTINTAS, usando los
pares de `master.csv`), reutilizando solo `face_analysis.py` y
`hair_segmentation.py` ya existentes -- sin modelo ni dependencia nueva.
Resultado: ninguno de los tres es fiable hoy.

- **Mandíbula** (`jawline_definition`): como proxy de "mandíbula marcada"
  se probó el ángulo de giro en el gonion (puntos 2-4-6 y 10-12-14 del
  contorno de mandíbula de los 68 landmarks, donde la línea pasa de
  vertical bajo la oreja a horizontal hacia la barbilla -- más agudo,
  mandíbula más marcada). Hay algo de señal (misma persona: 2,9° de
  diferencia media entre fotos, mediana 2,6°; personas distintas: 4,8° de
  media, mediana 4,2°, sobre un rango total de solo 18°), pero la mezcla
  es demasiado alta para fiarse: usando esa sola medida para adivinar si
  dos fotos son la misma persona acertaría solo 2 de cada 3 veces (AUC
  0,66 sobre 1,0 perfecto), muy lejos de lo que hace falta para
  clasificar a un cliente real sin que el barbero lo revise. Campo
  manual.
- **Cuello** (`neck_proportions`): con la clase `neck` de BiSeNet se
  probaron dos medidas (anchura de cuello / ancho de cara, y proporción
  de píxeles de cuello en la imagen). En ambas la diferencia entre dos
  fotos de la MISMA persona fue igual o mayor que entre personas
  distintas (0,38 vs 0,32 de media en la relación de anchura; 0,034 vs
  0,031 en la proporción de píxeles -- acertar por azar sería 0,5 de AUC,
  y salió 0,45 y 0,52 respectivamente), y en 2 de las 61 fotos no se
  detectó ni un solo píxel de cuello. La causa es el encuadre de la foto
  (cuánto cuello entra en el plano, el ángulo de la barbilla), no la
  persona: es ruido de cómo se hace la foto, no un rasgo suyo. Campo
  manual.
- **Mentón** (`chin_projection`): no se ha llegado a medir nada, por el
  mismo motivo de fondo que ya descartó el perfil de nariz -- cuánto
  sobresale el mentón hacia delante es información de profundidad
  (sagital), y una foto frontal 2D no la contiene; haría falta la foto de
  perfil, que ya se demostró arriba que ni siquiera detecta cara con los
  modelos actuales. Campo manual.

Con esto, los seis rasgos que Pedro pidió automatizar (mentón, orejas,
mandíbula, cuello, cejas, perfil) no se pueden rellenar con GEOMETRÍA
clásica: los tres de esta prueba (mentón, mandíbula, cuello) y los tres ya
descartados antes (orejas, cejas, perfil de nariz). No es un límite de
umbrales que se pueda recalibrar -- haría falta un modelo distinto
(landmarks/segmentación que funcionen de perfil, o un método que capture
profundidad) para retomarlo por esta vía. Sí se automatizaron por OTRA vía
-- IA con visión en vez de geometría -- ver la sección siguiente.

## Rasgos de perfil juzgados por IA con visión (sept 2026)

Pedro, tras ver que perfil/cejas/orejas/mentón/mandíbula/cuello seguían
pidiéndole rellenarlos a mano en producción: "pon la opción de que el
peluquero los edite, pero es el sistema el que debe responder a esas
preguntas de manera automática analizando el rostro en las 3 fotos".
Se le explicó lo de arriba (geometría clásica ya probada y descartada a
fondo, incluida una rama sin fusionar que lo intentó por contorno de
silueta y tampoco funcionó con fotos reales) y se le preguntó cómo seguir;
eligió usar la API de Claude (ya integrada para el informe de texto) para
que un modelo de IA con VISIÓN juzgue estos 6 rasgos directamente sobre
las 3 fotos, en vez de medir ángulos.

- **`app/pipeline/visagismo_vision_analysis.py`** (nuevo): envía las 3
  fotos (redimensionadas a 1536px de lado mayor, JPEG calidad 92 -- mismo
  criterio que `haircut_editor._to_jpeg`) a la API de Claude con
  `tool_choice` forzado a una única herramienta (`record_facial_traits`)
  cuyo `input_schema` solo admite los valores válidos de cada campo (o
  `null` si el modelo no tiene confianza) -- así no hace falta parsear
  texto libre ni fiarse de que el modelo no invente un valor fuera de la
  lista. El system prompt le pide explícitamente que actúe como lo haría
  un peluquero mirando a un cliente real (juicio visual, no medición de
  precisión clínica) y que devuelva `null` antes que adivinar si una foto
  no se lo permite.
- **Consentimiento**: se reutiliza `consent_ai_analysis` (el mismo que ya
  exigía el informe de texto de IA) en vez de crear uno nuevo, porque es
  la misma finalidad -- enviar datos a la API de Claude para análisis de
  visajismo -- ahora ampliada a incluir también las fotos. Antes este
  consentimiento solo se podía fijar al crear el cliente (`POST
  /api/clients`) y no tenía ninguna casilla en la interfaz -- ni siquiera
  el informe de texto era alcanzable desde la web, solo por API. Se
  añadió: `ConsentsIn.consent_ai_analysis` (antes solo tenía
  `consent_save_photo`/`consent_simulation`/`consent_3d_scan`),
  `repository.update_consents` acepta el cuarto parámetro, y
  `frontend/ficha.html` tiene ahora una casilla "Permiso para juzgar
  perfil/cejas/orejas/mentón/mandíbula/cuello con IA" junto a las
  herramientas de visajismo (mismo patrón que el resto de permisos de la
  ficha). Esto también destapa y resuelve, de paso, el informe de texto
  de IA, que hasta ahora era inalcanzable desde la interfaz.
- **`clients_routes.override_visagismo_auto_analysis`**: la foto frontal
  se sigue analizando siempre en local (gratis, sin consentimiento
  adicional). Las 2 fotos de perfil solo se decodifican y se envían a
  `visagismo_vision_analysis` si `client.consent_ai_analysis` es `true` --
  sin ese permiso, ni siquiera se procesan (no se toca ningún dato
  biométrico que no se vaya a usar, mismo criterio que antes cuando estas
  fotos no se analizaban nunca). Si falta el consentimiento, o falta
  `ANTHROPIC_API_KEY`, o la llamada a la API falla: el endpoint NO
  devuelve error -- sigue devolviendo 200 con el resultado local de la
  frontal, y añade un aviso explicando por qué esos 6 campos se quedaron
  sin rellenar. Deliberado: a diferencia de `.../visagismo-ai-report`
  (que es un endpoint 100% dependiente de la IA y por eso sí da 422 sin
  consentimiento), aquí hay trabajo útil que hacer tanto si se puede
  analizar por IA como si no, y bloquear todo el endpoint habría sido una
  regresión para cualquier cliente sin ese permiso.
- **Fusión**: se reutiliza `merge_detected_features` (misma función que
  ya fusionaba lo detectado por geometría) para no pisar lo que el
  barbero ya hubiera rellenado a mano -- se llama dos veces en cadena
  (geometría primero, IA después), cada una solo rellena huecos.
- **RGPD**: a diferencia de TODO el resto de este endpoint (que es 100%
  local), cuando hay consentimiento las 2 fotos de perfil SÍ salen del
  servidor hacia la API de Claude. Se procesan en memoria y se descartan
  justo después de la llamada, igual que el resto -- nunca se guardan en
  disco ni en la base de datos.
- Tests: `backend/tests/test_visagismo_vision_analysis.py`, mismo patrón
  que `test_visagismo_ai_advisor.py` (mockeando `anthropic.Anthropic`,
  sin llamada real ni fotos reales): valores válidos se guardan, valores
  fuera de lista se descartan con aviso, campos `null` del modelo generan
  aviso de "revísalo a mano", sin `ANTHROPIC_API_KEY` lanza
  `VisionAnalysisNotConfigured`, fallo de la API lanza `VisionAnalysisError`.
- Pendiente honesto: esto no se ha probado todavía con fotos reales de un
  cliente (a diferencia de la calibración geométrica de arriba, que sí se
  probó con 61 fotos reales + las de Pedro). Un modelo de IA con visión
  juzgando como un peluquero es cualitativamente distinto de medir
  ángulos, así que los fallos esperables también son distintos (más
  parecido a que un peluquero nuevo se equivoque clasificando un rasgo
  ambiguo, que a un bug de medición) -- conviene revisar los primeros
  resultados reales con ojo crítico antes de confiar en ellos a ciegas.
- **`frontend/visagismo.html`**: la leyenda de la sección "Rasgos" (icono
  ⓘ) distinguía solo dos casos -- ✨ "lo rellena el análisis de las fotos"
  y ✋ "se marca a mano: las fotos no permiten detectarlo con fiabilidad".
  Como ahora hay un tercer caso, se añadió un tercer icono
  (`wand-sparkles`, mismo que "Simular corte") para perfil/cejas/orejas/
  mentón/mandíbula/cuello, con su propia línea en la leyenda explicando
  que depende del consentimiento de IA del cliente. El resto de la
  página no cambia: `applyClientToPage` ya repinta los selects con
  cualquier valor que llegue en `visagismo_profile`, así que estos 6
  campos se rellenan solos en cuanto el backend los devuelve, sin tocar
  el JS de renderizado.
- **Ajuste de orden (sept 2026, feedback de Pedro tras probarlo con fotos
  reales)**: `#results-section` (la nota de confianza de la IA + los
  avisos) vivía justo debajo del botón "Analizar", ANTES de la sección
  "3 Rasgos" -- se leía el comentario antes de ver a qué campos se
  refería. Se movió (solo HTML, ningún cambio de JS: los `getElementById`
  no dependen de la posición en el DOM) a después de la rejilla de
  campos y de "Usa gafas"/notas, justo antes del botón "Guardar" --
  ahora primero se ven los rasgos ya rellenados y debajo el porqué.
- **Nota de la IA incoherente con los campos que rellena (sept 2026,
  bug real visto con fotos reales de Pedro)**: la nota que devuelve el
  modelo (`confidence_notes`, "Nota de la IA sobre estas fotos") describía
  rasgos que CONTRADECÍAN el valor guardado en los propios campos -- p.ej.
  la nota decía "la mandíbula se aprecia poco marcada" mientras el campo
  Mandíbula quedaba en "Definida", o "cejas rectas y relativamente bajas"
  mientras el campo Cejas quedaba en "Arqueada". Causa: con `tool_choice`
  forzado, el modelo genera el JSON completo de una vez, sin ningún paso
  de razonamiento visible antes; `confidence_notes` iba como ÚLTIMO campo
  del schema, así que nada obligaba a que ese texto libre describiera lo
  mismo que los 6 valores categóricos ya elegidos -- ambos se generaban de
  forma independiente y podían divergir. Arreglo en
  `visagismo_vision_analysis.py` (`_TOOL_SCHEMA`, `SYSTEM_PROMPT`): se
  puso `confidence_notes` como PRIMER campo del schema (antes de los 6
  campos categóricos) y se le pide explícitamente que describa ahí primero
  lo que ve, y que los 6 campos siguientes sean coherentes con esa
  descripción -- aprovecha que Claude genera el JSON de una herramienta en
  el orden de sus propiedades, así que escribir la nota antes obliga a que
  los valores categóricos que vienen después estén condicionados por ella,
  en vez de ser dos respuestas independientes que puedan contradecirse. El
  `SYSTEM_PROMPT` remacha además que si hay duda sobre un rasgo concreto,
  la forma correcta de expresarlo es `null` en ese campo, no describir un
  valor distinto en la nota. Esto reduce mucho el riesgo, pero sigue sin
  ser una garantía absoluta (es un modelo de lenguaje, no una regla
  determinista) -- conviene seguir revisando la nota contra los campos de
  vez en cuando. No se han tocado los tests (siguen mockeando el `dict` de
  respuesta, no dependen del orden de las claves) -- 118/118 en verde.
- **El permiso de IA se pregunta al abrir Visajismo, no en la ficha (sept
  2026)**: Pedro, tras localizar por fin la casilla "Permiso para juzgar
  perfil/cejas/orejas/mentón/mandíbula/cuello con IA" en `ficha.html` (ver
  el historial de confusión más arriba, en "Rasgos de perfil juzgados por
  IA con visión"): pidió que esa pregunta viva DENTRO de Visajismo, y que
  al abrir un cliente sin ese permiso solo se vea la pregunta -- nada de
  fotos ni de campos de rasgos -- hasta que el peluquero responda.
  - **`frontend/visagismo.html`**: nueva sección `#ai-gate` (checkbox +
    "Aceptar y continuar" + "Seguir sin IA (rellenar a mano)"), oculta por
    defecto. `openClient(client)` (llamada tanto al llegar con
    `client_id` en la URL como al pulsar "Abrir ficha") decide: si
    `client.consent_ai_analysis` ya es `true`, sigue el flujo normal
    (`applyClientToPage`); si no, `showAiGate(client)` oculta
    `#search-section`, `#photos-card` y `#analysis-form` y muestra SOLO
    `#ai-gate`. "Aceptar y continuar" hace `PATCH
    /api/clients/{id}/consents` con `consent_ai_analysis: true` (mismo
    endpoint que ya usaba la casilla de `ficha.html`) y revela el flujo
    normal con el cliente ya actualizado; "Seguir sin IA" revela el mismo
    flujo normal SIN cambiar el consentimiento (queda en `false`, y esos 6
    campos se siguen rellenando a mano, exactamente el comportamiento por
    defecto que ya existía antes de este permiso). Deliberado que exista
    esta segunda opción aunque Pedro pidiera que "solo aparezca eso":
    RGPD exige que un consentimiento sea libre y no puede convertirse en
    un requisito para poder usar el resto de la página (fotos, análisis
    local, edición manual de los 6 campos no dependen de este permiso) --
    sin una salida, un cliente que no quisiera dar este permiso concreto
    dejaría al peluquero sin poder usar Visajismo en absoluto para él.
  - **`frontend/ficha.html`**: se quita la casilla "Permiso para juzgar...
    con IA" y su cableado (`q("c-ai")...`) -- ya no hay dos sitios
    distintos preguntando por el mismo consentimiento, que es lo que
    causaba la confusión original de Pedro. El resto de casillas de
    consentimiento de esa página (foto, simulación) no se tocan.
  - Nada cambia en el backend: sigue siendo el mismo
    `consent_ai_analysis` y el mismo endpoint `PATCH
    /api/clients/{id}/consents` de siempre, solo cambia qué pantalla lo
    pregunta.
  - Verificado con Playwright contra un servidor local real (no
    simulado): al abrir `visagismo.html?client_id=...` de un cliente sin
    el permiso, solo se ve `#ai-gate` (search-section y photos-card
    ocultos); "Aceptar y continuar" hace el PATCH, oculta la pregunta y
    muestra el resto de la página, y una recarga posterior ya no vuelve a
    preguntar (el consentimiento quedó guardado); "Seguir sin IA" oculta
    la pregunta y muestra el resto de la página sin tocar el
    consentimiento (confirmado leyendo el cliente por la API:
    `consent_ai_analysis` sigue en `false`). Sin errores de consola
    achacables a este cambio. Suite completa: 118/118.

- **Frente/nariz/labios/papada pasan de manuales a juzgados por IA (sept
  2026)**: `intellectual_zone_forehead`, `nose_size`, `lip_thickness` y
  `has_double_chin` se habían añadido antes como campos SOLO manuales
  (ver la sección de barba y bigote más abajo), porque en ese momento no
  hacía falta más. Pedro, tras usar la página y ver el icono ✋ en estos 4
  campos junto al ✨ de los otros 6, pidió explícitamente: "quiero que no
  sean campos manuales, que se puedan editar pero que te de una respuesta
  la IA" -- es decir, mismo tratamiento que perfil/cejas/orejas/mentón/
  mandíbula/cuello (la IA los rellena si hay consentimiento y confianza,
  el barbero puede corregirlos siempre).
  - `visagismo_vision_analysis.py`: `_VALID_VALUES` gana `nose_size` y
    `lip_thickness` (parecidos a los 6 originales, valen igual);
    `intellectual_zone_forehead` también entra pero se marca aparte en
    `_ZONE_FIELDS` porque vive en `facial_horizontal_zones_ratio`, un
    diccionario HERMANO de `facial_features_profile` (ver `schemas.py`,
    `AnatomicalMetricsIn`) -- no en el mismo, así que `VisionTraitsResult`
    ahora reparte el resultado entre dos diccionarios en vez de uno solo.
    `has_double_chin` se marca en `_BOOL_FIELDS` porque es un booleano, no
    una categoría de texto -- se valida y se añade a `_TOOL_SCHEMA` por
    separado del resto (`{"type": ["boolean", "null"]}` en vez del
    `enum` de string que usa `_VALID_VALUES`). `SYSTEM_PROMPT` y
    `_TOOL_SCHEMA` pasan de describir 6 rasgos a 10, con un ejemplo de
    coherencia para `has_double_chin` igual que ya tenían los otros 6
    (para el mismo problema de nota-contra-campos que se documenta arriba).
  - `clients_routes.override_visagismo_auto_analysis`: además de fusionar
    `facial_features_profile` (como ya hacía), ahora también lee, fusiona
    (`merge_detected_features`, sin pisar lo manual) y guarda
    `facial_horizontal_zones_ratio` -- antes este endpoint no tocaba ese
    diccionario en absoluto.
  - `facial_traits_analysis.merge_detected_features`: `has_double_chin`
    se añade al caso especial que ya tenía `has_glasses` (un `False` de
    checkbox sin marcar cuenta como "vacío", no como un valor puesto a
    mano, porque una casilla no distingue "no" de "sin especificar").
  - `frontend/visagismo.html`: los 4 campos (Frente, Nariz, Labios,
    Papada / doble mentón) cambian su icono de ✋ `tag-hand` a
    `wand-sparkles` `tag-ai`, y las dos leyendas explicativas (la del
    encabezado y la de la sección "Rasgos") pasan de listar 6 rasgos por
    IA a los 10. Ningún cambio en los `<select>` ni en `applyClientToPage`:
    los valores del enum ya coincidían, solo cambiaba de dónde podían
    venir.
  - Tests ampliados: `test_visagismo_vision_analysis.py` (rutas de los 4
    campos nuevos a su diccionario correcto, `has_double_chin` con
    `True`/`False`/valor no booleano) y
    `test_sessions.py::test_visagismo_auto_analysis_gated_by_ai_consent_and_key`
    (fusión de extremo a extremo contra el endpoint real, incluida la
    corrección manual previa sin pisar en ambos diccionarios). 143/143 en
    verde.

Pendiente / no cubierto a propósito en `frontend/visagismo.html`: la
página nueva solo cubre `facial_features_profile` y
`facial_horizontal_zones_ratio` (los campos de esta sección) --
`hair_physical_metrics`, `lifestyle_and_preferences`, `cranial_morphology`
y `facial_geometry` del resto de `visagismo_profile` siguen sin tener UI
dedicada (mismo estado "solo API" que ya tenían antes de esta feature, ver
nota al final de la sección de reglas de visagismo).

## Guía de visajismo de perfil y cámara con marco (sept 2026)

**`frontend/guia-visagismo.html`** (solo se llega desde `visagismo.html`:
botón "❓ Ayuda: cómo hacer las fotos" junto al título y "¿Cómo hacer las
fotos?" en el paso 2, ambos a `#fotos`; a petición de Pedro ya no tiene
tarjeta propia en `inicio.html`, y su nav lleva de vuelta al análisis.
Como los dibujos se cargan con `fetch` después del salto al ancla, la
página repite el `scrollIntoView` al terminar de pintarlos): guía para valorar a mano tres rasgos de perfil
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
- La página va en tres pasos: 1. Cliente, 2. Fotos, 3. Rasgos del
  cliente. Las fotos están SIEMPRE visibles: en la primera versión iban
  dentro del formulario que solo aparece tras cargar o crear el cliente, y
  Pedro no encontraba la cámara (las pruebas forzaban el formulario visible
  y no lo detectaron). Se pueden hacer las fotos antes de elegir cliente;
  solo "Analizar" lo necesita, y si falta lo avisa sin perder las fotos.
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

## Criterio visual de la web: poco texto, iconos y ⓘ (sept 2026)

Pedro: "no puede haber mucho ruido visual en la web/app, tiene que ser
dinámica e intuitiva, sustituye el exceso de información por caracteres
intuitivos". Eligió aplicarlo a toda la web, con iconos de línea (no
emojis) y guardando las explicaciones detrás de un ⓘ en vez de borrarlas.
Reglas para cualquier página nueva o cambio:

- **Iconos**: `frontend/assets/ui.js` lleva los iconos de Lucide (licencia
  ISC, uso comercial permitido) que usa la web. En el HTML,
  `<span data-icon="camera"></span>`; en JS, `UI.icon("check")`. Para uno
  nuevo, copiar el interior de su SVG desde el paquete `lucide-static` al
  objeto `ICONS`. Nada de emojis en la interfaz (se ven distinto en cada
  móvil).
- **Explicaciones**: nunca un párrafo a la vista. Van en
  `<details class="info"><summary></summary><div class="info-body">…</div></details>`
  junto al título o campo al que se refieren (dentro de un `.head-row`,
  que hace de ancla para el globo). Solo uno abierto a la vez y se cierra
  al tocar fuera (lo hace `ui.js`). Estilos en `theme.css`.
- **Textos**: títulos de 1-2 palabras, botones con icono + verbo corto
  ("Abrir ficha", "Analizar", "Guardar"), mensajes de estado cortos con
  icono (`say("ok" | "warn" | "busy", texto)` en cada página).
- **Menú superior**: una fila, icono + palabra; en pantallas de menos de
  520 px solo iconos (salvo `.keep-label`, p.ej. "Ayuda").
- Cambios concretos: la portada es solo icono + palabra (Peluquero /
  Cliente, y un mosaico de accesos); Visajismo muestra las 3 fotos como
  casillas con la silueta que toca, botones de cámara/galería y ✓ verde
  al completar cada paso, y marca con ✨ los campos que rellena el análisis
  y con ✋ los manuales; el catálogo y las recomendaciones enseñan foto,
  nombre y etiquetas (la descripción va al ampliar o en el ⓘ), en 2
  columnas en el móvil; las recomendaciones se agrupan por foto igual que
  el catálogo (antes por familia, y la tarjeta enseñaba la foto de otro
  corte). La guía (ahora "Ayuda") empieza por cómo hacer las fotos, con
  textos de una línea (regenerados con `tools/generar_perfiles_guia.py`)
  y las fuentes plegadas.

## Simulación del corte con API externa (`app/pipeline/haircut_editor.py`, sept 2026)

Fase 2 hecha por otra vía: en vez de difusión + ControlNet propios (hace
falta GPU y Railway no la tiene), la foto del cliente se manda a un modelo
externo de edición de imagen con una instrucción construida a partir del
corte del catálogo. Pedro eligió esta vía ("API externa de edición"),
dejar DOS proveedores para compararlos con fotos reales, y un
consentimiento nuevo específico.

- **Proveedores** (solo aparecen si tienen clave; `GET /api/simulate/providers`):
  - `gemini`: Google Gemini, modelo de imagen (`GEMINI_IMAGE_MODEL`, por
    defecto `gemini-3.1-flash-image`, ~0,04-0,05 $/imagen). Recibe la foto
    del cliente + la foto de referencia del corte (`reference_image` del
    catálogo, leída de `frontend/`) + la instrucción. SDK `google-genai`.
  - `flux`: FLUX.1 Kontext [pro] vía fal.ai (`FAL_MODEL`, ~0,04 $/imagen).
    Solo foto + instrucción; con `FAL_USE_REFERENCE=1` usa el modelo [max]
    multi con la foto de referencia (~0,08 $). SDK `fal-client`, con
    `sync_mode` para que el resultado no quede en su historial.
- **Instrucción** (`build_edit_prompt`, en inglés): nombre y descripción
  del corte, largo arriba/laterales/nuca en mm, degradado, color si se
  pide, y una lista explícita de lo que NO debe cambiar (cara, identidad,
  barba, fondo, luz, encuadre). El tipo de pelo solo se incluye si lo ha
  indicado una persona (selector o ficha): la heurística automática falla
  (el pelo liso despeinado de Pedro sale "afro") y "mantén su pelo afro"
  le cambiaría el pelo. Con referencia, se avisa de que la segunda foto
  solo vale para la forma del corte (lleva la cara de otra persona).
- **`POST /api/simulate`**: campos nuevos `provider`,
  `consent_external_photo` y `record_visit`. Si hay algún proveedor
  configurado y no llega el consentimiento → 422 (se comprueba antes de
  abrir la foto). El resultado del modelo ya es la imagen final (no pasa
  por `compositor.py`, que mezcla con la máscara de pelo original). Errores
  del proveedor → 502; sin clave → 503. La respuesta dice qué `provider`
  la generó.
- **Web** (`index.html`): selector Gemini / FLUX / Comparar, casilla de
  consentimiento (el ⓘ dice a qué empresa va la foto) y resultado en
  mosaico Antes / Gemini / FLUX, cada uno con su estado (generando, hecho,
  error) y ampliable al tocar. En "Comparar" se lanzan las dos peticiones a
  la vez y solo la primera se apunta en el historial del cliente
  (`record_visit`). El catálogo tiene "Probar en un cliente" al ampliar
  una foto, que abre el simulador con ese corte (`index.html?style=<id>`).
- **RGPD**: es la primera vez que la FOTO sale del servidor (el informe de
  IA solo manda categorías). Consentimiento propio por simulación, aparte
  del de la ficha. No se guarda ni la foto ni el resultado. Gemini: usar
  una clave CON facturación (en el nivel gratuito Google puede usar lo
  enviado para mejorar sus productos). Revisar los términos de tratamiento
  de datos de cada proveedor antes de usarlo con clientes reales.
- **Configurar** (Railway → Variables): `GEMINI_API_KEY` y/o `FAL_KEY`.
  Sin ninguna, todo funciona como antes y la página dice "Simulación no
  activada".
- **Sin probar todavía con los modelos reales**: desde el entorno de
  desarrollo no hay salida a esas APIs (403 del proxy) ni claves. Probado:
  tests con los SDK simulados (`tests/test_haircut_editor.py`) y la web de
  principio a fin con un servidor que sustituye a los proveedores. Lo
  primero con las claves: comparar los dos con fotos reales, revisar que
  no cambian la cara, y afinar la instrucción.

### Puesta en marcha real + qué tan "hiperrealista" se puede pedir (sept 2026)

Pedro pidió activar esto de verdad y que sea "lo más hiperrealista posible".
Sigue sin haber clave configurada ni salida de red desde este entorno de
desarrollo, así que esto sigue siendo research + guía de configuración, NO
un cambio de código verificado contra las APIs reales (mismo criterio que
el resto del proyecto: no se toca el modelo por defecto sin poder probarlo).

- **Por qué Gemini es el punto de partida recomendado para "no tocar la
  cara"**: comparativas de 2026 (no solo la documentación oficial) coinciden
  en que la familia Gemini de edición de imagen ("Nano Banana") conserva
  mejor la identidad facial y el detalle real (pelo, piel) que FLUX Kontext,
  que tiende a re-tocar la cara "a lo IA de belleza" aunque se le pida no
  tocarla. Encaja con la prioridad número uno del prompt de este proyecto
  (`build_edit_prompt`: "Keep everything else exactly the same... same
  person and identity"). Además Gemini ya recibe SIEMPRE la foto de
  referencia del corte (`edit_haircut` solo se la quita a FLUX si no está
  `FAL_USE_REFERENCE=1`), así que ya está configurado en su modo más fiel.
- **Dos calidades de Gemini, ambas ya soportadas sin tocar código** (solo
  cambiando la variable `GEMINI_IMAGE_MODEL` en Railway):
  - `gemini-3.1-flash-image` (el valor por defecto actual): más barato,
    ~0,067 $/imagen a 1K de resolución.
  - `gemini-3-pro-image-preview` ("Nano Banana Pro"): calidad/realismo
    superior según las mismas comparativas, ~0,134 $/imagen (el doble) a
    1K/2K. Es un modelo "preview": no hay forma de confirmar desde aquí que
    responda exactamente igual a la llamada actual (`generate_content` con
    `response_modalities=["IMAGE"]`) sin probarlo con una clave real — si al
    activarlo diera error, quitar la variable vuelve a `flash` sin más.
  No se ha cambiado el valor por defecto en `app/config.py` a propósito: es
  una decisión de coste/calidad de Pedro, no algo que deba decidir el
  código, y cambiarlo a ciegas sin poder probarlo iría contra el criterio
  de este proyecto de no dar por buena una función sin verificarla.
- **FLUX.1 Kontext (lo que ya está integrado) ya no es la versión más
  reciente de Black Forest Labs** — existe FLUX.2 (Pro/Max/Flex/Klein, en
  fal.ai como `fal-ai/flux-2-pro` y similares) con mejor consistencia de
  identidad que FLUX.1 Kontext. NO se ha integrado: cambiar de familia de
  modelo (no solo de variante) implica una llamada distinta a fal.ai que
  habría que verificar con una clave real, y hoy solo hay una de FLUX
  documentada (`fal-ai/flux-pro/kontext`, sept 2026, ver arriba). Pendiente
  honesto si en algún momento se quiere seguir comparando con FLUX en vez
  de solo con Gemini.
- **Para arrancarlo de verdad** (pasos para Pedro, no algo que se pueda
  hacer desde este entorno): crear una clave en Google AI Studio
  (https://aistudio.google.com/apikey) con un proyecto de Google Cloud que
  tenga facturación activada (imprescindible: en el nivel gratuito Google
  puede usar las fotos para mejorar sus productos, ver RGPD arriba), y
  añadirla como `GEMINI_API_KEY` en Railway → Variables del servicio. Sin
  tocar `GEMINI_IMAGE_MODEL` usa `flash` (más barato); añadiendo
  `GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview` prueba la versión más
  realista. Ninguna clave debe pegarse nunca en el chat de Claude ni
  guardarse en el repo -- solo en las Variables de Railway (o en un `.env`
  local con gitignore para probar en el Mac de Pedro primero).
- Precios/nombres de modelo de esta nota son un apunte de investigación de
  sept 2026, no una garantía: esta parte del mercado cambia cada pocos
  meses -- conviene revisar precio y disponibilidad en la documentación
  oficial de cada proveedor antes de activarlo con clientes reales.

## Recomendaciones explicadas: rasgos, cuestionario y "por qué" (sept 2026)

Pedro pidió tres cosas a la vez: que los rasgos de la ficha cuenten en las
recomendaciones, una pantalla para que el cliente rellene lo que falta, y
que cada recomendación diga por qué encaja.

- **Rasgos → reglas** (`app/pipeline/trait_rules.py`): orejas prominentes
  (fade alto/a piel o laterales < 6 mm = aviso; laterales con largo o fade
  bajo con volumen = a favor), mentón retraído (pelo hacia atrás o recogido
  = aviso; algo de largo en la nuca = a favor), mentón prominente
  (flequillo = a favor), mandíbula poco definida (laterales cortos con largo
  arriba = a favor), perfil convexo/nariz prominente (hacia atrás = aviso;
  volumen arriba = a favor), perfil cóncavo (flequillo a favor, hacia atrás
  aviso), cuello corto (nuca larga = aviso; nuca despejada = a favor),
  cuello largo (media melena en la nuca = a favor). Fuentes en el
  docstring: Luc Vincent, Book of Barbering (orejas), Castlebeard y Beard
  Resource (barba). "Hacia atrás", "flequillo" y "volumen" se leen del
  nombre/descripción del corte (regex); "parte de atrás" no cuenta como
  peinado hacia atrás (test).
- **Consejo de barba** (`beard_advice`): mentón retraído y/o mandíbula poco
  definida. Va aparte de los cortes (`beard_advice` en
  `GET .../recommendations`); si el cliente dijo "afeitado", se da como
  sugerencia. Sept 2026: sustituido por un motor mucho más completo de
  barba Y bigote -- ver la sección "Barba y bigote: motor de recomendación
  completo, 18 estilos ilustrados" más abajo. El campo de la API
  (`beard_advice`) y el contrato (`list[dict]` con `label`/`detail`) no
  cambiaron, solo lo que hay detrás.
- **Campos nuevos** en `facial_features_profile`: `chin_projection`
  (retruded/balanced/prominent) y `jawline_definition` (defined/soft), a
  mano en `visagismo.html` (con ⓘ-enlace a la sección de la guía). El
  campo "Perfil de nariz" pasa a llamarse "Perfil" (mismo `profile_type`).
- **Por qué** (`app/pipeline/rule_effects.py`): cada regla devuelve
  `Effect(score, label, detail)`; puntuación negativa = razón a favor,
  positiva = aviso. `recommend_styles` devuelve `reasons` y `warnings` por
  corte y ordena por la suma. Las 5 reglas de `visagismo_rules.py` llevan
  ahora también `label`; sus razones a favor YA NO van en `note` (antes se
  mezclaban con los avisos y la web las pintaba como aviso). Remolinos y
  forma de cara ganan razones a favor ("Disimula los remolinos", "Alarga
  la cara", "Acorta la cara").
- **Web** (`recomendaciones.html`): máximo 3 etiquetas por tarjeta (avisos
  primero, verde = a favor, ámbar = aviso, "+N" el resto); el ⓘ de cada
  corte tiene la descripción y la explicación de cada etiqueta (en el
  móvil se abre como hoja abajo, a todo lo ancho). Arriba, fila "Barba"
  con su ⓘ, y chip "Mi perfil" que abre el cuestionario.
- **Cuestionario** (`frontend/cuestionario.html`, "Mi perfil" en la
  portada del cliente): 5 preguntas, una por pantalla, con dibujos SVG
  propios en vez de texto: forma de cara, nacimiento del pelo (recto,
  entradas, pico, frente alta), tiempo de peinado (reloj), cada cuántas
  semanas viene (calendario) y barba. "No lo sé" salta la pregunta. Guarda
  la forma de cara en `face_shape_override` (y `facial_geometry`) y el
  resto en `visagismo_profile`, partiendo de lo ya guardado para no borrar
  los rasgos. Se entra con `?client_id=` (desde Visajismo, Recomendaciones)
  o buscando el nombre; si no hay ficha, pide darse de alta en el
  mostrador (el consentimiento se recoge ahí).
- Tests: `tests/test_trait_rules.py`.

## Flujo "cliente esperando en el sillón": alta propia, sala de espera y ficha (sept 2026)

Idea de Pedro: el cliente normalmente espera mientras el peluquero termina
con otro. La web se organiza alrededor de eso.

- **Primera visita**: el cliente, en la tablet de la barbería o en su
  móvil (QR), se da de alta él mismo (`cliente.html` → "Soy nuevo":
  nombre, teléfono y consentimientos, que ahora da él), responde "Mi
  perfil" (6 preguntas con dibujos, incluida "¿Cómo es tu pelo?") y ve
  recomendaciones y catálogo, marcando favoritos con el corazón. Queda
  apuntado solo en la lista de espera de hoy.
- **Peluquero**: entra con PIN (`peluquero.html`) a la **sala de espera**
  (`sala.html`, se refresca sola; quién espera, "Nuevo", qué ha rellenado,
  favoritos; Atender/Terminado; buscador de todos los clientes; QR para
  imprimir) y abre la **ficha** (`ficha.html?client_id=`): lo que ha
  contado el cliente, lo que le gusta, tipo de pelo (manda sobre lo que
  dijo el cliente), enlaces a Remolinos y Visajismo con vuelta a la ficha,
  y la **foto para simular** (se guarda con permiso en
  `CLIENT_PHOTOS_DIR/<id>/simulation.jpg`, dentro del Volume).
- **Siguientes visitas**: entra con su teléfono y tiene recomendaciones
  completas, favoritos y **Probar cortes** (`probar.html`) sobre su foto
  guardada, con tope diario (`MAX_CLIENT_SIMULATIONS_PER_DAY`, 6). El
  peluquero usa la misma página desde la ficha, sin tope.
- **Dispositivos** (decisión de Pedro): con QR (`cliente.html?qr=1`, queda
  recordado en ese navegador) solo la parte del cliente; en la tablet, las
  dos (la del peluquero con PIN). En la tablet el cliente se sale solo a
  los 3 min sin tocar (aviso de 20 s, `assets/session.js`).
- **QR a la vista** (sept 2026, petición de Pedro: que el peluquero no
  tenga que dejar de cortar para enseñarlo): el QR del cliente sale en la
  portada (`inicio.html`, debajo de Peluquero/Cliente) y en la bienvenida
  del cliente (`cliente.html`, sin sesión). En la sala del peluquero ya NO
  está el QR del cliente (Pedro lo cambió): hay un QR para el PELUQUERO
  ("Abrir en tu móvil", `/api/qr.svg?path=/peluquero.html?pel=1`), para
  llevar la sala y las fichas en su móvil; pide el PIN igual. `?pel=1`
  quita el "modo cliente" que ese móvil pudiera tener guardado de haber
  entrado antes por el QR del cliente (`session.js`). Tocarlo lo amplía a pantalla completa (`assets/qr.js`, que trae
  sus propios estilos porque la portada no carga `theme.css`). En el móvil
  que ya entró por QR no se muestra.
- **Acceso** (`app/api/auth.py`): cookies firmadas (HMAC, sin
  dependencias nuevas). Peluquero: `BARBER_PIN`, caduca a los
  `BARBER_IDLE_MINUTES` (15) sin uso; las recargas automáticas de la sala
  (`X-Background: 1`) no cuentan como uso. Todo `/api/clients/*`,
  `/api/simulate` y `/api/waiting*` lo exigen. Cliente: solo teléfono
  (decisión de Pedro; quien sepa el teléfono de otro puede entrar en su
  ficha), cookie de `CLIENT_SESSION_DAYS` (30), y solo ve lo suyo por
  `/api/me/*`. Entrar como uno cierra la sesión del otro en ese navegador.
  Se quitó la búsqueda por nombre de Recomendaciones y del cuestionario:
  dejaba ver cualquier ficha sabiendo el nombre.
- **Tipo de pelo**: el que marca el peluquero (`hair_texture_override`)
  manda; si no, el que dijo el cliente (`hair_pattern_shape`); si ninguno,
  se ordena todo el catálogo en vez de dar error (`client_service.py`).
  A la simulación solo se le pasa el del peluquero.
- **Datos nuevos**: `clients.phone` (único, normalizado sin +34),
  `consent_simulation`, `simulation_photo_path`, `liked_styles`; tablas
  `waiting` y `simulation_log`. Retirar el permiso de guardar la foto la
  borra.
- **Configurar en Railway**: `BARBER_PIN` (obligatorio: sin él nadie entra
  en la parte del peluquero) y `APP_SECRET` (recomendado; si no, se crea
  uno en `data/.app_secret`, dentro del Volume). `segno` nuevo en
  requirements (QR).
- Pendiente / límites: la "lista de hoy" usa la fecha UTC; quien entra por
  QR desde casa también se apunta en la lista (el peluquero lo marca
  Terminado); no hay borrado de ficha desde la web (se pide en el
  mostrador); el texto de los consentimientos debe revisarlo alguien con
  conocimiento de RGPD.
- Tests: `tests/test_sessions.py` (recorrido completo con la app real y
  base de datos temporal). Probado además con Playwright: alta en tablet,
  cuestionario, favoritos, PIN, sala, ficha con foto, simulación (con
  proveedor simulado), segunda visita por teléfono, modo QR y salida
  automática.

## Historial de cortes del cliente (sept 2026)

Pedro: que al terminar cada corte el peluquero registre en la ficha una
foto del resultado y el corte hecho, para que el cliente tenga su historial
y pueda enseñar fácilmente "quiero este", con un máximo por cliente.

- **Registrar** (`ficha.html`, sección "Historial de cortes"): al pulsar
  "Terminado" en la ficha se abre "Corte de hoy" (foto del resultado con la
  cámara trasera, corte del catálogo con los favoritos y el pedido arriba,
  u "Otro" escrito, y notas técnicas de hasta 500 caracteres) con "Guardar y
  terminar" o "Terminar sin guardar". También "Registrar corte de hoy" en
  cualquier momento. `POST /api/clients/{id}/history` (multipart).
- **Máximo**: `MAX_HAIRCUT_HISTORY` (12). Al pasar, se borran los más
  antiguos con su foto. Fotos reducidas a 1400 px, JPEG 85, en
  `CLIENT_PHOTOS_DIR/<id>/history/` (Volume).
- **Fotos solo con `consent_save_photo`**; sin él se guarda el corte y las
  notas, sin foto (la web lo avisa). Retirar ese permiso (cliente o
  peluquero) borra TODAS sus fotos: la de simular y las del historial (los
  cortes se conservan). Sin foto propia, se enseña la del catálogo marcada
  como tal.
- **Cliente** (`mis-cortes.html`, "Mis cortes" en su espacio): sus cortes
  con foto, fecha y nombre; tocar amplía; puede borrar uno. "Quiero este"
  (`PUT /api/me/request`) lo apunta en la entrada de hoy de la lista de
  espera (`waiting.requested_history_id`; si no estaba, le apunta).
- **Peluquero**: en la sala sale "Quiere repetir: <corte>" y en la ficha un
  aviso con la foto, el corte y las notas de aquel día; al registrar el
  corte de hoy, ese corte y sus notas vienen ya rellenos.
- Tabla `haircut_history`. Tests en `tests/test_sessions.py` (máximo y
  borrado de fotos, petición, que un cliente no ve cortes de otro, permiso).

## Maniquí v4, flechas con el dedo, crecimiento en las recomendaciones y frecuencia de retoque (sept 2026)

Pedro pidió mejorar el maniquí de remolinos y el sistema de flechas, que
las recomendaciones tengan en cuenta el crecimiento (y la forma de cara), y
evitar que los clientes "aguanten" demasiado entre visitas. Eligió: cabeza
masculina, zona del pelo pintada, pelo que sigue las flechas, vistas
rápidas para tablet, y flechas trazadas deslizando el dedo (en vez del
disco por zona). Para las visitas no eligió; se hizo frecuencia visible +
aviso + desempate hacia retoque frecuente sin ir contra lo que dice el
cliente.

- **Maniquí v4** (`frontend/assets/head.glb`, script
  `tools/construir_cabeza_masculina.py`, que SÍ queda en el repo): parte
  del maniquí neutro v3 y le aplica los targets CC0 de MakeHuman de hombre
  joven (media de las tres etnias = "género 100 %") y unos pocos de rasgos
  (cabeza algo cuadrada, mentón/mandíbula marcados, cejas más bajas). Cada
  vértice se mueve con el triángulo del base mesh deformado (baricéntricas)
  y el desplazamiento se suaviza 8 pasadas (sin eso se ven facetas y
  arrugas del base mesh). El cráneo se realinea con el v3 (error ~2 cm a
  escala real), así los mapas ya guardados siguen cayendo encima; además la
  página los pega a la superficie al cargar. Se pintan por vértice la zona
  del pelo (como pelo al cero), cejas y sombra de barba, y la máscara del
  cuero cabelludo va en el atributo `_SCALP` (Three.js lo lee como
  `geometry.attributes._scalp`). La línea del nacimiento del pelo es una
  tabla ángulo→altura ajustada a ojo (`HAIRLINE` en el script).
  Intentos descartados: interpolar el desplazamiento por vecinos (IDW)
  dejaba la cabeza con bultos; suavizar 40 pasadas borraba lo masculino.
- **v4.1: ojos mejorados y sin pestañas** (`tools/mejorar_ojos_cabeza.py`,
  se ejecuta después del script anterior). Pedro pidió un maniquí como el
  "AlexV2 Male Head" de TurboSquid o, si no se podía, mejorar los ojos y
  quitar las pestañas. El de TurboSquid no se usó: es de pago (31 $) y la
  licencia estándar de TurboSquid no permite servir el modelo en una web de
  la que cualquiera puede descargar el .glb. Se probó también el escaneado
  gratuito de Lee Perry-Smith (Infinite-Realities, CC BY 3.0, el de los
  ejemplos de Three.js): textura de piel muy realista, pero tiene los ojos
  CERRADOS y le faltan datos de textura en la coronilla (justo donde se
  marcan los remolinos), así que se descartó. Cambios del v4.1: sin malla de
  pestañas (se veían como una banda negra arriba y gris abajo), iris marrón
  natural en vez de rojizo y esclerótica más limpia (más claros de lo que
  parecería necesario porque la escena no corrige gamma), una línea suave
  más oscura en el borde de los párpados en lugar de pestañas (color de
  vértice, contorno de la abertura buscado con rayos de frente), y el ojo
  subdividido y llevado a su esfera para que no se vean facetas.
- **`frontend/growth-map.html`** (reescrita): modos Girar / Flecha /
  Horario / Antihorario / Borrar; deshacer y borrar todo (dos toques)
  sobre el lienzo; vistas Frente/Izq./Dcha./Atrás/Arriba ("Izq." = lado
  izquierdo del cliente, +x); botón "Pelo".
  - Flecha: se desliza el dedo sobre el cuero cabelludo (solo donde
    `_scalp` > 0,35 al empezar); los puntos se remuestrean (máx. 24) y se
    dibuja un tubo con punta. En los modos de marcar, un dedo dibuja y
    girar queda para "Girar", las vistas o el pellizco (el listener va en
    fase de captura para adelantarse a OrbitControls).
  - Pelo: ~2600 mechones de 6 tramos con raíz en el cuero cabelludo; la
    dirección sale de un campo interpolado (por defecto desde la coronilla
    hacia fuera; cerca de una flecha manda la flecha, σ = 0,16; cerca de un
    remolino gira a su alrededor). Para pegarse a la superficie usa una
    rejilla de vértices, no raycasts (serían miles por redibujo).
  - Se guardan `points` (lista de [x,y,z], coordenadas de mundo, maniquí x2
    y bajado 0,3) además de x1..z2. Los mapas antiguos (una flecha recta
    por zona) se cargan como flechas normales.
  - Debajo del lienzo, chips con cómo lo interpreta el backend
    (`POST /api/growth-map/summary`, solo peluquero, no guarda).
- **`app/pipeline/growth_analysis.py`**: zona de cada punto por su
  dirección desde el centro de la cabeza (frente, arriba, coronilla,
  laterales, nuca; umbrales ajustados pintando las zonas sobre el
  maniquí), dirección dominante por zona, remolinos por zona y raya
  natural (remolino de la coronilla horario → izquierda, antihorario →
  derecha; si no, el lado contrario al que va el pelo de la frente).
- **`app/pipeline/growth_rules.py`** sustituye la regla vieja de remolinos
  (que avisaba con < 1 cm arriba, al revés de lo que dicen las guías):
  coronilla: 2,5-7,5 cm arriba = aviso salvo con textura; < 2,5 o > 7,5 a
  favor; doble remolino: undercut o textura. Frente: remolino → aviso a
  flequillo liso, a favor de tupé, flequillo con textura o raya en el
  remolino; crecimiento hacia delante → flequillo a favor, hacia atrás en
  contra (y al revés). Raya lateral del lado natural a favor. Nuca con
  remolino o crecimiento de lado: línea recta/cuadrada en contra, largo o
  degradado a favor. Fuentes en el docstring (Book of Barbering, Cadmen
  Academy, Fellow Barber). La forma de cara sigue igual y se suma.
- **`app/pipeline/combined_rules.py` (sept 2026, cruces crecimiento +
  visajismo/forma de cara).** Pedro, tras terminar de dejar el maniquí
  como herramienta de dibujo pura: "olvida las entradas ahora mismo para
  el modelo estándar de maniquí... ahora lo que debemos hacer es que la
  dirección del pelo sirva para recomendar peinados y cortes de pelo que
  encajen con esa dirección y además que encajen con su perfil de
  visajismo". Esto último (cruzar crecimiento + visajismo en la misma
  puntuación) ya estaba conectado de punta a punta desde la v4
  (`recommend_styles` ya suma `growth_rules` + `visagismo_rules`/
  `trait_rules` + forma de cara sobre el mismo corte); lo que pidió
  después, al elegir entre las opciones que le planteé, fue añadir reglas
  NUEVAS que reconozcan cuándo dos señales de esos módulos van a la vez
  a favor o en contra, en vez de dejar solo notas sueltas. Tres reglas,
  todas opcionales (piden `face_shape_override` y/o `visagismo_profile`
  y/o el mapa de crecimiento — lo que falte, simplemente no se activa):
  cara redonda + raya natural (por remolino o por hacia dónde va el pelo
  de la frente) + corte de raya lateral, a favor (asimetría que además
  aguanta sin producto); cara alargada + remolino en la coronilla en el
  rango de largo donde ya "gana el remolino" (2,5-7,5 cm) + volumen arriba
  sin textura, aviso reforzado (dos motivos independientes empujan a lo
  mismo: no dar más altura); entradas en M + remolino en la frente + tupé
  largo hacia atrás, aviso explicando que aquí las dos reglas sueltas
  tiran en direcciones CONTRARIAS (`growth_rules` anima a aprovechar el
  remolino, `visagismo_rules` avisa de que expone las entradas) y por qué
  gana taparlas. Ninguna es un hecho nuevo sin base: la de raya
  lateral+redonda es la misma recomendación de asimetría que ya usan
  varias guías generalistas de barbería (la misma familia de motivo por
  la que aquí ya se prioriza raya lateral en cara alargada, solo que por
  anchura en vez de por asimetría); las otras dos son composiciones
  directas de reglas que ya existían por separado, no fuentes nuevas. Ver
  el docstring de `combined_rules.py` para el porqué de no cubrir más
  combinaciones (solo las que no estén ya cubiertas por una regla
  individual, para no puntuar dos veces lo mismo).
- **Frecuencia de retoque** (`app/pipeline/maintenance.py`): semanas por
  corte (degradado alto/a piel 2-3, bajo/medio 3-4, corto 3-4, medio 4-6,
  largo 6-8; guías de barbería citadas en el docstring), en
  `StyleOut.maintenance_weeks` y como etiqueta "2-3 sem." en catálogo y
  recomendaciones. En `recommend_styles` solo cuenta como desempate:
  entre cortes con la misma puntuación, primero los de retoque más
  frecuente. **La frecuencia de visitas del cliente ya no influye en las
  recomendaciones** (Pedro lo pidió, sept 2026): se quitaron los efectos
  "Se verá crecido antes" / "Encaja con sus visitas" de `recommender.py` y
  la regla de frecuencia de visitas de `visagismo_rules.py` ("Aguanta entre
  visitas" / "Se nota crecido pronto"). La pregunta del cuestionario se
  mantiene porque sirve para calcular el próximo corte (abajo).
  El cliente ve "Próximo corte en N días" / "Te toca corte" en su espacio
  (`GET /api/me/next-visit`), y la sala tiene "Les toca volver"
  (`GET /api/return-due`, solo peluquero): quién se ha pasado de fecha (o
  le toca en 2 días) y no está hoy, con WhatsApp (mensaje ya escrito,
  `wa.me`, prefijo 34 si el número tiene 9 cifras) y llamada. La fecha sale
  del último corte registrado (con su intervalo) o del último "Terminado"
  (intervalo por defecto 3-5), acortada si el cliente dijo que viene más a
  menudo (`repository.last_visits`).
- Tests: `tests/test_growth.py` (zonas, direcciones, raya, reglas,
  semanas, desempate, y las reglas cruzadas de `combined_rules.py`) y dos
  nuevos en `tests/test_sessions.py` (lista de vuelta y próxima visita,
  endpoint de resumen). Probado con Playwright:
  dibujo con ratón y con toques reales (CDP), remolinos, vistas, borrar,
  guardar y recargar, mapa antiguo, sala, espacio del cliente y
  recomendaciones.
- Pendiente: la línea del pelo del maniquí es genérica (no la del
  cliente); el mapa sigue siendo sobre la cabeza genérica, no sobre su
  foto.

## Maniquí personalizado del cliente (v5, sept 2026)

Pedro: "que el maniquí vaya cambiando en función de las características del
cliente: si tiene el pelo afro, que tenga pelo afro y sus físicas, así con
todos los tipos de cabello, la longitud y las características de visajismo
(mandíbula, ojos, simetría, orejas, entradas...)". Eligió verlo en Remolinos
y en la ficha, largo automático + ajustable, balanceo al girar y selector de
color.

- **Parámetros** (`app/pipeline/avatar.py`, `GET /api/clients/{id}/avatar`,
  solo peluquero): tipo de pelo (el del peluquero, si no el del cliente, si
  no liso), color (`clients.hair_color`), densidad y línea del pelo
  (`hair_physical_metrics`), largo de hoy y pesos de los rasgos.
- **Largo de hoy**: largo puesto a mano (`clients.current_length` +
  `current_length_at`, con el degradado que conserve) o el del último corte
  del historial que sea del catálogo, más lo crecido (0,41 mm/día, máx. 6
  meses); si no hay nada, 4/1,5/1,5 cm. `PATCH /api/clients/{id}/appearance`
  guarda color y largo (`current_length: null` = volver a automático).
- **Rasgos → morphs** (tabla en `avatar.morph_weights`): forma de cara
  (override del peluquero/cliente o `facial_geometry`, 7 formas), cráneo
  braqui/dolicocéfalo, mentón retraído/prominente, mandíbula suave/definida,
  orejas salientes, ojos juntos/separados, asimetría ocular (solo si la nota
  del análisis dice qué ojo está más abierto; si no, no se inventa el lado),
  perfil convexo/cóncavo, cuello corto/largo, arco superciliar marcado y
  frente alta/baja. Los morphs son targets CC0 de MakeHuman calculados sobre
  el maniquí (`frontend/assets/rasgos/*.bin`, int16 dispersos + índice,
  2,3 MB en total pero cada cliente descarga solo 2-5 archivos pequeños);
  los genera `tools/construir_cabeza_masculina.py` (tabla `MORPHS`). Se probó
  meterlos como morph targets del .glb: 5 MB, y WebGL solo mezcla 8 a la vez;
  por eso van aparte y se suman en la CPU al cargar.
- **Línea del pelo y color** (`assets/avatar.js`): el .glb v5 ya no trae el
  pelo pintado; trae `_HAIRH` (distancia a la línea del pelo), `_HAIRTH`
  (ángulo) y `_EARS`, y la página pinta el color del cliente y mueve la
  línea: entradas en M (sube en las sienes), frente alta, pico de viuda.
  Densidad baja = menos mechones y cuero cabelludo más claro.
- **Pelo por tipo** (`Avatar.buildHair`): línea central que sigue el campo de
  flechas/remolinos junto a la raíz, luego gravedad (liso 1, ondulado 0,75,
  rizado 0,3, afro 0) o hacia fuera (rizado 0,5, afro 1,6), sin atravesar la
  cabeza; encima, ondas (ondulado) o espiral (rizado R=1,6 cm·escala, afro
  espiral pequeña y apretada); "encogimiento" del rizo (rizado 0,6, afro
  0,7). Largo por zona (arriba/laterales/nuca, mezcla suave) y degradado
  (bajo/medio/alto/piel: por debajo de su altura el largo baja a lo crecido).
  Por delante de la cara el pelo largo se aparta a los lados para no taparla.
  ~2.600 mechones (afro ×1,6, rizado ×1,25); 50-250 ms por redibujo.
- **Balanceo**: muelle amortiguado movido por la velocidad de giro de la
  cámara; desplaza las puntas en el shader (atributo `sway`, más en pelo
  largo y liso, casi nada en afro), sin rehacer los mechones.
- **Remolinos** (`growth-map.html`): nuevos controles Color y "Largo de hoy"
  (3 deslizadores, chip con el origen del largo y botón Auto). "Guardar"
  guarda también color/largo. Al aplicar rasgos se rehace la rejilla y se
  vuelven a pegar flechas y remolinos a la superficie.
- **Ficha**: vista del maniquí incrustada (`growth-map.html?embed=1` en un
  iframe, solo lienzo y vistas) con "Editar maniquí".
- Tests: `tests/test_avatar.py` (rasgos, lado de la asimetría, que cada morph
  tenga su archivo, largo automático/manual/tope) y
  `test_sessions.test_avatar_and_appearance`. Capturas probadas: afro con
  cara redonda, orejas salientes y mentón retraído; liso largo rubio con
  entradas; rizado con cara alargada, nariz convexa, cuello largo y un ojo
  más pequeño; ondulado con degradado alto, pico y densidad baja.
- Límites: la barba del cliente todavía no se dibuja; la escala 1 mm =
  0,0041 supone una cabeza de ~15,5 cm de ancho.

**v5.1: pelo natural y largo que dice el cliente (sept 2026).** Pedro: "dale
forma natural al pelo: aunque sea liso tiene un poco de textura, al peinarlo
puede tapar entradas; más densidad, no tanta separación entre cabellos", y
"pregunta también cómo de largo tiene el pelo, para el maniquí".

- **Mechones en vez de líneas** (`Avatar.buildHair`, devuelve un
  `THREE.Group` con `userData.uSway`; liberar con `Avatar.disposeHair`):
  ~3.000 raíces (afro ×1,2), cada una una cinta ancha (liso 0,024 ≈ 6 mm,
  ondulado 0,022, rizado 0,014, afro 0,011) con `MeshStandardMaterial`
  iluminado, llena casi hasta el final y con punta afilada (`taper`), más una
  hebra fina encima (líneas) con algo de brillo. Las cintas se solapan y
  tapan el cuero cabelludo; la pintura del cuero cabelludo sube a 0,92.
- **Textura natural también en el liso**: cada mechón sale algo desviado del
  peinado, con una ondulación suave e irregular (suma de dos senos), se
  retuerce un poco, y los largos varían ±15 % (puntas desiguales).
- **Peinado que tapa**: la gravedad va sobre todo a lo largo de la piel (el
  pelo se apoya en la cabeza y cae por los lados) y cada tramo queda algo
  más separado de la piel, en capas; el pelo largo de arriba cae sobre la
  frente y las entradas. El apartado de la cara solo actúa por debajo de las
  cejas (un flequillo sobre la frente sí se ve).
- Rendimiento (swiftshader): 70 mm liso ~160 k vértices / 230 ms; melena
  larga ~340 k / 700 ms; afro ~350 k / 320 ms.
- **Pregunta "¿Cómo de largo tienes el pelo?"** en `cuestionario.html` (7
  preguntas): rapado, lados cortos, corto, medio, por la barbilla, por los
  hombros, con dibujos de la cabeza con el pelo hasta donde llega. Se guarda
  en `hair_physical_metrics.hair_length` + `hair_length_at` (la fecha solo
  cambia si cambia la respuesta). Tabla de mm en `avatar.HAIR_LENGTHS`.
- **Largo de hoy** (`avatar.current_length`): manda el puesto a mano por el
  peluquero; si no, lo más reciente entre el último corte y la respuesta del
  cliente (fuente "cliente", chip "Dice él + N mm"), más lo crecido. Sale en
  la ficha como chip.
- Tests: `test_avatar.test_client_answer_vs_last_cut` y el final de
  `test_sessions.test_avatar_and_appearance`.

**v5.2: menos aspecto de "púas" (sept 2026).** Pedro, viendo el maniquí:
"mejora el tipo de pelo... que se vea mejor/más realista". Comparando
capturas de las 4 texturas en varios largos (frente, perfil y cenital), el
problema estaba sobre todo en liso/ondulado: cerca de la coronilla y en la
raya del flequillo salía "de púas" (mechones cortos apuntando en
direcciones casi al azar) en vez de peinado. Causas encontradas en
`Avatar.buildHair` (`assets/avatar.js`):
1. El campo de direcciones (`flowField` en `growth-map.html`, "irradia
   desde la coronilla" cuando no hay flechas dibujadas) ya devuelve un
   vector UNITARIO, así que el jitter por mechón (para que no salgan todos
   paralelos) competía con él en la misma magnitud -- y justo cerca de la
   coronilla el campo cambia muy rápido de una raíz a la vecina (es una
   singularidad, irradia desde un punto), así que el jitter dominaba y cada
   mechón salía disparado en una dirección casi al azar. Se bajó su peso
   (0,5 -> 0,16 al multiplicar el vector, y su influencia decae algo más
   rápido con la distancia a la raíz).
2. La gravedad/peinado tardaba en pesar (rampa hasta el 4 % del largo del
   mechón): el primer tramo, el más visible cerca de la raíz, dependía solo
   del campo + jitter sin nada que lo estabilizara. Ahora pesa antes (2 %).
3. La punta del mechón se afinaba casi hasta una aguja (10 % del ancho de
   la raíz); con mechones finos eso se lee como pincho. Se dejó una punta
   más redondeada (33 % del ancho).
4. La variación de largo entre mechones ("puntas desiguales", para que no
   parezca un casco) era de hasta ±15 %; con pelo corto (2-3 cm) ese ±15 %
   ya se notaba como si fueran mechones sueltos de largos muy distintos.
   Se dejó más discreta (±7 %).
- Resultado, comparado con capturas antes/después: mejora clara en pelo
  medio/largo (perfil, cae en un flujo continuo en vez de disparado) y en
  ondulado/rizado/afro (se ven las ondas/rizos, no ruido); el pelo liso muy
  corto visto de frente (flequillo/coronilla en cámara) TODAVÍA se ve algo
  puntiagudo -- es en parte una limitación de la técnica (cada mechón es
  una cinta plana; visto casi de canto, cerca de donde apunta, se ve su
  punta en vez de su lado ancho, más notorio cuanto más corto y más recto
  es el pelo). Arreglarlo del todo pediría agrupar mechones en "mechones
  visuales" más anchos o suavizar el campo de direcciones cerca de la
  coronilla en vez de solo bajar el ruido -- no se ha hecho por no
  sobre-ajustar sin que Pedro vea antes el resultado actual.
- Sin tests automáticos (es ajuste visual de Three.js, sin lógica de
  backend de por medio); verificado con capturas de Playwright contra un
  servidor local en las 4 texturas, tres largos (corto/medio/largo) y tres
  ángulos (frente/perfil/cenital), sin errores de consola.

**v5.3: disimular entradas al peinar hacia un lado (sept 2026).** Pedro:
"cuando tenga entradas, al poner el pelo hacia un lado, se disimulen, o que
no parezca que tiene tantas entradas, porque si no no parece real".

- El nacimiento del pelo con entradas (`hairlineShift("m_shaped_receding",
  th)`, ver más arriba) YA hacía bien lo suyo: en la zona de las sienes no
  nace ningún mechón (es lo correcto, ahí no hay pelo). El problema era que,
  sin nada más, el pelo de al lado no llegaba a taparlo: cada mechón seguía
  el peinado normal (irradia desde la coronilla + algo de ruido) sin "saber"
  que un poco más allá la piel está calva, así que la entrada se veía como
  un hueco limpio en vez de disimulada, que es lo que haría un peinado real.
- **Cobertura de entradas** (`Avatar.buildHair`): cada mechón mira, una vez
  al nacer (no en cada tramo: la piel no cambia mientras crece), si la piel
  más adelante en la dirección en la que ya se peina (0,09 y 0,2 unidades
  de escena, ~2 y ~5 cm: una entrada real es ancha, hay que mirar bastante
  más lejos que un solo tramo) está calva. Si lo está, ESE mechón se peina
  de forma más disciplinada hacia allí (menos ruido/desviación propia, más
  apoyado en la piel) para llegar a cubrirla, en vez de dejar que el resto
  del peinado lo desvíe. Nuevo parámetro opcional `ctx.maskAt(punto) ->
  0-1` (cuánto pelo hay ahí "de por sí"): lo da `growth-map.html`
  reutilizando la rejilla de `nearestVertex` que ya tenía para pegar
  flechas y remolinos a la superficie; si no se pasa, se comporta como
  antes (ningún otro llamador de `Avatar.buildHair` lo necesita).
  Importante: esto NO inventa pelo donde no lo hay ni cambia el dato
  guardado de la entrada -- sigue sin nacer ningún mechón ahí --, solo
  peina mejor el pelo de alrededor para que la disimule, como en la vida
  real.
- Verificado comparando capturas con la cobertura activada/desactivada (se
  puede anular pasando `ctx.maskAt: undefined`) en corto (2,6 cm) y medio
  (6,5 cm) de largo, con y sin una "raya lateral" dibujada a mano: en los
  dos casos la entrada se nota bastante menos; con pelo demasiado corto (el
  mechón no llega a cruzar toda la anchura de la entrada) sigue notándose
  algo, como pasaría de verdad.

**v5.3.1: que quede repeinado, no solo tapado (sept 2026).** Pedro: "si el
flequillo es hacia abajo (liso/ondulado) y con entradas, al peinarlo hacia
un lado quede repeinado". La cobertura de arriba ya llevaba el mechón hasta
tapar la entrada, pero seguía con su ondulación/textura suelta de siempre
-- se notaba que era pelo normal que por casualidad pasaba por ahí, no un
peinado hecho a propósito para disimular.

- Mismo mechón "en cobertura" de la v5.3 (variable `coverage`), pero solo
  para liso/ondulado (`slick = coverage` en esos dos; en rizado/afro no se
  toca -- no se alisan para tapar una entrada, no tendría sentido). En esos
  mechones: menos capas/separación de la piel (más pegado, `minH` crece
  hasta un 55 % menos con el largo), menos variación de altura entre
  mechones (`r.layer` a la mitad), la onda de `ondulado` hasta un 75 %
  más suave y el punteado/ruido natural de la textura hasta un 60 % menos.
  El resto del pelo (el que no está tapando nada) sigue con su textura
  normal -- solo el mechón "de peluquería" queda liso y pegado.
- Sin tests automáticos, igual que el resto de ajustes visuales de
  `Avatar.buildHair`; verificado con capturas en liso y ondulado, largo
  medio, con entradas y con/sin raya lateral dibujada, sin regresión en el
  resto de combinaciones (texturas/largos sin entradas se quedan igual,
  `slick` sale 0).

**v5.4: dejar dibujar flechas sobre la frente para el flequillo largo (sept
2026).** Pedro: "el problema es que para el flequillo aunque dibuje las
líneas al ser tan bajo y no poder pintar flechas tan abajo se queda igual".
Con capturas se vio el problema exacto: en `growth-map.html`,
`pointerdown`/`pointermove` solo dejaban empezar o seguir una flecha donde
`_scalp` (atributo horneado en `head.glb`, 0-1 según la distancia a la
línea del pelo "de fábrica") fuera >= 0,35 / >= 0,2. Esa máscara es casi 0
en cuanto se baja un poco de la línea del pelo -- es decir, en TODA la
frente por debajo del nacimiento, que es justo donde cae visualmente un
flequillo largo (puede llegar hasta cerca de las cejas). El barbero podía
dibujar la flecha por encima del nacimiento, pero nunca lo bastante abajo
como para redirigir el propio flequillo; de ahí que "se quede igual" por
mucho que dibujara.

- No hacía falta tocar el campo de direcciones (`flowField()` ya actúa por
  distancia en el espacio, no le importa si el punto es "cuero cabelludo"
  o no) ni el crecimiento del pelo (`Avatar.buildHair`): el único cuello de
  botella era la propia herramienta de dibujo.
- Arreglo: además de `_scalp`, `growth-map.html` ahora también lee del
  `.glb` los atributos `_hairh`/`_hairth` (los mismos que usa
  `Avatar.hairMask` para dibujar entradas/pico/frente alta) y, con la
  misma tabla `HAIRLINE` de `tools/construir_cabeza_masculina.py` portada
  a JS (`HAIRLINE_BASE`/`hairlineBase()`), reconstruye la altura real del
  punto tocado (0 = altura de los ojos, 1 = arriba de la cabeza). Un punto
  se acepta también como "sobre el pelo" a efectos de dibujar flecha
  (`hit.onForehead`) si esa altura es >= 0,10 (justo por encima de las
  cejas) y está en la mitad frontal de la cara (`_hairth < 70`, el mismo
  corte que ya separa cara de sienes/orejas en otras máscaras). Por fuera
  de esa zona (sienes, nuca, o ya en cejas/ojos/nariz/boca) sigue mandando
  el umbral de `_scalp` de siempre -- no cambia nada ahí.
- Los remolinos (`whorl-cw`/`whorl-ccw`) no se tocan: siguen exigiendo
  `_scalp` real, porque un remolino sí representa un punto de nacimiento
  del pelo, no un peinado por encima de piel sin pelo.
- Verificado sin red (Playwright + servidor local, ver
  `scratchpad/hair_flequillo_*.py` de esta sesión): con el arreglo
  desactivado a mano (`hit.onForehead = false`), el mismo gesto de arrastre
  desde el nacimiento hasta encima de las cejas no llega a crear ninguna
  flecha (exactamente el bug de Pedro); con el arreglo, la misma flecha se
  crea completa, con puntos que llegan hasta un `y` a la altura de las
  cejas.

## Gemelo digital 3D del cliente (Tripo AI, sept 2026)

Pedro pasó un anuncio de ILTONIF ("Tu corte, calculado") y pidió integrar lo
que sale en él: foto -> modelo 3D de su cara -> "tu gemelo, en 360°" ->
medidas de cada rasgo -> corte recomendado con % -> "ficha para tu barbero".
Eligió: todo el flujo del vídeo, usar el .glb de Tripo tal cual (no adaptar
el maniquí de MakeHuman a la forma del cliente), generarlo con las 3 fotos
guiadas de visajismo, y que lo lance el peluquero con un botón.

- **Proveedor** (`app/pipeline/avatar3d.py`): Tripo AI
  (`https://api.tripo3d.ai/v2/openapi`), tarea `multiview_to_model` con las
  vistas en el orden [frontal, izquierda, atrás, derecha] (la de atrás va
  vacía: no se le hace foto a la nuca). Se sube cada foto, se sondea la
  tarea y se descarga el `.glb`. Coste ~20-30 créditos = ~0,20-0,30 $ por
  modelo. Configurar `TRIPO_API_KEY` en Railway (cuenta de pago: en el plan
  gratuito los modelos llevan CC BY y obligan a citar a Tripo).
  **Sin probar contra la API real** (desde el entorno de desarrollo no hay
  salida a esa red ni clave, igual que pasó con Gemini/FLUX): si el formato
  de subida o de respuesta no coincide, lo que hay que tocar es
  `_upload_photo`/`_output_url`. Los tests usan la red simulada.
- **Límite honesto**: Tripo es un modelo GENERATIVO, no un escáner.
  Reconstruye una cabeza plausible; la nuca, las orejas y el pelo son en
  parte inventados y dos fotos de la misma persona no dan la misma malla.
  Por eso todo lo medido sale marcado como estimación.
- **Medidas** (`app/pipeline/mesh_metrics.py`, solo numpy, sin dependencias
  nuevas): lee el `.glb` a mano (cabecera + JSON + búfer, con las
  transformaciones de los nodos), orienta la cabeza buscando el plano de
  simetría y de qué lado está la nariz (saliente estrecho) y mide sobre la
  malla: proporción alto/ancho, simetría %, convexidad del perfil (los
  mismos ángulos de `guia-visagismo.html`), anchos de frente/pómulos/
  mandíbula/cuello y un índice de definición mandibular. `classify()` los
  traduce a `facial_geometry`, `profile_type`, `jawline_definition` y
  `neck_proportions`, que se copian a la ficha SIN pisar lo que el
  peluquero haya puesto a mano (misma fusión que el análisis 2D).
  A propósito NO se da un "ángulo mandibular" en grados como en el vídeo:
  el gonion real no se puede localizar en esta malla y un número inventado
  con decimales es peor que no darlo. Los umbrales están puestos a ojo y
  solo validados contra el maniquí del repo -- hay que revisarlos con
  clientes reales, como se hizo con los umbrales 2D.
- **Endpoints** (peluquero): `POST /api/clients/{id}/avatar3d` (multipart,
  frontal obligatoria y los dos perfiles opcionales), `GET .../avatar3d`
  (estado + medidas), `GET .../avatar3d/model.glb`, `DELETE .../avatar3d`,
  y `GET .../barber-sheet[?style_id=]` (también `/api/me/barber-sheet`).
- **% de encaje** (`recommender.match_percent`): la puntuación de las
  reglas escrita de forma legible (60 % sin señales, ±10 por razón o
  aviso, límites 20-99). No es una probabilidad, y cortes con las mismas
  señales salen con el mismo porcentaje.
- **Ficha "cómo pedirlo"** (`app/pipeline/barber_sheet.py`): laterales
  (degradado + mm), parte superior (largo + acabado según textura), nuca,
  peinado (del mapa de crecimiento), producto y cada cuántas semanas
  volver. Todo sale de datos que ya estaban en la ficha.
- **Web**: `frontend/gemelo.html` (visor del `.glb` con vistas y giro,
  chips de rasgos, top 3 con %, ficha para el barbero, y el bloque para
  crearlo con las 3 fotos y el permiso). Enlace desde la ficha ("Gemelo
  3D") y botón "Crear gemelo 3D" en Visajismo, que reutiliza las fotos ya
  hechas con la cámara con marco.
- **RGPD**: consentimiento propio `consent_3d_scan` (columnas
  `consent_3d_scan`/`_at`), porque es una finalidad nueva: la foto va a un
  tercero (Tripo) Y el modelo 3D de su cara se guarda
  (`CLIENT_PHOTOS_DIR/<id>/avatar3d/model.glb`, dentro del Volume). Las
  fotos no se guardan en ningún momento. Retirar el permiso (cliente o
  peluquero) borra el modelo. Un modelo 3D de la cara es dato biométrico:
  revisar el texto del consentimiento con alguien de RGPD antes de usarlo
  con clientes reales.
- Tests: `tests/test_avatar3d.py` (lectura del .glb, medidas sobre el
  maniquí, invariancia a giro/escala, cliente de Tripo simulado),
  `test_sessions.test_avatar3d_needs_consent_and_a_key` y
  `test_growth.BarberSheetTest`. Probado con Playwright contra un servidor
  con Tripo simulado: estado vacío, permiso, creación, visor, rasgos,
  porcentajes y ficha.
- Pendiente: el gemelo no sustituye todavía al maniquí de remolinos (el
  `.glb` de Tripo trae su pelo pegado y no tiene cuero cabelludo marcado,
  que es lo que necesita el sistema de pelo/flechas); la textura del pelo
  y los remolinos siguen saliendo de la foto y de lo que marca el
  peluquero, no de la malla.

**Simular un corte del catálogo sobre el gemelo (sept 2026).** Pedro pidió
además que el gemelo sirviera para "extraer resultados de visajismo... y
proponer cortes" (YA cubierto por lo de arriba: `mesh_metrics` ya
alimenta `visagismo_profile`, que ya matiza `recommend_styles` -- no hizo
falta código nuevo) y para "simular cortes del catálogo" sobre el propio
modelo 3D. Antes de montarlo se investigó si el retexturizado de Tripo
(`texture_model`, con `part_names` para apuntar a una pieza segmentada
con `mesh_segmentation`) serviría para esto, y se descartó: solo cambia
el COLOR/patrón de la superficie, no la geometría, así que no puede darle
volumen a un corte largo -- como mucho serviría para algo muy rapado y
pegado al cráneo, sin ninguna garantía de que el pelo salga como pieza
segmentable en un escaneo fotorrealista (viene horneado en la piel, ver
más arriba), y cada intento cuesta créditos aparte de los del gemelo.
Pedro, tras ver esto, eligió reutilizar el editor de fotos por IA que ya
existía (`haircut_editor.py`) en vez de tocar Tripo para esto:

- **`client_service.simulate_on_avatar3d`**: misma función que
  `simulate_with_stored_photo` (mismo `haircut_editor.edit_haircut`,
  mismo `consent_simulation`, mismo tope diario para el cliente -- se
  compartieron los chequeos en `_simulate_checks`), pero la imagen de
  partida es una captura del VISOR del gemelo (una foto de lo que se ve
  en pantalla, no la malla en sí), no la foto guardada. Exige que el
  cliente ya tenga un gemelo 3D creado (409 si no). Deliberadamente NO
  usa `consent_3d_scan` para esto: ese consentimiento ya cubre crear y
  guardar el modelo, no reenviar una vista suya a un proveedor externo de
  edición -- es la misma finalidad de tratamiento que la simulación sobre
  foto, así que comparte su consentimiento, no crea uno nuevo.
- **Endpoint**: `POST /api/clients/{id}/avatar3d/simulate` (multipart:
  `style_id`, `provider` opcional, `render` la captura en sí). Solo
  peluquero (mismo criterio que el resto de `clients_routes.py`).
- **`frontend/gemelo.html`**: cada corte de "Su corte" tiene un botón
  "Probar" que gira la cámara a Frente (`setView("front")`), espera dos
  frames a que la escena termine de moverse, cambia `scene.background` a
  un gris neutro SOLO para ese frame (el canvas es transparente para que
  se vea el degradado del visor, pero un JPEG no tiene canal alfa: sin
  esto la captura saldría con fondo negro), lee el canvas con
  `renderer.domElement.toBlob(...)` y lo manda al endpoint de arriba. El
  resultado se enseña debajo, en una sección nueva ("Simulado").
- Tests: `tests/test_sessions.test_avatar3d_simulate_reuses_the_photo_editor`
  (sin gemelo 409, sin proveedor 503, sin `consent_simulation` 422, éxito
  con el editor simulado). Probado además con Playwright de extremo a
  extremo contra un servidor con Tripo Y el editor de fotos simulados
  (este último devuelve la misma imagen con un texto superpuesto, para
  poder comprobar visualmente que la captura del gemelo llegó y volvió):
  captura, envío, resultado en pantalla, sin errores de consola, en
  escritorio y en móvil.

**Un único paso de análisis: se quita Visajismo como paso manual aparte
(sept 2026).** Pedro: "que tan solo con el gemelo digital el sistema
analice todo... que tan solo el peluquero pueda editar resultados una vez
extraídos por el sistema, de esa manera el peluquero no pierde tanto
tiempo". Pedido en dos partes, resueltas cada una por su lado (se
preguntó primero con AskUserQuestion qué paso manual quitar de los dos que
había, y qué hacer con los remolinos si el gemelo no puede darlos):

- **Visajismo (`facial_traits_analysis.py`) se fusiona en la creación del
  gemelo.** `POST /clients/{id}/avatar3d` ahora, con la MISMA foto frontal
  que ya sube a Tripo, llama también a `facial_traits_analysis.
  analyze_facial_traits` (simetría/separación de ojos, gafas) y fusiona su
  resultado con `merge_detected_features` -- lo mismo que ya hacía con los
  rasgos de la malla 3D (`mesh_metrics.classify`), así que los dos
  análisis (2D y malla) conviven en `facial_features_profile` sin pisarse
  ni pisar lo que el peluquero haya puesto a mano. Ya no hace falta pasar
  por `visagismo.html` para tener rasgos: se quitó del menú y de
  `ficha.html` (el endpoint `POST .../visagismo-auto-analysis` se deja tal
  cual por si hace falta repetir solo esa parte con otra foto). Si el
  análisis 2D falla no rompe la creación del gemelo: se captura la
  excepción y se añade a `avisos`, igual que el resto de avisos de este
  endpoint.
  **Aviso de sandbox, no de código**: en este entorno de desarrollo la
  parte 2D falla siempre con `<urlopen error Tunnel connection failed: 403
  Forbidden>`, porque `facial_traits_analysis` usa BiSeNet para segmentar
  la cara (detectar gafas) y BiSeNet baja los pesos de ResNet-18 de
  `download.pytorch.org` la primera vez que se construye -- y ese dominio
  está bloqueado aquí (mismo tipo de restricción que Tripo/Gemini/FAL, ya
  documentada). El código en sí está verificado: la ruta de fusión
  funciona (test con `analyze_facial_traits` simulado, ver abajo) y el
  fallo se captura sin tirar el resto de la petición (test con
  `analyze_facial_traits` fallando). **Falta comprobar en Railway/el Mac
  de Pedro** si esa descarga funciona ahí (debería, si hay salida a
  internet normal, y una vez bajados los pesos quedan cacheados) -- no dar
  por hecho que los rasgos 2D van a aparecer hasta probarlo con una clave
  real.
- **Remolinos: el gemelo NO puede darlos, se investigó y se descarta.**
  Se miró si la malla o la textura de Tripo podían dar alguna pista sobre
  hacia dónde crece el pelo o dónde hay remolinos, para no depender de que
  el peluquero los dibuje a mano en `growth-map.html`. Conclusión: no, por
  dos motivos independientes, no por falta de un algoritmo mejor:
  1. **La nuca/coronilla, que es donde están casi todos los remolinos,
     nunca se fotografía.** `avatar3d.py` manda la vista "atrás" VACÍA a
     Tripo a propósito (no se le pide al cliente una foto de su nuca) --
     así que esa parte de la cabeza no se reconstruye a partir de nada
     real, se INVENTA con lo que Tripo considera plausible para una
     cabeza genérica. Cualquier "remolino" que se viera ahí en la malla o
     la textura sería una alucinación del modelo generativo, no una
     medida del cliente -- justo el motivo por el que `mesh_metrics.py`
     ya avisa de que todo lo suyo es una estimación, pero aquí es peor:
     no hay ninguna foto real detrás de esa zona con la que contrastar.
  2. **El pelo no existe como capa aparte.** El pelo de Tripo viene
     "horneado" en la piel/textura del modelo (ver la nota de arriba
     sobre por qué el retexturizado no sirve para simular cortes): no hay
     cuero cabelludo separado ni ninguna estructura de mechones o campo
     de direcciones que analizar. `growth_analysis.py` necesita un dato
     que este `.glb` no representa en absoluto, no uno que esté ahí pero
     sea difícil de leer.
  Por eso `growth-map.html` (y el maniquí de pelo de `avatar.js` que lo
  usa) se queda como estaba: un paso MANUAL del peluquero, no sustituible
  por el gemelo. La diferencia con Visajismo es justo esta: los rasgos de
  cara sí estaban en la foto (solo hacía falta juntar dos análisis en una
  llamada), la dirección del pelo nunca lo estuvo. Como ya era opcional
  antes de este cambio (`growth_rules.evaluate_growth` y
  `barber_sheet._styling` ya funcionan con `growth=None`, sin tocar
  código), no hizo falta nada más que dejarlo así explícitamente en vez de
  quitarlo.
- **El gemelo se ve y se guarda en la ficha del cliente.** `ficha.html`
  incrusta `gemelo.html?embed=1` (mismo patrón `?embed=1` que ya usaba
  `growth-map.html`: una clase `body.embed` que oculta todo menos el
  visor y lo hace ocupar el hueco) en una sección nueva ("Gemelo 3D"),
  con un botón "Crear gemelo 3D" cuando todavía no existe. El modelo en sí
  ya se guardaba (`avatar3d_path`, ver más arriba); lo que cambia es que
  ahora se ve directamente en la ficha en vez de solo detrás de un enlace.
- Tests: `tests/test_sessions.
  test_avatar3d_also_runs_the_2d_facial_analysis` (fusión 2D+malla sin
  pisarse; y que un fallo del análisis 2D queda en `avisos` sin romper la
  creación del gemelo). Probado además con Playwright extremo a extremo
  contra un servidor con Tripo Y el análisis 2D simulados: sin la entrada
  de Visajismo en el menú, "Crear gemelo 3D" visible antes de crearlo,
  tras crearlo con una foto real aparecen rasgos de LOS DOS análisis en la
  ficha (`eye_symmetry`/`eye_spacing` del 2D, `profile_type`/
  `jawline_definition` de la malla) y el visor incrustado sustituye al
  botón de crear.

**Opción ocultada temporalmente (sept 2026).** Pedro pidió quitar la
opción de "Gemelo 3D" del flujo del peluquero por ahora, para reintegrarla
más adelante cuando estén pulidos los servicios de los que depende
(recuérdese lo ya documentado como pendiente arriba: Tripo nunca se ha
probado contra la API real desde este entorno, y los umbrales de
`mesh_metrics.py` solo están validados contra el maniquí de MakeHuman del
repo, no con clientes reales). Cambio puramente de interfaz, reversible,
sin tocar backend ni borrar nada:

- `frontend/ficha.html`: la sección `#gemelo-sec` (visor incrustado + "Crear
  gemelo 3D") y la llamada a `loadGemelo()` en `reload()` quedan
  comentadas, con nota. La función `loadGemelo()` se deja definida (sin
  llamarla) para no tener que reescribirla al reactivarlo.
- `frontend/visagismo.html`: el botón `#twin-btn` ("Crear gemelo 3D") y su
  `addEventListener` quedan comentados. **Importante para quien lo
  reactive**: si solo se descomenta el botón sin descomentar también el
  `addEventListener`, o al revés, `document.getElementById("twin-btn")`
  devuelve `null` y `.addEventListener` sobre `null` rompe el script
  entero de la página (ningún botón de la página funcionaría, incluido
  "Analizar") -- los dos bloques se comentan y se descomentan juntos.
- Nada más cambia: `avatar3d.py`, `mesh_metrics.py`, `gemelo.html`, los
  endpoints `/avatar3d*` y los tests de este apartado siguen intactos y en
  verde (108/108 en `test_growth.py`+`test_avatar3d.py`+`test_sessions.py`,
  no se ha tocado ni un test). `gemelo.html` sigue funcionando si se abre
  por URL directa; solo se han quitado los DOS enlaces que llevaban hasta
  ahí.
- **Efecto secundario para el peluquero, para que no sorprenda**: como el
  botón "Crear gemelo 3D" era el único camino hasta `mesh_metrics.classify`
  (forma de cara, perfil, mandíbula y cuello desde la malla 3D), mientras
  esté oculto esos cuatro campos vuelven a depender solo de lo que rellene
  el peluquero a mano en Visajismo. El botón "Analizar" de esa misma
  página sigue funcionando y sigue rellenando lo suyo (simetría/separación
  de ojos, gafas) vía `facial_traits_analysis.py`, que no depende del
  gemelo.

## El maniquí deja de representar las entradas (sept 2026)

Pedro: "vamos a eliminar que el maniquí tenga entradas. Si el cliente dice
que tiene entradas que se utilice para la recomendación pero para el
maniquí tan solo longitud de pelo y dirección de las diferentes zonas."
Petición clara de separar dos usos que hasta ahora compartían el mismo
dato (`hair_physical_metrics.frontal_hairline_shape`): dejarlo tal cual
para las recomendaciones (`visagismo_rules.py`, `combined_rules.py`,
`visagismo_ai_advisor.py`, la ficha) y quitarlo del maniquí 3D, que a
partir de ahora solo se mueve por largo por zona (arriba/laterales/nuca) y
dirección de crecimiento (mapa de remolinos) -- lo que ya pedía el título
de esta misma sección antes de la v5 ("olvida las entradas ahora mismo
para el modelo estándar de maniquí", ver más arriba en "combined_rules.py
(sept 2026, cruces crecimiento + visajismo/forma de cara)"): aquello
dejó la recomendación aparte pero el maniquí de la v5 sí que había vuelto
a representarlas (`hairlineShift`, "Cobertura de entradas" de v5.3/v5.3.1)
para que el peinado se viera más realista. Esta vez la petición es
explícita y va sobre el propio maniquí, así que se retira del todo ahí:

- **`backend/app/pipeline/avatar.py`** (`avatar_params()`): ya no incluye
  la clave `"hairline"` en el diccionario que consume el frontend. El dato
  sigue existiendo tal cual en `visagismo_profile` y lo siguen leyendo
  exactamente igual `visagismo_rules.py`/`combined_rules.py` -- no se ha
  tocado nada de la parte de recomendación.
- **`frontend/assets/avatar.js`**: se quita `hairlineShift(type, th)`
  (desplazaba el borde del cuero cabelludo según `m_shaped_receding` /
  `high_forehead` / `widows_peak`) y `hairMask(skin)` pierde su parámetro
  `hairline` -- ahora pinta siempre el nacimiento "de fábrica" del propio
  `.glb` (atributo `_hairh`), igual para todos los clientes. También se
  quita entera la lógica de "Cobertura de entradas" de `buildHair()`
  (v5.3: `coverage` vía `ctx.maskAt`, peinaba el pelo de al lado para
  disimular el hueco calvo; v5.3.1: `slick`, alisaba ese mismo mechón para
  que pareciera repeinado a propósito) y sus cuatro usos aguas abajo
  (jitter, peso de la gravedad, altura mínima sobre el cuero cabelludo,
  amplitud de onda/textura) -- sin `hairlineShift` ya no hay ningún hueco
  calvo que disimular, así que esa lógica quedaba sin sentido, no solo sin
  uso.
- **`frontend/growth-map.html`**: se quita `hairline` del objeto `look`
  por defecto, se quita la función `maskAt()` (el único llamador de
  `Avatar.buildHair` que la pasaba) y las dos llamadas a
  `Avatar.hairMask(skinMesh, look.hairline)` pasan a
  `Avatar.hairMask(skinMesh)`. **Ojo, para quien lea el código**: esto NO
  es lo mismo que `hairlineBase()`/`HAIRLINE_BASE` (la tabla ángulo→altura
  del nacimiento "de fábrica" del propio modelo, usada para decidir dónde
  se puede dibujar una flecha de flequillo, v5.4 más arriba) ni que
  `_estimate_hairline`/`hairline_points` de `face_analysis.py` (una
  aproximación 2D del nacimiento del pelo por landmarks, para el mapa de
  crecimiento por defecto en `head_mesh.py`, sin relación con el maniquí
  de cliente) -- ninguna de esas dos se ha tocado, siguen haciendo lo que
  ya hacían.
- `backend/tests/test_avatar.py`: `test_params` comprueba ahora
  explícitamente que `"hairline"` NO está en lo que devuelve
  `avatar_params()` aunque el cliente tenga marcadas las entradas.
- Verificado con Playwright (servidor local + un cliente con
  `frontal_hairline_shape: "m_shaped_receding"`): el maniquí sale con el
  nacimiento del pelo normal, sin hueco ni mechón "repeinado" de más, en
  frontal y de perfil (que es donde más se notaba antes); al mismo tiempo,
  `GET /api/clients/{id}/recommendations` sigue devolviendo la razón
  "Disimula las entradas" en los cortes recomendados -- confirma que la
  separación pedida (maniquí vs. recomendación) quedó como se pidió, no
  solo a nivel de código sino de comportamiento real.
- Suite completa: 108/108 (`python -m unittest discover -s tests -q`).

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
- ~~Integrar un modelo de difusión con ControlNet en `generator.py`~~ Sustituido por modelos externos de edición por API (`haircut_editor.py`, ver su sección). Pendiente: probar con claves reales y elegir proveedor.
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

## Selector de cortes con búsqueda + alta de cortes nuevos desde ahí (sept 2026)

Pedro: "a la hora de seleccionar cortes o de registrar cortes ya realizados
para añadirlos a la base de datos, en el seleccionador de cortes, deja la
posibilidad de poder hacer una búsqueda del corte, escribiendo letras, o
registrar un nuevo corte que no aparece en las opciones." Con el catálogo ya
en 104 entradas, el `<select>` plano obligaba a desplazarse a ciegas. Se
preguntó dónde aplicarlo y qué hacer con un corte nuevo; Pedro eligió los dos
sitios donde se elige un corte (el simulador y "Corte de hoy" en la ficha) y
que un corte nuevo **se añada también al catálogo general** (no solo quede
en el historial de ese cliente), aceptando que eso implica pedir algo más
que un nombre (largo por zona, degradado, para qué tipos de pelo vale).

- **Dónde vive un corte añadido por el peluquero**: en la base de datos
  (tabla `custom_styles`), NO en `app/pipeline/catalog_data/styles.json`. El
  JSON vive fuera del Volume de Railway a propósito (ver el incidente
  documentado en "Despliegue en la nube" más abajo: cualquier fichero ahí se
  actualiza con cada `git push`, así que NO puede persistir algo añadido en
  producción — se perdería en el siguiente despliegue); la base de datos
  (`backend/data/clients.db`) sí está dentro del Volume y sí sobrevive a los
  despliegues. Por eso un corte nuevo del peluquero tenía que ir a la base
  de datos si de verdad iba a "quedarse".
- **Catálogo combinado, un único punto de fusión** (`app/pipeline/
  style_catalog.py`): `load_catalog()`/`get_style_by_id()` (el JSON puro) se
  dejan completamente intactos — los usan `test_style_catalog.py` y los
  scripts de importación, y romper su pureza (sin depender de la base de
  datos) habría sido un cambio innecesariamente arriesgado para lo que se
  pedía. En su lugar se añadieron `load_full_catalog()` (JSON + `custom_
  styles`) y `get_style_by_id_anywhere()`, y se cambiaron los sitios donde
  de verdad hace falta ver también los cortes propios: `POST /api/simulate`,
  `GET /api/styles`, `recommender.recommend_styles`, y las rutas de sesión/
  historial (`session_routes.py`, `client_service.py`).
- **`repository.list_custom_styles()` tolera una base de datos sin
  `init_db()`** (captura `sqlite3.OperationalError` y devuelve `[]`): varios
  tests (`test_growth.py`, etc.) llaman a `recommend_styles()` directamente
  sin arrancar la app ni crear ninguna tabla — sin esto, añadir la tabla
  `custom_styles` habría roto esos tests con un error de "no existe la
  tabla" en vez de tratarlo, correctamente, como "todavía no hay cortes
  propios".
- **`POST /api/styles`** (solo peluquero, `StyleCreateIn` en
  `app/api/schemas.py`): nombre, descripción opcional, largo de las tres
  zonas (mm), degradado y una lista de tipos de pelo para los que vale. El
  formulario de "añadir corte" (`frontend/assets/style-picker.js`) empieza
  con los CUATRO tipos de pelo marcados por defecto — no vacío ni solo uno —
  porque `suitable_hair_types` es un filtro DURO en `recommend_styles`
  (excluye el corte de las recomendaciones para quien no tenga ese tipo
  marcado, no solo lo reordena): un alta rápida a mitad de un corte real que
  se dejara con un solo tipo marcado por descuido haría que ese corte nunca
  volviera a aparecer para el resto de clientes.
- **`frontend/assets/style-picker.js`** (componente nuevo, vainilla JS, sin
  framework): sustituye al `<select>` en `index.html` (simulador) y
  `ficha.html` ("Corte de hoy"). Busca por nombre/descripción/familia (con
  normalización de acentos — "degradado" encuentra "degradado" aunque se
  escriba sin tilde). Si lo escrito no coincide con ningún corte, aparece
  "+ Añadir «texto» como corte nuevo", que abre el formulario de arriba
  (`POST /api/styles`) y selecciona el corte recién creado al guardar. En
  `ficha.html` convive con el "Otro (uso puntual)" que ya existía (para un
  corte de una sola vez que NO se quiere añadir al catálogo general) vía la
  opción `extraActions` del componente — las dos cosas conviven porque son
  necesidades distintas: "Otro" es una anotación de esa visita, "+ Añadir
  como corte nuevo" es alta real en el catálogo.
- **Fallo de apilamiento (z-index) encontrado al probarlo de verdad, no solo
  a ojo**: en `ficha.html`, el desplegable del selector (`position:
  absolute`, `.sp-menu` en `theme.css`) quedaba TAPADO por la tarjeta
  siguiente ("Historial de cortes") en vez de mostrarse por encima, aunque
  su `z-index` fuera mayor. Causa: `.glass-card` usa `backdrop-filter`, que
  crea su propio contexto de apilamiento aunque el elemento no tenga
  `z-index` propio — un `z-index` interno solo compite dentro de SU contexto
  de apilamiento, no contra el de una tarjeta hermana que va después en el
  HTML. Se arregló dándole a `#log-sec` (`position: relative; z-index: 5;`)
  su propio nivel de apilamiento frente a las tarjetas siguientes. Se
  encontró probando el flujo completo con Playwright contra un servidor
  real (ver abajo), no revisando el código a ojo — con captura de pantalla
  sola no se nota porque el desplegable sí se pinta, solo que detrás.
- Backend: 111/111 tests (`python -m unittest discover -s tests -q`), 3
  nuevos (`test_style_catalog.py`: catálogo combinado sin base de datos
  iniciada y con un corte propio ya guardado; `test_sessions.py`: alta de un
  corte desde el selector, que requiere sesión de peluquero, que aparece
  luego en `/api/styles` y en el historial, y que respeta el filtro de tipo
  de pelo en las recomendaciones).
- Verificado con Playwright de extremo a extremo contra un servidor local
  real (no simulado): en `index.html`, buscar "buzz cut" filtra a un único
  resultado, elegirlo lo selecciona, y escribir un nombre nuevo + guardar
  crea el corte y lo deja seleccionado; en `ficha.html`, el flujo "Otro"
  sigue mostrando el campo de texto libre y guardándose como antes, buscar
  y elegir un corte del catálogo (p.ej. "mullet") oculta ese campo, y crear
  un corte nuevo desde ahí también funciona y lo dio de alta en el
  catálogo general (visible por el peluquero para cualquier cliente,
  filtrado igual que el resto por tipo de pelo). Sin errores de consola
  achacables a este cambio (los únicos errores que salían en `ficha.html`
  eran de `growth-map.html` intentando cargar Three.js desde
  `cdn.jsdelivr.net`, bloqueado en este entorno de desarrollo — mismo tipo
  de restricción de red ya documentada para Tripo/Gemini/FAL/
  `download.pytorch.org`, no relacionado con este cambio).
- Pendiente/límite honesto: el buscador no tiene ninguna función de
  "corregir/fusionar" un corte que el peluquero añadió por error con nombre
  duplicado o casi idéntico a uno ya existente (p.ej. "Fade alto" y "fade
  Alto"); si eso pasa en el uso real, hoy quedarían como dos cortes
  distintos en el catálogo.

## Revisión del catálogo: descripciones reales, deduplicado y fotos fuera (sept 2026)

Pedro pidió revisar `styles.json` antes de la demo con la peluquería real. Tres cambios:

- **Descripciones reales para los 100 cortes importados de Esquire**: desde el
  import (`scripts/import_esquire_styles.py`) el campo `description` de esos
  100 cortes era literalmente el `name` repetido (el artículo de Esquire no
  traía una descripción separada). Se escribió a mano una descripción propia
  y distinta para cada uno, en el mismo estilo que los 4 cortes originales
  (qué se ve, no solo el nombre): p.ej. "Corte mullet con flequillo tazón y
  patilla larga" pasó de describirse como sí mismo a "Mullet con flequillo
  recto estilo tazón por delante, degradado bajo en los laterales y patillas
  largas marcadas."
- **Dos duplicados exactos fusionados**: el ranking de Esquire traía dos
  entradas para "Semirrecogido con moño" (puestos #62 y #86) y dos para
  "Peinado de efecto recién levantado" (#25 y #98) — mismo nombre, mismos mm,
  mismo `fade_type`, mismo tipo de pelo, sin ningún dato real que los
  distinguiera más allá de la foto (que además se ha quitado, ver abajo). Se
  eliminó el duplicado de mayor número de puesto de cada par
  (`esq2023-086-...`, `esq2023-098-...`) en vez de inventarles una diferencia
  artificial. El catálogo pasa de 104 a **102 cortes**. Tests actualizados en
  `backend/tests/test_style_catalog.py` (104→102, 105→103) y
  `backend/tests/test_sessions.py` (`test_haircut_history_max_request_and_privacy`
  ya no exige `reference_image` en el historial, solo `has_photo`, por el
  punto siguiente).
- **Fotos de stock retiradas de los 102 cortes** (decisión de Pedro): las
  fotos de Unsplash documentadas más arriba ("Catálogo de cortes con foto
  propia") eran un placeholder hasta tener fotos reales. Con la demo con la
  peluquería ya decidida, Pedro prefiere construir el catálogo de fotos desde
  cero con cortes reales hechos allí en vez de seguir con las de stock. Se
  puso `reference_image` a `null` en las 102 entradas de `styles.json` y se
  borraron las 116 fotos de `frontend/assets/style_photos/` (las 104 propias
  de cada corte más las 12 huérfanas que ya no usaba ningún corte desde la
  pasada de "casi 1 foto por corte"). No hizo falta tocar código: todo el
  frontend que pinta `reference_image` (`catalogo.html`, `probar.html`,
  `ficha.html`, `mis-cortes.html`, `recomendaciones.html`) ya usaba `|| ""` o
  comprobaba el campo antes de usarlo, y `haircut_editor.load_reference_photo`
  ya devolvía `None` con elegancia si el fichero no existe — así que el
  simulador sigue funcionando (Gemini/FLUX generan a partir del texto del
  corte, sin foto de referencia) y el catálogo/selector muestran las
  tarjetas sin imagen en vez de romperse. Pendiente cuando haya fotos reales:
  rellenar `reference_image` corte a corte (o los que se vayan fotografiando)
  y quitar este apunte.
- Verificado con la suite completa: 111/111 tests
  (`python -m unittest discover -s tests -q`).

## Recomendaciones: simular corte desde la tarjeta y cortes parecidos uno al lado del otro (sept 2026)

Pedro: "que en la sección de recomendación aparezca una opción de simular corte, además de que aparezcan los cortes de pelo similares al lado pero no debajo" -- dos cambios en `frontend/recomendaciones.html`, sin tocar backend.

- **"Simular corte" en cada tarjeta**: enlaza a `probar.html` (mismo destino
  que ya usan el catálogo y la ficha para repetir un corte del historial)
  con ese corte ya elegido (`?style=<id>`) y, en modo peluquero, con el
  `client_id` de la ficha abierta (`simulateHref()`, reutiliza el helper
  `ficha()` que ya existía para el resto de enlaces de la página). Abre
  directamente sobre la foto guardada del cliente -- no hace falta pasar
  por el catálogo ni escribir el nombre del corte a mano.
- **Cortes parecidos uno al lado del otro, no apilados**: antes,
  `groupByFamily()` metía todos los cortes de una misma familia visual
  dentro de una ÚNICA tarjeta, en una lista vertical con un "+N" plegado
  (pensado para cuando compartían la misma foto de referencia de stock).
  Desde que esas fotos se retiraron del catálogo (ver "Revisión del
  catálogo" más abajo, `reference_image` siempre `null`), ese agrupado ya
  solo servía para amontonar cortes de una misma familia dentro de una
  tarjeta cada vez más larga. Se sustituyó por `orderWithFamiliesAdjacent()`:
  cada corte tiene ahora su PROPIA tarjeta completa (foto/badges/por qué/
  Simular corte), pero el orden en el que se pintan mantiene los de una
  misma familia SEGUIDOS, así que la rejilla (`display:grid`, columnas en
  fila) los deja uno al lado del otro de forma natural -- sin necesitar
  ningún contenedor especial para "agrupar visualmente". Cada tarjeta de un
  corte con hermanos lleva además un aviso pequeño ("Parecido a: <familia>")
  para que se entienda por qué están juntos.
- `groupByFamily()` se mantiene tal cual (mismo criterio de agrupado por
  `reference_image`/`style_family`/`id`) porque sigue haciendo falta para
  decidir el orden y la etiqueta de familia -- lo que cambió es que ya no
  se usa para renderizar una tarjeta por familia, sino una por corte.
- Se quitó del CSS lo que solo servía para la lista apilada de antes
  (`cut-list`, `cut-item`, `details.more`) y dos reglas de `con-aviso` que
  ya estaban muertas desde antes de este cambio (ninguna función de JS
  llegaba a aplicar esas clases).
- Sin tests de backend que tocar (cambio 100% de frontend). Verificado con
  Playwright contra un servidor local real (peluquero, ficha con cliente
  sin filtros de pelo/cara -- el caso con más cortes a la vez, 102):
  sin errores de consola, 102 tarjetas individuales, los 3 primeros cortes
  de la familia "Fade / undercut con textura" cayendo en la misma fila uno
  junto a otro (captura de pantalla), y el botón "Simular corte" de la
  primera tarjeta generando el enlace correcto
  (`probar.html?client_id=<id>&style=undercut-flequillo`).

## Visajismo: enlace directo desde la ficha (no solo vía mapa de remolinos) (sept 2026)

Pedro: "no aparece el apartado en el que tomar las 3 fotos para que te haga
el análisis de visajismo" -- `visagismo.html` existía y funcionaba
(guía de las 3 fotos, análisis automático de la frontal, etc.) pero se
había quedado sin ningún enlace directo desde `ficha.html`: el único
camino era abrir "Remolinos" (`growth-map.html`) y desde ahí, si el
peluquero sabía que el icono de la esquina llevaba a visajismo, pulsarlo
-- y aun así ese enlace estaba fijo a `visagismo.html` sin `client_id`,
así que al llegar ahí tocaba buscar al cliente otra vez a mano. Dos
arreglos, sin tocar backend:

- **`frontend/ficha.html`**: nuevo botón "Visajismo" en la fila de
  herramientas (`.tools`), junto a "Remolinos" y "Recomendaciones", usando
  el mismo patrón que ya tenían esos dos (`with_id("visagismo.html")`, el
  helper que añade el `client_id` de la ficha abierta a la URL). Verificado
  con Playwright: al pulsarlo se llega a `visagismo.html?client_id=<id>`
  con el cliente ya cargado (el formulario de análisis visible, sin el
  aviso de "abre la ficha").
- **`frontend/growth-map.html`**: el enlace `#visagismo-link` de su propia
  barra de navegación era estático (`href="visagismo.html"`, sin
  `client_id`). Se añadió una línea dentro de `applyClientToScene(client)`
  para fijarlo dinámicamente en cuanto se carga el cliente
  (`visagismo-link.href = "visagismo.html?client_id=" + client.id`), igual
  que ya hacía el botón de Recomendaciones un poco más abajo en el mismo
  archivo. Comprobado con `node --check` sobre el script embebido (sintaxis
  válida); no se pudo verificar en caliente con Playwright en este entorno
  porque `growth-map.html` carga Three.js desde un CDN bloqueado en el
  sandbox (limitación ya conocida y documentada, no relacionada con este
  cambio) -- el cambio replica exactamente el patrón ya usado y probado en
  `recommendationsBtn` unas líneas más arriba del mismo fichero.
- Sin tests de backend que tocar (cambio 100% de frontend, ningún endpoint
  ni dato nuevo).

## Barba y bigote: motor de recomendación completo, 18 estilos ilustrados (sept 2026)

Pedro pidió integrar un análisis y recomendación profesional de barba y
bigote "aparte de toda la información que tienes agregada ya en el sistema
de recomendación tras el análisis de visajismo", a partir de vídeos y texto
que él mismo transcribió/resumió: 13 tipos de bigote (chevron, Dalí,
inglés, húngaro, fu manchu, horizontal/lápiz, herradura, imperial,
piramidal, morsa/walrus, mosquetero, revolucionario, corto), 5 estilos
base de barba (en collar, completa/clásica, perilla, chiva/chivita, de
varios días/media sombra -- cualquier barba moderna es combinación de
estas 5), la correlación de cada uno con la forma de rostro, y
correcciones por frente, papada/cuello, nariz, mandíbula y labios. También
dio un PDF ("VISAGISMO MASCULINO, tipos de rostros") con la transcripción
de un vídeo sobre cómo dibujar/clasificar la forma del rostro por
proporciones (5 líneas verticales -> "dos unidades y media", 5
horizontales -> "tres unidades y media").

Aviso importante sobre las fuentes: Pedro dio los enlaces a varios vídeos
de YouTube además del texto y las imágenes, pero **este entorno no tiene
capacidad de ver vídeos** (un intento de `WebFetch` sobre una de esas URLs
devolvió un rechazo del proxy, `PROXY_REJECTED`/429, con instrucción
explícita de no reintentarlo). Todo lo que se implementó sale del texto
que Pedro pegó él mismo, el PDF transcrito y las imágenes de guía que
mandó -- no de haber "visto" los vídeos. Si algún matiz de un vídeo no
llegó a estar en el texto/PDF que Pedro pegó, no está cubierto aquí.

- **Motor de reglas nuevo** (`app/pipeline/beard_mustache_rules.py`,
  función `advice(profile, face_shape)`): sustituye a la antigua
  `trait_rules.beard_advice` (que solo cubría mentón retraído y mandíbula
  poco definida). Mismo contrato hacia la API (`list[dict]` con
  `label`/`detail`, expuesto en `RecommendationsOut.beard_advice`), pero
  mucho más completo:
  - **Por forma de rostro** (`face_shape_override`, ahora con 6 valores --
    ver más abajo): ovalada (cualquier estilo vale), alargada (barba
    rebajada + bigote horizontal/lápiz que acorta), redonda (barba de
    candado angulosa o chivita/perilla alargada + bigote grande), diamante
    (barba de candado + bigote con pelo bajo el labio, tipo mosquetero),
    triangular invertida (barba completa y densa en mentón y mejillas),
    triangular (perilla/chivita muy corta y pulida, evitar formas rectas).
    **"cuadrada" se deja SIN regla a propósito**: ninguna fuente que dio
    Pedro la cubre, y el criterio del proyecto es dejar un hueco honesto
    en vez de inventar una regla sin fuente (mismo criterio que ya se
    sigue en `recommender.py`/`combined_rules.py`).
  - **Correcciones** (fusionadas con las reglas de barba que ya existían
    para el mismo campo, en vez de duplicarlas): mentón retraído
    (`chin_projection == "retruded"`, ya existía) + mandíbula prominente
    (`"prominent"`, nueva); mandíbula poco definida (`jawline_definition
    == "soft"`, ya existía, ahora también con el matiz de "dibuja una
    línea de afeitado que alargue el cuello"); papada (`has_double_chin`,
    campo nuevo); frente pequeña/ancha (`intellectual_zone_forehead`,
    campo que YA EXISTÍA en el schema desde antes pero ninguna regla lo
    leía hasta ahora) y entradas (`frontal_hairline_shape ==
    "m_shaped_receding"`, reutilizado de `hair_physical_metrics`); nariz
    grande/pequeña (`nose_size`, campo nuevo); labios prominentes/finos
    (`lip_thickness`, campo nuevo).
- **`face_shape_override` pasa de 4 a 7 valores posibles** (sigue siendo
  un `str` libre, sin enum en el servidor): se añaden `"diamante"`,
  `"triangular"` y `"triangular_invertida"` a los ya existentes
  `"ovalada"|"redonda"|"cuadrada"|"alargada"`. El maniquí 3D YA TENÍA los
  morphs correspondientes desde antes (`face-diamond`/`face-triangle`/
  `face-heart` en `tools/construir_cabeza_masculina.py`, alcanzables solo
  vía el campo inglés `facial_geometry`, nunca usado en la práctica) --
  solo hacía falta mapear los 3 valores nuevos de `face_shape_override` a
  esos mismos morphs en `avatar._FACE_OVERRIDE`, sin tocar el maniquí.
  `triangular` -> cara con la mandíbula más ancha que la frente
  (`head-triangular`); `triangular_invertida` -> frente más ancha que la
  mandíbula (`head-invertedtriangular`, la misma malla que el corazón
  inglés "heart", que es la forma equivalente).
- **Campos nuevos** en `FacialFeaturesProfileIn` (`app/api/schemas.py`):
  `has_double_chin` (bool), `nose_size`
  (`"small"|"proportional"|"large"`), `lip_thickness`
  (`"thin"|"proportional"|"prominent"`). Se rellenan a mano en
  `visagismo.html` (icono "mano", igual que ya se documentaba en la guía
  de esa página para rasgos que las fotos no permiten detectar con
  fiabilidad) -- **no** se añadieron al análisis por visión de IA
  (`visagismo_vision_analysis.py`): esos 6 campos ya estaban acotados y
  probados (ver la sección de ese módulo), y ampliar ese análisis no se
  pidió ni se ha calibrado, así que se quedan como rasgos manuales, igual
  que `has_glasses`.
- **`intellectual_zone_forehead`** (`facial_horizontal_zones_ratio`): campo
  que ya existía en el schema desde hace tiempo sin ninguna UI ni regla.
  Ahora tiene un select en `visagismo.html` (Pequeña/Proporcional/Ancha) y
  alimenta la corrección de frente de `beard_mustache_rules.py`. Los otros
  dos campos del mismo objeto (`affective_zone_mid_face`,
  `sensitive_zone_jaw_chin`) siguen sin UI ni regla -- no se pidieron.
- **Frontend**:
  - `visagismo.html`: nuevos campos Frente/Nariz/Labios (selects) y Papada
    (checkbox) en el formulario de rasgos, guardados en
    `facial_features_profile` (Papada/Nariz/Labios) y
    `facial_horizontal_zones_ratio` (Frente) -- este segundo objeto se
    fusiona igual que el primero (se parte de lo ya guardado, se sobrescribe
    solo `intellectual_zone_forehead`, sin tocar los otros dos campos que no
    tienen UI).
  - `cuestionario.html`: el selector de forma de rostro pasa de 4 a 7
    opciones (se añaden diamante/triangular/triangular invertida, con SVG
    nuevo para cada una siguiendo el mismo patrón que las 4 que ya
    había -- `face('<path d="..."/>')`), y `FACE_TO_GEOMETRY` se amplía con
    los 3 valores nuevos.
  - `recomendaciones.html`: la sección "Barba" pasa a llamarse "Barba y
    bigote" (el campo de la API sigue siendo `beard_advice`), con un
    enlace ⓘ a la guía ilustrada nueva.
- **Guía ilustrada nueva** (`frontend/guia-barba-bigote.html` +
  `tools/generar_barba_bigote_guia.py` + `frontend/assets/guia/barba-bigote.json`):
  18 ilustraciones (13 bigotes + 5 barbas) generadas por script, siguiendo
  el mismo patrón que `tools/generar_perfiles_guia.py` (código en vez de
  SVG pegado a mano). Diferencia importante con aquel script: los perfiles
  se generan a partir de ÁNGULOS con definición publicada y el script
  COMPRUEBA que cada dibujo mide lo que dice; un tipo de bigote o barba no
  es una medida angular, así que aquí no hay ningún número que verificar
  al final -- la única verificación posible fue visual (capturas de
  pantalla con Playwright contra un servidor estático local, corrigiendo a
  ojo dos bugs de geometría: los mechones con voluta enroscada de Dalí e
  Imperial dibujaban una mancha grande en vez de una punta fina hasta que
  se separó la voluta en un segundo mechón encadenado, y la barba
  completa/media sombra tapaba los ojos hasta que se bajó el borde
  superior de la zona de relleno). Estilo visual: el mismo lenguaje simple
  de los iconos de `cuestionario.html` (trazo/relleno de un color, viewBox
  100x100), no las siluetas anatómicas detalladas de
  `generar_perfiles_guia.py` -- esto es un catálogo de referencia visual,
  no una herramienta de medición de fotos reales. La página también
  documenta la técnica de proporciones del PDF de Pedro como contenido de
  referencia/educativo, sin convertirla en medición automática (mismo
  criterio que el resto de rasgos de visagismo, ver
  `app/pipeline/face_analysis.py` y el historial de `wip-perfil-automatico`
  en la sección de "Rasgos de perfil juzgados por IA con visión").
- **Tests**: `backend/tests/test_beard_mustache_rules.py` (nuevo, sustituye
  al `test_beard_advice` que había en `test_trait_rules.py`) cubre las 6
  formas de rostro con regla, que "cuadrada" no tiene ninguna, y cada
  corrección por separado y combinada (sin duplicar entradas cuando dos
  campos apuntan a la misma corrección). Suite completa: 140 tests, sin
  romper nada de lo que ya había.
- **RGPD**: los 3 campos nuevos (`has_double_chin`, `nose_size`,
  `lip_thickness`) son datos biométricos del mismo tipo que el resto de
  `facial_features_profile` -- se guardan bajo el mismo `consent_history`
  que ya cubre esa ficha, no hace falta un consentimiento aparte (mismo
  razonamiento que ya se documenta para `VisagismoProfileIn` más arriba).
- **Segunda entrega de Pedro (mismo día): capturas del vídeo + PDF
  re-subido.** Tras entregar lo anterior, Pedro mandó 14 capturas de
  pantalla (incluida una del propio YouTube, confirmando que el vídeo de
  proporciones es "Human Face. How to draw a face with correct
  proportions // VISUPLAS") y un PDF re-subido con el mismo nombre. El PDF
  se cortaba exactamente en el mismo punto que el primero ("En las
  corre..."): **no es un fallo de Pedro al copiar, el propio export de la
  transcripción se corta ahí siempre** -- ese hueco concreto no tiene
  arreglo, y no hay que volver a pedirle el mismo PDF esperando que
  cambie. Lo que sí aportaban las capturas eran dos detalles que el texto
  nombra pero no dibuja, y que se han incorporado a
  `tools/generar_barba_bigote_guia.py` -> `proportions_diagram()` (sección
  "Proporciones" de `guia-barba-bigote.html`):
  - **Eje de simetría**: el vídeo marca la línea vertical central (una de
    las 5 de la proporción) en rojo y sin discontinuar, aparte de las
    otras 4, porque también sirve para comparar un lado de la cara con el
    otro. Antes las 5 verticales se dibujaban todas igual; ahora la
    central tiene su propia clase CSS (`.bb-guide-axis`).
  - **Zona intelectual**: el vídeo resalta como banda propia el tramo de
    la frente entre el nacimiento del pelo y las cejas. Es exactamente la
    franja que ya pregunta `intellectual_zone_forehead` (usada en la
    sección "Frente" de `beard_mustache_rules.py`) -- las capturas
    confirman la definición visual de ese campo, no añaden ninguna regla
    nueva. Se añadió como un rectángulo resaltado (`.bb-guide-zone`).
  El resto de capturas (comparativa de proporción infantil por edades --
  2, 6 y 12 años -- y detalles de trazo de ceja/frente) son material de
  dibujo para un ilustrador, no de clasificación de un rostro adulto ya
  formado: no aportan ninguna regla de barba/bigote o de forma de rostro
  distinta de las que Pedro ya dio por texto, así que no se ha tocado
  `_POR_FORMA` ni el resto del motor de reglas. Verificado visualmente con
  una captura de Playwright de la tarjeta "Proporciones" (mismo criterio
  del resto del módulo: no hay ángulo que comprobar, solo inspección
  visual) y con la suite completa (140 tests, sin romper nada).
- **Tercera entrega de Pedro (mismo día): ~19 fotos de famosos con nombre
  de estilo, y un tercer PDF re-subido.** El PDF se sigue cortando en el
  mismo punto de siempre ("En las corre...") -- tercera confirmación de
  que ese hueco no depende de Pedro y no hay que volver a insistir en
  pedírselo. Las fotos sí traían nombres nuevos: como el volumen y el
  riesgo de inventar contenido sin fuente eran altos (meten una categoría
  entera, patillas, que no existía), se preguntó a Pedro con
  `AskUserQuestion` qué quería hacer con ellas -- eligió "añadirlas como
  ejemplos + crear sección de patillas", ambas sin regla de recomendación
  nueva (ninguna foto traía texto de corrección).
  - **Alias plegados en tarjetas ya existentes** (mismo estilo, otro
    nombre en la foto de Pedro, no un dibujo nuevo): "bigote con perilla"
    = mosquetero; "barba de fin de semana" = varios días/media sombra.
  - **`BEARDS_VARIANTES`** (nueva lista en
    `tools/generar_barba_bigote_guia.py`, sub-sección "Otros nombres que
    vas a escuchar" dentro de `#barbas` en `guia-barba-bigote.html`, NO
    mezclada con las 5 bases de `BEARDS`): candado extendido, Van Dycke
    (perilla + bigote fino sin conectar), perilla larga, barba imperial
    (perilla + tira fina de mandíbula + bigote), duck tail (completa con
    pico marcado), completa corta. Dos de ellas, "triangular (forma de la
    barba)" y "cuadrada (forma de la barba)", usan la MISMA palabra que
    dos valores de `face_shape_override` pero describen la forma que se
    le da al pelo, no la cara del cliente -- cada `detalle` dice
    explícitamente "OJO: no es una recomendación para el rostro
    triangular/cuadrado" para que no se lean como si por fin hubiera una
    regla para la cara cuadrada (sigue sin haberla, a propósito, ver más
    arriba).
  - **`SIDEBURNS`** (patillas: cuadrada/corta/larga, sección `#patillas`
    nueva en la guía, con su propio enlace en el menú de saltos):
    categoría sin precedente en el proyecto. Solo catálogo visual, sin
    ninguna correlación con forma de rostro -- ninguna fuente la trae.
  - **Sin fotos reales**: las ilustraciones son dibujos de línea propios
    en el mismo lenguaje visual del resto de la guía (`face()`,
    `_sliver`, `_mirror`, clases `bb-fill*`), nunca las fotos de los
    famosos que mandó Pedro -- usar esas fotos directamente habría sido
    distribuir imágenes con derechos de imagen de personas reales.
  - **Dos bugs encontrados y corregidos por inspección visual con
    Playwright** (mismo criterio del resto del módulo: no hay ángulo que
    comprobar): la tira de la barba imperial arrancaba a la altura de los
    ojos (`y=50`) en vez de a la altura de la mandíbula, cruzando la
    mejilla como un "bigote de gato" -- se corrigió el punto de partida a
    `y=62`, siguiendo el borde de la mandíbula. La barba cuadrada, con un
    único punto de barbilla suavizado por `_catmull_rom`, no se
    distinguía de "completa"/"duck tail" -- se corrigió añadiendo varios
    puntos casi a la misma altura en la base para que la curva se pegue a
    un tramo recto en vez de redondear una punta.
  - Verificado con capturas de Playwright de la sección completa y de
    cada icono ambiguo por separado, y con la suite completa (140 tests,
    sin romper nada -- este cambio no toca ningún archivo de `backend/`).

## Doce detalles de la web (sept 2026)

Lista de 12 ajustes que Pedro pidió de golpe (capturas de `cliente.html` y
`catalogo.html` + una lista numerada), la mayoría de UI pero con dos
cambios de esquema. Se listan en el mismo orden en que los pidió.

1. **Dejar claro que el perfil cliente es el del cliente**: subtítulo
   `.page-sub` (clase ya existía en `theme.css`, sin usar en ningún sitio)
   bajo el título de `cliente.html#welcome`, aclarando que al escanear el
   QR se entra en el perfil del cliente.
2. **Teléfono a exactamente 9 dígitos**: `maxlength="9"` +
   `inputmode="numeric"` + filtro de no-dígitos en `#reg-phone`/
   `#login-phone` (`cliente.html`), y en el backend `_validate_phone()`
   (`session_routes.py`) pasa de `len(normalized) < 9` a `!= 9`.
3. **"No lo sé" del mismo tamaño**: en `cuestionario.html`, deja de ser un
   link de texto aparte (`.skip`) y pasa a ser un `button.opt` más dentro
   de la rejilla de opciones (mismo tamaño que el resto), con un dibujo
   propio de una persona con los brazos abiertos (`SHRUG_ART`, mismo
   lenguaje visual -- SVG 100x100, `stroke="currentColor"` -- que
   `FACE_ART`/`HAIR_ART`/etc.; no se usó ningún icono de `ui.js` porque no
   hay ninguno de "brazos abiertos"/accesibilidad en ese set y arriesgarse
   a inventar un `path` de Lucide podía salir mal).
4. **Borrar una foto de visajismo ya hecha**: botón `trash-2` en la
   esquina de cada una de las 3 casillas de foto (`visagismo.html`),
   visible solo cuando esa casilla tiene foto (`.photo-slot.done
   .slot-delete`); nueva función `clearSlot(slot)`.
5. **Guardar rasgos cierra la ficha sola**: `visagismo.html`, nueva función
   `closeClientSection()` (resetea las 3 fotos, el buscador de cliente y
   los 3 pasos) llamada 900ms después de un "Guardado" en el formulario de
   rasgos, para que el peluquero pase directo al siguiente cliente sin
   tocar nada más.
6. **Casilla de "enviar foto para simular" antes de tener foto**: en
   `cliente.html`, `#c-sim` empieza deshabilitada y se deshabilita/
   desmarca sola mientras `#c-photo` no esté marcada (con `#c-photo`
   marcada más tarde por el peluquero en su primera visita) -- antes se
   podía marcar sin sentido durante el alta, cuando el cliente todavía no
   tiene ninguna foto.
7. **Catálogo: una tarjeta por corte, no apiladas**: `catalogo.html` tenía
   su propio `renderFamilyCard` que metía varios cortes dentro de la misma
   tarjeta con un "+N" plegado -- justo el patrón que `recomendaciones.html`
   ya había dejado atrás (ver "Recomendaciones: simular corte desde la
   tarjeta..." más arriba). Se sustituye por el mismo patrón de esa página:
   `renderStyleCard`/`orderWithFamiliesAdjacent`, una tarjeta por corte y
   un aviso "Parecido a: `<familia>`" cuando comparte foto/familia con
   otros. Se borra el CSS/JS que ya no se usa (`cut-list`, `cut-item`,
   `MAX_VISIBLE_PER_FAMILY`, `cutItemHtml`, `renderFamilyCard`).
8. **El cliente manda un corte del catálogo al peluquero mientras
   espera**: antes solo existía "quiero repetir este" sobre un corte YA
   hecho (`waiting.requested_history_id`, ver "Historial de cortes del
   cliente"). Se añade una columna hermana `waiting.requested_style_id`
   (migración en `database.py`, campo en `WaitingEntry`) para pedir un
   corte del catálogo que el cliente **todavía no se ha hecho**. Son
   mutuamente excluyentes: `repository.set_requested_history`/
   `set_requested_style` limpian la otra columna al fijar la suya.
   `HaircutRequestIn` (`schemas.py`) gana un `style_id` opcional junto al
   `history_id` que ya tenía; `PUT /api/me/request` acepta cualquiera de
   los dos (o ninguno, para quitar la petición) y `GET /api/me/request`
   devuelve ambos. `_waiting_out()` construye, cuando hay
   `requested_style_id`, un `HaircutOut` "sintético" con `id="style:<id>"`
   (sin foto ni fecha real) para que `sala.html`/`ficha.html` lo muestren
   igual que un corte repetido -- ambos comprueban ese prefijo para decir
   "Quiere probar" en vez de "Quiere repetir". Botón "Quiero este"/"Pedido"
   (mismo patrón que ya usaba `mis-cortes.html` con `history_id`) añadido
   en el lightbox de `catalogo.html` y en cada tarjeta de
   `recomendaciones.html` (solo para el cliente, no en modo peluquero).
   Tests: `test_request_catalog_style_not_yet_in_history` en
   `test_sessions.py`.
9. **"Atender" abre la ficha sola**: `sala.html`, el botón ya no se queda
   en la propia sala tras el PATCH a `in_service` -- ahora navega a
   `ficha.html?client_id=...&w=...` justo después.
10. **"Terminado" pregunta antes por una foto, en su propia página**: el
    formulario "Corte de hoy" vivía como una sección más dentro de
    `ficha.html` (`#log-sec`, con su botón "Registrar corte de hoy" en
    Historial y otra apertura automática al pulsar "Terminado"). Pedro
    pidió que fuera una página aparte y que esa página dijera cuántas
    fotos hay ya guardadas de ese cliente. Se creó
    **`frontend/registrar-corte.html`** (mismo formulario/lógica que
    tenía `#log-sec`: `StylePicker`, foto opcional, notas, preselección
    del corte pedido) y se borró la sección, el botón y las funciones
    `openLog`/`finishWaiting` de `ficha.html`. El estado de la sala
    (`waiting.status`) pasa a `"done"` solo al salir de esa página
    ("Guardar y terminar" o "Terminar sin guardar"), nunca antes; hay
    también un enlace "Volver a la ficha sin terminar" para el clic
    accidental. Tanto `sala.html` como `ficha.html` navegan ahí en vez de
    marcar "done" directamente.
11. **"Simular corte" como botón grande con desplegable**: en `index.html`
    (la única página de simular), el botón de enviar pasa de "Generar" a
    "Simular corte" con más tamaño (`min-height: 56px`, `font-size: 16px`),
    y la fila de proveedores (antes botones en pastilla, `.seg`) pasa a un
    `<select id="provider-select">` -- un desplegable real, oculto cuando
    solo hay un proveedor configurado (no hay nada que elegir).
12. **Maniquí de remolinos más grande para iPad**: `growth-map.html`
    tenía `#three-canvas` a una altura fija de 560px (o 62vh en móvil,
    ≤600px) -- una tablet como iPad (768-1024px) caía en el rango de
    escritorio con esa altura fija, demasiado pequeña para dibujar
    remolinos con el dedo con precisión. Se amplía a `min(720px, 76vh)`
    en general y se añade un rango específico de tablet (601-1080px) a
    `min(820px, 82vh)`; el ancho de página también sube de 760px a 980px.
    `resizeRenderer()` ya escuchaba `resize` y lee el tamaño real del
    canvas, así que no hizo falta tocar el JS del render 3D.

Verificado con la suite completa de backend (145 tests, incluida la nueva
`test_phone_must_be_exactly_9_digits` para el punto 2) y comprobando a
mano (`node -e "new Function(...)"` sobre cada `<script>` inline) que
ningún HTML tocado quedó con un error de sintaxis tras los borrados.

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
