// Service worker mínimo: solo existe para que Chrome/Android considere la
// web "instalable" (uno de los requisitos técnicos del criterio de
// instalación de PWA es tener un service worker con un listener de
// "fetch"). No cachea nada a propósito: todo pasa a la red tal cual,
// para no servir nunca una versión vieja del simulador ni de los datos
// de la peluquería.

// IMPORTANTE (detectado al desplegar un cambio y comprobar que no se
// veía ni con recarga forzada): `fetch(event.request)` a secas NO
// basta. Reenvía la petición tal cual, y el modo de caché de una
// petición de navegación normal (incluso una recarga forzada del
// usuario) puede seguir siendo "default" en cuanto pasa por aquí, así
// que el navegador puede reutilizar una copia ya cacheada sin ni
// siquiera preguntarle al servidor -- con lo cual nunca llega a ver la
// cabecera `Cache-Control: no-cache` que manda el backend (ver
// `_no_stale_cache` en `backend/app/main.py`). Reconstruir la petición
// con `cache: "no-store"` fuerza a que la petición vaya siempre a la
// red de verdad, sin pasar por la caché HTTP del navegador en ningún
// sentido (ni leerla ni guardar la respuesta).
self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(new Request(event.request, { cache: "no-store" })));
});
