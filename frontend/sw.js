// Service worker mínimo: solo existe para que Chrome/Android considere la
// web "instalable" (uno de los requisitos técnicos del criterio de
// instalación de PWA es tener un service worker con un listener de
// "fetch"). No cachea nada a propósito: todo pasa a la red tal cual,
// para no servir nunca una versión vieja del simulador ni de los datos
// de la peluquería.
self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
