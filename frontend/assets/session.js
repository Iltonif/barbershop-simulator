/* Sesiones de la web (ver backend/app/api/auth.py y session_routes.py).

   Dos formas de usar la parte del cliente:
   - Tablet de la barbería (por defecto): se ven las dos partes, cliente y
     peluquero (esta con PIN). El cliente se sale solo tras unos minutos sin
     tocar la pantalla, para que el siguiente no vea su ficha.
   - Su propio móvil, entrando por el QR (cliente.html?qr=1): solo la parte
     del cliente; no aparece el acceso del peluquero y no hay salida
     automática.

   Uso:
     Session.requireBarber()   -> en páginas del peluquero (redirige al PIN)
     Session.requireClient()   -> en páginas del cliente (devuelve su ficha
                                  o redirige a cliente.html)
     Session.optionalClient()  -> ficha del cliente si hay sesión, si no null
     Session.isQr()            -> true si entró por el QR */
(function () {
  const MODE_KEY = "modo";
  const CLIENT_IDLE_MS = 3 * 60 * 1000;   // tablet: salida automática del cliente
  const WARN_MS = 20 * 1000;              // aviso antes de salir

  function store(key, value) {
    try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { return null; }
  }

  // ?qr=1 fija el modo "móvil del cliente" en este navegador.
  if (new URLSearchParams(location.search).get("qr") === "1") store(MODE_KEY, "qr");
  const isQr = () => store(MODE_KEY) === "qr";

  async function getJson(url) {
    const res = await fetch(url, { credentials: "same-origin" });
    if (!res.ok) { const err = new Error(res.statusText); err.status = res.status; throw err; }
    return res.json();
  }

  function goLogin() {
    const next = encodeURIComponent(location.pathname.replace(/^\//, "") + location.search);
    location.href = `peluquero.html?next=${next}`;
  }

  // En páginas del peluquero: cualquier 401 de la API (la sesión caduca a
  // los 15 min sin uso) lleva otra vez al PIN.
  function guardBarberFetch() {
    const original = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const res = await original(...args);
      const url = String(args[0] && args[0].url ? args[0].url : args[0]);
      if (res.status === 401 && url.includes("/api/") && !url.includes("/api/me")) goLogin();
      return res;
    };
  }

  async function requireBarber() {
    if (isQr()) { location.href = "cliente.html"; return new Promise(() => {}); }
    guardBarberFetch();
    try {
      const s = await getJson("/api/barber/session");
      if (!s.logged_in) { goLogin(); return new Promise(() => {}); }
      return s;
    } catch (e) { goLogin(); return new Promise(() => {}); }
  }

  async function optionalClient() {
    try { return await getJson("/api/me"); } catch (e) { return null; }
  }

  async function requireClient() {
    const me = await optionalClient();
    if (!me) { location.href = "cliente.html"; return new Promise(() => {}); }
    if (!isQr()) startClientIdle();
    return me;
  }

  async function logoutClient() {
    try { await fetch("/api/me/logout", { method: "POST" }); } catch (e) { /* sin red */ }
    location.href = isQr() ? "cliente.html" : "inicio.html";
  }

  async function logoutBarber() {
    try { await fetch("/api/barber/logout", { method: "POST" }); } catch (e) { /* sin red */ }
    location.href = "inicio.html";
  }

  // Tablet: tras 3 min sin tocar, aviso de 20 s y salida.
  let idleTimer = null, warnTimer = null, overlay = null;
  function startClientIdle() {
    const reset = () => {
      clearTimeout(idleTimer); clearTimeout(warnTimer);
      if (overlay) { overlay.remove(); overlay = null; }
      idleTimer = setTimeout(warn, CLIENT_IDLE_MS);
    };
    function warn() {
      overlay = document.createElement("div");
      overlay.className = "idle-overlay";
      overlay.innerHTML = `<div class="idle-box glass-card"><strong>¿Sigues ahí?</strong>
        <span>Por tu privacidad, se cerrará tu perfil en unos segundos.</span>
        <button type="button">Sigo aquí</button></div>`;
      overlay.querySelector("button").addEventListener("click", reset);
      document.body.appendChild(overlay);
      warnTimer = setTimeout(logoutClient, WARN_MS);
    }
    ["pointerdown", "keydown", "scroll"].forEach((ev) => window.addEventListener(ev, () => { if (!overlay) reset(); }, { passive: true }));
    reset();
  }

  // Favoritos del cliente (corazón en catálogo y recomendaciones).
  let liked = new Set();
  function setLikes(ids) { liked = new Set(ids || []); }
  function heartHtml(styleId) {
    const on = liked.has(styleId);
    return `<button type="button" class="heart-btn" data-like="${styleId}" aria-pressed="${on}" aria-label="Me gusta">${UI.icon("heart")}</button>`;
  }
  // Delegado: un solo listener para todos los corazones de la página.
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-like]");
    if (!btn) return;
    e.preventDefault(); e.stopPropagation();
    const id = btn.dataset.like;
    if (liked.has(id)) liked.delete(id); else liked.add(id);
    document.querySelectorAll(`[data-like="${CSS.escape(id)}"]`).forEach((b) => b.setAttribute("aria-pressed", liked.has(id)));
    try {
      await fetch("/api/me/likes", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ style_ids: [...liked] }) });
    } catch (err) { /* se reintenta en el siguiente toque */ }
  }, true);

  // En las herramientas del peluquero abiertas desde una ficha
  // (?client_id=), enlace de vuelta a esa ficha al principio del menú.
  function fichaLink() {
    const id = new URLSearchParams(location.search).get("client_id");
    const nav = document.querySelector("nav.top-nav");
    if (!id || !nav) return;
    nav.insertAdjacentHTML("afterbegin",
      `<a class="keep-label" href="ficha.html?client_id=${encodeURIComponent(id)}">${UI.icon("arrow-left")}<span class="lbl">Ficha</span></a>`);
  }

  window.Session = { isQr, fichaLink, requireBarber, requireClient, optionalClient, logoutClient, logoutBarber,
                     setLikes, heartHtml, isLiked: (id) => liked.has(id) };
})();
