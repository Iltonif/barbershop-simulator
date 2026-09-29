/* Onboarding "medio forzado" para cliente y peluquero (sept 2026).

   Pedro: que la primera vez que alguien use la web en la barbería sepa qué
   hacer y en qué orden, tanto en la parte del cliente como en la del
   peluquero. Decisiones (preguntadas a Pedro):
   - Tour guiado sobre la interfaz REAL (resalta un botón/zona de verdad
     con un recuadro con brillo), no un carrusel de dibujos aparte.
   - Obligatorio la primera vez (no se puede saltar ni cerrar hasta llegar
     al final); después queda un botón "¿Cómo funciona?" para volver a
     verlo cuando quieran -- ese repaso sí se puede cerrar en cualquier
     paso.
   - Sin tocar la base de datos: cuando el paso explica algo que no existe
     todavía en pantalla (p.ej. "Atender" cuando no hay nadie esperando),
     no se inventa ningún cliente de prueba -- la tarjeta se muestra
     centrada, sin señalar nada, en vez de resaltar un elemento que no
     está. Si el elemento SÍ existe de verdad en ese momento (p.ej. si hay
     alguien esperando), se resalta el real.

   Progreso guardado en localStorage (por dispositivo/navegador, no por
   cuenta -- el PIN del peluquero es compartido y el cliente entra con
   teléfono desde varios móviles, así que "ya lo vio" solo puede
   comprobarse por dispositivo).

   Uso:
     Onboarding.maybeRun(steps, { storageKey: "ob_x_v1" })
       -> lo muestra solo si no se ha visto antes en este navegador;
          no se puede cerrar hasta el último paso.
     Onboarding.replay(steps, { storageKey: "ob_x_v1" })
       -> lo muestra siempre (botón "¿Cómo funciona?"); se puede cerrar
          en cualquier momento (✕ o tocando fuera de la tarjeta).

   Cada paso: { selector?, icon, title, text }. `selector` es opcional
   (CSS selector de un elemento real de la página); si no se da, o el
   elemento no existe/no es visible en ese momento, el paso se muestra
   como tarjeta centrada sin resaltar nada. */
(function () {
  function isDone(key) {
    try { return localStorage.getItem(key) === "1"; } catch (e) { return false; }
  }
  function markDone(key) {
    try { localStorage.setItem(key, "1"); } catch (e) { /* sin almacenamiento: se repetirá, no es grave */ }
  }

  function run(steps, opts) {
    if (!steps || !steps.length) return;
    const storageKey = opts && opts.storageKey;
    const dismissible = !!(opts && opts.dismissible);
    const onDone = opts && opts.onDone;

    const overlay = document.createElement("div");
    overlay.className = "ob-overlay";
    const prevOverflow = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    document.body.appendChild(overlay);
    if (dismissible) {
      overlay.addEventListener("click", (e) => { if (e.target === overlay) finish(); });
    }

    let i = 0;
    let card = null;

    function cleanup() {
      window.removeEventListener("resize", reposition);
      document.documentElement.style.overflow = prevOverflow;
      overlay.remove();
    }
    function finish() {
      if (storageKey) markDone(storageKey);
      cleanup();
      if (onDone) onDone();
    }

    function reposition() {
      overlay.querySelectorAll(".ob-highlight").forEach((h) => h.remove());
      const step = steps[i];
      const target = step.selector ? document.querySelector(step.selector) : null;
      const visible = !!(target && target.offsetParent !== null && target.getBoundingClientRect().width > 0);
      if (visible) {
        const r = target.getBoundingClientRect();
        const hi = document.createElement("div");
        hi.className = "ob-highlight";
        hi.style.top = `${r.top - 8}px`;
        hi.style.left = `${r.left - 8}px`;
        hi.style.width = `${r.width + 16}px`;
        hi.style.height = `${r.height + 16}px`;
        overlay.appendChild(hi);
        const placeBelow = window.innerHeight - r.bottom > 200 || r.top < 200;
        card.style.position = "fixed";
        card.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - 336))}px`;
        card.style.top = placeBelow ? `${Math.min(r.bottom + 16, window.innerHeight - 200)}px` : "";
        card.style.bottom = placeBelow ? "" : `${Math.max(12, window.innerHeight - r.top + 16)}px`;
      } else {
        card.style.position = "";
        card.style.left = "";
        card.style.top = "";
        card.style.bottom = "";
      }
    }

    function renderStep() {
      const step = steps[i];
      const last = i === steps.length - 1;
      overlay.innerHTML = "";
      card = document.createElement("div");
      card.className = "ob-card glass-card";
      card.innerHTML = `
        ${dismissible ? `<button type="button" class="ob-close icon-btn" aria-label="Cerrar">${UI.icon("x")}</button>` : ""}
        <div class="ob-icon">${UI.icon(step.icon || "info")}</div>
        <h3 class="ob-title">${step.title}</h3>
        <p class="ob-text">${step.text}</p>
        <div class="ob-foot">
          <span class="ob-step">${i + 1}/${steps.length}</span>
          <button type="button" class="ob-next">${last ? "Entendido, empezar" : "Siguiente"}</button>
        </div>`;
      overlay.appendChild(card);
      card.querySelector(".ob-next").addEventListener("click", () => {
        if (last) finish(); else { i += 1; renderStep(); }
      });
      if (dismissible) card.querySelector(".ob-close").addEventListener("click", finish);
      reposition();
    }

    window.addEventListener("resize", reposition);
    renderStep();
  }

  window.Onboarding = {
    isDone,
    maybeRun(steps, opts) {
      const key = opts && opts.storageKey;
      if (key && isDone(key)) return;
      run(steps, Object.assign({}, opts, { dismissible: false }));
    },
    replay(steps, opts) {
      run(steps, Object.assign({}, opts, { dismissible: true }));
    },
  };
})();
