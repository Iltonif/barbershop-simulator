/* Acordeón progresivo compartido (sept 2026, a petición de Pedro: ficha
   del cliente y espacio del cliente reorganizados en "capas lógicas" con
   pocas opciones a la vez, con libertad para saltar de una capa a otra
   sin forzar un orden estricto).

   HTML esperado, dentro de un contenedor (p.ej. <div class="layers">):
     <section class="glass-card layer" data-layer="clave">
       <button type="button" class="layer-head">
         <span class="ico" data-icon="..."></span>
         <span>Título</span>
         <span class="chev" data-icon="chevron-right"></span>
       </button>
       <div class="layer-body" hidden> ...contenido... </div>
     </section>

   Uso:
     Layers.init(document.getElementById("layers"))
       -> abre la primera capa, pliega el resto, y engancha los clics.
     Layers.open(container, "clave")
       -> abre esa capa a mano (p.ej. desde el onboarding, para poder
          resaltar un botón real que vive dentro de una capa plegada). */
(function () {
  function layersOf(container) {
    return Array.from(container.children).filter((el) => el.classList.contains("layer"));
  }

  function setOpen(layer, open) {
    const head = layer.querySelector(".layer-head");
    const body = layer.querySelector(".layer-body");
    layer.dataset.open = open ? "true" : "false";
    if (body) body.hidden = !open;
    if (head) head.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function open(container, key) {
    layersOf(container).forEach((layer) => setOpen(layer, layer.dataset.layer === key));
  }

  function init(container, opts) {
    if (!container) return;
    const layers = layersOf(container);
    if (!layers.length) return;
    const startKey = (opts && opts.open) || layers[0].dataset.layer;
    layers.forEach((layer) => {
      setOpen(layer, layer.dataset.layer === startKey);
      const head = layer.querySelector(".layer-head");
      if (head) head.addEventListener("click", () => open(container, layer.dataset.layer));
    });
  }

  window.Layers = { init, open };
})();
