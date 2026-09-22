// QR de la parte del cliente a pantalla completa, para escanearlo desde el
// móvil sin coger la tablet (así el peluquero no tiene que dejar de
// cortar). El SVG lo genera el backend: /api/qr.svg. Los estilos van aquí
// porque la portada (inicio.html) no carga theme.css.
window.QR = {
  // path: dirección que abre el QR (por defecto, la parte del cliente).
  zoom(path = "/cliente.html?qr=1", caption = "Escanéalo con la cámara del móvil") {
    if (!document.getElementById("qr-zoom-style")) {
      const st = document.createElement("style");
      st.id = "qr-zoom-style";
      st.textContent = `.qr-zoom{position:fixed;inset:0;z-index:1000;background:rgba(3,4,10,.94);display:flex;flex-direction:column;
        align-items:center;justify-content:center;gap:16px;color:rgba(255,255,255,.75);font:15px system-ui,sans-serif;text-align:center;padding:16px}
        .qr-zoom img{width:min(78vw,72vh,420px);height:auto;background:#fff;border-radius:18px;padding:16px}`;
      document.head.appendChild(st);
    }
    const z = document.createElement("div");
    z.className = "qr-zoom";
    z.innerHTML = `<img src="/api/qr.svg?path=${encodeURIComponent(path)}" alt="Código QR" /><div>${caption}<br>Toca para cerrar</div>`;
    const close = () => { z.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    z.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(z);
  },
};
