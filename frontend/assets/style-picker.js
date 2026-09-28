/* Selector de cortes con búsqueda por texto + "añadir corte nuevo".

   Sustituye al <select> plano de toda la vida en los dos sitios donde el
   peluquero elige un corte de un catálogo que ya ronda los 100+ entradas
   (sept 2026, petición de Pedro): el simulador (frontend/index.html) y
   "Corte de hoy" en la ficha (frontend/ficha.html, registrar en el
   historial). Un <select> con tantas opciones obliga a desplazarse a
   ciegas; esto deja escribir unas letras y filtra en vivo.

   Si el corte que se busca no existe todavía, "+ Añadir «texto» como
   corte nuevo" abre un formulario mínimo (largo por zona, degradado, para
   qué tipos de pelo vale) y lo guarda con POST /api/styles -- a partir de
   ahí es un corte más del catálogo (ver style_catalog.load_full_catalog
   en el backend), no solo una anotación suelta para esa visita.

   Uso:
     const picker = StylePicker.mount(document.getElementById("hueco"), {
       styles,                 // lista de {id, name, description, style_family}
       value: null,            // id ya elegido, si lo hay
       placeholder: "Busca un corte…",
       allowCreate: true,      // deja añadir uno nuevo al catálogo
       extraActions: [{ id: "__otro", label: "Otro (uso puntual, no se guarda en el catálogo)", icon: "pen-line" }],
       onChange(styleId, style) {},   // al elegir uno del catálogo (o recién creado)
       onExtraAction(id) {},          // al tocar una de extraActions
     });
     picker.getValue();  // id elegido, o null
     picker.clearExtra(); // vuelve a mostrar el nombre del corte elegido (deshace el estado "extra")
*/
(function () {
  const FADE_TYPES = [
    ["ninguno", "Sin degradado"], ["bajo", "Bajo"], ["medio", "Medio"], ["alto", "Alto"], ["skin", "A piel"],
  ];
  const HAIR_TYPES = [["liso", "Liso"], ["ondulado", "Ondulado"], ["rizado", "Rizado"], ["afro", "Afro"]];

  function normalize(s) {
    return String(s ?? "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "");
  }

  function matches(style, needle) {
    if (!needle) return true;
    const hay = normalize(`${style.name} ${style.description || ""} ${style.style_family || ""}`);
    return hay.includes(needle);
  }

  function mount(container, opts) {
    const state = {
      styles: opts.styles.slice(),
      value: opts.value || null,
      extra: null, // id de extraActions activo (p.ej. "__otro"), o null
      open: false,
      creating: false,
    };

    // El <label for="..."> de fuera apunta al id del contenedor (un <div>
    // no es "labelable"), así que el input real hereda ese mismo id para
    // que tocar la etiqueta lo enfoque igual que con un <select>.
    const inputId = container.id || "";
    container.innerHTML = `
      <div class="style-picker">
        <input type="text" id="${inputId}" class="sp-input" placeholder="${opts.placeholder || "Busca un corte…"}" autocomplete="off" />
        <div class="sp-menu" hidden></div>
      </div>
    `;
    const root = container.querySelector(".style-picker");
    const input = root.querySelector(".sp-input");
    const menu = root.querySelector(".sp-menu");

    function styleById(id) {
      return state.styles.find((s) => s.id === id) || null;
    }

    function showValueInInput() {
      if (state.extra) {
        const a = (opts.extraActions || []).find((x) => x.id === state.extra);
        input.value = a ? a.label : "";
        return;
      }
      const s = styleById(state.value);
      input.value = s ? s.name : "";
    }

    function closeMenu() {
      state.open = false;
      state.creating = false;
      menu.hidden = true;
      menu.innerHTML = "";
    }

    function selectStyle(style) {
      state.value = style.id;
      state.extra = null;
      showValueInInput();
      closeMenu();
      if (opts.onChange) opts.onChange(style.id, style);
    }

    function selectExtra(id) {
      state.extra = id;
      state.value = null;
      showValueInInput();
      closeMenu();
      if (opts.onExtraAction) opts.onExtraAction(id);
    }

    function renderCreateForm(prefillName) {
      state.creating = true;
      menu.innerHTML = `
        <div class="sp-create">
          <label>Nombre</label>
          <input type="text" class="sp-c-name" value="${escAttr(prefillName)}" />
          <label>Descripción (opcional)</label>
          <input type="text" class="sp-c-desc" placeholder="Una frase corta" />
          <div class="sp-c-lengths">
            <div><label>Arriba (mm)</label><input type="number" min="0" max="600" class="sp-c-top" /></div>
            <div><label>Laterales (mm)</label><input type="number" min="0" max="600" class="sp-c-sides" /></div>
            <div><label>Nuca (mm)</label><input type="number" min="0" max="600" class="sp-c-back" /></div>
          </div>
          <label>Degradado</label>
          <select class="sp-c-fade">${FADE_TYPES.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>
          <label>Vale para</label>
          <div class="sp-c-hairtypes">
            ${HAIR_TYPES.map(([v, l]) => `
              <label class="checkbox-row"><input type="checkbox" value="${v}" checked /><span>${l}</span></label>
            `).join("")}
          </div>
          <p class="hint sp-c-err" hidden></p>
          <div class="sp-c-actions">
            <button type="button" class="secondary sp-c-cancel">Cancelar</button>
            <button type="button" class="sp-c-save">${UI.icon("save")}Guardar corte</button>
          </div>
        </div>
      `;
      menu.hidden = false;
      state.open = true;
      menu.querySelector(".sp-c-name").focus();
      menu.querySelector(".sp-c-cancel").addEventListener("click", () => { closeMenu(); input.focus(); });
      menu.querySelector(".sp-c-save").addEventListener("click", async () => {
        const name = menu.querySelector(".sp-c-name").value.trim();
        const top = menu.querySelector(".sp-c-top").value;
        const sides = menu.querySelector(".sp-c-sides").value;
        const back = menu.querySelector(".sp-c-back").value;
        const hairTypes = [...menu.querySelectorAll(".sp-c-hairtypes input:checked")].map((c) => c.value);
        const errEl = menu.querySelector(".sp-c-err");
        const fail = (msg) => { errEl.textContent = msg; errEl.hidden = false; };
        errEl.hidden = true;
        if (!name) return fail("Escribe un nombre para el corte.");
        if (top === "" || sides === "" || back === "") return fail("Rellena el largo de las tres zonas (en mm).");
        if (!hairTypes.length) return fail("Marca al menos un tipo de pelo.");
        const payload = {
          name, description: menu.querySelector(".sp-c-desc").value.trim(),
          length_top_mm: Number(top), length_sides_mm: Number(sides), length_back_mm: Number(back),
          fade_type: menu.querySelector(".sp-c-fade").value, suitable_hair_types: hairTypes,
        };
        const saveBtn = menu.querySelector(".sp-c-save");
        saveBtn.disabled = true;
        try {
          const res = await fetch((opts.createEndpoint || "/api/styles"), {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(" ") : data.detail;
            fail(detail || "No se pudo guardar el corte.");
            saveBtn.disabled = false;
            return;
          }
          state.styles.push(data);
          if (opts.onCreate) opts.onCreate(data);
          selectStyle(data);
        } catch (e) {
          fail(`Sin conexión: ${e.message}`);
          saveBtn.disabled = false;
        }
      });
    }

    function escAttr(s) {
      return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    }

    function renderMenu() {
      const needle = normalize(input.value);
      const found = state.styles.filter((s) => matches(s, needle)).slice(0, 30);
      const rows = [];
      for (const a of opts.extraActions || []) {
        rows.push(`<button type="button" class="sp-row sp-extra" data-extra="${a.id}">${UI.icon(a.icon || "pen-line")}<span>${escAttr(a.label)}</span></button>`);
      }
      for (const s of found) {
        rows.push(`<button type="button" class="sp-row" data-id="${s.id}">
          <span class="sp-row-main"><b>${escAttr(s.name)}</b>${s.description ? `<small>${escAttr(s.description)}</small>` : ""}</span>
        </button>`);
      }
      const typed = input.value.trim();
      const exact = found.some((s) => normalize(s.name) === needle);
      if (opts.allowCreate && typed && !exact) {
        rows.push(`<button type="button" class="sp-row sp-add" data-add="1">${UI.icon("plus")}<span>Añadir «${escAttr(typed)}» como corte nuevo</span></button>`);
      }
      if (!rows.length) {
        menu.innerHTML = `<p class="sp-empty hint">Sin resultados${opts.allowCreate ? "" : " para esa búsqueda"}.</p>`;
      } else {
        menu.innerHTML = rows.join("");
      }
      menu.hidden = false;
      state.open = true;
      menu.querySelectorAll(".sp-row[data-id]").forEach((btn) => {
        btn.addEventListener("click", () => selectStyle(styleById(btn.dataset.id)));
      });
      menu.querySelectorAll(".sp-row[data-extra]").forEach((btn) => {
        btn.addEventListener("click", () => selectExtra(btn.dataset.extra));
      });
      const addBtn = menu.querySelector(".sp-row[data-add]");
      if (addBtn) addBtn.addEventListener("click", () => renderCreateForm(typed));
    }

    input.addEventListener("focus", renderMenu);
    input.addEventListener("input", renderMenu);
    // Tocar fuera cierra el menú de resultados (y descarta la búsqueda a
    // medio escribir); mientras se está rellenando el formulario de "+
    // Añadir corte nuevo" no se cierra solo, para no perder lo escrito por
    // un toque accidental fuera -- solo sus propios botones lo cierran.
    document.addEventListener("click", (e) => {
      if (!state.open || state.creating) return;
      if (!root.contains(e.target)) { closeMenu(); showValueInInput(); }
    });

    showValueInInput();

    return {
      getValue: () => state.value,
      getExtra: () => state.extra,
      clear: () => { state.value = null; state.extra = null; showValueInInput(); },
      setStyles: (styles) => { state.styles = styles.slice(); },
      // Selección programática (p.ej. llegar desde el catálogo con
      // ?style=<id> ya elegido, o preseleccionar el pedido/favorito al
      // abrir "Corte de hoy"). No dispara onChange -- es solo pintar el
      // estado inicial, no una elección del peluquero.
      selectById: (id) => {
        const s = styleById(id);
        if (!s) return false;
        state.value = s.id;
        state.extra = null;
        showValueInInput();
        return true;
      },
    };
  }

  window.StylePicker = { mount };
})();
