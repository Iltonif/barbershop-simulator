// Maniquí personalizado del cliente (growth-map.html y su vista en la ficha).
//
// - Rasgos de la cara: se suman al maniquí los "morph targets" de
//   assets/rasgos/*.bin (MakeHuman, CC0; los genera
//   tools/construir_cabeza_masculina.py) con los pesos que manda el backend
//   (GET /api/clients/{id}/avatar, ver backend/app/pipeline/avatar.py).
// - Línea del pelo (recta, entradas, pico, frente alta) y color: se pintan
//   en la piel con los atributos _hairh / _hairth / _ears del .glb.
// - Pelo: miles de mechones con forma según el tipo (liso cae por
//   gravedad, ondulado hace ondas, rizado tirabuzones, afro sale hacia fuera
//   en espiral apretada), largo por zona (arriba / laterales / nuca, con el
//   degradado del último corte), densidad y color. Siguen el campo de
//   direcciones de las flechas y remolinos (lo da growth-map.html).
// - Balanceo: al girar la cabeza, las puntas se retrasan y vuelven con un
//   muelle amortiguado (en el shader, sin rehacer los mechones).
window.Avatar = (function () {
  const MM = 0.0041; // 1 mm en unidades de la escena (cabeza de ~15,5 cm = 0,64)
  const COLORS = {
    negro: [0.10, 0.08, 0.07], castano_oscuro: [0.22, 0.15, 0.11], castano: [0.34, 0.23, 0.15],
    castano_claro: [0.50, 0.36, 0.24], rubio: [0.78, 0.62, 0.40], pelirrojo: [0.55, 0.23, 0.10],
    canoso: [0.72, 0.71, 0.69],
  };
  const DEFAULT_COLOR = "castano_oscuro";
  const TEXTURES = {
    liso:     { shrink: 1.0,  gravity: 1.0,  outward: 0.0,  wave: 0,     coil: 0,      pitch: 0,     stiff: 0.05 },
    ondulado: { shrink: 0.9,  gravity: 0.75, outward: 0.05, wave: 0.010, waveLen: 0.075, coil: 0, pitch: 0, stiff: 0.2 },
    rizado:   { shrink: 0.6,  gravity: 0.3,  outward: 0.5,  wave: 0,     coil: 0.016,  pitch: 0.034, stiff: 0.45 },
    // Afro: sale hacia fuera casi en línea recta desde la raíz y forma una
    // bola; el rizo es muy apretado (espiral pequeña y de paso corto).
    afro:     { shrink: 0.7,  gravity: 0.0,  outward: 1.6,  wave: 0,     coil: 0.009,  pitch: 0.016, stiff: 0.85 },
  };
  const DENSITY = { low_thinning: 0.55, medium: 1.0, high_dense: 1.25 };
  // Altura (u.y desde el centro de la cabeza) donde empieza el degradado.
  const FADE_Y = { bajo: -0.38, medio: -0.12, alto: 0.12, skin: 0.22 };

  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  // ---------- Rasgos ----------
  let morphIndex = null;
  const morphCache = {};
  async function loadIndex() {
    if (!morphIndex) morphIndex = await (await fetch("assets/rasgos/index.json")).json();
    return morphIndex;
  }
  async function loadMorph(name) {
    if (morphCache[name]) return morphCache[name];
    const buf = await (await fetch(`assets/rasgos/${name}.bin`)).arrayBuffer();
    const dv = new DataView(buf);
    const n = dv.getUint32(0, true), scale = dv.getFloat32(4, true);
    const idx = new Uint32Array(buf, 8, n);
    const q = new Int16Array(buf.slice(8 + 4 * n, 8 + 4 * n + 6 * n));
    const vals = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i++) vals[i] = (q[i] / 32767) * scale;
    return (morphCache[name] = { idx, vals });
  }
  // Pone el maniquí base y le suma los rasgos pedidos (weights: {nombre: 0-1}).
  async function applyMorphs(skin, eyes, weights) {
    const index = await loadIndex();
    const P = skin.geometry.attributes.position;
    if (!skin.userData.basePos) skin.userData.basePos = P.array.slice();
    P.array.set(skin.userData.basePos);
    const E = eyes ? eyes.geometry.attributes.position : null;
    if (E && !eyes.userData.basePos) eyes.userData.basePos = E.array.slice();
    if (E) E.array.set(eyes.userData.basePos);
    for (const [name, w] of Object.entries(weights || {})) {
      if (!index.morphs[name] || !w) continue;
      const m = await loadMorph(name);
      for (let k = 0; k < m.idx.length; k++) {
        const i = m.idx[k] * 3;
        P.array[i] += w * m.vals[k * 3]; P.array[i + 1] += w * m.vals[k * 3 + 1]; P.array[i + 2] += w * m.vals[k * 3 + 2];
      }
      if (E) {
        const [dl, dr] = index.morphs[name].eyes;
        for (let i = 0; i < E.count; i++) {
          const d = E.array[i * 3] > 0 ? dl : dr;
          E.array[i * 3] += w * d[0]; E.array[i * 3 + 1] += w * d[1]; E.array[i * 3 + 2] += w * d[2];
        }
      }
    }
    P.needsUpdate = true;
    skin.geometry.computeVertexNormals();
    skin.geometry.computeBoundingSphere();
    if (E) { E.needsUpdate = true; eyes.geometry.computeVertexNormals(); eyes.geometry.computeBoundingSphere(); }
  }

  // ---------- Línea del pelo y color de la zona del pelo ----------
  // Desplazamiento de la línea del pelo según su forma (en la misma escala
  // que _hairh: 0 = ojos, 1 = parte más alta de la cabeza).
  function hairlineShift(type, th) {
    const g = (c, w) => Math.exp(-(((th - c) / w) ** 2));
    if (type === "m_shaped_receding") return 0.24 * g(36, 12);
    if (type === "high_forehead") return 0.13 * (1 - smooth(40, 60, th));
    if (type === "widows_peak") return -0.07 * g(0, 7) + 0.05 * g(24, 10);
    return 0;
  }
  function hairMask(skin, hairline) {
    const a = skin.geometry.attributes;
    const H = a._hairh, TH = a._hairth, EAR = a._ears, S = a._scalp;
    const n = a.position.count, mask = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (!H) { mask[i] = S ? S.getX(i) : 0; continue; }
      const h = H.getX(i) - hairlineShift(hairline, TH.getX(i));
      mask[i] = smooth(-0.035, 0.035, h) * (1 - EAR.getX(i));
    }
    return mask;
  }
  function paintScalp(skin, mask, colorName, density) {
    const C = skin.geometry.attributes.color;
    if (!C) return;
    if (!skin.userData.baseColor) skin.userData.baseColor = C.array.slice();
    const base = skin.userData.baseColor;
    const col = COLORS[colorName] || COLORS[DEFAULT_COLOR];
    const k = C.normalized || C.array instanceof Uint8Array ? 255 : 1;
    const cover = 0.92 * (density === "low_thinning" ? 0.65 : 1);
    for (let i = 0; i < C.count; i++) {
      const a = cover * mask[i];
      for (let c = 0; c < 3; c++) C.array[i * C.itemSize + c] = base[i * C.itemSize + c] * (1 - a) + col[c] * 0.8 * k * a;
    }
    C.needsUpdate = true;
  }

  // ---------- Pelo ----------
  // ctx: { grid: {pos, nrm}, triIndex, snap(p, lift) -> {p, n}, field(p, n) -> Vector3|null,
  //        headCenter, mask, params: {hair_texture, hair_color, hair_density, length} }
  function lengthAt(u, L) {
    const ws = smooth(0.45, 0.75, Math.abs(u.x)) * (1 - smooth(0.55, 0.85, u.y));
    const wb = smooth(0.2, 0.6, -u.z) * (1 - smooth(0.35, 0.75, u.y)) * (1 - ws);
    const wt = Math.max(0, 1 - ws - wb);
    let mm = wt * L.top + ws * L.sides + wb * L.back;
    const fy = FADE_Y[L.fade];
    if (fy !== undefined && ws + wb > 0.3) mm = L.fade_mm + (mm - L.fade_mm) * smooth(fy - 0.25, fy, u.y);
    return mm;
  }

  function sampleRoots(ctx, count, seed) {
    const rnd = mulberry32(seed);
    const { pos, nrm } = ctx.grid, idx = ctx.triIndex, mask = ctx.mask;
    const tris = [], cum = [];
    let total = 0;
    const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
    for (let f = 0; f < idx.length; f += 3) {
      const a = idx[f], b = idx[f + 1], c = idx[f + 2];
      if ((mask[a] + mask[b] + mask[c]) / 3 < 0.5) continue;
      A.fromArray(pos, a * 3); B.fromArray(pos, b * 3); C.fromArray(pos, c * 3);
      total += B.clone().sub(A).cross(C.clone().sub(A)).length() / 2;
      tris.push(f); cum.push(total);
    }
    const roots = [];
    if (!tris.length) return roots;
    for (let k = 0; k < count; k++) {
      const r = rnd() * total;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
      const f = tris[lo];
      let u = rnd(), v = rnd(); if (u + v > 1) { u = 1 - u; v = 1 - v; }
      const p = new THREE.Vector3(), n = new THREE.Vector3();
      for (const [i, w] of [[idx[f], 1 - u - v], [idx[f + 1], u], [idx[f + 2], v]]) {
        p.x += pos[i * 3] * w; p.y += pos[i * 3 + 1] * w; p.z += pos[i * 3 + 2] * w;
        n.x += nrm[i * 3] * w; n.y += nrm[i * 3 + 1] * w; n.z += nrm[i * 3 + 2] * w;
      }
      // lenVar: variación de largo entre mechones ("puntas desiguales", para
      // que no parezca un casco de una sola pieza). Con pelo corto un ±15 %
      // ya se notaba demasiado (parecía un peinado de púas); se deja más
      // discreto (antes 0,82-1,12, un ±15 % de spread).
      roots.push({ p, n: n.normalize(), shade: 0.8 + rnd() * 0.4, phase: rnd() * Math.PI * 2, width: 0.75 + rnd() * 0.5, layer: rnd(), lenVar: 0.91 + rnd() * 0.14 });
    }
    return roots;
  }

  // Material con balanceo: desplaza cada vértice uSway * sway en el shader.
  function withSway(mat, uSway) {
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uSway = uSway;
      sh.vertexShader = "attribute float sway;\nuniform vec3 uSway;\n" +
        sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\ntransformed += uSway * sway;");
    };
    return mat;
  }

  // Ruido suave 1D (suma de senos con fases por mechón): textura natural.
  // Ancho a lo largo del mechón: lleno casi hasta el final y punta redondeada
  // (no una aguja -- una punta demasiado afilada es lo que hacía que el pelo
  // liso se leyera como púas, ver la nota de `buildHair`).
  const taper = (fr) => 1 - 0.22 * fr - 0.45 * Math.pow(fr, 3);
  const wobble = (s, ph, len) => Math.sin((2 * Math.PI * s) / len + ph) + 0.45 * Math.sin((2 * Math.PI * s) / (len * 0.43) + ph * 1.7);

  // El pelo son MECHONES: cada raíz es una cinta ancha (tapa el cuero
  // cabelludo como el pelo real, sin huecos entre pelos) con unas hebras
  // finas encima que le dan detalle y brillo. Pedro: "aunque sea liso tiene
  // un poco de textura, al peinarlo puede tapar entradas; más densidad, no
  // tanta separación entre cabellos".
  function buildHair(ctx) {
    const P = ctx.params;
    const texName = TEXTURES[P.hair_texture] ? P.hair_texture : "liso";
    const tex = TEXTURES[texName];
    const col = COLORS[P.hair_color] || COLORS[DEFAULT_COLOR];
    const dens = DENSITY[P.hair_density] || 1;
    const count = Math.round(3000 * dens * (texName === "afro" ? 1.2 : 1));
    // Ancho del mechón en la raíz (unidades de escena; 0,01 ≈ 2,4 mm).
    const W0 = { liso: 0.024, ondulado: 0.022, rizado: 0.014, afro: 0.011 }[texName] * (dens < 1 ? 0.85 : 1);
    const roots = sampleRoots(ctx, count, 7);
    const DOWN = new THREE.Vector3(0, -1, 0);
    const tmp = new THREE.Vector3();
    const rootC = col.map((c) => c * 0.6), tipC = col.map((c) => Math.min(1, c * 1.15 + 0.03));
    const hiC = col.map((c) => Math.min(1, c * 1.45 + 0.06));
    // Cinta: posiciones, normales, colores, balanceo, índices. Hebras: líneas.
    const rp = [], rn = [], rc = [], rs = [], ri = [];
    const lp = [], lc = [], ls = [];
    const rnd = mulberry32(11);
    const swayOf = (s) => Math.pow(Math.min(1, s / 0.35), 1.5) * (1 - tex.stiff);

    for (const r of roots) {
      const u = r.p.clone().sub(ctx.headCenter).normalize();
      const mm = lengthAt(u, P.length);
      if (mm < 2.5) continue;
      const Lw = mm * MM * tex.shrink * (mm > 15 ? r.lenVar : 1);   // puntas desiguales
      const stubble = mm < 12;
      const segs = Math.max(2, Math.min(28, Math.ceil(Lw / 0.015)));
      const ds = Lw / segs;
      // Cada mechón sale un poco desviado del peinado (no todos paralelos),
      // pero POCO: el campo de direcciones ya es un vector unitario (ver
      // `flowField` en growth-map.html), así que una desviación del mismo
      // orden de magnitud competía con él y, sobre todo cerca de la
      // coronilla -- donde el campo "irradia desde un punto" y por tanto
      // cambia muy rápido de una raíz a la vecina --, el resultado era que
      // cada mechón salía disparado en una dirección casi al azar (aspecto
      // de "púas"/erizo en vez de peinado). 0.5 -> 0.16.
      const jit = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
      jit.sub(r.n.clone().multiplyScalar(jit.dot(r.n))).multiplyScalar(0.16);
      // "Cobertura de entradas": si justo por delante de la raíz -- siguiendo
      // la dirección en la que ya se peina -- la piel está calva (entradas,
      // sin remolino que la tape), esta hebra se peina hacia ahí de forma
      // más disciplinada (menos ruido, más apoyada en la piel) para
      // disimularla, en vez de dejar que el ruido normal del peinado la
      // desvíe (Pedro: "que al poner el pelo hacia un lado, las entradas se
      // disimulen"). Solo hace falta mirarlo una vez por mechón, no en cada
      // tramo: la piel no cambia mientras crece un mechón de 2-7 cm.
      // `ctx.maskAt` es opcional para no romper otros usos de `buildHair`.
      // Una entrada de verdad es ancha (varios cm entre el nacimiento normal
      // y el retrocedido), así que hay que mirar bastante más lejos que un
      // solo tramo de mechón para encontrarla; se prueba a dos distancias.
      let coverage = 0;
      if (ctx.maskAt) {
        const ahead0 = ctx.field(r.p, r.n) || DOWN.clone().sub(r.n.clone().multiplyScalar(r.n.y)).normalize();
        for (const dist of [0.09, 0.2]) {
          const m = ctx.maskAt(r.p.clone().add(ahead0.clone().multiplyScalar(dist)));
          coverage = Math.max(coverage, Math.max(0, Math.min(1, 1 - m)));
        }
      }
      // "Repeinado": Pedro -- si el pelo es liso/ondulado (con rizado/afro
      // no se pide, y tampoco tendría sentido: no se alisan para tapar una
      // entrada) y ese mechón concreto está tapando una entrada, que quede
      // repeinado de verdad (pegado, sin la textura/ondulación suelta de
      // siempre), como un peinado de peluquería y no como pelo suelto que
      // por casualidad cae por ahí.
      const slick = (texName === "liso" || texName === "ondulado") ? coverage : 0;
      // 1) Línea central: sigue el crecimiento cerca de la raíz y luego cae
      //    por gravedad (o sale hacia fuera en rizado/afro), sin atravesar la
      //    cabeza. Cada tramo queda algo más separado de la piel que el
      //    anterior, así el pelo se superpone en capas y el de arriba tapa
      //    zonas sin pelo (entradas, coronilla clara) si es lo bastante largo.
      const pts = [r.p.clone().add(r.n.clone().multiplyScalar(0.002))], nrms = [r.n.clone()], ss = [0];
      let p = pts[0].clone(), n = r.n.clone(), s = 0;
      for (let i = 0; i < segs; i++) {
        const f = ctx.field(p, n) || DOWN.clone().sub(n.clone().multiplyScalar(n.y)).normalize();
        f.add(jit.clone().multiplyScalar((1 - smooth(0, 0.05, s) * 0.8) * (1 - 0.85 * coverage)));
        // Gravedad "peinada": sobre todo a lo largo de la piel (el pelo se
        // apoya en la cabeza y cae por los lados), y algo de caída libre.
        // Empieza a pesar un poco antes que antes (0.04 -> 0.02): así el
        // peinado estabiliza la dirección enseguida, en vez de dejar que el
        // primer tramo (el más visible, cerca de la raíz) dependa solo del
        // campo de direcciones y del jitter. Con cobertura de entradas pesa
        // aún un poco más: ese mechón tiene que llegar hasta tapar la calva.
        const gw = tex.gravity * smooth(0.0, 0.02, s) * (1 + 0.5 * coverage);
        const tDown = DOWN.clone().sub(n.clone().multiplyScalar(n.y));
        const dir = f.clone().multiplyScalar(1 - 0.6 * gw).add(tDown.multiplyScalar(1.3 * gw)).add(DOWN.clone().multiplyScalar(0.35 * gw));
        dir.add(n.clone().multiplyScalar(stubble ? 0.5 + tex.outward : tex.outward));
        // Cara despejada: por delante de la cara, el pelo largo se aparta a
        // los lados (como con raya), en vez de caer como una cortina.
        const rel = tmp.copy(p).sub(ctx.headCenter);
        // Solo por debajo de las cejas: un flequillo sobre la frente sí vale.
        if (rel.z > 0.12 && rel.y < 0.2 && Math.abs(rel.x) < 0.3) {
          const k = smooth(0.12, 0.3, rel.z) * (1 - smooth(0.22, 0.3, Math.abs(rel.x))) * (1 - smooth(0.06, 0.16, rel.y));
          dir.x += (r.p.x >= ctx.headCenter.x ? 1 : -1) * 2.2 * k;
          dir.z -= 0.6 * k;
        }
        dir.normalize();
        const q = p.clone().add(dir.multiplyScalar(ds));
        const sn = ctx.snap(q, 0, false);
        // Repeinado: capas más pegadas a la piel (menos "esponjado") y
        // menos variación entre mechones -- el efecto peinado/con producto.
        const minH = 0.0025 + (0.018 + 0.06 * tex.outward) * (1 - 0.55 * slick) * Math.min(s, 0.25)
          + 0.004 * r.layer * (1 - 0.5 * slick);
        const hgt = tmp.copy(q).sub(sn.p).dot(sn.n);
        if (hgt < minH) q.add(sn.n.clone().multiplyScalar(minH - hgt));
        s += ds; p = q; n = sn.n;
        pts.push(q.clone()); nrms.push(n.clone()); ss.push(s);
      }
      // 2) Forma: espiral (rizado/afro), ondas (ondulado) y, en todos, una
      //    textura suave e irregular (también en el liso).
      const sub = tex.coil ? 3 : tex.wave ? 2 : 1;
      const line = [], lnrm = [], lss = [], side = [];
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const T = b.clone().sub(a).normalize();
        const Bv = new THREE.Vector3().crossVectors(T, nrms[i]);
        if (Bv.lengthSq() < 1e-8) Bv.set(1, 0, 0); else Bv.normalize();
        const N2 = new THREE.Vector3().crossVectors(Bv, T).normalize();
        for (let j = (i === 0 ? 0 : 1); j <= sub; j++) {
          const t = j / sub, sc = ss[i] + (ss[i + 1] - ss[i]) * t;
          const c = a.clone().lerp(b, t);
          const ramp = smooth(0, 0.012, sc);
          if (tex.coil) {
            const ph = (2 * Math.PI * sc) / tex.pitch + r.phase;
            c.add(Bv.clone().multiplyScalar(Math.cos(ph) * tex.coil * ramp)).add(N2.clone().multiplyScalar(Math.sin(ph) * tex.coil * ramp));
          } else if (tex.wave) {
            c.add(Bv.clone().multiplyScalar(Math.sin((2 * Math.PI * sc) / tex.waveLen + r.phase) * tex.wave * (1 - 0.75 * slick) * ramp));
          }
          // Textura natural: el liso no es una línea perfecta (repeinado: menos).
          const amp = (texName === "liso" ? 0.0024 : 0.0016) * (1 - 0.6 * slick) * ramp * Math.min(1, sc / 0.05);
          c.add(Bv.clone().multiplyScalar(amp * wobble(sc, r.phase, 0.11)))
           .add(N2.clone().multiplyScalar(0.5 * amp * wobble(sc, r.phase * 2.3, 0.07)));
          line.push(c); lnrm.push(nrms[i].clone().lerp(nrms[i + 1], t).normalize()); lss.push(sc); side.push(Bv);
        }
      }
      // 3) Cinta del mechón: ancha en la raíz y afinándose hacia la punta.
      const w0 = W0 * r.width, base = rp.length / 3;
      for (let k = 0; k < line.length; k++) {
        const c = line[k], sc = lss[k], fr = sc / Lw;
        const half = 0.5 * w0 * taper(fr);
        // Eje lateral de la cinta: perpendicular a la dirección del pelo y a la piel.
        const Tk = (k + 1 < line.length ? line[k + 1].clone().sub(c) : c.clone().sub(line[k - 1])).normalize();
        let sd = new THREE.Vector3().crossVectors(Tk, lnrm[k]);
        if (sd.lengthSq() < 1e-8) sd = side[k].clone(); else sd.normalize();
        const tw = Math.sin(r.phase + sc * 25) * 0.25;   // se retuerce un poco
        const nk = lnrm[k].clone().add(sd.clone().multiplyScalar(tw)).normalize();
        const shade = (0.5 + 0.5 * r.shade) * (0.94 + 0.06 * wobble(sc, r.phase, 0.09));
        for (const sgn of [-1, 1]) {
          rp.push(c.x + sd.x * half * sgn, c.y + sd.y * half * sgn, c.z + sd.z * half * sgn);
          rn.push(nk.x, nk.y, nk.z);
          for (let ch = 0; ch < 3; ch++) rc.push((rootC[ch] + (tipC[ch] - rootC[ch]) * fr) * shade);
          rs.push(swayOf(sc));
        }
        if (k > 0) {
          const a = base + 2 * (k - 1), b = base + 2 * k;
          ri.push(a, a + 1, b + 1, a, b + 1, b);
        }
      }
      // 4) Hebras finas sobre la cinta: detalle de pelo y reflejos.
      const nStr = 1;
      const lit = 1.25;   // las cintas se iluminan; las hebras no: igualar tono
      for (let h = 0; h < nStr; h++) {
        const off = (rnd() - 0.5) * 0.8, lift = 0.0006 + rnd() * 0.0008, ph = rnd() * 6.28;
        const bright = 0.92 + rnd() * 0.16;
        for (let k = 1; k < line.length; k++) {
          for (const kk of [k - 1, k]) {
            const fr = lss[kk] / Lw, half = 0.5 * w0 * taper(fr);
            const o = half * (off + 0.25 * Math.sin(lss[kk] * 9 + ph));
            const c = line[kk], sd = side[kk], nn = lnrm[kk];
            lp.push(c.x + sd.x * o + nn.x * lift, c.y + sd.y * o + nn.y * lift, c.z + sd.z * o + nn.z * lift);
            const hl = 0.5 + 0.5 * Math.sin(fr * 5 + ph);   // brillo repartido a lo largo
            for (let ch = 0; ch < 3; ch++) lc.push(((rootC[ch] + (tipC[ch] - rootC[ch]) * fr) * (1 - 0.25 * hl) + hiC[ch] * 0.25 * hl) * (0.5 + 0.5 * r.shade) * bright * lit);
            ls.push(swayOf(lss[kk]));
          }
        }
      }
    }
    const uSway = { value: new THREE.Vector3() };
    const group = new THREE.Group();
    group.userData.uSway = uSway;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(rp, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(rn, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(rc, 3));
    g.setAttribute("sway", new THREE.Float32BufferAttribute(rs, 1));
    g.setIndex(rp.length / 3 > 65535 ? new THREE.Uint32BufferAttribute(ri, 1) : new THREE.Uint16BufferAttribute(ri, 1));
    const ribbons = new THREE.Mesh(g, withSway(new THREE.MeshStandardMaterial({
      vertexColors: true, side: THREE.DoubleSide, roughness: 0.62, metalness: 0.0,
    }), uSway));
    ribbons.name = "hair-locks";
    const gl = new THREE.BufferGeometry();
    gl.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
    gl.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    gl.setAttribute("sway", new THREE.Float32BufferAttribute(ls, 1));
    const strands = new THREE.LineSegments(gl, withSway(new THREE.LineBasicMaterial({ vertexColors: true }), uSway));
    strands.name = "hair-strands";
    for (const o of [ribbons, strands]) { o.frustumCulled = false; group.add(o); }
    group.userData.vertexCount = rp.length / 3 + lp.length / 3;
    return group;
  }

  // Libera la memoria de un pelo ya dibujado.
  function disposeHair(hair) {
    if (!hair) return;
    hair.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
  }

  // ---------- Balanceo ----------
  // Muelle amortiguado movido por la velocidad de giro de la cámara: el
  // pelo se retrasa respecto al giro y vuelve oscilando un poco.
  function createSway() {
    return { x: 0, vx: 0, y: 0, vy: 0, az: null, el: null, t: performance.now() };
  }
  function updateSway(state, hair, camera, target) {
    const now = performance.now(), dt = Math.min(0.05, (now - state.t) / 1000); state.t = now;
    if (!hair || !dt) return;
    const d = camera.position.clone().sub(target);
    const az = Math.atan2(d.x, d.z), el = Math.atan2(d.y, Math.hypot(d.x, d.z));
    let wAz = 0, wEl = 0;
    if (state.az !== null) {
      let da = az - state.az; if (da > Math.PI) da -= 2 * Math.PI; if (da < -Math.PI) da += 2 * Math.PI;
      wAz = da / dt; wEl = (el - state.el) / dt;
    }
    state.az = az; state.el = el;
    const K = 60, C = 7, G = 1.2;
    state.vx += (-K * state.x - C * state.vx - G * wAz) * dt; state.x += state.vx * dt;
    state.vy += (-K * state.y - C * state.vy - G * wEl) * dt; state.y += state.vy * dt;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
    const amp = 0.8;   // ~1 cm en las puntas del pelo largo al girar a ritmo normal
    hair.userData.uSway.value.copy(right.multiplyScalar(state.x * amp)).add(up.multiplyScalar(state.y * amp));
  }

  return { MM, COLORS, TEXTURES, applyMorphs, hairMask, paintScalp, buildHair, disposeHair, createSway, updateSway };
})();
