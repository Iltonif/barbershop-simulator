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
    const cover = 0.82 * (density === "low_thinning" ? 0.6 : 1);
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
      roots.push({ p, n: n.normalize(), shade: 0.8 + rnd() * 0.4, phase: rnd() * Math.PI * 2 });
    }
    return roots;
  }

  function makeMaterial() {
    const mat = new THREE.LineBasicMaterial({ vertexColors: true });
    mat.userData.uSway = { value: new THREE.Vector3() };
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uSway = mat.userData.uSway;
      sh.vertexShader = "attribute float sway;\nuniform vec3 uSway;\n" +
        sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\ntransformed += uSway * sway;");
    };
    return mat;
  }

  function buildHair(ctx) {
    const P = ctx.params;
    const tex = TEXTURES[P.hair_texture] || TEXTURES.liso;
    const col = COLORS[P.hair_color] || COLORS[DEFAULT_COLOR];
    const dens = DENSITY[P.hair_density] || 1;
    const count = Math.round(2600 * dens * (P.hair_texture === "afro" ? 1.6 : P.hair_texture === "rizado" ? 1.25 : 1));
    const roots = sampleRoots(ctx, count, 7);
    const pos = [], colr = [], sway = [];
    const DOWN = new THREE.Vector3(0, -1, 0);
    const tmp = new THREE.Vector3();
    const rootC = col.map((c) => c * 0.55), tipC = col.map((c) => Math.min(1, c * 1.2 + 0.03));
    for (const r of roots) {
      const u = r.p.clone().sub(ctx.headCenter).normalize();
      const mm = lengthAt(u, P.length);
      if (mm < 2.5) continue;
      const Lw = mm * MM * tex.shrink;
      const stubble = mm < 12;
      const segs = Math.max(2, Math.min(36, Math.ceil(Lw / 0.012)));
      const ds = Lw / segs;
      // 1) Línea central: sigue el crecimiento cerca de la raíz y luego cae
      //    por gravedad (o sale hacia fuera en rizado/afro), sin atravesar la cabeza.
      const pts = [r.p.clone().add(r.n.clone().multiplyScalar(0.003))], nrms = [r.n.clone()], ss = [0];
      let p = pts[0].clone(), n = r.n.clone(), s = 0;
      for (let i = 0; i < segs; i++) {
        const f = ctx.field(p, n) || DOWN.clone().sub(n.clone().multiplyScalar(n.y)).normalize();
        const gw = tex.gravity * smooth(0.0, 0.05, s);
        const dir = f.clone().multiplyScalar(1 - gw).add(DOWN.clone().multiplyScalar(gw));
        dir.add(n.clone().multiplyScalar(stubble ? 0.5 + tex.outward : tex.outward));
        // Cara despejada: por delante de la cara, el pelo largo se aparta a
        // los lados (como con raya), en vez de caer como una cortina.
        const rel = tmp.copy(p).sub(ctx.headCenter);
        if (rel.z > 0.12 && rel.y < 0.42 && Math.abs(rel.x) < 0.3) {
          const k = smooth(0.12, 0.3, rel.z) * (1 - smooth(0.22, 0.3, Math.abs(rel.x)));
          dir.x += (r.p.x >= ctx.headCenter.x ? 1 : -1) * 2.2 * k;
          dir.z -= 0.6 * k;
        }
        dir.normalize();
        const q = p.clone().add(dir.multiplyScalar(ds));
        const sn = ctx.snap(q, 0, false);
        const minH = 0.003 + (0.025 + 0.06 * tex.outward) * Math.min(s, 0.25);
        const hgt = tmp.copy(q).sub(sn.p).dot(sn.n);
        if (hgt < minH) q.add(sn.n.clone().multiplyScalar(minH - hgt));
        s += ds; p = q; n = sn.n;
        pts.push(q.clone()); nrms.push(n.clone()); ss.push(s);
      }
      // 2) Ondas / espirales alrededor de la línea central.
      const sub = tex.coil ? 4 : tex.wave ? 2 : 1;
      let prev = null, prevS = 0;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const T = b.clone().sub(a).normalize();
        const Bv = new THREE.Vector3().crossVectors(T, nrms[i]);
        if (Bv.lengthSq() < 1e-8) Bv.set(1, 0, 0); else Bv.normalize();
        const N2 = new THREE.Vector3().crossVectors(Bv, T).normalize();
        for (let j = (i === 0 ? 0 : 1); j <= sub; j++) {
          const t = j / sub, sc = ss[i] + (ss[i + 1] - ss[i]) * t;
          const c = a.clone().lerp(b, t);
          const ramp = smooth(0, 0.01, sc);
          if (tex.coil) {
            const ph = (2 * Math.PI * sc) / tex.pitch + r.phase;
            c.add(Bv.clone().multiplyScalar(Math.cos(ph) * tex.coil * ramp)).add(N2.clone().multiplyScalar(Math.sin(ph) * tex.coil * ramp));
          } else if (tex.wave) {
            c.add(Bv.clone().multiplyScalar(Math.sin((2 * Math.PI * sc) / tex.waveLen + r.phase) * tex.wave * ramp));
          }
          if (prev) {
            pos.push(prev.x, prev.y, prev.z, c.x, c.y, c.z);
            for (const [ssv] of [[prevS], [sc]]) {
              const k = ssv / Lw;
              for (let ch = 0; ch < 3; ch++) colr.push((rootC[ch] + (tipC[ch] - rootC[ch]) * k) * r.shade);
              sway.push(Math.pow(Math.min(1, ssv / 0.35), 1.5) * (1 - tex.stiff));
            }
          }
          prev = c; prevS = sc;
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(colr, 3));
    g.setAttribute("sway", new THREE.Float32BufferAttribute(sway, 1));
    const obj = new THREE.LineSegments(g, makeMaterial());
    obj.frustumCulled = false;
    return obj;
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
    hair.material.userData.uSway.value.copy(right.multiplyScalar(state.x * amp)).add(up.multiplyScalar(state.y * amp));
  }

  return { MM, COLORS, TEXTURES, applyMorphs, hairMask, paintScalp, buildHair, createSway, updateSway };
})();
