/* ═══════════════════════════════════════════════════════════════════
   CardioSDF — Multi-page SPA client
   Upload → Viewer → Results
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ─── State ───
  const state = {
    caseId: null,
    caseMeta: null,
    currentZ: 0,
    currentFrame: 0,
    sliceWidth: 0,
    sliceHeight: 0,
    imageData: null,
    labelData: null,
    drawnData: null,
    drawMode: "draw",
    brushSize: 10,
    isDrawing: false,
    dirty: false,
    showGt: true,
    zoom: 1.0,
    mode: "3d",  // "3d" or "wt"
  };

  // ─── Selectors ───
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // ─── Pages & Navigation ───
  const pages = {
    upload:  $("#pageUpload"),
    viewer:  $("#pageViewer"),
    results: $("#pageResults"),
  };

  function navigateTo(page) {
    Object.values(pages).forEach((p) => p.classList.remove("active"));
    $$(".nav-tab").forEach((t) => t.classList.remove("active"));
    if (pages[page]) {
      pages[page].classList.remove("active");
      // Force reflow for animation
      void pages[page].offsetWidth;
      pages[page].classList.add("active");
    }
    const tab = document.querySelector(`.nav-tab[data-nav="${page}"]`);
    if (tab) tab.classList.add("active");
    // Set body mode class so CSS can show/hide wt-only sections
    document.body.classList.remove("mode-3d", "mode-wt");
    document.body.classList.add(state.mode === "wt" ? "mode-wt" : "mode-3d");
  }

  $$(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.disabled) return;
      navigateTo(tab.dataset.nav);
    });
  });

  $$("[data-nav]").forEach((el) => {
    if (el.tagName === "A") {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        navigateTo(el.dataset.nav);
      });
    }
  });

  function enableTab(name) {
    const tab = document.querySelector(`.nav-tab[data-nav="${name}"]`);
    if (tab) tab.disabled = false;
  }

  // ─── Loading overlay ───
  function showLoading(text) {
    const overlay = $("#loadingOverlay");
    const msg = $("#loadingText");
    if (msg) msg.textContent = text || "Processing…";
    overlay.classList.add("visible");
  }

  function hideLoading() {
    $("#loadingOverlay").classList.remove("visible");
  }

  // ─── Toast ───
  function toast(message, type = "info") {
    const container = $("#toastContainer");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(-8px)";
      el.style.transition = "all 0.3s ease";
      setTimeout(() => el.remove(), 300);
    }, 4000);
  }

  // ─── API helpers ───
  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: body instanceof FormData ? {} : { "Content-Type": "application/json" },
      body: body instanceof FormData ? body : JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  async function apiGet(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function setBusy(el, busy) {
    el.classList.toggle("busy", busy);
  }

  // ─── DOM refs ───
  const uploadForm     = $("#uploadForm");
  const mriFileInput   = $("#mriFile");
  const segFileInput   = $("#segFile");
  const caseNameInput  = $("#caseName");
  const demoButton     = $("#demoButton");
  const drawButton     = $("#drawButton");
  const eraseButton    = $("#eraseButton");
  const clearButton    = $("#clearButton");
  const brushSlider    = $("#brushSlider");
  const brushValue     = $("#brushValue");
  const saveMaskButton = $("#saveMaskButton");
  const inferButton    = $("#inferButton");
  const frameSlider    = $("#frameSlider");
  const frameValue     = $("#frameValue");
  const sliceSlider    = $("#sliceSlider");
  const sliceValue     = $("#sliceValue");
  const sliceCaption   = $("#sliceCaption");
  const canvasEmpty    = $("#canvasEmpty");
  const caseStatus     = $("#caseStatus");
  const inferenceSource = $("#inferenceSource");
  const meshStatus     = $("#meshStatus");
  const threeFallback  = $("#threeFallback");

  const canvas = $("#mriCanvas");
  const ctx    = canvas.getContext("2d");

  const shapeReadout   = $("#shapeReadout");
  const spacingReadout = $("#spacingReadout");
  const framesReadout  = $("#framesReadout");
  const gtReadout      = $("#gtReadout");

  // ─── Mode selector ───
  $$(".mode-card").forEach((card) => {
    card.addEventListener("click", () => {
      $$(".mode-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      state.mode = card.dataset.mode;
    });
  });

  // ─── GT Toggle ───
  const gtToggle = $("#gtToggle");
  gtToggle.addEventListener("change", () => {
    state.showGt = gtToggle.checked;
    renderCanvas();
  });

  // ─── Zoom ───
  const zoomSlider = $("#zoomSlider");
  const zoomValueEl = $("#zoomValue");
  const canvasWrapper = $("#canvasWrapper");
  zoomSlider.addEventListener("input", () => {
    state.zoom = parseInt(zoomSlider.value, 10) / 100;
    zoomValueEl.textContent = state.zoom.toFixed(1) + "×";
    canvasWrapper.style.transform = `scale(${state.zoom})`;
  });

  // ═══════════════════════════════════════════════════════════════
  // Case Loading
  // ═══════════════════════════════════════════════════════════════

  function onCaseLoaded(caseMeta) {
    state.caseId = caseMeta.id;
    state.caseMeta = caseMeta;
    state.currentZ = caseMeta.centerSlice;
    state.currentFrame = 0;

    // Status
    caseStatus.textContent = caseMeta.name;
    caseStatus.classList.add("loaded");

    // Readouts
    shapeReadout.textContent = caseMeta.shape.join(" × ");
    spacingReadout.textContent = caseMeta.spacing.map((v) => v.toFixed(2)).join(" × ");
    framesReadout.textContent = String(caseMeta.frames);
    gtReadout.textContent = caseMeta.hasSegmentation ? "Yes" : "No";
    gtReadout.style.color = caseMeta.hasSegmentation ? "var(--accent-green)" : "var(--ink-muted)";

    // Sliders
    frameSlider.max = Math.max(0, caseMeta.frames - 1);
    frameSlider.value = 0;
    frameValue.textContent = "0";
    sliceSlider.max = Math.max(0, caseMeta.slices - 1);
    sliceSlider.value = state.currentZ;
    sliceValue.textContent = String(state.currentZ);

    canvasEmpty.style.display = "none";

    // Enable tabs and navigate
    enableTab("viewer");
    enableTab("results");
    navigateTo("viewer");

    fetchSlice();
    toast(`Case "${caseMeta.name}" loaded`, "success");
  }

  // ─── Upload ───
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!mriFileInput.files.length) return;
    const formData = new FormData();
    formData.append("mri", mriFileInput.files[0]);
    if (segFileInput.files.length) formData.append("segmentation", segFileInput.files[0]);
    if (caseNameInput.value.trim()) formData.append("caseName", caseNameInput.value.trim());

    showLoading("Uploading and processing MRI…");
    try {
      const data = await apiPost("/api/upload", formData);
      onCaseLoaded(data.case);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  });

  // ─── Patient selector ───
  const patientSelect = $("#patientSelect");

  async function loadPatientList() {
    try {
      const data = await apiGet("/api/patients");
      if (data.patients && data.patients.length > 0) {
        data.patients.forEach((p) => {
          const opt = document.createElement("option");
          opt.value = p.id;
          opt.textContent = `${p.id} — ${p.group || "?"}`;
          patientSelect.appendChild(opt);
        });
      }
    } catch (_) { /* no patient list available */ }
  }
  loadPatientList();

  // ─── Demo ───
  demoButton.addEventListener("click", async () => {
    const patient = patientSelect.value || undefined;
    showLoading(`Loading ACDC patient${patient ? " " + patient : ""}…`);
    try {
      const body = patient ? { patient } : {};
      const data = await apiPost("/api/demo", body);
      onCaseLoaded(data.case);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  });

  // ═══════════════════════════════════════════════════════════════
  // Slice rendering
  // ═══════════════════════════════════════════════════════════════

  async function fetchSlice() {
    if (!state.caseId) return;
    try {
      const data = await apiGet(
        `/api/case/${state.caseId}/slice?z=${state.currentZ}&frame=${state.currentFrame}`
      );
      state.sliceWidth = data.width;
      state.sliceHeight = data.height;
      state.imageData = b64ToU8(data.image, data.width * data.height);
      state.labelData = b64ToU8(data.labels, data.width * data.height);
      state.drawnData = b64ToU8(data.drawn, data.width * data.height);
      state.dirty = false;

      canvas.width = data.width;
      canvas.height = data.height;
      renderCanvas();

      sliceCaption.textContent =
        `Slice ${data.z} / ${state.caseMeta.slices - 1}  ·  Frame ${data.frame}  ·  ${data.width} × ${data.height}`;
    } catch (err) {
      toast(`Slice error: ${err.message}`, "error");
    }
  }

  function b64ToU8(b64, len) {
    const raw = atob(b64);
    const a = new Uint8Array(len);
    for (let i = 0; i < Math.min(raw.length, len); i++) a[i] = raw.charCodeAt(i);
    return a;
  }

  function u8ToB64(a) {
    let s = "";
    for (let i = 0; i < a.length; i++) s += String.fromCharCode(a[i]);
    return btoa(s);
  }

  const LABEL_COLORS = [
    null,
    [59, 130, 246, 90],   // RV — blue
    [34, 197, 94, 90],    // MYO — green
    [231, 76, 94, 100],   // LV — red
  ];
  const DRAWN_COLOR = [245, 158, 11, 80]; // amber

  function renderCanvas() {
    if (!state.imageData) return;
    const w = state.sliceWidth, h = state.sliceHeight;
    const img = ctx.createImageData(w, h);
    const px = img.data;

    for (let i = 0; i < w * h; i++) {
      const g = state.imageData[i];
      const o = i * 4;
      px[o] = g; px[o + 1] = g; px[o + 2] = g; px[o + 3] = 255;

      if (state.showGt) {
        const lbl = state.labelData ? state.labelData[i] : 0;
        if (lbl > 0 && lbl < LABEL_COLORS.length && LABEL_COLORS[lbl]) {
          const c = LABEL_COLORS[lbl], a = c[3] / 255;
          px[o]     = Math.round(px[o]     * (1 - a) + c[0] * a);
          px[o + 1] = Math.round(px[o + 1] * (1 - a) + c[1] * a);
          px[o + 2] = Math.round(px[o + 2] * (1 - a) + c[2] * a);
        }
      }

      if (state.drawnData && state.drawnData[i] > 0) {
        const c = DRAWN_COLOR, a = c[3] / 255;
        px[o]     = Math.round(px[o]     * (1 - a) + c[0] * a);
        px[o + 1] = Math.round(px[o + 1] * (1 - a) + c[1] * a);
        px[o + 2] = Math.round(px[o + 2] * (1 - a) + c[2] * a);
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // ─── Sliders ───
  frameSlider.addEventListener("input", () => {
    state.currentFrame = parseInt(frameSlider.value, 10);
    frameValue.textContent = frameSlider.value;
    fetchSlice();
  });

  sliceSlider.addEventListener("input", () => {
    state.currentZ = parseInt(sliceSlider.value, 10);
    sliceValue.textContent = sliceSlider.value;
    fetchSlice();
  });

  // ─── Drawing ───
  drawButton.addEventListener("click", () => {
    state.drawMode = "draw";
    drawButton.classList.add("active");
    eraseButton.classList.remove("active");
  });

  eraseButton.addEventListener("click", () => {
    state.drawMode = "erase";
    eraseButton.classList.add("active");
    drawButton.classList.remove("active");
  });

  clearButton.addEventListener("click", () => {
    if (!state.drawnData) return;
    state.drawnData.fill(0);
    state.dirty = true;
    renderCanvas();
  });

  brushSlider.addEventListener("input", () => {
    state.brushSize = parseInt(brushSlider.value, 10);
    brushValue.textContent = brushSlider.value;
  });

  function canvasXY(e) {
    const r = canvas.getBoundingClientRect();
    return {
      x: Math.floor((e.clientX - r.left) * canvas.width / r.width),
      y: Math.floor((e.clientY - r.top) * canvas.height / r.height),
    };
  }

  function paintBrush(cx, cy) {
    if (!state.drawnData) return;
    const r = Math.max(1, Math.floor(state.brushSize / 2));
    const val = state.drawMode === "draw" ? 1 : 0;
    const w = state.sliceWidth, h = state.sliceHeight;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;
        const px = cx + dx, py = cy + dy;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          state.drawnData[py * w + px] = val;
        }
      }
    }
    state.dirty = true;
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!state.caseId || !state.drawnData) return;
    state.isDrawing = true;
    canvas.setPointerCapture(e.pointerId);
    const { x, y } = canvasXY(e);
    paintBrush(x, y);
    renderCanvas();
  });

  canvas.addEventListener("pointermove", (e) => {
    const cursor = $("#canvasCrosshair");
    if (state.caseId && state.drawnData) {
      const r = canvas.getBoundingClientRect();
      const scale = r.width / canvas.width;
      cursor.style.display = "block";
      // Position relative to canvasWrapper
      const wr = canvasWrapper.getBoundingClientRect();
      cursor.style.left = `${e.clientX - wr.left}px`;
      cursor.style.top = `${e.clientY - wr.top}px`;
      const sz = state.brushSize * scale;
      cursor.style.width = `${sz}px`;
      cursor.style.height = `${sz}px`;
    }
    if (!state.isDrawing) return;
    const { x, y } = canvasXY(e);
    paintBrush(x, y);
    renderCanvas();
  });

  canvas.addEventListener("pointerup", () => { state.isDrawing = false; });
  canvas.addEventListener("pointerleave", () => {
    state.isDrawing = false;
    $("#canvasCrosshair").style.display = "none";
  });

  // Scroll to change slice
  canvas.addEventListener("wheel", (e) => {
    if (!state.caseId) return;
    e.preventDefault();
    const d = e.deltaY > 0 ? 1 : -1;
    const nz = Math.max(0, Math.min(state.caseMeta.slices - 1, state.currentZ + d));
    if (nz !== state.currentZ) {
      state.currentZ = nz;
      sliceSlider.value = nz;
      sliceValue.textContent = String(nz);
      fetchSlice();
    }
  }, { passive: false });

  // ─── Save mask ───
  saveMaskButton.addEventListener("click", async () => {
    if (!state.caseId || !state.drawnData || !state.dirty) {
      toast("Nothing to save", "info");
      return;
    }
    try {
      await apiPost(`/api/case/${state.caseId}/mask`, {
        z: state.currentZ,
        width: state.sliceWidth,
        height: state.sliceHeight,
        mask: u8ToB64(state.drawnData),
      });
      state.dirty = false;
      toast(`Mask saved for slice ${state.currentZ}`, "success");
    } catch (err) {
      toast(`Save failed: ${err.message}`, "error");
    }
  });

  // ═══════════════════════════════════════════════════════════════
  // Inference
  // ═══════════════════════════════════════════════════════════════

  inferButton.addEventListener("click", async () => {
    if (!state.caseId) { toast("Load a case first", "error"); return; }

    // Auto-save
    if (state.dirty && state.drawnData) {
      try {
        await apiPost(`/api/case/${state.caseId}/mask`, {
          z: state.currentZ,
          width: state.sliceWidth,
          height: state.sliceHeight,
          mask: u8ToB64(state.drawnData),
        });
        state.dirty = false;
      } catch (_) {}
    }

    showLoading("Running cardiac inference…");
    try {
      const result = await apiPost(`/api/case/${state.caseId}/infer`, {
        frame: state.currentFrame,
      });
      displayMetrics(result);
      if (result.meshes) build3DScene(result.meshes);
      inferenceSource.textContent = result.source || "Complete";
      navigateTo("results");
      toast("Inference complete", "success");
    } catch (err) {
      toast(`Inference failed: ${err.message}`, "error");
    } finally {
      hideLoading();
    }
  });

  // ─── Display metrics ───
  function displayMetrics(result) {
    const m = result.metrics || {};

    $$(".metric-value[data-key]").forEach((el) => {
      const v = m[el.dataset.key];
      el.textContent = v != null ? (typeof v === "number" ? v.toFixed(1) : String(v)) : "—";
    });

    // EF color
    const efEl = $('[data-key="ejectionFractionPct"]');
    if (efEl && m.ejectionFractionPct != null) {
      const ef = m.ejectionFractionPct;
      efEl.style.color = ef < 40 ? "#ef4444" : ef < 55 ? "var(--accent-amber)" : "var(--primary)";
    }

    // AHA 17-segment bullseye
    const aha17 = result.aha17 || [];
    if (aha17.length === 17) {
      drawBullseye(aha17);
    }

    // Regional list (use aha17 if available, fallback to regionalThickness)
    const segments = aha17.length === 17 ? aha17 : (result.regionalThickness || []);
    const list = $("#regionalList");
    list.innerHTML = "";
    segments.forEach((r) => {
      const row = document.createElement("div");
      row.className = "regional-row";
      row.dataset.status = r.status || "unavailable";
      const mm = r.meanMm != null ? r.meanMm : 0;
      const pct = Math.min(100, (mm / 18) * 100);
      const idLabel = r.id ? `<span class="regional-id">${r.id}</span>` : "";
      row.innerHTML = `
        ${idLabel}
        <span class="regional-name">${r.name}</span>
        <div class="regional-bar"><span style="width:${pct}%"></span></div>
        <span class="regional-value">${r.meanMm != null ? r.meanMm.toFixed(1) + " mm" : "—"}</span>
      `;
      list.appendChild(row);
    });

    meshStatus.textContent = result.meshMethod ? result.meshMethod : "ready";
  }

  // ─── AHA 17-Segment Bullseye ───
  function drawBullseye(aha17) {
    const canvas = $("#bullseyeCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const R = Math.min(cx, cy) - 20;

    ctx.clearRect(0, 0, W, H);

    // Ring radii: basal(outer), mid, apical, apex(center)
    const r0 = R;           // outer edge
    const r1 = R * 0.72;    // basal/mid boundary
    const r2 = R * 0.46;    // mid/apical boundary
    const r3 = R * 0.22;    // apical/apex boundary

    // Basal ring: segments 1-6, each 60°
    for (let i = 0; i < 6; i++) {
      const startAngle = -Math.PI / 2 + i * (Math.PI / 3);
      const endAngle = startAngle + Math.PI / 3;
      drawArc(ctx, cx, cy, r1, r0, startAngle, endAngle, aha17[i]);
    }

    // Mid ring: segments 7-12, each 60°
    for (let i = 0; i < 6; i++) {
      const startAngle = -Math.PI / 2 + i * (Math.PI / 3);
      const endAngle = startAngle + Math.PI / 3;
      drawArc(ctx, cx, cy, r2, r1, startAngle, endAngle, aha17[6 + i]);
    }

    // Apical ring: segments 13-16, each 90°
    for (let i = 0; i < 4; i++) {
      const startAngle = -Math.PI / 2 + i * (Math.PI / 2);
      const endAngle = startAngle + Math.PI / 2;
      drawArc(ctx, cx, cy, r3, r2, startAngle, endAngle, aha17[12 + i]);
    }

    // Apex: segment 17 (center circle)
    const apexVal = aha17[16].meanMm;
    ctx.beginPath();
    ctx.arc(cx, cy, r3, 0, 2 * Math.PI);
    ctx.fillStyle = apexVal != null ? thicknessColor(apexVal) : "#e8ecf1";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (apexVal != null) {
      ctx.fillStyle = apexVal > 8 ? "#fff" : "#1a1f36";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(apexVal.toFixed(1), cx, cy);
    }
  }

  function drawArc(ctx, cx, cy, rInner, rOuter, startAngle, endAngle, segment) {
    const val = segment ? segment.meanMm : null;
    ctx.beginPath();
    ctx.arc(cx, cy, rOuter, startAngle, endAngle);
    ctx.arc(cx, cy, rInner, endAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = val != null ? thicknessColor(val) : "#e8ecf1";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    const midAngle = (startAngle + endAngle) / 2;
    const midR = (rInner + rOuter) / 2;
    const tx = cx + midR * Math.cos(midAngle);
    const ty = cy + midR * Math.sin(midAngle);

    if (val != null) {
      ctx.fillStyle = val > 8 ? "#fff" : "#1a1f36";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(val.toFixed(1), tx, ty);
    }

    // Segment number
    if (segment && segment.id) {
      const numR = rOuter - (rOuter - rInner) * 0.18;
      const nx = cx + numR * Math.cos(midAngle);
      const ny = cy + numR * Math.sin(midAngle);
      ctx.fillStyle = val != null && val > 8 ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.2)";
      ctx.font = "500 9px Inter, sans-serif";
      ctx.fillText(String(segment.id), nx, ny);
    }
  }

  // Thickness → color (blue→cyan→green→yellow→red, 0-15mm)
  function thicknessColor(mm) {
    const t = Math.max(0, Math.min(1, mm / 15));
    const stops = [
      [0, [59, 130, 246]],
      [0.25, [34, 211, 238]],
      [0.5, [34, 197, 94]],
      [0.75, [245, 158, 11]],
      [1, [231, 76, 94]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i][0] && t <= stops[i + 1][0]) {
        const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        const r = Math.round(stops[i][1][0] + f * (stops[i + 1][1][0] - stops[i][1][0]));
        const g = Math.round(stops[i][1][1] + f * (stops[i + 1][1][1] - stops[i][1][1]));
        const b = Math.round(stops[i][1][2] + f * (stops[i + 1][1][2] - stops[i][1][2]));
        return `rgb(${r},${g},${b})`;
      }
    }
    return `rgb(231,76,94)`;
  }

  // ═══════════════════════════════════════════════════════════════
  // THREE.JS — 3D Cardiac Viewer
  // ═══════════════════════════════════════════════════════════════

  let scene, camera, renderer, controls, animId;

  function initThree() {
    const el = $("#heartViewport");
    if (!el || typeof THREE === "undefined") return;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    scene.fog = new THREE.FogExp2(0x0d1117, 0.002);

    camera = new THREE.PerspectiveCamera(45, el.clientWidth / Math.max(el.clientHeight, 1), 0.1, 2000);
    camera.position.set(80, 60, 120);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(el.clientWidth, el.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    el.appendChild(renderer.domElement);

    if (THREE.OrbitControls) {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.rotateSpeed = 0.6;
      controls.minDistance = 20;
      controls.maxDistance = 500;
    }

    // Lighting — dramatic medical look
    scene.add(new THREE.AmbientLight(0x1a2540, 0.5));
    scene.add(new THREE.HemisphereLight(0x0066aa, 0x221133, 0.4));

    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(50, 80, 60);
    key.castShadow = true;
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x4488cc, 0.45);
    fill.position.set(-40, 30, -30);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0x00ccff, 0.3);
    rim.position.set(0, -20, -60);
    scene.add(rim);

    // Subtle grid
    const grid = new THREE.GridHelper(200, 30, 0x1a2233, 0x111822);
    grid.position.y = -40;
    scene.add(grid);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", resize);
    new ResizeObserver(resize).observe(el);

    (function loop() {
      animId = requestAnimationFrame(loop);
      if (controls) controls.update();
      renderer.render(scene, camera);
    })();
  }

  function build3DScene(meshes) {
    if (!scene) initThree();
    if (!scene) return;
    threeFallback.style.display = "none";

    // Clear old
    const rm = [];
    scene.traverse((c) => { if (c.isMesh && c.userData.cardiac) rm.push(c); });
    rm.forEach((m) => { m.geometry.dispose(); m.material.dispose(); scene.remove(m); });

    let center = new THREE.Vector3(), count = 0;

    // Endocardium — SOLID, opaque, with wall thickness colors
    if (meshes.endo && meshes.endo.vertices.length > 0) {
      const m = buildMesh(meshes.endo, {
        color: 0xe74c5e,
        emissive: 0x440015,
        emissiveIntensity: 0.25,
        opacity: 1.0,
        transparent: false,
        roughness: 0.3,
        metalness: 0.12,
        values: meshes.endo.values,
        cmMin: 2, cmMax: 14,
      });
      if (m) {
        scene.add(m);
        m.geometry.computeBoundingBox();
        center.add(m.geometry.boundingBox.getCenter(new THREE.Vector3()));
        count++;
      }
    }

    // Epicardium — semi-transparent shell
    if (meshes.epi && meshes.epi.vertices.length > 0) {
      const m = buildMesh(meshes.epi, {
        color: 0x3b82f6,
        emissive: 0x0a1530,
        emissiveIntensity: 0.15,
        opacity: 0.15,
        transparent: true,
        roughness: 0.6,
        metalness: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      if (m) {
        scene.add(m);
        m.geometry.computeBoundingBox();
        center.add(m.geometry.boundingBox.getCenter(new THREE.Vector3()));
        count++;
      }
    }

    if (count > 0) {
      center.divideScalar(count);
      camera.position.set(center.x + 80, center.y + 60, center.z + 120);
      if (controls) { controls.target.copy(center); controls.update(); }
    }
  }

  function buildMesh(data, opts) {
    if (!data.vertices || data.vertices.length < 9) return null;
    const verts = new Float32Array(data.vertices);
    const idx = new Uint32Array(data.faces);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();

    let mat;
    if (opts.values && opts.values.length > 0) {
      const n = verts.length / 3;
      const cols = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const v = i < opts.values.length ? opts.values[i] : 0;
        const t = Math.max(0, Math.min(1, (v - (opts.cmMin || 3)) / ((opts.cmMax || 15) - (opts.cmMin || 3))));
        const c = wtColor(t);
        cols[i * 3] = c[0]; cols[i * 3 + 1] = c[1]; cols[i * 3 + 2] = c[2];
      }
      geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      mat = new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        emissive: new THREE.Color(opts.emissive || 0),
        emissiveIntensity: opts.emissiveIntensity || 0,
        opacity: opts.opacity ?? 1,
        transparent: !!opts.transparent,
        roughness: opts.roughness ?? 0.4,
        metalness: opts.metalness ?? 0.1,
        side: opts.side || THREE.FrontSide,
        depthWrite: opts.depthWrite !== false,
        clearcoat: 0.12,
        clearcoatRoughness: 0.35,
      });
    } else {
      mat = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(opts.color || 0xcccccc),
        emissive: new THREE.Color(opts.emissive || 0),
        emissiveIntensity: opts.emissiveIntensity || 0,
        opacity: opts.opacity ?? 1,
        transparent: !!opts.transparent,
        roughness: opts.roughness ?? 0.4,
        metalness: opts.metalness ?? 0.1,
        side: opts.side || THREE.FrontSide,
        depthWrite: opts.depthWrite !== false,
        clearcoat: 0.15,
        clearcoatRoughness: 0.3,
      });
    }

    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData.cardiac = true;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  // Colormap: blue → cyan → green → yellow → red
  function wtColor(t) {
    const stops = [
      [0, [0.23, 0.51, 0.96]],
      [0.25, [0.13, 0.76, 0.76]],
      [0.5, [0.13, 0.77, 0.37]],
      [0.75, [0.96, 0.62, 0.04]],
      [1, [0.91, 0.30, 0.37]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i][0] && t <= stops[i + 1][0]) {
        const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        return [
          stops[i][1][0] + f * (stops[i + 1][1][0] - stops[i][1][0]),
          stops[i][1][1] + f * (stops[i + 1][1][1] - stops[i][1][1]),
          stops[i][1][2] + f * (stops[i + 1][1][2] - stops[i][1][2]),
        ];
      }
    }
    return stops[stops.length - 1][1];
  }

  // ─── File drop visuals ───
  $$(".file-drop").forEach((drop) => {
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
    drop.addEventListener("drop", () => drop.classList.remove("dragover"));
  });

  // ─── Init 3D (lazy) ───
  // Don't init until results page is first shown — saves GPU
  const resultsObserver = new MutationObserver(() => {
    if (pages.results.classList.contains("active") && !scene) {
      initThree();
    }
  });
  resultsObserver.observe(pages.results, { attributes: true, attributeFilter: ["class"] });

})();
