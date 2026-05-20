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
    phase: "ed", // "ed" or "es"
    hasEs: false,
    resultsEd: null,   // stored inference result for ED
    resultsEs: null,    // stored inference result for ES
    resultsDifference: null,
    results: null,
    resultsPhase: "ed", // which phase to show in results
    sliceContours: null, // 3D slice contour data
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

  function phaseInfo(phase) {
    const meta = state.caseMeta || {};
    if (meta.phases && meta.phases[phase]) return meta.phases[phase];
    if (phase === "es") {
      return {
        slices: meta.esSlices || meta.slices || 1,
        frames: meta.esFrames || 1,
        shape: meta.shape || [],
        spacing: meta.spacing || [],
        hasSegmentation: !!meta.hasEsSeg,
        centerSlice: Math.floor((meta.esSlices || meta.slices || 1) / 2),
      };
    }
    return {
      slices: meta.slices || 1,
      frames: meta.frames || 1,
      shape: meta.shape || [],
      spacing: meta.spacing || [],
      hasSegmentation: !!meta.hasSegmentation,
      centerSlice: meta.centerSlice || 0,
    };
  }

  function setViewerPhase(phase, resetPosition = false) {
    const info = phaseInfo(phase);
    state.phase = phase;
    if (resetPosition) {
      state.currentZ = info.centerSlice || Math.floor((info.slices || 1) / 2);
      state.currentFrame = 0;
    } else {
      state.currentZ = Math.max(0, Math.min((info.slices || 1) - 1, state.currentZ));
      state.currentFrame = Math.max(0, Math.min((info.frames || 1) - 1, state.currentFrame));
    }

    shapeReadout.textContent = (info.shape || []).join(" × ") || "—";
    spacingReadout.textContent = (info.spacing || []).map((v) => Number(v).toFixed(2)).join(" × ") || "—";
    framesReadout.textContent = String(info.frames || 1);
    gtReadout.textContent = info.hasSegmentation ? "Yes" : "No";
    gtReadout.style.color = info.hasSegmentation ? "var(--accent-green)" : "var(--ink-muted)";

    frameSlider.max = Math.max(0, (info.frames || 1) - 1);
    frameSlider.value = state.currentFrame;
    frameValue.textContent = String(state.currentFrame);
    sliceSlider.max = Math.max(0, (info.slices || 1) - 1);
    sliceSlider.value = state.currentZ;
    sliceValue.textContent = String(state.currentZ);

    $$(".phase-btn[data-phase]").forEach((b) => b.classList.toggle("active", b.dataset.phase === phase));
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

  async function hydrateSdfResult(result, label = "SDF") {
    if (!result || !result.sdfResultUrl || result.meshes) return result;
    try {
      showLoading(`Loading ${label} 3D model from cloud…`);
      const sdfData = await fetch(result.sdfResultUrl).then((r) => {
        if (!r.ok) throw new Error(`GCS fetch failed: ${r.status}`);
        return r.json();
      });
      result.meshes = sdfData.meshes;
      result.meshMethod = sdfData.meshMethod;
      result.regionalThickness = sdfData.regionalThickness;
      result.aha17 = sdfData.aha17 || [];
    } catch (err) {
      console.warn(`${label} mesh fetch failed, using voxel meshes:`, err.message);
      toast(`${label} 3D model unavailable, using voxel meshes`, "warning");
    }
    return result;
  }

  function setBusy(el, busy) {
    el.classList.toggle("busy", busy);
  }

  // Poll a background job until done; rejects on error.
  async function pollJob(jobId, label) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 2500));
      const job = await apiGet(`/api/jobs/${jobId}`);
      if (job.status === "done") return job.result;
      if (job.status === "error") throw new Error(job.error || "Inference failed");
      showLoading(`${label}…`);
    }
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
      // Toggle upload panels
      const single = $("#uploadSingle");
      const dual = $("#uploadDual");
      if (state.mode === "wt") {
        single.style.display = "none";
        dual.style.display = "block";
      } else {
        single.style.display = "block";
        dual.style.display = "none";
      }
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
    state.phase = "ed";
    state.hasEs = !!caseMeta.hasEs;
    state.resultsEd = null;
    state.resultsEs = null;
    state.resultsDifference = null;
    state.results = null;
    state.resultsPhase = "ed";
    state.sliceContours = null;

    // Status
    caseStatus.textContent = caseMeta.name;
    caseStatus.classList.add("loaded");

    setViewerPhase("ed", true);

    canvasEmpty.style.display = "none";

    // Show/hide phase toggle in viewer
    const phaseSection = $("#phaseToggleSection");
    if (state.hasEs) {
      phaseSection.style.display = "block";
      $$(".phase-btn[data-phase]").forEach((b) => b.classList.toggle("active", b.dataset.phase === "ed"));
    } else {
      phaseSection.style.display = "none";
    }

    // Enable tabs and navigate
    enableTab("viewer");
    enableTab("results");
    navigateTo("viewer");

    fetchSlice();
    fetchSliceContours();
    toast(`Case "${caseMeta.name}" loaded`, "success");
  }

  // ─── Upload ───
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData();

    if (state.mode === "wt") {
      // Dual upload mode
      const edMri = $("#mriFileEd");
      const edSeg = $("#segFileEd");
      const esMri = $("#mriFileEs");
      const esSeg = $("#segFileEs");
      if (!edMri.files.length) { toast("Upload an ED MRI file", "error"); return; }
      formData.append("mri", edMri.files[0]);
      if (edSeg.files.length) formData.append("segmentation", edSeg.files[0]);
      if (esMri.files.length) formData.append("mri_es", esMri.files[0]);
      if (esSeg.files.length) formData.append("segmentation_es", esSeg.files[0]);
    } else {
      if (!mriFileInput.files.length) return;
      formData.append("mri", mriFileInput.files[0]);
      if (segFileInput.files.length) formData.append("segmentation", segFileInput.files[0]);
    }

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
        `/api/case/${state.caseId}/slice?z=${state.currentZ}&frame=${state.currentFrame}&phase=${state.phase}`
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

      const info = phaseInfo(state.phase);
      sliceCaption.textContent =
        `${state.phase.toUpperCase()}  ·  Slice ${data.z} / ${(info.slices || 1) - 1}  ·  Frame ${data.frame}  ·  ${data.width} × ${data.height}`;
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
    const info = phaseInfo(state.phase);
    const nz = Math.max(0, Math.min((info.slices || 1) - 1, state.currentZ + d));
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
        phase: state.phase,
        width: state.sliceWidth,
        height: state.sliceHeight,
        mask: u8ToB64(state.drawnData),
      });
      state.dirty = false;
      toast(`Mask saved for ${state.phase.toUpperCase()} slice ${state.currentZ}`, "success");
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
          phase: state.phase,
          width: state.sliceWidth,
          height: state.sliceHeight,
          mask: u8ToB64(state.drawnData),
        });
        state.dirty = false;
      } catch (_) {}
    }

    showLoading("Starting inference…");
    try {
      if (state.mode === "wt" && state.hasEs) {
        const edFrame = state.phase === "ed" ? state.currentFrame : 0;
        const esFrame = state.phase === "es" ? state.currentFrame : 0;
        showLoading("Queuing paired ED+ES inference…");
        const { jobId } = await apiPost(`/api/case/${state.caseId}/infer-paired/start`, {
          edFrame,
          esFrame,
        });
        const paired = await pollJob(jobId, "Running paired ED+ES inference (may take a few minutes)");

        await Promise.all([
          hydrateSdfResult(paired.ed, "ED SDF"),
          hydrateSdfResult(paired.es, "ES SDF"),
        ]);

        paired.ed.sliceContours = await apiGet(`/api/case/${state.caseId}/slice-contours?phase=ed&frame=${edFrame}`);
        paired.es.sliceContours = await apiGet(`/api/case/${state.caseId}/slice-contours?phase=es&frame=${esFrame}`);
        state.results = paired;
        state.resultsEd = paired.ed;
        state.resultsEs = paired.es;
        state.resultsDifference = paired.difference;
        state.resultsPhase = "difference";
        setResultsToggleVisible(true);
        setResultsPhaseButtons("difference");
        showResultsForPhase();
        inferenceSource.textContent = paired.source || "Complete";
      } else {
        showLoading(`Queuing ${state.phase.toUpperCase()} inference…`);
        const { jobId } = await apiPost(`/api/case/${state.caseId}/infer/start`, {
          frame: state.currentFrame,
          phase: state.phase,
        });
        const result = await pollJob(jobId, `Running ${state.phase.toUpperCase()} inference`);
        await hydrateSdfResult(result, `${state.phase.toUpperCase()} SDF`);
        state.sliceContours = await apiGet(
          `/api/case/${state.caseId}/slice-contours?phase=${state.phase}&frame=${state.currentFrame}`
        );
        result.sliceContours = state.sliceContours;
        state.results = { ed: result, es: null, difference: null };
        state.resultsEd = result;
        state.resultsEs = null;
        state.resultsDifference = null;
        state.resultsPhase = "ed";
        setResultsToggleVisible(false);
        displayMetrics(result, { mode: "wallThickness" });
        if (result.meshes) build3DScene(result.meshes, result.sliceContours, { meshMode: "wallThickness" });
        inferenceSource.textContent = result.source || "Complete";
      }

      navigateTo("results");
      showResultsForPhase();
      toast("Inference complete", "success");
    } catch (err) {
      toast(`Inference failed: ${err.message}`, "error");
    } finally {
      hideLoading();
    }
  });

  function setResultsToggleVisible(visible) {
    const resultsToggle = $("#resultsPhaseToggle");
    if (resultsToggle) resultsToggle.style.display = visible ? "flex" : "none";
    const summary = $("#comparisonSummary");
    if (summary) summary.style.display = visible ? "grid" : "none";
  }

  function setResultsPhaseButtons(phase) {
    $$(".phase-btn[data-results-phase]").forEach((b) => b.classList.toggle("active", b.dataset.resultsPhase === phase));
  }

  function formatSigned(value, digits = 1) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    const n = Number(value);
    return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
  }

  function updateComparisonSummary(difference) {
    const metrics = difference && difference.metrics ? difference.metrics : {};
    $$('[data-comparison-key]').forEach((el) => {
      const key = el.dataset.comparisonKey;
      const value = metrics[key];
      el.textContent = key === "meanDeltaWallThicknessMm" ? formatSigned(value, 1) : (value != null ? Number(value).toFixed(1) : "—");
    });
  }

  function setWallLabels(mode) {
    const difference = mode === "difference";
    const title = $("#wallSectionTitle");
    const bullseyeTitle = $("#bullseyeTitle");
    const meshTitle = $("#meshTitle");
    const minLabel = $("#bullseyeMinLabel");
    const maxLabel = $("#bullseyeMaxLabel");
    const gradient = $("#bullseyeGradient");
    if (title) title.textContent = difference ? "Wall Thickening — AHA 17-Segment" : "Wall Thickness — AHA 17-Segment";
    if (bullseyeTitle) bullseyeTitle.textContent = difference ? "Difference Bullseye" : "Bullseye Plot";
    if (meshTitle) meshTitle.textContent = difference ? "LV ED→ES Shape Change" : "LV 3D Reconstruction";
    if (minLabel) minLabel.textContent = difference ? "-5 mm" : "0 mm";
    if (maxLabel) maxLabel.textContent = difference ? "+5 mm" : "15 mm";
    if (gradient) gradient.classList.toggle("difference", difference);
  }

  // ─── Display metrics ───
  function displayMetrics(result, opts = {}) {
    const mode = opts.mode || (result && result.kind === "difference" ? "difference" : "wallThickness");
    setWallLabels(mode);
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

    const aha17 = mode === "difference" ? (result.aha17Delta || []) : (result.aha17 || []);
    if (aha17.length === 17) drawBullseye(aha17, { mode });

    const segments = aha17.length === 17 ? aha17 : (result.regionalThickness || []);
    const list = $("#regionalList");
    list.innerHTML = "";
    segments.forEach((r) => {
      const row = document.createElement("div");
      row.className = "regional-row";
      row.dataset.status = r.status || "unavailable";
      const value = mode === "difference" ? r.deltaMm : r.meanMm;
      const pct = mode === "difference" ? Math.min(100, (Math.abs(value || 0) / 6) * 100) : Math.min(100, ((value || 0) / 18) * 100);
      const idLabel = r.id ? `<span class="regional-id">${r.id}</span>` : "";
      if (mode === "difference") {
        row.innerHTML = `
          ${idLabel}
          <span class="regional-name">${r.name}</span>
          <div class="regional-bar difference"><span style="width:${pct}%"></span></div>
          <span class="regional-value">${formatSigned(value, 1)} mm</span>
        `;
        row.title = `ED ${r.edMeanMm ?? "—"} mm · ES ${r.esMeanMm ?? "—"} mm · ${r.relativeThickeningPct ?? "—"}%`;
      } else {
        row.innerHTML = `
          ${idLabel}
          <span class="regional-name">${r.name}</span>
          <div class="regional-bar"><span style="width:${pct}%"></span></div>
          <span class="regional-value">${value != null ? Number(value).toFixed(1) + " mm" : "—"}</span>
        `;
      }
      list.appendChild(row);
    });

    meshStatus.textContent = result.valueLabel
      ? `${result.valueLabel} · ${result.meshMethod || "ready"}`
      : (result.meshMethod ? result.meshMethod : "ready");
    if (mode === "difference") updateComparisonSummary(result);
  }

  // ─── AHA 17-Segment Bullseye ───
  function drawBullseye(aha17, opts = {}) {
    const mode = opts.mode || "wallThickness";
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
      drawArc(ctx, cx, cy, r1, r0, startAngle, endAngle, aha17[i], mode);
    }

    // Mid ring: segments 7-12, each 60°
    for (let i = 0; i < 6; i++) {
      const startAngle = -Math.PI / 2 + i * (Math.PI / 3);
      const endAngle = startAngle + Math.PI / 3;
      drawArc(ctx, cx, cy, r2, r1, startAngle, endAngle, aha17[6 + i], mode);
    }

    // Apical ring: segments 13-16, each 90°
    for (let i = 0; i < 4; i++) {
      const startAngle = -Math.PI / 2 + i * (Math.PI / 2);
      const endAngle = startAngle + Math.PI / 2;
      drawArc(ctx, cx, cy, r3, r2, startAngle, endAngle, aha17[12 + i], mode);
    }

    // Apex: segment 17 (center circle)
    const apexVal = mode === "difference" ? aha17[16].deltaMm : aha17[16].meanMm;
    ctx.beginPath();
    ctx.arc(cx, cy, r3, 0, 2 * Math.PI);
    ctx.fillStyle = apexVal != null ? valueColor(apexVal, mode) : "#e8ecf1";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (apexVal != null) {
      ctx.fillStyle = apexVal > 8 ? "#fff" : "#1a1f36";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(mode === "difference" ? formatSigned(apexVal, 1) : apexVal.toFixed(1), cx, cy);
    }
  }

  function drawArc(ctx, cx, cy, rInner, rOuter, startAngle, endAngle, segment, mode = "wallThickness") {
    const val = segment ? (mode === "difference" ? segment.deltaMm : segment.meanMm) : null;
    ctx.beginPath();
    ctx.arc(cx, cy, rOuter, startAngle, endAngle);
    ctx.arc(cx, cy, rInner, endAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = val != null ? valueColor(val, mode) : "#e8ecf1";
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
      ctx.fillText(mode === "difference" ? formatSigned(val, 1) : val.toFixed(1), tx, ty);
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

  function differenceColor(mm) {
    const t = Math.max(0, Math.min(1, (mm + 5) / 10));
    const stops = [
      [0, [37, 99, 235]],
      [0.5, [248, 250, 252]],
      [1, [220, 38, 38]],
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
    return "rgb(220,38,38)";
  }

  function valueColor(value, mode) {
    return mode === "difference" ? differenceColor(value) : thicknessColor(value);
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

    // Lighting — clean medical look with strong coverage for solid surfaces
    scene.add(new THREE.AmbientLight(0x334466, 0.7));
    scene.add(new THREE.HemisphereLight(0x6688cc, 0x332244, 0.5));

    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(50, 80, 60);
    key.castShadow = true;
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x6699cc, 0.6);
    fill.position.set(-40, 30, -30);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0x00ccff, 0.4);
    rim.position.set(0, -20, -60);
    scene.add(rim);

    const bottom = new THREE.DirectionalLight(0x334455, 0.3);
    bottom.position.set(0, -50, 0);
    scene.add(bottom);

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

  function build3DScene(meshes, contourData, opts = {}) {
    // Cleanup dual viewports if they exist
    if (dualScenes.ed || dualScenes.es) cleanupDualViewports();

    if (!scene) initThree();
    if (!scene) return;
    threeFallback.style.display = "none";

    // Clear all cardiac objects
    clearCardiacObjects();

    // First pass: compute mesh bounding box for centering everything
    const allVerts = [];
    if (meshes.endo && meshes.endo.vertices.length > 0) allVerts.push(meshes.endo.vertices);
    if (meshes.epi && meshes.epi.vertices.length > 0) allVerts.push(meshes.epi.vertices);

    const meshCenter = new THREE.Vector3();
    if (allVerts.length > 0) {
      let mx = 0, my = 0, mz = 0, n = 0;
      allVerts.forEach((v) => {
        for (let i = 0; i < v.length; i += 3) {
          mx += v[i]; my += v[i+1]; mz += v[i+2]; n++;
        }
      });
      meshCenter.set(mx/n, my/n, mz/n);
    }

    let count = 0;

    // Endocardium — SOLID opaque surface, centered
    if (meshes.endo && meshes.endo.vertices.length > 0) {
      const m = buildMesh(meshes.endo, {
        color: 0xe74c5e, emissive: 0x330011, emissiveIntensity: 0.3,
        opacity: 1.0, transparent: false, roughness: 0.25, metalness: 0.15,
        values: meshes.endo.values,
        cmMin: opts.cmMin ?? (opts.meshMode === "difference" ? -5 : 2),
        cmMax: opts.cmMax ?? (opts.meshMode === "difference" ? 5 : 14),
        colorMode: opts.meshMode || "wallThickness",
        side: THREE.DoubleSide,
      }, meshCenter);
      if (m) { scene.add(m); count++; }
    }

    // Epicardium — semi-transparent shell, centered
    if (opts.showEpi !== false && meshes.epi && meshes.epi.vertices.length > 0) {
      const m = buildMesh(meshes.epi, {
        color: 0x3b82f6, emissive: 0x0a1530, emissiveIntensity: 0.15,
        opacity: 0.18, transparent: true, roughness: 0.5, metalness: 0.05,
        side: THREE.DoubleSide, depthWrite: false,
      }, meshCenter);
      if (m) { scene.add(m); count++; }
    }

    if (count > 0) {
      camera.position.set(60, 50, 100);
      if (controls) { controls.target.set(0, 0, 0); controls.update(); }
    }

    // Add slice contours centered with the same origin as the mesh
    if (contourData && contourData.slices && contourData.slices.length > 0) {
      addSliceContours(contourData, meshCenter);
    }
  }

  function clearCardiacObjects() {
    const toRemove = [];
    scene.traverse((c) => {
      if (c.userData.cardiac) toRemove.push(c);
    });
    toRemove.forEach((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (obj.material.map) obj.material.map.dispose();
        obj.material.dispose();
      }
      scene.remove(obj);
    });
  }

  function addSliceContours(contourData, worldCenter) {
    if (!scene || !contourData || !contourData.slices) return;
    const COLORS = { myo: 0x22c55e, lv: 0xe74c5e };
    const cx = worldCenter ? worldCenter.x : 0;
    const cy = worldCenter ? worldCenter.y : 0;
    const cz = worldCenter ? worldCenter.z : 0;

    contourData.slices.forEach((sliceInfo) => {
      sliceInfo.contours.forEach((contour) => {
        const pts = contour.points;
        if (!pts || pts.length < 4) return;

        // Points are already in world-mm space [x,y,z], center them
        const points = pts.map((p) => new THREE.Vector3(
          p[0] - cx, p[1] - cy, p[2] - cz
        ));

        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const color = COLORS[contour.label] || 0xffffff;
        const mat = new THREE.LineBasicMaterial({
          color: color,
          opacity: 0.6,
          transparent: true,
        });
        const line = new THREE.LineLoop(geo, mat);
        line.userData.cardiac = true;
        scene.add(line);
      });
    });
  }

  function buildDual3DScene(edMeshes, esMeshes) {
    // For "both" mode, use two separate side-by-side viewports
    buildDualViewports(edMeshes, esMeshes);
  }

  // ─── Dual viewport system ───
  let dualScenes = { ed: null, es: null };

  function getMeshCenter(meshes) {
    const center = new THREE.Vector3();
    let n = 0;
    if (meshes && meshes.endo && meshes.endo.vertices.length > 0) {
      const v = meshes.endo.vertices;
      for (let i = 0; i < v.length; i += 3) { center.x += v[i]; center.y += v[i+1]; center.z += v[i+2]; n++; }
    }
    if (meshes && meshes.epi && meshes.epi.vertices.length > 0) {
      const v = meshes.epi.vertices;
      for (let i = 0; i < v.length; i += 3) { center.x += v[i]; center.y += v[i+1]; center.z += v[i+2]; n++; }
    }
    if (n > 0) center.divideScalar(n);
    return center;
  }

  function buildDualViewports(edMeshes, esMeshes) {
    const container = $("#heartViewport");
    if (!container || typeof THREE === "undefined") return;
    threeFallback.style.display = "none";

    // Remove existing canvases
    container.querySelectorAll("canvas").forEach((c) => c.remove());
    container.querySelectorAll(".dual-viewport-wrapper").forEach((c) => c.remove());

    // Stop existing animation loop
    if (animId) cancelAnimationFrame(animId);

    // Cleanup old single scene
    if (scene) {
      clearCardiacObjects();
    }

    // Create wrapper
    const wrapper = document.createElement("div");
    wrapper.className = "dual-viewport-wrapper";
    wrapper.style.cssText = "display:flex;width:100%;height:100%;gap:2px;";
    container.appendChild(wrapper);

    // Create ED viewport
    const edDiv = document.createElement("div");
    edDiv.style.cssText = "flex:1;position:relative;min-width:0;";
    const edLabel = document.createElement("div");
    edLabel.className = "viewport-label";
    edLabel.textContent = "ED (Diastole)";
    edDiv.appendChild(edLabel);
    wrapper.appendChild(edDiv);

    // Create ES viewport
    const esDiv = document.createElement("div");
    esDiv.style.cssText = "flex:1;position:relative;min-width:0;";
    const esLabel = document.createElement("div");
    esLabel.className = "viewport-label viewport-label-es";
    esLabel.textContent = "ES (Systole)";
    esDiv.appendChild(esLabel);
    wrapper.appendChild(esDiv);

    // Build two separate scenes
    const edScene = createSubScene(edDiv, edMeshes, state.resultsEd, 0xe74c5e, 0x3b82f6);
    const esScene = createSubScene(esDiv, esMeshes, state.resultsEs, 0xf59e0b, 0x8b5cf6);

    dualScenes = { ed: edScene, es: esScene };

    // Shared animation loop
    function dualLoop() {
      animId = requestAnimationFrame(dualLoop);
      if (edScene) {
        if (edScene.controls) edScene.controls.update();
        edScene.renderer.render(edScene.scene, edScene.camera);
      }
      if (esScene) {
        if (esScene.controls) esScene.controls.update();
        esScene.renderer.render(esScene.scene, esScene.camera);
      }
    }
    dualLoop();
  }

  function createSubScene(container, meshes, resultData, endoColor, epiColor) {
    if (!meshes) return null;

    const scn = new THREE.Scene();
    scn.background = new THREE.Color(0x0d1117);

    const w = container.clientWidth || 400;
    const h = container.clientHeight || 400;

    const cam = new THREE.PerspectiveCamera(45, w / Math.max(h, 1), 0.1, 2000);
    cam.position.set(60, 50, 100);

    const ren = new THREE.WebGLRenderer({ antialias: true });
    ren.setSize(w, h);
    ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    ren.shadowMap.enabled = true;
    ren.toneMapping = THREE.ACESFilmicToneMapping;
    ren.toneMappingExposure = 1.2;
    container.appendChild(ren.domElement);

    let ctrl = null;
    if (THREE.OrbitControls) {
      ctrl = new THREE.OrbitControls(cam, ren.domElement);
      ctrl.enableDamping = true;
      ctrl.dampingFactor = 0.08;
      ctrl.rotateSpeed = 0.6;
    }

    // Lighting
    scn.add(new THREE.AmbientLight(0x334466, 0.7));
    scn.add(new THREE.HemisphereLight(0x6688cc, 0x332244, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(50, 80, 60);
    scn.add(key);
    const fill = new THREE.DirectionalLight(0x6699cc, 0.6);
    fill.position.set(-40, 30, -30);
    scn.add(fill);

    const grid = new THREE.GridHelper(200, 30, 0x1a2233, 0x111822);
    grid.position.y = -40;
    scn.add(grid);

    // Center
    const center = getMeshCenter(meshes);

    // Endo
    if (meshes.endo && meshes.endo.vertices.length > 0) {
      const m = buildMesh(meshes.endo, {
        color: endoColor, emissive: 0x330011, emissiveIntensity: 0.3,
        opacity: 1.0, transparent: false, roughness: 0.25, metalness: 0.15,
        values: meshes.endo.values, cmMin: 2, cmMax: 14, colorMode: "wallThickness", side: THREE.DoubleSide,
      }, center);
      if (m) scn.add(m);
    }

    // Epi
    if (meshes.epi && meshes.epi.vertices.length > 0) {
      const m = buildMesh(meshes.epi, {
        color: epiColor, emissive: 0x0a1530, emissiveIntensity: 0.15,
        opacity: 0.18, transparent: true, roughness: 0.5, metalness: 0.05,
        side: THREE.DoubleSide, depthWrite: false,
      }, center);
      if (m) scn.add(m);
    }

    // Slice contours
    if (resultData && resultData.sliceContours && resultData.sliceContours.slices) {
      addSliceContoursToScene(scn, resultData.sliceContours, center);
    }

    if (ctrl) { ctrl.target.set(0, 0, 0); ctrl.update(); }

    // Resize handling
    const resize = () => {
      const nw = container.clientWidth, nh = container.clientHeight;
      if (!nw || !nh) return;
      cam.aspect = nw / nh;
      cam.updateProjectionMatrix();
      ren.setSize(nw, nh);
    };
    new ResizeObserver(resize).observe(container);

    return { scene: scn, camera: cam, renderer: ren, controls: ctrl };
  }

  function addSliceContoursToScene(targetScene, contourData, worldCenter) {
    if (!contourData || !contourData.slices) return;
    const COLORS = { myo: 0x22c55e, lv: 0xe74c5e };
    const cx = worldCenter ? worldCenter.x : 0;
    const cy = worldCenter ? worldCenter.y : 0;
    const cz = worldCenter ? worldCenter.z : 0;

    contourData.slices.forEach((sliceInfo) => {
      sliceInfo.contours.forEach((contour) => {
        const pts = contour.points;
        if (!pts || pts.length < 4) return;
        const points = pts.map((p) => new THREE.Vector3(p[0] - cx, p[1] - cy, p[2] - cz));
        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
          color: COLORS[contour.label] || 0xffffff,
          opacity: 0.6, transparent: true,
        });
        const line = new THREE.LineLoop(geo, mat);
        line.userData.cardiac = true;
        targetScene.add(line);
      });
    });
  }

  function addLabel3D(text, x, y, z) {
    const canvas2 = document.createElement("canvas");
    canvas2.width = 256; canvas2.height = 64;
    const c = canvas2.getContext("2d");
    c.clearRect(0, 0, 256, 64);
    c.font = "bold 22px Inter, sans-serif";
    c.fillStyle = "#ffffff";
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillText(text, 128, 32);
    const tex = new THREE.CanvasTexture(canvas2);
    const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(x, y, z);
    sprite.scale.set(40, 10, 1);
    sprite.userData.cardiac = true;
    scene.add(sprite);
  }

  function buildMesh(data, opts, worldCenter) {
    if (!data.vertices || data.vertices.length < 9) return null;
    const srcVerts = data.vertices;
    const verts = new Float32Array(srcVerts.length);
    const cx = worldCenter ? worldCenter.x : 0;
    const cy = worldCenter ? worldCenter.y : 0;
    const cz = worldCenter ? worldCenter.z : 0;
    // Subtract world center so mesh sits at origin
    for (let i = 0; i < srcVerts.length; i += 3) {
      verts[i]     = srcVerts[i]     - cx;
      verts[i + 1] = srcVerts[i + 1] - cy;
      verts[i + 2] = srcVerts[i + 2] - cz;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));

    // If we have face indices, use them for a proper solid mesh
    if (data.faces && data.faces.length >= 3) {
      const idx = new Uint32Array(data.faces);
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
    } else {
      // No faces — generate a convex hull-like solid from vertices using Delaunay-ish triangulation
      // Fallback: create a point-based surface estimation
      const n = verts.length / 3;
      if (n < 4) return null;
      // Use indexed faces if we can estimate them, otherwise render as solid points
      // For now, build faces from sequential vertex triplets (assumes mesh-ordered vertices)
      const faces = [];
      for (let i = 0; i < n - 2; i += 1) {
        faces.push(i, i + 1, i + 2);
      }
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
    }

    geo.computeVertexNormals();
    // Smooth the normals for a more solid appearance
    geo.computeBoundingSphere();

    let mat;
    if (opts.values && opts.values.length > 0) {
      const n = verts.length / 3;
      const cols = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const v = i < opts.values.length ? opts.values[i] : 0;
        const t = Math.max(0, Math.min(1, (v - (opts.cmMin || 3)) / ((opts.cmMax || 15) - (opts.cmMin || 3))));
        const c = opts.colorMode === "difference" ? diffColor(t) : wtColor(t);
        cols[i * 3] = c[0]; cols[i * 3 + 1] = c[1]; cols[i * 3 + 2] = c[2];
      }
      geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      mat = new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        emissive: new THREE.Color(opts.emissive || 0),
        emissiveIntensity: opts.emissiveIntensity || 0,
        opacity: opts.opacity ?? 1,
        transparent: !!opts.transparent,
        roughness: opts.roughness ?? 0.3,
        metalness: opts.metalness ?? 0.1,
        side: opts.side || THREE.DoubleSide,
        depthWrite: opts.depthWrite !== false,
        clearcoat: 0.2,
        clearcoatRoughness: 0.25,
        flatShading: false,
      });
    } else {
      mat = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(opts.color || 0xcccccc),
        emissive: new THREE.Color(opts.emissive || 0),
        emissiveIntensity: opts.emissiveIntensity || 0,
        opacity: opts.opacity ?? 1,
        transparent: !!opts.transparent,
        roughness: opts.roughness ?? 0.3,
        metalness: opts.metalness ?? 0.1,
        side: opts.side || THREE.DoubleSide,
        depthWrite: opts.depthWrite !== false,
        clearcoat: 0.2,
        clearcoatRoughness: 0.25,
        flatShading: false,
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

  function diffColor(t) {
    const stops = [
      [0, [0.15, 0.39, 0.92]],
      [0.5, [0.97, 0.98, 0.99]],
      [1, [0.86, 0.15, 0.15]],
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

  // ─── Phase toggle (viewer) ───
  $$(".phase-btn[data-phase]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setViewerPhase(btn.dataset.phase, true);
      fetchSlice();
      fetchSliceContours();
    });
  });

  // ─── Phase toggle (results) ───
  $$(".phase-btn[data-results-phase]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".phase-btn[data-results-phase]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.resultsPhase = btn.dataset.resultsPhase;
      showResultsForPhase();
    });
  });

  function showResultsForPhase() {
    const phase = state.resultsPhase;

    cleanupDualViewports();

    if (phase === "difference") {
      const result = state.resultsDifference;
      if (result) {
        displayMetrics(result, { mode: "difference" });
        const scale = result.colorScale || { min: -5, max: 5 };
        const minLabel = $("#bullseyeMinLabel");
        const maxLabel = $("#bullseyeMaxLabel");
        if (minLabel) minLabel.textContent = `${Number(scale.min).toFixed(1)} mm`;
        if (maxLabel) maxLabel.textContent = `${Number(scale.max).toFixed(1)} mm`;
        if (result.meshes) build3DScene(result.meshes, null, {
          meshMode: "difference",
          cmMin: scale.min,
          cmMax: scale.max,
          showEpi: false,
        });
      }
    } else {
      const result = phase === "es" ? state.resultsEs : state.resultsEd;
      if (result) {
        displayMetrics(result, { mode: "wallThickness" });
        if (result.meshes) build3DScene(result.meshes, result.sliceContours, { meshMode: "wallThickness" });
      }
    }
  }

  function cleanupDualViewports() {
    // Dispose dual renderers and remove their DOM elements
    if (dualScenes.ed) {
      dualScenes.ed.renderer.dispose();
      if (dualScenes.ed.controls) dualScenes.ed.controls.dispose();
    }
    if (dualScenes.es) {
      dualScenes.es.renderer.dispose();
      if (dualScenes.es.controls) dualScenes.es.controls.dispose();
    }
    dualScenes = { ed: null, es: null };

    const container = $("#heartViewport");
    container.querySelectorAll(".dual-viewport-wrapper").forEach((w) => w.remove());
    container.querySelectorAll("canvas").forEach((c) => c.remove());

    // Re-init the single scene for single-phase views
    scene = null;
    renderer = null;
    controls = null;
    if (animId) cancelAnimationFrame(animId);
    animId = null;
  }

  // ─── Fetch slice contours for 3D visualization ───
  async function fetchSliceContours() {
    if (!state.caseId) return;
    try {
      const data = await apiGet(`/api/case/${state.caseId}/slice-contours?phase=${state.phase}&frame=${state.currentFrame}`);
      state.sliceContours = data;
    } catch (_) {
      state.sliceContours = null;
    }
  }

  // ─── Init 3D (lazy) ───
  // Don't init until results page is first shown — saves GPU
  const resultsObserver = new MutationObserver(() => {
    if (pages.results.classList.contains("active") && !scene) {
      initThree();
    }
  });
  resultsObserver.observe(pages.results, { attributes: true, attributeFilter: ["class"] });

})();
