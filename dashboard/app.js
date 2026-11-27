(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const channels = [
    { name: "Temperature", code: "t2m", unit: "K" },
    { name: "Precipitation", code: "tp", unit: "mm/day" },
    { name: "Eastward wind", code: "u10", unit: "m/s" },
    { name: "Northward wind", code: "v10", unit: "m/s" },
    { name: "Pressure", code: "msl", unit: "Pa" },
    { name: "Water vapour", code: "tcwv", unit: "kg/m²" }
  ];
  const state = {
    input: null,
    fileName: "",
    baseline: null,
    shock: null,
    ready: false,
    busy: "",
    selection: null
  };
  const map = $("map");
  const ctx = map.getContext("2d");
  const canvasWidth = map.width;
  const canvasHeight = map.height;
  const overviewMap = $("overview-map");
  const overviewCtx = overviewMap.getContext("2d");
  const overviewWidth = overviewMap.width;
  const overviewHeight = overviewMap.height;
  // Schematic coastline in the fixed 65.5-100.5E, 6.5-37.5N model domain.
  const indiaOutline = [
    [75.5, 37.3], [76.2, 36.3], [77.2, 35.5], [78.0, 34.6], [78.8, 33.6],
    [79.1, 32.5], [78.7, 31.4], [78.5, 30.5], [79.0, 29.7], [79.8, 29.2],
    [80.8, 28.9], [81.7, 28.5], [82.6, 27.8], [83.6, 27.7], [84.5, 27.9],
    [85.5, 27.7], [86.5, 27.5], [87.5, 27.2], [88.3, 26.9], [89.0, 26.5],
    [89.7, 26.1], [90.5, 26.0], [91.3, 25.9], [92.0, 26.4], [92.8, 27.0],
    [93.7, 27.4], [94.6, 27.8], [95.5, 28.1], [96.4, 28.3], [97.4, 28.1],
    [97.1, 27.2], [96.3, 26.4], [95.3, 25.7], [94.7, 24.8], [94.2, 23.6],
    [93.2, 22.1], [92.4, 21.8], [92.4, 23.3], [91.7, 24.0], [90.8, 24.1],
    [90.0, 23.8], [89.6, 22.8], [89.0, 22.0], [88.2, 21.8], [87.6, 21.8],
    [86.8, 20.8], [85.8, 19.8], [84.6, 19.0], [83.7, 17.5], [82.5, 16.8],
    [81.5, 15.8], [80.5, 14.5], [80.2, 12.9], [80.1, 11.4], [79.5, 10.3],
    [78.6, 8.5], [77.5, 8.1], [76.5, 8.5], [75.7, 9.4], [75.1, 10.6],
    [74.6, 12.0], [74.1, 13.3], [73.7, 14.8], [73.3, 16.5], [72.9, 18.3],
    [72.6, 19.8], [72.2, 20.8], [71.3, 21.8], [70.2, 22.5], [69.2, 22.8],
    [68.7, 23.2], [68.1, 23.8], [68.1, 24.6], [68.5, 25.5], [69.2, 26.3],
    [70.1, 27.1], [71.1, 27.8], [72.0, 28.5], [72.9, 29.4], [73.8, 30.4],
    [74.8, 31.5], [75.3, 32.5], [74.8, 33.6], [74.5, 34.8], [74.8, 35.8],
    [75.5, 37.3]
  ];
  let dragStart = null;

  function setMessage(text, kind) {
    const message = $("message");
    message.textContent = text;
    message.classList.toggle("error", kind === "error");
    message.classList.toggle("success", kind === "success");
  }

  function setApiStatus(ready, text) {
    state.ready = ready;
    $("status-dot").classList.toggle("ready", ready);
    $("status-text").textContent = text;
  }

  function detailMessage(data, fallback) {
    const detail = data && data.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail.reason === "string") return detail.reason;
    return fallback;
  }

  function assertShapeAndFinite(value, shape, path) {
    if (!Array.isArray(value) || value.length !== shape[0]) {
      throw new Error(path + " must have shape [" + shape.join(", ") + "]");
    }

    if (shape.length === 1) {
      for (let i = 0; i < value.length; i++) {
        if (typeof value[i] !== "number" || !Number.isFinite(value[i])) {
          throw new Error(path + "[" + i + "] must be a finite number");
        }
      }
      return;
    }

    for (let i = 0; i < value.length; i++) {
      assertShapeAndFinite(value[i], shape.slice(1), path + "[" + i + "]");
    }
  }

  function validateInput(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error("The JSON file must contain an object.");
    }
    assertShapeAndFinite(data.init_states, [3, 6, 32, 36], "init_states");
    if (!Array.isArray(data.doy_sincos) || data.doy_sincos.length !== 2 ||
        !data.doy_sincos.every(Number.isFinite)) {
      throw new Error("doy_sincos must contain two finite numbers: [sin, cos].");
    }
    if (data.perturbation !== undefined && data.perturbation !== null) {
      throw new Error("Upload only init_states and doy_sincos. Shocks are set in this dashboard.");
    }
    return { init_states: data.init_states, doy_sincos: data.doy_sincos };
  }

  async function loadWindow(file) {
    if (!file || state.busy) return;
    state.input = null;
    state.fileName = "";
    state.baseline = null;
    state.shock = null;
    $("baseline-time").textContent = "";
    $("shock-time").textContent = "";

    try {
      const data = JSON.parse(await file.text());
      state.input = validateInput(data);
      state.fileName = file.name;
      $("file-title").textContent = file.name;
      $("file-meta").textContent = "Window loaded · ready for a baseline";
      $("file-drop-area").classList.add("has-file");
      $("day").value = "1";
      $("day").max = $("steps").value;
      $("view-mode").value = "baseline";
      setMessage("Window loaded. Run a baseline forecast to continue.", "success");
    } catch (error) {
      $("file-title").textContent = "Choose a JSON window";
      $("file-meta").textContent = "or drop the file here";
      $("file-drop-area").classList.remove("has-file");
      setMessage(error.message || "Could not read this JSON window.", "error");
    }

    syncControls();
    draw();
  }

  function syncControls() {
    const busy = Boolean(state.busy);
    $("window-file").disabled = busy;
    $("steps").disabled = busy;
    $("channel").disabled = busy;
    $("day").disabled = busy || !state.baseline;
    $("view-mode").disabled = busy;
    $("shock-preset").disabled = busy;
    $("run-baseline").disabled = busy || !state.input;
    $("run-shock").disabled = busy || !state.baseline;
    $("run-zero").disabled = busy || !state.baseline;

    const mode = $("view-mode");
    mode.options[1].disabled = !state.shock;
    mode.options[2].disabled = !state.shock;
    if ((mode.value === "shock" || mode.value === "difference") && !state.shock) {
      mode.value = "baseline";
    }

    $("run-baseline").textContent = state.busy === "baseline"
      ? "Running baseline…" : "Run baseline forecast";
    $("run-shock").textContent = state.busy === "shock"
      ? "Running counterfactual…" : "Run counterfactual";
    $("run-zero").textContent = state.busy === "zero"
      ? "Checking…" : "Verify zero shock";
  }

  async function checkHealth() {
    try {
      const response = await fetch("/health", { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== "ready") {
        throw new Error(detailMessage(data, "The model is not ready."));
      }
      setApiStatus(true, "Ready · " + data.model_name);
      if (!state.input) setMessage("Choose a real three-day JSON window to begin.");
    } catch (error) {
      setApiStatus(false, "Local model unavailable");
      if (!state.input) {
        setMessage("The local API is unavailable or the model is not ready. Start the FORKCAST server, then try again.", "error");
      }
    }
    syncControls();
  }

  function makeShock() {
    if (!state.input) throw new Error("Load a real weather window before running a scenario.");
    const preset = $("shock-preset").value;
    const perturbation = Array.from({ length: 6 }, () =>
      Array.from({ length: 32 }, () => Array(36).fill(0)));
    const lastDay = state.input.init_states[2];

    if (preset === "warm_ocean" || preset === "monsoon_stall") {
      const latStart = 8;
      const latEnd = 20;
      const lonStart = preset === "warm_ocean" ? 82 : 65;
      const lonEnd = preset === "warm_ocean" ? 94 : 75;
      for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 36; x++) {
          const lat = 6.5 + y;
          const lon = 65.5 + x;
          if (lat < latStart || lat > latEnd || lon < lonStart || lon > lonEnd) continue;
          if (preset === "warm_ocean") {
            perturbation[0][y][x] = 3;
            perturbation[5][y][x] = 0.15 * lastDay[5][y][x];
          } else {
            perturbation[2][y][x] = -0.4 * lastDay[2][y][x];
            perturbation[3][y][x] = -0.4 * lastDay[3][y][x];
          }
        }
      }
      return perturbation;
    }

    const channel = Number($("shock-channel").value);
    const delta = Number($("shock-delta").value);
    const latStart = Number($("lat-start").value);
    const latEnd = Number($("lat-end").value);
    const lonStart = Number($("lon-start").value);
    const lonEnd = Number($("lon-end").value);
    if (!Number.isInteger(channel) || channel < 0 || channel >= channels.length ||
        ![delta, latStart, latEnd, lonStart, lonEnd].every(Number.isFinite)) {
      throw new Error("Enter a finite change and valid grid coordinates.");
    }
    if (latStart > latEnd || lonStart > lonEnd) {
      throw new Error("Each start coordinate must be less than or equal to its end.");
    }
    if (latStart < 6.5 || latEnd > 37.5 || lonStart < 65.5 || lonEnd > 100.5) {
      throw new Error("Keep the custom area inside the India model grid.");
    }

    let touchedCells = 0;
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 36; x++) {
        const lat = 6.5 + y;
        const lon = 65.5 + x;
        if (lat >= latStart && lat <= latEnd && lon >= lonStart && lon <= lonEnd) {
          perturbation[channel][y][x] = delta;
          touchedCells++;
        }
      }
    }
    if (!touchedCells) {
      throw new Error("That custom area does not include any grid-cell centres.");
    }
    return perturbation;
  }

  async function callSim(perturbation, steps) {
    const response = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        init_states: state.input.init_states,
        doy_sincos: state.input.doy_sincos,
        steps: steps,
        perturbation: perturbation
      })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(detailMessage(data, "Simulation request failed (HTTP " + response.status + ")."));
    }
    assertShapeAndFinite(data.rollout, [steps, 6, 32, 36], "rollout");
    if (typeof data.inference_ms !== "number" || !Number.isFinite(data.inference_ms) || data.inference_ms < 0) {
      throw new Error("The API returned an invalid inference time.");
    }
    return { rollout: data.rollout, inferenceMs: data.inference_ms, steps: steps };
  }

  function valueAt(rollout, day, channel, y, x) {
    return rollout[day][channel][y][x];
  }

  function maxDifference(left, right, day, channel) {
    let maximum = 0;
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 36; x++) {
        maximum = Math.max(maximum, Math.abs(
          valueAt(left, day, channel, y, x) - valueAt(right, day, channel, y, x)
        ));
      }
    }
    return maximum;
  }

  function maxAllDifference(left, right) {
    let maximum = 0;
    for (let day = 0; day < left.length; day++) {
      for (let channel = 0; channel < channels.length; channel++) {
        maximum = Math.max(maximum, maxDifference(left, right, day, channel));
      }
    }
    return maximum;
  }

  function interpolate(start, end, amount) {
    return start.map((value, index) => Math.round(value + (end[index] - value) * amount));
  }

  function colorAt(value, low, high, difference) {
    const stops = difference
      ? [[40, 126, 163], [169, 197, 198], [238, 229, 212], [233, 196, 125], [189, 77, 53]]
      : [[23, 107, 155], [88, 167, 179], [229, 191, 81], [216, 84, 56]];
    const t = Math.max(0, Math.min(1, (value - low) / Math.max(high - low, 1e-12)));
    const scaled = t * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    return interpolate(stops[index], stops[index + 1], scaled - index);
  }

  function formatValue(value) {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(value) >= 100000 || (value !== 0 && Math.abs(value) < 0.001)) {
      return value.toExponential(2);
    }
    return value.toFixed(2);
  }

  function setMetric(id, value, unit) {
    $(id).textContent = Number.isFinite(value) ? formatValue(value) + " " + unit : "—";
  }

  function drawGridBase(target, width, height) {
    target.fillStyle = "#172020";
    target.fillRect(0, 0, width, height);
    target.strokeStyle = "rgba(240, 235, 220, .16)";
    target.lineWidth = Math.max(.6, width / canvasWidth);
    for (let x = 0; x <= 36; x += 6) {
      const px = x * width / 36;
      target.beginPath();
      target.moveTo(px, 0);
      target.lineTo(px, height);
      target.stroke();
    }
    for (let y = 0; y <= 32; y += 4) {
      const py = y * height / 32;
      target.beginPath();
      target.moveTo(0, py);
      target.lineTo(width, py);
      target.stroke();
    }
  }

  function drawIndiaOutline(target, width, height, fillLand) {
    target.beginPath();
    indiaOutline.forEach(([longitude, latitude], index) => {
      const x = (longitude - 65.5) / 35 * width;
      const y = (37.5 - latitude) / 31 * height;
      if (index === 0) target.moveTo(x, y);
      else target.lineTo(x, y);
    });
    target.closePath();
    target.save();
    if (fillLand) {
      target.fillStyle = "rgba(248, 243, 232, .12)";
      target.fill();
    }
    target.strokeStyle = "rgba(248, 243, 232, .88)";
    target.lineWidth = Math.max(1.2, width / canvasWidth * 2.2);
    target.shadowColor = "rgba(4, 11, 12, .5)";
    target.shadowBlur = Math.max(1, width / canvasWidth * 3);
    target.stroke();
    target.restore();
  }

  function draw() {
    drawGridBase(ctx, canvasWidth, canvasHeight);
    drawGridBase(overviewCtx, overviewWidth, overviewHeight);
    const mode = $("view-mode").value;
    const day = Math.max(0, Number($("day").value) - 1);
    const channel = Number($("channel").value);
    const source = mode === "shock" ? state.shock : state.baseline;
    const hasDifference = mode === "difference" && state.baseline && state.shock;
    const hasField = Boolean(source || hasDifference);
    drawIndiaOutline(ctx, canvasWidth, canvasHeight, !hasField);
    drawIndiaOutline(overviewCtx, overviewWidth, overviewHeight, !hasField);
    $("overview-status").textContent = state.baseline
      ? (mode === "difference" ? "DIFFERENCE" : mode === "shock" ? "COUNTERFACTUAL" : "BASELINE") +
        " / DAY " + (day + 1)
      : state.input ? "WINDOW READY" : "AWAITING WINDOW";

    $("map-empty").hidden = Boolean(source || hasDifference);
    if (!source && !hasDifference) {
      $("result-label").textContent = "Waiting for a baseline";
      $("legend-min").textContent = "Lower";
      $("legend-max").textContent = "Higher";
      $("legend-gradient").classList.remove("difference");
      $("metric-min").textContent = "—";
      $("metric-max").textContent = "—";
      $("metric-diff").textContent = "—";
      drawSelection(ctx, canvasWidth, canvasHeight);
      drawSelection(overviewCtx, overviewWidth, overviewHeight);
      return;
    }

    const rollout = mode === "difference" ? state.baseline.rollout : source.rollout;
    if (day >= rollout.length) {
      $("day").value = String(rollout.length);
      return draw();
    }

    const values = [];
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 36; x++) {
        values.push(mode === "difference"
          ? valueAt(state.shock.rollout, day, channel, y, x) - valueAt(state.baseline.rollout, day, channel, y, x)
          : valueAt(rollout, day, channel, y, x));
      }
    }

    let low = Math.min(...values);
    let high = Math.max(...values);
    if (mode === "difference") {
      const extent = Math.max(Math.abs(low), Math.abs(high), 1e-9);
      low = -extent;
      high = extent;
      $("legend-gradient").classList.add("difference");
    } else {
      if (low === high) { low -= 1; high += 1; }
      $("legend-gradient").classList.remove("difference");
    }

    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 36; x++) {
        const value = mode === "difference"
          ? valueAt(state.shock.rollout, day, channel, y, x) - valueAt(state.baseline.rollout, day, channel, y, x)
          : valueAt(rollout, day, channel, y, x);
        const rgb = colorAt(value, low, high, mode === "difference");
        const color = "rgb(" + rgb.join(",") + ")";
        [ [ctx, canvasWidth, canvasHeight], [overviewCtx, overviewWidth, overviewHeight] ]
          .forEach(([target, width, height]) => {
            target.fillStyle = color;
            target.fillRect(
              x * width / 36,
              (31 - y) * height / 32,
              Math.ceil(width / 36),
              Math.ceil(height / 32)
            );
          });
      }
    }

    drawGridLines(ctx, canvasWidth, canvasHeight);
    drawGridLines(overviewCtx, overviewWidth, overviewHeight);
    drawIndiaOutline(ctx, canvasWidth, canvasHeight, false);
    drawIndiaOutline(overviewCtx, overviewWidth, overviewHeight, false);
    drawSelection(ctx, canvasWidth, canvasHeight);
    drawSelection(overviewCtx, overviewWidth, overviewHeight);
    $("result-label").textContent =
      (mode === "difference" ? "Difference" : mode === "shock" ? "Counterfactual" : "Baseline") +
      " · Day " + (day + 1);
    $("legend-min").textContent = formatValue(low) + " " + channels[channel].unit;
    $("legend-max").textContent = formatValue(high) + " " + channels[channel].unit;
    setMetric("metric-min", low, channels[channel].unit);
    setMetric("metric-max", high, channels[channel].unit);
    $("metric-diff").textContent = state.shock
      ? formatValue(maxDifference(state.baseline.rollout, state.shock.rollout, day, channel)) + " " + channels[channel].unit
      : "—";
  }

  function drawGridLines(target, width, height) {
    target.strokeStyle = "rgba(246, 241, 231, .28)";
    target.lineWidth = Math.max(.6, width / canvasWidth);
    for (let x = 0; x <= 36; x += 6) {
      const px = x * width / 36;
      target.beginPath();
      target.moveTo(px, 0);
      target.lineTo(px, height);
      target.stroke();
    }
    for (let y = 0; y <= 32; y += 4) {
      const py = y * height / 32;
      target.beginPath();
      target.moveTo(0, py);
      target.lineTo(width, py);
      target.stroke();
    }
  }

  function drawSelection(target, width, height) {
    if (!state.selection) return;
    const cellWidth = width / 36;
    const cellHeight = height / 32;
    const minX = Math.min(state.selection.x0, state.selection.x1);
    const maxX = Math.max(state.selection.x0, state.selection.x1);
    const minY = Math.min(state.selection.y0, state.selection.y1);
    const maxY = Math.max(state.selection.y0, state.selection.y1);
    target.strokeStyle = "#f4c655";
    target.lineWidth = Math.max(1.5, width / canvasWidth * 3);
    target.strokeRect(
      minX * cellWidth,
      (31 - maxY) * cellHeight,
      (maxX - minX + 1) * cellWidth,
      (maxY - minY + 1) * cellHeight
    );
  }

  function updateScenarioDescription() {
    const preset = $("shock-preset").value;
    const custom = preset === "custom";
    $("custom-controls").hidden = !custom;
    if (custom) $("custom-controls").open = true;
    else $("custom-controls").open = false;

    const descriptions = {
      warm_ocean: "Bay of Bengal · 8°–20°N, 82°–94°E · +3 K temperature and +15% water vapour.",
      monsoon_stall: "Arabian Sea · 8°–20°N, 65°–75°E · −40% eastward and northward winds.",
      custom: "Choose one variable and a region. Or drag across the forecast grid to set the area."
    };
    $("shock-description").textContent = descriptions[preset];
  }

  function invalidateShock() {
    if (state.shock) {
      state.shock = null;
      $("shock-time").textContent = "";
      $("view-mode").value = "baseline";
      syncControls();
      draw();
    }
  }

  async function runBaseline() {
    if (!state.input) {
      setMessage("Choose a valid three-day JSON window first.", "error");
      return;
    }
    const steps = Number($("steps").value);
    if (!Number.isInteger(steps) || steps < 1 || steps > 14) {
      setMessage("Forecast length must be a whole number from 1 to 14 days.", "error");
      return;
    }

    state.busy = "baseline";
    syncControls();
    setMessage("Running the baseline on the local model…");
    try {
      const result = await callSim(null, steps);
      state.baseline = result;
      state.shock = null;
      $("baseline-time").textContent = "Model inference: " + result.inferenceMs.toFixed(1) + " ms";
      $("shock-time").textContent = "";
      $("day").max = String(steps);
      $("day").value = "1";
      $("view-mode").value = "baseline";
      setMessage("Baseline received from the live FORKCAST model.", "success");
    } catch (error) {
      state.baseline = null;
      state.shock = null;
      $("view-mode").value = "baseline";
      setMessage(error.message || "The baseline request failed.", "error");
    } finally {
      state.busy = "";
      syncControls();
      draw();
    }
  }

  async function runShock() {
    if (!state.baseline) {
      setMessage("Run a baseline forecast before the counterfactual.", "error");
      return;
    }
    state.busy = "shock";
    syncControls();
    setMessage("Running the counterfactual on the local model…");
    try {
      const result = await callSim(makeShock(), state.baseline.steps);
      state.shock = result;
      $("shock-time").textContent = "Model inference: " + result.inferenceMs.toFixed(1) + " ms";
      $("view-mode").value = "shock";
      setMessage("Counterfactual received. Choose a view to compare it with the baseline.", "success");
    } catch (error) {
      state.shock = null;
      $("view-mode").value = "baseline";
      setMessage(error.message || "The counterfactual request failed.", "error");
    } finally {
      state.busy = "";
      syncControls();
      draw();
    }
  }

  async function verifyZeroShock() {
    if (!state.baseline) {
      setMessage("Run a baseline forecast before the model check.", "error");
      return;
    }
    state.busy = "zero";
    syncControls();
    setMessage("Checking the same input with a zero-valued shock…");
    try {
      const zero = Array.from({ length: 6 }, () =>
        Array.from({ length: 32 }, () => Array(36).fill(0)));
      const result = await callSim(zero, state.baseline.steps);
      const difference = maxAllDifference(state.baseline.rollout, result.rollout);
      $("view-mode").value = "baseline";
      setMessage(
        "Zero-shock maximum raw difference: " + difference.toExponential(3) +
        " across channels with different physical units.",
        "success"
      );
    } catch (error) {
      setMessage(error.message || "The zero-shock check failed.", "error");
    } finally {
      state.busy = "";
      syncControls();
      draw();
    }
  }

  function gridPoint(event) {
    const rect = map.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = Math.max(0, Math.min(35, Math.floor((event.clientX - rect.left) / rect.width * 36)));
    const topY = Math.max(0, Math.min(31, Math.floor((event.clientY - rect.top) / rect.height * 32)));
    return { x: x, y: 31 - topY };
  }

  function applySelectionToCustomArea() {
    const selection = state.selection;
    const minX = Math.min(selection.x0, selection.x1);
    const maxX = Math.max(selection.x0, selection.x1);
    const minY = Math.min(selection.y0, selection.y1);
    const maxY = Math.max(selection.y0, selection.y1);
    $("lon-start").value = String(65.5 + minX);
    $("lon-end").value = String(65.5 + maxX);
    $("lat-start").value = String(6.5 + minY);
    $("lat-end").value = String(6.5 + maxY);
    $("shock-preset").value = "custom";
    updateScenarioDescription();
    invalidateShock();
    setMessage("Custom area selected. Run the counterfactual to model this change.");
    draw();
  }

  $("window-file").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) loadWindow(file);
  });
  $("file-drop-area").addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.busy) $("file-drop-area").classList.add("dragging");
  });
  $("file-drop-area").addEventListener("dragleave", () => {
    $("file-drop-area").classList.remove("dragging");
  });
  $("file-drop-area").addEventListener("drop", (event) => {
    event.preventDefault();
    $("file-drop-area").classList.remove("dragging");
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) loadWindow(file);
  });

  $("run-baseline").addEventListener("click", runBaseline);
  $("run-shock").addEventListener("click", runShock);
  $("run-zero").addEventListener("click", verifyZeroShock);
  $("shock-preset").addEventListener("change", () => {
    updateScenarioDescription();
    invalidateShock();
  });
  ["shock-channel", "shock-delta", "lat-start", "lat-end", "lon-start", "lon-end"].forEach((id) => {
    $(id).addEventListener("change", invalidateShock);
  });
  ["channel", "day", "view-mode"].forEach((id) => $(id).addEventListener("change", draw));
  $("steps").addEventListener("change", () => {
    $("day").max = $("steps").value;
    if (Number($("day").value) > Number($("steps").value)) $("day").value = $("steps").value;
    if (state.baseline || state.shock) {
      state.baseline = null;
      state.shock = null;
      $("baseline-time").textContent = "";
      $("shock-time").textContent = "";
      $("view-mode").value = "baseline";
      setMessage("Forecast length changed. Run a new baseline to update the results.");
      syncControls();
      draw();
    }
  });

  map.addEventListener("pointerdown", (event) => {
    const point = gridPoint(event);
    if (!point) return;
    dragStart = point;
    state.selection = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
    map.setPointerCapture(event.pointerId);
    draw();
  });
  map.addEventListener("pointermove", (event) => {
    if (!dragStart) return;
    const point = gridPoint(event);
    if (!point) return;
    state.selection.x1 = point.x;
    state.selection.y1 = point.y;
    draw();
  });
  map.addEventListener("pointerup", (event) => {
    if (!dragStart) return;
    const point = gridPoint(event);
    if (point) {
      state.selection.x1 = point.x;
      state.selection.y1 = point.y;
    }
    dragStart = null;
    applySelectionToCustomArea();
  });
  map.addEventListener("pointercancel", () => {
    dragStart = null;
    draw();
  });

  updateScenarioDescription();
  syncControls();
  draw();
  checkHealth();
})();
