import { MODELS } from "./models.js";
import { CHIPS, chipById, cloneChip, chipRatios } from "./chips.js";
import { defaultWorkload, simulatePair, opTable, tableTotal, PRECISIONS, supportsPrecision, sweepVecRatio, sweepBwRatio, sweepHeatmap } from "./engine.js";
import { fmtMs, fmtTps, boundLabel, shortBound } from "./conclusions.js";
import * as charts from "./charts.js";

const CATALOG_IDS = CHIPS.filter((c) => c.id !== "custom").map((c) => c.id);

const state = {
  modelId: "qwen3.8-27b",
  chip: chipById("custom"),
  workload: defaultWorkload(),
  pins: [],
  pinSeq: 0,
  catalogOn: Object.fromEntries(CATALOG_IDS.map((id) => [id, true])),
};

let lastPair = null;
let lastCompare = [];
let catalogCache = { key: "", rows: [] };

function $(id) {
  return document.getElementById(id);
}

function num(id) {
  const v = parseFloat($(id).value);
  return Number.isFinite(v) ? v : 0;
}

function fillChipSelect() {
  const sel = $("chip-id");
  sel.innerHTML = "";
  const vendors = [...new Set(CHIPS.map((c) => c.vendor))];
  for (const v of vendors) {
    const g = document.createElement("optgroup");
    g.label = v;
    CHIPS.filter((c) => c.vendor === v).forEach((c) => {
      const o = document.createElement("option");
      o.value = c.id;
      o.textContent = c.name;
      g.appendChild(o);
    });
    sel.appendChild(g);
  }
}

function setChipFields(chip) {
  const ids = new Set(CHIPS.map((c) => c.id));
  $("chip-id").value = ids.has(chip.id) ? chip.id : "custom";
  $("bf16").value = chip.bf16Tflops;
  $("fp8").value = chip.fp8Tflops;
  $("fp4").value = chip.fp4Tflops;
  $("fp32").value = chip.fp32Tflops;
  $("vector").value = chip.vectorTflops;
  $("bw").value = chip.hbmBandwidthGBs;
  $("cap").value = chip.hbmCapacityGB;
  const r = chipRatios(chip);
  $("vec-ratio").value = (1 / r.vectorToMatmul).toFixed(2);
  $("bw-ratio").value = r.bwPerMatmul.toFixed(3);
  $("fp8-mul").value = (chip.fp8Tflops / Math.max(chip.bf16Tflops, 1e-9)).toFixed(2);
  $("fp4-mul").value = (chip.fp4Tflops / Math.max(chip.bf16Tflops, 1e-9)).toFixed(2);
  renderChipNotes(chip);
}

function renderChipNotes(chip) {
  const el = $("chip-notes");
  const note = chip.notes || "";
  if (chip.sourceUrl) {
    el.innerHTML = `${escapeHtml(note)} <a class="src" href="${escapeAttr(chip.sourceUrl)}" target="_blank" rel="noopener">${escapeHtml(chip.sourceLabel || "来源")}</a>`;
  } else {
    el.textContent = note;
  }
}

function readChipFields() {
  const chip = cloneChip(state.chip);
  chip.id = $("chip-id").value;
  chip.bf16Tflops = num("bf16");
  chip.fp8Tflops = num("fp8");
  chip.fp4Tflops = num("fp4");
  chip.fp32Tflops = num("fp32") || chip.fp32Tflops;
  chip.vectorTflops = num("vector");
  chip.hbmBandwidthGBs = num("bw");
  chip.hbmCapacityGB = num("cap");
  return chip;
}

function readWorkload() {
  const w = { ...defaultWorkload(), ...state.workload };
  w.batch = Math.max(1, Math.round(num("batch") || 1));
  w.decodeBatch = Math.max(1, Math.round(num("decode-batch") || 1));
  w.seq = Math.max(1, Math.round(num("seq") || 1));
  w.context = Math.max(1, Math.round(num("context") || 1));
  w.precision = $("precision").value;
  w.cubeVectorGamma = Math.min(1, Math.max(0, num("cube-vec-gamma")));
  w.overlapAlpha = Math.min(1, Math.max(0, num("overlap-alpha")));
  w.epilogueHbmFrac = Math.min(1, Math.max(0, num("epilogue-hbm")));
  return w;
}

function applyVecRatio() {
  const n = Math.max(0.1, num("vec-ratio"));
  const bf16 = num("bf16");
  if (bf16 > 0) $("vector").value = (bf16 / n).toFixed(4);
}

function applyBwRatio() {
  const n = Math.max(0.01, num("bw-ratio"));
  const bf16 = num("bf16");
  if (bf16 > 0) $("bw").value = (bf16 * n).toFixed(2);
}

function applyMuls() {
  const bf16 = num("bf16");
  $("fp8").value = (bf16 * num("fp8-mul")).toFixed(4);
  $("fp4").value = (bf16 * num("fp4-mul")).toFixed(4);
}

function syncRatiosFromAbs() {
  const chip = readChipFields();
  const r = chipRatios(chip);
  $("vec-ratio").value = (1 / r.vectorToMatmul).toFixed(2);
  $("bw-ratio").value = r.bwPerMatmul.toFixed(3);
}

function run() {
  state.chip = readChipFields();
  state.workload = readWorkload();
  render();
  $("live").textContent = "已重算 " + new Date().toLocaleTimeString();
}

let timer = 0;
function scheduleRun() {
  $("live").textContent = "计算中…";
  clearTimeout(timer);
  timer = setTimeout(run, 80);
}

function bind() {
  fillChipSelect();
  $("chip-id").addEventListener("change", () => {
    state.chip = chipById($("chip-id").value);
    setChipFields(state.chip);
    if (!$("config-name").value.trim()) {
      $("config-name").placeholder = chipTitle(state.chip);
    }
    run();
  });
  $("reset-chip").addEventListener("click", () => {
    state.chip = chipById($("chip-id").value);
    setChipFields(state.chip);
    run();
  });

  const onInput = (el, before) => {
    el.addEventListener("input", () => {
      if (before) before();
      scheduleRun();
    });
    el.addEventListener("change", () => {
      if (before) before();
      run();
    });
  };

  onInput($("vec-ratio"), applyVecRatio);
  onInput($("bw-ratio"), applyBwRatio);
  onInput($("fp8-mul"), applyMuls);
  onInput($("fp4-mul"), applyMuls);
  onInput($("bf16"), () => {
    applyMuls();
    applyVecRatio();
    applyBwRatio();
  });
  ["fp8", "fp4", "vector", "bw", "cap"].forEach((id) => {
    onInput($(id), syncRatiosFromAbs);
  });
  ["batch", "decode-batch", "seq", "context", "cube-vec-gamma", "overlap-alpha", "epilogue-hbm"].forEach((id) => onInput($(id)));
  $("precision").addEventListener("change", run);
  $("config-name").addEventListener("change", run);
  $("pin-config").addEventListener("click", pinCurrent);
  $("catalog-all").addEventListener("click", () => setCatalogAll(true));
  $("catalog-none").addEventListener("click", () => setCatalogAll(false));
  $("dl-prefill").addEventListener("click", () => downloadOpCsv("prefill"));
  $("dl-decode").addEventListener("click", () => downloadOpCsv("decode"));
  $("dl-compare").addEventListener("click", downloadCompareCsv);

  window.addEventListener("resize", () => charts.resizeCharts());

  setChipFields(state.chip);
  $("batch").value = state.workload.batch;
  $("decode-batch").value = state.workload.decodeBatch;
  $("seq").value = state.workload.seq;
  $("context").value = state.workload.context;
  $("precision").value = state.workload.precision;
  $("cube-vec-gamma").value = state.workload.cubeVectorGamma;
  $("overlap-alpha").value = state.workload.overlapAlpha;
  $("epilogue-hbm").value = state.workload.epilogueHbmFrac;
}

function fmtFlops(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TF`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GF`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MF`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)} kF`;
  return n === 0 ? "0" : n.toFixed(0);
}

function fmtBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(3)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} kB`;
  return `${n.toFixed(0)} B`;
}

function fmtOpMs(s) {
  const ms = s * 1e3;
  if (ms === 0) return "0";
  if (ms < 0.001) return ms.toExponential(1);
  if (ms < 1) return ms.toFixed(3);
  return ms.toFixed(2);
}

function chipTitle(chip) {
  const cat = CHIPS.find((c) => c.id === chip.id);
  return cat ? cat.name : chip.name || chip.id;
}

function currentName() {
  const typed = $("config-name").value.trim();
  return typed || `当前 · ${chipTitle(state.chip)}`;
}

function configLabel(chip, wl) {
  const r = chipRatios(chip);
  return `${chipTitle(chip)} · ${wl.precision} · C:V ${r.cubeToVector.toFixed(0)}:1 · BW${r.bwPerMatmul.toFixed(1)}`;
}

function configDetail(chip, wl) {
  const r = chipRatios(chip);
  const fp8 = chip.fp8Tflops > 0 ? `FP8 ${chip.fp8Tflops}T` : "无FP8";
  const fp4 = chip.fp4Tflops > 0 ? `FP4 ${chip.fp4Tflops}T` : "无FP4";
  return [
    `Cube BF16 ${chip.bf16Tflops}T`,
    `Vec BF16 ${chip.vectorTflops}T`,
    `Cube:Vector ${r.cubeToVector.toFixed(1)}:1`,
    `HBM ${chip.hbmBandwidthGBs}GB/s (${r.bwPerMatmul.toFixed(2)}:1)`,
    fp8,
    fp4,
    `HBM ${chip.hbmCapacityGB}GB`,
    wl.precision,
    `Bp=${wl.batch} Bd=${wl.decodeBatch ?? wl.batch} S=${wl.seq} C=${wl.context}`,
  ].join(" · ");
}

function fingerprint(chip, wl) {
  return JSON.stringify({
    id: chip.id,
    bf16: chip.bf16Tflops,
    fp8: chip.fp8Tflops,
    fp4: chip.fp4Tflops,
    vec: chip.vectorTflops,
    bw: chip.hbmBandwidthGBs,
    cap: chip.hbmCapacityGB,
    b: wl.batch,
    db: wl.decodeBatch ?? wl.batch,
    s: wl.seq,
    c: wl.context,
    p: wl.precision,
    g: wl.cubeVectorGamma,
    a: wl.overlapAlpha,
    e: wl.epilogueHbmFrac,
  });
}

function workloadKey(wl) {
  return `${wl.batch}|${wl.decodeBatch ?? wl.batch}|${wl.seq}|${wl.context}|${wl.precision}|${wl.cubeVectorGamma}|${wl.epilogueHbmFrac}|${wl.overlapAlpha}`;
}

function snapshotFrom(chip, wl, pair, extra = {}) {
  return {
    key: fingerprint(chip, wl),
    chipId: chip.id,
    name: extra.name || configLabel(chip, wl),
    label: configLabel(chip, wl),
    detail: configDetail(chip, wl),
    bf16: chip.bf16Tflops,
    vector: chip.vectorTflops,
    bw: chip.hbmBandwidthGBs,
    fp4: chip.fp4Tflops,
    fp8: chip.fp8Tflops,
    cap: chip.hbmCapacityGB,
    precision: wl.precision,
    batch: wl.batch,
    decodeBatch: wl.decodeBatch ?? wl.batch,
    seq: wl.seq,
    context: wl.context,
    prefillMs: pair.prefill.ms,
    decodeTps: pair.decode.tokPerSec,
    decodeMs: pair.decode.ms,
    prefillBound: pair.prefill.primary,
    decodeBound: pair.decode.primary,
    oom: pair.prefill.mem.oom || pair.decode.mem.oom,
    sourceUrl: chip.sourceUrl || "",
    sourceLabel: chip.sourceLabel || "",
  };
}

function catalogSnapshots(model, wl) {
  const key = workloadKey(wl);
  if (catalogCache.key === key) return catalogCache.rows;
  const rows = CATALOG_IDS.map((id) => {
    const chip = chipById(id);
    const pair = simulatePair(model, chip, wl);
    return { ...snapshotFrom(chip, wl, pair, { name: chip.name }), kind: "catalog", id: `cat-${id}` };
  });
  catalogCache = { key, rows };
  return rows;
}

function missingDtypeLabel(precision) {
  const d = (PRECISIONS[precision] || {}).gemmDtype;
  if (d === "fp4") return "无原生 FP4";
  if (d === "fp8") return "无原生 FP8";
  return "";
}

function rowSupportsPrecision(r, precision) {
  return supportsPrecision({ fp4Tflops: r.fp4, fp8Tflops: r.fp8 }, precision);
}

function visibleCompare(current, catalog, precision) {
  const out = [];
  const cur = { ...current, kind: "current", id: "current" };
  if (rowSupportsPrecision(cur, precision)) out.push(cur);
  for (const p of state.pins) {
    if (p.precision && p.precision !== precision) continue;
    if (!rowSupportsPrecision(p, precision)) continue;
    out.push({ ...p, kind: "pin" });
  }
  for (const c of catalog) {
    if (!state.catalogOn[c.chipId]) continue;
    if (c.key === current.key) continue;
    if (!rowSupportsPrecision(c, precision)) continue;
    out.push(c);
  }
  return out;
}

function pinCurrent() {
  const model = MODELS[state.modelId];
  state.chip = readChipFields();
  state.workload = readWorkload();
  const pair = simulatePair(model, state.chip, state.workload);
  const typed = $("config-name").value.trim();
  const snap = snapshotFrom(state.chip, state.workload, pair, {
    name: typed || `自定义 ${state.pinSeq + 1}`,
  });
  if (state.pins.some((p) => p.key === snap.key)) {
    $("live").textContent = "该配置已在对比中";
    return;
  }
  snap.id = `pin-${++state.pinSeq}`;
  snap.kind = "pin";
  state.pins.push(snap);
  render();
  $("live").textContent = "已加入对比 " + snap.name;
}

function unpin(id) {
  state.pins = state.pins.filter((p) => p.id !== id);
  render();
}

function setCatalogAll(on) {
  for (const id of CATALOG_IDS) state.catalogOn[id] = on;
  render();
}

function kindLabel(kind) {
  if (kind === "current") return "当前";
  if (kind === "pin") return "钉住";
  return "目录";
}

function shownInChart(r, currentKey, precision) {
  if (!rowSupportsPrecision(r, precision)) return false;
  if (r.kind === "pin" && r.precision && r.precision !== precision) return false;
  if (r.kind === "current") return true;
  if (r.kind === "pin") return true;
  return !!state.catalogOn[r.chipId] && r.key !== currentKey;
}

function fillCompareTable(current, catalog) {
  const rows = [
    { ...current, kind: "current", id: "current" },
    ...state.pins.map((p) => ({ ...p, kind: "pin" })),
    ...catalog,
  ];
  lastCompare = rows;
  const el = $("tbl-compare");
  el.classList.add("compare");
  const precision = state.workload.precision;
  const miss = missingDtypeLabel(precision);
  const body = rows.map((r) => {
    const native = rowSupportsPrecision(r, precision);
    const onCell =
      r.kind === "current"
        ? (native ? "—" : miss)
        : r.kind === "pin"
          ? `<button type="button" data-unpin="${r.id}" aria-label="移除">移除</button>`
          : native
            ? `<input class="check" type="checkbox" data-catalog="${r.chipId}" ${state.catalogOn[r.chipId] ? "checked" : ""} />`
            : `<span class="muted">${miss}</span>`;
    const nameCell =
      r.kind === "pin"
        ? `<input class="rename" data-pin="${r.id}" value="${escapeAttr(r.name)}" />`
        : escapeHtml(r.name);
    const metrics = native
      ? `<td>${r.prefillMs.toFixed(2)}</td>
      <td>${r.decodeTps.toFixed(1)}</td>
      <td class="bound-${r.prefillBound}">${shortBound(r.prefillBound)}</td>
      <td class="bound-${r.decodeBound}">${shortBound(r.decodeBound)}</td>`
      : `<td colspan="4" class="detail">${miss}，GEMM 会回退 BF16，不进入对比图</td>`;
    return `<tr class="${native ? "" : "unsupported"}">
      <td class="kind">${kindLabel(r.kind)}</td>
      <td>${onCell}</td>
      <td>${nameCell}</td>
      <td class="detail" title="${escapeAttr(r.detail)}">${escapeHtml(r.detail)}</td>
      <td>${r.sourceUrl ? `<a class="src" href="${escapeAttr(r.sourceUrl)}" target="_blank" rel="noopener">${escapeHtml(r.sourceLabel || "来源")}</a>` : "—"}</td>
      ${metrics}
    </tr>`;
  }).join("");
  el.innerHTML = `<thead><tr>
      <th>类型</th><th>显示</th><th>名称</th><th>配置</th><th>数据来源</th>
      <th>Prefill ms</th><th>Decode tok/s</th><th>Prefill 瓶颈</th><th>Decode 瓶颈</th>
    </tr></thead><tbody>${body}</tbody>`;
  el.querySelectorAll("[data-unpin]").forEach((btn) => {
    btn.addEventListener("click", () => unpin(btn.getAttribute("data-unpin")));
  });
  el.querySelectorAll("[data-catalog]").forEach((box) => {
    box.addEventListener("change", () => {
      state.catalogOn[box.getAttribute("data-catalog")] = box.checked;
      render();
    });
  });
  el.querySelectorAll(".rename").forEach((inp) => {
    inp.addEventListener("change", () => {
      const p = state.pins.find((x) => x.id === inp.getAttribute("data-pin"));
      if (p) {
        p.name = inp.value.trim() || p.label;
        render();
      }
    });
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function downloadText(filename, text) {
  const blob = new Blob(["\ufeff" + text], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function opCsv(result) {
  const rows = opTable(result);
  const tot = tableTotal(rows);
  const headers = [
    "算子", "层", "shape", "计算", "Cube FLOP", "Vector FLOP", "搬运 bytes",
    "Cube s", "Vector s", "带宽 s", "流水 s", "占比", "瓶颈",
  ];
  const line = (r, share) => [
    r.name, r.n, r.shape || "", r.formula || "", r.flopsCube, r.flopsVector, r.bytes,
    r.tCube, r.tVector, r.tBw, r.tPipe, share, r.bound,
  ];
  const body = [
    ...rows.map((r) => line(r, ((100 * r.t) / result.t).toFixed(4) + "%")),
    line(tot, "100%"),
  ];
  return [headers, ...body].map((row) => row.map(csvEscape).join(",")).join("\n");
}

function downloadOpCsv(phase) {
  if (!lastPair) return;
  downloadText(`qwen38-27b-${phase}.csv`, opCsv(lastPair[phase]));
}

function downloadCompareCsv() {
  const headers = [
    "类型", "名称", "配置", "数据来源", "来源URL", "chip", "BF16 Cube TFLOPS", "Vector BF16 TFLOPS", "HBM GB/s",
    "FP8 TFLOPS", "FP4 TFLOPS", "HBM GB", "precision", "batch", "decodeBatch", "seq", "context",
    "Prefill ms", "Decode tok/s", "Decode ms", "Prefill 瓶颈", "Decode 瓶颈",
  ];
  const currentKey = lastCompare.find((r) => r.kind === "current")?.key;
  const precision = state.workload.precision;
  const headersWithOn = ["显示", ...headers];
  const body = lastCompare.map((r) => [
    shownInChart(r, currentKey, precision) ? "是" : "否",
    kindLabel(r.kind), r.name, r.detail, r.sourceLabel || "", r.sourceUrl || "", r.chipId, r.bf16, r.vector, r.bw,
    r.fp8, r.fp4, r.cap, r.precision, r.batch, r.decodeBatch, r.seq, r.context,
    r.prefillMs, r.decodeTps, r.decodeMs, r.prefillBound, r.decodeBound,
  ]);
  downloadText("chip-compare.csv", [headersWithOn, ...body].map((row) => row.map(csvEscape).join(",")).join("\n"));
}

function fillTable(id, result) {
  const rows = opTable(result);
  const tot = tableTotal(rows);
  const totalT = result.t || 1e-15;
  const tr = (r, share, cls = "") => `<tr class="${cls}">
      <td>${r.name}</td>
      <td>${r.n}</td>
      <td class="shape">${r.shape || "—"}</td>
      <td class="formula">${escapeHtml(r.formula || "—")}</td>
      <td>${fmtFlops(r.flopsCube)}</td>
      <td>${fmtFlops(r.flopsVector)}</td>
      <td>${fmtBytes(r.bytes)}</td>
      <td>${fmtOpMs(r.tCube)}</td>
      <td>${fmtOpMs(r.tVector)}</td>
      <td>${fmtOpMs(r.tBw)}</td>
      <td>${fmtOpMs(r.tPipe)}</td>
      <td>${share}</td>
      <td class="bound-${r.bound}">${boundLabel(r.bound)}</td>
    </tr>`;
  $(id).innerHTML = `<thead><tr>
      <th>算子</th><th>层</th><th>shape</th><th>计算</th>
      <th>Cube FLOP</th><th>Vector FLOP</th><th>搬运</th>
      <th>Cube ms</th><th>Vector ms</th><th>带宽 ms</th><th>流水 ms</th><th>占比</th><th>瓶颈</th>
    </tr></thead>
    <tbody>${rows.map((r) => tr(r, ((100 * r.t) / totalT).toFixed(1) + "%")).join("")}</tbody>
    <tfoot>${tr(tot, "100%", "total")}</tfoot>`;
  $(id).classList.add("ops");
}

function render() {
  const model = MODELS[state.modelId];
  const pair = simulatePair(model, state.chip, state.workload);
  const r = chipRatios(state.chip);
  const current = snapshotFrom(state.chip, state.workload, pair, { name: currentName() });
  const catalog = catalogSnapshots(model, state.workload);
  const compare = visibleCompare(current, catalog, state.workload.precision);
  lastPair = pair;

  $("kpi-prefill").textContent = fmtMs(pair.prefill.ms);
  $("kpi-prefill-tps").textContent = fmtTps(pair.prefill.tokPerSec);
  $("kpi-pbound").textContent = shortBound(pair.prefill.primary);
  $("kpi-decode").textContent = fmtTps(pair.decode.tokPerSec);
  $("kpi-decode-ms").textContent = `${fmtMs(pair.decode.ms)} / token`;
  $("kpi-dbound").textContent = shortBound(pair.decode.primary);
  $("kpi-vec").textContent = `${r.cubeToVector.toFixed(1)}:1`;
  $("kpi-bw").textContent = `${r.bwPerMatmul.toFixed(2)}:1`;
  const mem = pair.decode.mem;
  $("kpi-mem").textContent = mem.oom
    ? `${mem.totalGB.toFixed(1)} GB 超出 ${mem.capGB} GB`
    : `${mem.totalGB.toFixed(1)} / ${mem.capGB} GB`;
  $("kpi-kv").textContent = `${mem.kvGB.toFixed(2)} GB @ C=${state.workload.context} Bd=${state.workload.decodeBatch}（16 层 Attention）`;
  $("hero-prefill").classList.toggle("warn", pair.prefill.mem.oom);
  $("hero-decode").classList.toggle("warn", mem.oom);

  fillCompareTable(current, catalog);
  charts.drawCompare("chart-compare", compare);
  const vecRows = sweepVecRatio(model, state.chip, state.workload, [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]);
  charts.drawRatioSweep("chart-sweep-vec", {
    xName: "Cube:Vector",
    rows: vecRows,
    currentX: r.cubeToVector,
  });
  charts.drawKeyOpsSweep("chart-sweep-ops", {
    rows: vecRows,
    currentX: r.cubeToVector,
  });
  charts.drawRatioSweep("chart-sweep-bw", {
    xName: "带宽:Cube",
    rows: sweepBwRatio(model, state.chip, state.workload, [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]),
    currentX: r.bwPerMatmul,
  });
  charts.drawHeat("chart-decode-heat", sweepHeatmap(model, state.chip, state.workload, { phase: "decode" }), {
    cubeToVector: r.cubeToVector,
    bwPerMatmul: r.bwPerMatmul,
  });
  charts.drawWalls("chart-prefill-walls", pair.prefill);
  charts.drawWalls("chart-decode-walls", pair.decode);
  charts.drawWallShare("chart-prefill-share", pair.prefill);
  charts.drawWallShare("chart-decode-share", pair.decode);
  fillTable("tbl-prefill", pair.prefill);
  fillTable("tbl-decode", pair.decode);
  requestAnimationFrame(() => charts.resizeCharts());
}

bind();
run();
