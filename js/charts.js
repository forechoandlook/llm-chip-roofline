import { PRECISIONS, peakTflops } from "./engine.js";
import { boundLabel, fmtMs, fmtTps } from "./conclusions.js";

const C = {
  tensor: "#7eb8b0",
  vector: "#d4a574",
  bw: "#8b9cc8",
  ink: "#e8e4dc",
  muted: "#9aa3ab",
  line: "#2c3238",
  warn: "#e08a6a",
};

const FAMILY_LABEL = {
  embed: "Embedding",
  gdn_proj: "GDN 投影",
  gdn_conv: "GDN Conv",
  gdn_state: "GDN 状态",
  attn_proj: "Attn 投影",
  flashattn: "FlashAttention",
  ffn: "FFN",
  norm: "Norm/RoPE",
  residual: "Residual",
  lm_head: "LM Head",
};

const charts = new Map();

function baseTheme() {
  return {
    backgroundColor: "transparent",
    textStyle: { color: C.ink, fontFamily: "PingFang SC, sans-serif" },
    grid: { left: 52, right: 18, top: 36, bottom: 42 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#1c2024",
      borderColor: C.line,
      textStyle: { color: C.ink, fontSize: 12 },
    },
    legend: { textStyle: { color: C.muted }, top: 4 },
  };
}

export function setChart(id, option) {
  const el = document.getElementById(id);
  if (!el || !window.echarts) return;
  let ch = charts.get(id);
  if (!ch) {
    ch = window.echarts.init(el, null, { renderer: "canvas" });
    charts.set(id, ch);
  }
  ch.setOption(option, true);
}

export function resizeCharts() {
  for (const ch of charts.values()) ch.resize();
}

function famEntries(byFamily) {
  return Object.entries(byFamily)
    .map(([k, v]) => ({ k, label: FAMILY_LABEL[k] || k, ...v }))
    .sort((a, b) => b.t - a.t);
}

export function drawWalls(id, result) {
  const cube = result.tCube ?? result.tTensor;
  const rows = [
    { name: "Cube", value: cube * 1e3, color: C.tensor },
    { name: "Vector", value: result.tVector * 1e3, color: C.vector },
    { name: "带宽", value: result.tBw * 1e3, color: C.bw },
  ];
  const pipe = result.t * 1e3;
  setChart(id, {
    ...baseTheme(),
    tooltip: {
      ...baseTheme().tooltip,
      trigger: "item",
      formatter: (p) => `${p.name}<br/>${p.value.toFixed(3)} ms`,
    },
    grid: { left: 56, right: 16, top: 28, bottom: 28 },
    xAxis: { type: "category", data: rows.map((r) => r.name), axisLabel: { color: C.ink } },
    yAxis: {
      type: "value",
      name: "ms",
      axisLabel: { color: C.muted },
      splitLine: { lineStyle: { color: C.line } },
    },
    series: [{
      type: "bar",
      data: rows.map((r) => ({ value: r.value, itemStyle: { color: r.color } })),
      barMaxWidth: 42,
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: C.ink, type: "dashed", width: 1.5 },
        label: { color: C.ink, formatter: `流水 ${pipe.toFixed(2)} ms` },
        data: [{ yAxis: pipe }],
      },
    }],
  });
}

export function drawWallShare(id, result) {
  const cube = result.tCube ?? result.tTensor;
  setChart(id, {
    ...baseTheme(),
    tooltip: { trigger: "item", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    legend: { show: false },
    series: [{
      type: "pie",
      radius: ["42%", "68%"],
      label: { color: C.ink, fontSize: 11, formatter: "{b}\n{d}%" },
      data: [
        { name: "Cube", value: cube, itemStyle: { color: C.tensor } },
        { name: "Vector", value: result.tVector, itemStyle: { color: C.vector } },
        { name: "带宽", value: result.tBw, itemStyle: { color: C.bw } },
      ],
    }],
  });
}

export function drawKeyOpsSweep(id, spec) {
  const { rows, currentX } = spec;
  if (!rows.length || rows[0].matmulMs == null) return;
  let mark = 0;
  let best = Infinity;
  rows.forEach((r, i) => {
    const d = Math.abs(r.x - currentX);
    if (d < best) {
      best = d;
      mark = i;
    }
  });
  setChart(id, {
    ...baseTheme(),
    tooltip: {
      ...baseTheme().tooltip,
      axisPointer: { type: "cross" },
      formatter: (items) => {
        if (!items || !items.length) return "";
        const r = rows[items[0].dataIndex];
        if (!r) return "";
        return `Cube:Vector ${r.x}:1<br/>FFN MatMul+quant ${r.matmulMs.toFixed(1)} ms（${boundLabel(r.matmulBound)}）<br/>FlashAttention ${r.attnMs.toFixed(1)} ms（${boundLabel(r.attnBound)}）<br/>GDN 线性注意力 ${r.gdnMs.toFixed(1)} ms（${boundLabel(r.gdnBound)}）`;
      },
    },
    legend: { ...baseTheme().legend, data: ["FFN MatMul+quant", "FlashAttention", "GDN 线性注意力"] },
    grid: { left: 56, right: 24, top: 36, bottom: 40 },
    xAxis: {
      type: "category",
      name: "Cube:Vector",
      data: rows.map((r) => String(r.x)),
      axisLabel: { color: C.ink },
      nameTextStyle: { color: C.muted },
    },
    yAxis: { type: "value", name: "Prefill ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    series: [
      {
        name: "FFN MatMul+quant",
        type: "line",
        data: rows.map((r) => +r.matmulMs.toFixed(2)),
        itemStyle: { color: C.tensor },
        lineStyle: { width: 2 },
        markPoint: {
          data: [{ coord: [String(rows[mark].x), +rows[mark].matmulMs.toFixed(2)], name: "当前" }],
          symbol: "diamond",
          symbolSize: 11,
          itemStyle: { color: C.warn },
          label: { show: false },
        },
      },
      {
        name: "FlashAttention",
        type: "line",
        data: rows.map((r) => +r.attnMs.toFixed(2)),
        itemStyle: { color: C.warn },
        lineStyle: { width: 2 },
        markPoint: {
          data: [{ coord: [String(rows[mark].x), +rows[mark].attnMs.toFixed(2)], name: "当前" }],
          symbol: "diamond",
          symbolSize: 11,
          itemStyle: { color: C.warn },
          label: { show: false },
        },
      },
      {
        name: "GDN 线性注意力",
        type: "line",
        data: rows.map((r) => +r.gdnMs.toFixed(2)),
        itemStyle: { color: C.vector },
        lineStyle: { width: 2 },
        markPoint: {
          data: [{ coord: [String(rows[mark].x), +rows[mark].gdnMs.toFixed(2)], name: "当前" }],
          symbol: "diamond",
          symbolSize: 11,
          itemStyle: { color: C.warn },
          label: { color: C.warn, formatter: "当前", fontSize: 10, offset: [14, -8] },
        },
      },
    ],
  });
}

export function drawRatioSweep(id, spec) {
  const { xName, rows, currentX } = spec;
  let mark = 0;
  let best = Infinity;
  rows.forEach((r, i) => {
    const d = Math.abs(r.x - currentX);
    if (d < best) {
      best = d;
      mark = i;
    }
  });
  setChart(id, {
    ...baseTheme(),
    tooltip: { ...baseTheme().tooltip, axisPointer: { type: "cross" } },
    legend: { ...baseTheme().legend, data: ["Prefill ms", "Decode tok/s"] },
    grid: { left: 56, right: 64, top: 36, bottom: 40 },
    xAxis: {
      type: "category",
      name: xName,
      data: rows.map((r) => String(r.x)),
      axisLabel: { color: C.ink },
      nameTextStyle: { color: C.muted },
    },
    yAxis: [
      { type: "value", name: "Prefill ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
      { type: "value", name: "Decode tok/s", axisLabel: { color: C.muted }, splitLine: { show: false } },
    ],
    series: [
      {
        name: "Prefill ms",
        type: "line",
        data: rows.map((r) => +r.prefillMs.toFixed(2)),
        itemStyle: { color: C.tensor },
        lineStyle: { width: 2 },
        markPoint: {
          data: [{ coord: [String(rows[mark].x), +rows[mark].prefillMs.toFixed(2)], name: "当前" }],
          symbol: "diamond",
          symbolSize: 11,
          itemStyle: { color: C.warn },
          label: { show: false },
        },
      },
      {
        name: "Decode tok/s",
        type: "line",
        yAxisIndex: 1,
        data: rows.map((r) => +r.decodeTps.toFixed(1)),
        itemStyle: { color: C.vector },
        lineStyle: { width: 2 },
        markPoint: {
          data: [{ coord: [String(rows[mark].x), +rows[mark].decodeTps.toFixed(1)], name: "当前" }],
          symbol: "diamond",
          symbolSize: 11,
          itemStyle: { color: C.warn },
          label: { color: C.warn, formatter: "当前", fontSize: 10, offset: [14, -8] },
        },
      },
    ],
  });
}

export function drawCompare(id, rows) {
  if (!rows.length) {
    setChart(id, {
      ...baseTheme(),
      title: {
        text: "当前精度下没有可对比的芯片（需要原生 Tensor 峰值）",
        left: "center",
        top: "middle",
        textStyle: { color: C.muted, fontSize: 13, fontWeight: 400 },
      },
      xAxis: { show: false },
      yAxis: { show: false },
      series: [],
    });
    return;
  }
  setChart(id, {
    ...baseTheme(),
    tooltip: {
      ...baseTheme().tooltip,
      axisPointer: { type: "shadow" },
      formatter: (items) => {
        if (!items || !items.length) return "";
        const i = items[0].dataIndex;
        const r = rows[i];
        if (!r) return "";
        const lines = items.map((p) => `${p.marker}${p.seriesName} ${p.value}`);
        return `<b>${r.name}</b><br/>${r.detail || ""}<br/>${lines.join("<br/>")}`;
      },
    },
    legend: { ...baseTheme().legend, data: ["Prefill ms", "Decode tok/s"] },
    grid: { left: 56, right: 64, top: 36, bottom: rows.length > 4 ? 96 : 56 },
    xAxis: {
      type: "category",
      data: rows.map((r) => r.name),
      axisLabel: { color: C.ink, fontSize: 10, rotate: rows.length > 3 ? 38 : 0, interval: 0 },
    },
    yAxis: [
      { type: "value", name: "Prefill ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
      { type: "value", name: "Decode tok/s", axisLabel: { color: C.muted }, splitLine: { show: false } },
    ],
    series: [
      {
        name: "Prefill ms",
        type: "bar",
        data: rows.map((r) => +r.prefillMs.toFixed(2)),
        itemStyle: { color: C.tensor },
        barMaxWidth: 28,
      },
      {
        name: "Decode tok/s",
        type: "bar",
        yAxisIndex: 1,
        data: rows.map((r) => +r.decodeTps.toFixed(1)),
        itemStyle: { color: C.vector },
        barMaxWidth: 28,
      },
    ],
  });
}

export function drawFamilyStack(id, result) {
  const rows = famEntries(result.byFamily);
  setChart(id, {
    ...baseTheme(),
    tooltip: { ...baseTheme().tooltip, trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { ...baseTheme().legend, data: ["MatMul 时间", "Vector 时间", "带宽时间"] },
    grid: { left: 112, right: 16, top: 36, bottom: 24 },
    xAxis: { type: "value", name: "ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    yAxis: { type: "category", data: rows.map((r) => r.label).reverse(), axisLabel: { color: C.ink } },
    series: [
      bar("MatMul 时间", rows.map((r) => r.tTensor * 1e3).reverse(), C.tensor),
      bar("Vector 时间", rows.map((r) => r.tVector * 1e3).reverse(), C.vector),
      bar("带宽时间", rows.map((r) => r.tBw * 1e3).reverse(), C.bw),
    ],
  });
}

function bar(name, data, color) {
  return { name, type: "bar", stack: "t", data, itemStyle: { color }, barMaxWidth: 18 };
}

export function drawTopOps(id, result) {
  const rows = Object.entries(result.byName)
    .map(([name, v]) => ({ name, ...v }))
    .sort((a, b) => b.t - a.t)
    .slice(0, 12)
    .reverse();
  const color = (b) => (b === "bandwidth" ? C.bw : b === "vector" ? C.vector : C.tensor);
  setChart(id, {
    ...baseTheme(),
    grid: { left: 110, right: 16, top: 16, bottom: 28 },
    tooltip: { trigger: "item", backgroundColor: "#1c2024", borderColor: C.line, textStyle: { color: C.ink } },
    xAxis: { type: "value", name: "ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    yAxis: { type: "category", data: rows.map((r) => r.name), axisLabel: { color: C.ink, fontSize: 11 } },
    series: [{
      type: "bar",
      data: rows.map((r) => ({ value: r.t * 1e3, itemStyle: { color: color(r.bound) } })),
      barMaxWidth: 16,
    }],
  });
}

export function drawDonut(id, result) {
  const rows = famEntries(result.byFamily);
  setChart(id, {
    ...baseTheme(),
    tooltip: { trigger: "item", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    legend: { show: false },
    series: [{
      type: "pie",
      radius: ["42%", "68%"],
      label: { color: C.ink, fontSize: 11, formatter: "{b}\n{d}%" },
      data: rows.map((r) => ({ name: r.label, value: r.t })),
    }],
  });
}

export function drawBytes(id, mem) {
  setChart(id, {
    ...baseTheme(),
    tooltip: { trigger: "item", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    series: [{
      type: "pie",
      radius: ["40%", "66%"],
      label: { color: C.ink, formatter: "{b}\n{c} GB" },
      data: [
        { name: "权重", value: +mem.weightGB.toFixed(2) },
        { name: "KV cache", value: +mem.kvGB.toFixed(2) },
        { name: "GDN state", value: +mem.stateGB.toFixed(2) },
        { name: "激活下限", value: +mem.actGB.toFixed(2) },
      ],
    }],
  });
}

export function drawFlop(id, result) {
  setChart(id, {
    ...baseTheme(),
    tooltip: { trigger: "item", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    series: [{
      type: "pie",
      radius: ["40%", "66%"],
      label: { color: C.ink, formatter: "{b}\n{d}%" },
      data: [
        { name: "Tensor FLOP", value: result.flopsT, itemStyle: { color: C.tensor } },
        { name: "Vector FLOP", value: result.flopsV, itemStyle: { color: C.vector } },
      ],
    }],
  });
}

export function drawUtil(id, result) {
  const item = (name, val, color) => ({
    name,
    type: "gauge",
    center: name === "MatMul" ? ["16%", "55%"] : name === "Vector" ? ["50%", "55%"] : ["84%", "55%"],
    radius: "70%",
    startAngle: 210,
    endAngle: -30,
    min: 0,
    max: 100,
    pointer: { show: false },
    progress: { show: true, width: 10, itemStyle: { color } },
    axisLine: { lineStyle: { width: 10, color: [[1, "#2c3238"]] } },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { show: false },
    title: { offsetCenter: [0, "68%"], color: C.muted, fontSize: 11 },
    detail: { valueAnimation: false, offsetCenter: [0, "8%"], fontSize: 16, color: C.ink, formatter: "{value}%" },
    data: [{ value: Math.round(val * 100), name }],
  });
  setChart(id, {
    ...baseTheme(),
    series: [
      item("MatMul", result.utilTensor, C.tensor),
      item("Vector", result.utilVector, C.vector),
      item("带宽", result.utilBw, C.bw),
    ],
  });
}

export function drawRoofline(id, result, chip, workload) {
  const prec = PRECISIONS[workload.precision];
  const peak = peakTflops(chip, prec.gemmDtype || "bf16");
  const points = Object.entries(result.byName).map(([name, v]) => {
    const intensity = v.intensity;
    const achieved = (v.flopsTensor + v.flopsVector) / Math.max(v.t, 1e-15) / 1e12;
    return { name, value: [intensity, achieved], bound: v.bound };
  });
  const xs = [0.2, 1, 4, 16, 64, 256, 1024, 4096, 16384];
  const bwLine = xs.map((x) => [x, (chip.hbmBandwidthGBs * 1e9 * x) / 1e12]);
  const tensorLine = xs.map((x) => [x, peak]);
  const vecLine = xs.map((x) => [x, chip.vectorTflops]);
  const colorOf = (b) => (b === "bandwidth" ? C.bw : b === "vector" ? C.vector : C.tensor);
  setChart(id, {
    ...baseTheme(),
    legend: { ...baseTheme().legend, data: ["算子", "带宽墙", "MatMul 墙", "Vector 墙"] },
    tooltip: {
      trigger: "item",
      backgroundColor: "#1c2024",
      textStyle: { color: C.ink },
      formatter: (p) => {
        if (p.seriesName !== "算子") return p.seriesName;
        return `${p.data.name}<br/>强度 ${p.data.value[0].toFixed(1)} FLOP/B<br/>${p.data.value[1].toFixed(1)} TFLOPS`;
      },
    },
    xAxis: {
      type: "log",
      name: "算术强度 FLOP/byte",
      min: 0.2,
      axisLabel: { color: C.muted },
      splitLine: { lineStyle: { color: C.line } },
    },
    yAxis: {
      type: "log",
      name: "TFLOPS",
      min: 0.1,
      axisLabel: { color: C.muted },
      splitLine: { lineStyle: { color: C.line } },
    },
    series: [
      {
        name: "算子",
        type: "scatter",
        data: points.map((p) => ({ ...p, itemStyle: { color: colorOf(p.bound) }, symbolSize: 11 })),
      },
      { name: "带宽墙", type: "line", data: bwLine, showSymbol: false, lineStyle: { color: C.bw, width: 2 } },
      { name: "MatMul 墙", type: "line", data: tensorLine, showSymbol: false, lineStyle: { color: C.tensor, type: "dashed" } },
      { name: "Vector 墙", type: "line", data: vecLine, showSymbol: false, lineStyle: { color: C.vector, type: "dotted" } },
    ],
  });
}

export function drawPhase(id, pair) {
  const keys = ["tTensor", "tVector", "tBw"];
  const names = ["MatMul", "Vector", "带宽"];
  const colors = [C.tensor, C.vector, C.bw];
  setChart(id, {
    ...baseTheme(),
    tooltip: { ...baseTheme().tooltip, axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: ["Prefill", "Decode / step"], axisLabel: { color: C.ink } },
    yAxis: { type: "value", name: "ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    series: keys.map((k, i) => ({
      name: names[i],
      type: "bar",
      stack: "p",
      data: [pair.prefill[k] * 1e3, pair.decode[k] * 1e3],
      itemStyle: { color: colors[i] },
    })),
  });
}

export function drawSeq(id, sweep) {
  setChart(id, {
    ...baseTheme(),
    legend: { ...baseTheme().legend, data: ["Prefill ms", "Decode ms/step", "Decode tok/s"] },
    tooltip: { ...baseTheme().tooltip },
    xAxis: { type: "category", data: sweep.map((s) => String(s.seq)), name: "seq / context", axisLabel: { color: C.ink } },
    yAxis: [
      { type: "value", name: "ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
      { type: "value", name: "tok/s", axisLabel: { color: C.muted }, splitLine: { show: false } },
    ],
    series: [
      { name: "Prefill ms", type: "line", data: sweep.map((s) => +s.prefill.ms.toFixed(2)), itemStyle: { color: C.tensor } },
      { name: "Decode ms/step", type: "line", data: sweep.map((s) => +s.decode.ms.toFixed(3)), itemStyle: { color: C.vector } },
      { name: "Decode tok/s", type: "line", yAxisIndex: 1, data: sweep.map((s) => +s.decode.tokPerSec.toFixed(1)), itemStyle: { color: C.bw } },
    ],
  });
}

export function drawBatch(id, sweep) {
  setChart(id, {
    ...baseTheme(),
    xAxis: { type: "category", data: sweep.map((s) => String(s.batch)), name: "decode batch", axisLabel: { color: C.ink } },
    yAxis: { type: "value", name: "tok/s", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    series: [{
      type: "bar",
      data: sweep.map((s) => ({
        value: +s.decode.tokPerSec.toFixed(1),
        itemStyle: { color: s.decode.primary === "bandwidth" ? C.bw : s.decode.primary === "vector" ? C.vector : C.tensor },
      })),
      barMaxWidth: 28,
    }],
  });
}

export function drawPrecision(id, sweep) {
  setChart(id, {
    ...baseTheme(),
    tooltip: { ...baseTheme().tooltip, axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: sweep.map((s) => s.label), axisLabel: { color: C.ink, fontSize: 11 } },
    yAxis: [
      { type: "value", name: "Prefill ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
      { type: "value", name: "Decode tok/s", axisLabel: { color: C.muted }, splitLine: { show: false } },
    ],
    series: [
      { name: "Prefill ms", type: "bar", data: sweep.map((s) => +s.prefill.ms.toFixed(1)), itemStyle: { color: C.tensor }, barMaxWidth: 22 },
      { name: "Decode tok/s", type: "bar", yAxisIndex: 1, data: sweep.map((s) => +s.decode.tokPerSec.toFixed(1)), itemStyle: { color: C.vector }, barMaxWidth: 22 },
    ],
  });
}

export function drawHeat(id, heat, current = {}) {
  const yLabels = heat.vecRatios.map((v) => `${Math.round(1 / v)}:1`);
  const xLabels = heat.bwRatios.map((b) => String(b));
  const curCv = current.cubeToVector;
  const curBw = current.bwPerMatmul;
  let mark = null;
  if (Number.isFinite(curCv) && Number.isFinite(curBw)) {
    let best = Infinity;
    heat.cells.forEach((c) => {
      const d = Math.abs(1 / c.vecRatio - curCv) + Math.abs(c.bwRatio - curBw);
      if (d < best) {
        best = d;
        mark = c;
      }
    });
  }
  const data = heat.cells.map((c) => ({
    value: [
      heat.bwRatios.indexOf(c.bwRatio),
      heat.vecRatios.indexOf(c.vecRatio),
      +c.tokPerSec.toFixed(1),
    ],
    itemStyle: mark && c.vecRatio === mark.vecRatio && c.bwRatio === mark.bwRatio
      ? { borderColor: C.warn, borderWidth: 2 }
      : undefined,
  }));
  const vals = heat.cells.map((c) => c.tokPerSec);
  setChart(id, {
    ...baseTheme(),
    tooltip: {
      trigger: "item",
      backgroundColor: "#1c2024",
      textStyle: { color: C.ink, fontSize: 12 },
      formatter: (p) => {
        const xi = p.data.value[0];
        const yi = p.data.value[1];
        const c = heat.cells.find(
          (x) => heat.bwRatios.indexOf(x.bwRatio) === xi && heat.vecRatios.indexOf(x.vecRatio) === yi
        );
        if (!c) return "";
        const here = mark && c.vecRatio === mark.vecRatio && c.bwRatio === mark.bwRatio ? " · 当前" : "";
        return `Cube:Vector ${Math.round(1 / c.vecRatio)}:1 · 带宽:Cube ${c.bwRatio}:1${here}<br/>${fmtTps(c.tokPerSec)} · ${fmtMs(c.ms)}/step<br/>墙 ${boundLabel(c.primary)}`;
      },
    },
    grid: { left: 72, right: 80, top: 28, bottom: 48 },
    xAxis: {
      type: "category",
      data: xLabels,
      name: "带宽:Cube",
      axisLabel: { color: C.ink },
      nameTextStyle: { color: C.muted },
    },
    yAxis: {
      type: "category",
      data: yLabels,
      name: "Cube:Vector",
      axisLabel: { color: C.ink },
      nameTextStyle: { color: C.muted },
    },
    visualMap: {
      min: Math.min(...vals),
      max: Math.max(...vals),
      calculable: true,
      orient: "vertical",
      right: 8,
      top: 40,
      text: ["快", "慢"],
      inRange: { color: ["#e08a6a", "#d4a574", "#7eb8b0"] },
      textStyle: { color: C.muted },
    },
    series: [{
      type: "heatmap",
      data,
      label: {
        show: true,
        color: C.ink,
        fontSize: 10,
        formatter: (p) => Math.round(p.data.value[2]),
      },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
    }],
  });
}

export function drawChips(id, rows) {
  const sorted = [...rows].sort((a, b) => b.decodeTps - a.decodeTps);
  setChart(id, {
    ...baseTheme(),
    grid: { left: 210, right: 16, top: 28, bottom: 28 },
    tooltip: { trigger: "item", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    xAxis: { type: "value", name: "Decode tok/s", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    yAxis: { type: "category", data: sorted.map((r) => r.name), axisLabel: { color: C.ink, fontSize: 11 } },
    series: [{
      type: "bar",
      data: sorted.map((r) => ({
        value: r.oom ? 0 : +r.decodeTps.toFixed(1),
        itemStyle: { color: r.oom ? C.warn : C.tensor },
      })),
      barMaxWidth: 14,
    }],
  });
}

export function drawSens(id, rows) {
  setChart(id, {
    ...baseTheme(),
    grid: { left: 88, right: 16, top: 28, bottom: 28 },
    tooltip: { trigger: "axis", backgroundColor: "#1c2024", textStyle: { color: C.ink } },
    xAxis: {
      type: "value",
      name: "时间变化",
      axisLabel: { color: C.muted, formatter: (v) => `${(v * 100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: C.line } },
    },
    yAxis: { type: "category", data: rows.map((r) => r.label), axisLabel: { color: C.ink } },
    series: [
      { name: "+20% 资源 → 变快", type: "bar", data: rows.map((r) => +r.dUp.toFixed(3)), itemStyle: { color: C.tensor } },
      { name: "-20% 资源 → 变慢", type: "bar", data: rows.map((r) => +r.dDown.toFixed(3)), itemStyle: { color: C.warn } },
    ],
  });
}

export function drawLayers(id, result) {
  const n = 64;
  const mix = new Array(n).fill(0);
  const ffn = new Array(n).fill(0);
  for (const op of result.ops) {
    if (op.layer < 0) continue;
    if (op.family === "ffn" || op.family === "residual" || op.family === "norm") ffn[op.layer] += op.t * 1e3;
    else mix[op.layer] += op.t * 1e3;
  }
  setChart(id, {
    ...baseTheme(),
    tooltip: { ...baseTheme().tooltip, axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: [...Array(n).keys()].map(String), axisLabel: { color: C.muted, interval: 7 } },
    yAxis: { type: "value", name: "ms", axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.line } } },
    series: [
      { name: "混合层 (GDN/Attn)", type: "bar", stack: "l", data: mix, itemStyle: { color: C.vector }, barMaxWidth: 8 },
      { name: "FFN+Norm", type: "bar", stack: "l", data: ffn, itemStyle: { color: C.tensor }, barMaxWidth: 8 },
    ],
  });
}

export function drawBound(id, result) {
  setChart(id, {
    ...baseTheme(),
    series: [{
      type: "pie",
      radius: ["40%", "66%"],
      label: { color: C.ink, formatter: "{b}\n{c} 个算子" },
      data: [
        { name: "MatMul 墙", value: result.boundCounts.tensor, itemStyle: { color: C.tensor } },
        { name: "Vector 墙", value: result.boundCounts.vector, itemStyle: { color: C.vector } },
        { name: "带宽墙", value: result.boundCounts.bandwidth, itemStyle: { color: C.bw } },
      ],
    }],
  });
}

export { FAMILY_LABEL, C };
