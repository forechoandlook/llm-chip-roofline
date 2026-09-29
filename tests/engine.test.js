import { MODELS, gdnDims, attnDims, layerTypes } from "../js/models.js";
import {
  paramInventory,
  simulate,
  simulatePair,
  defaultWorkload,
  PRECISIONS,
  expandOps,
  pipelineTime,
  opTable,
  tableTotal,
  supportsPrecision,
  sweepVecRatio,
  sweepBwRatio,
  sweepHeatmap,
} from "../js/engine.js";
import { chipById, chipRatios } from "../js/chips.js";
import { buildExplain, renderExplainHtml } from "../js/explain.js";

const model = MODELS["qwen3.8-27b"];
let failed = 0;
let passed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed += 1;
    console.error("FAIL:", msg);
  } else {
    passed += 1;
  }
}

function close(a, b, rel, msg) {
  const ok = Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));
  assert(ok, `${msg}: got ${a}, expected ${b}`);
}

{
  const types = layerTypes(model);
  assert(types.length === 64, "64 layers");
  assert(types.filter((t) => t === "gdn").length === 48, "48 GDN");
  assert(types.filter((t) => t === "attn").length === 16, "16 attn");
  assert(types[3] === "attn" && types[0] === "gdn", "pattern 3 GDN then 1 attn");
}

{
  const g = gdnDims(model);
  assert(g.keyDim === 2048, "gdn key dim");
  assert(g.valueDim === 6144, "gdn value dim");
  assert(g.convDim === 10240, "gdn conv dim");
  assert(g.stateElems === 48 * 128 * 128, "gdn state elems per layer");
}

{
  const a = attnDims(model);
  assert(a.qDim === 6144, "q dim 24*256");
  assert(a.kvDim === 1024, "kv dim 4*256");
  assert(a.groupSize === 6, "GQA 24/4");
}

{
  const inv = paramInventory(model);
  close(inv.ffn, 64 * 3 * 5120 * 17408, 0, "FFN params");
  close(inv.kvElemsPerToken, 16 * 4 * 256 * 2, 0, "KV elems/token");
  close(inv.gdnStateElems, 48 * 48 * 128 * 128, 0, "GDN state elems");
  assert(inv.total > 2.6e10 && inv.total < 2.9e10, `total params ~27B, got ${inv.total}`);
}

{
  const chip = chipById("custom");
  const wl = { ...defaultWorkload(), phase: "decode", batch: 4, context: 8192, precision: "bf16" };
  const inv = paramInventory(model);
  const kvBytes = inv.kvElemsPerToken * 8192 * 4 * 2;
  const r = simulate(model, chip, wl);
  close(r.mem.kvGB, kvBytes / 1e9, 0.02, "decode KV GB bf16");
  close(r.mem.stateGB, (4 * (inv.gdnStateElems * 4 + inv.gdnConvStateElems * 2)) / 1e9, 0.02, "GDN rec FP32 + conv BF16");
}

{
  const chip = chipById("b300-sxm");
  const pair = simulatePair(model, chip, {
    ...defaultWorkload(),
    batch: 4,
    seq: 8192,
    context: 8192,
    precision: "nvfp4",
  });
  assert(pair.prefill.t > pair.decode.t, "prefill 8K takes longer than one decode step");
  assert(pair.decode.primary === "bandwidth" || pair.decode.utilBw > 0.4, "decode is bandwidth-ish at B=4");
  const ffn = pair.decode.byFamily.ffn.t;
  const fa = pair.decode.byFamily.flashattn.t;
  assert(ffn > fa, `decode FFN (${ffn}) should exceed FA (${fa}) at 8K B=4`);
}

{
  const chip = chipById("b300-sxm");
  const d8k = simulate(model, chip, { ...defaultWorkload(), phase: "decode", context: 8192, batch: 4 });
  const d128k = simulate(model, chip, { ...defaultWorkload(), phase: "decode", context: 131072, batch: 4 });
  assert(d128k.t > d8k.t, "longer context decode is slower");
  const kvGrowth = d128k.mem.kvGB / d8k.mem.kvGB;
  close(kvGrowth, 131072 / 8192, 0.01, "KV scales with C");
  close(d128k.mem.stateGB, d8k.mem.stateGB, 0.01, "GDN state independent of C");
}

{
  const chip = chipById("custom");
  const ops = expandOps(model, chip, { ...defaultWorkload(), phase: "prefill", seq: 1024, batch: 1 });
  const fa = ops.filter((o) => o.name === "flash_attn");
  assert(fa.length === 16, "16 FA ops");
  const gdnState = ops.filter((o) => o.name === "gdn_delta_rule");
  assert(gdnState.length === 48, "48 GDN state ops");
  const g0 = gdnState[0];
  const M = 1024;
  const d2 = 48 * 128 * 128;
  close(g0.flopsTensor, 6 * d2 * M, 0, "GDN Cube BF16 6 M Vh D²");
  close(g0.flopsVector, 2 * d2 * M, 0, "GDN Vec FP32 2 M Vh D² (decay)");
}

{
  const weakVec = chipById("custom");
  weakVec.vectorTflops = 1;
  const strongVec = chipById("custom");
  strongVec.vectorTflops = 500;
  const wl = { ...defaultWorkload(), phase: "prefill", seq: 8192, batch: 1, precision: "bf16" };
  const a = simulate(model, weakVec, wl);
  const b = simulate(model, strongVec, wl);
  assert(a.t > b.t, "more vector tflops should not slow prefill");
}

{
  const chip = chipById("910c");
  const r = simulate(model, chip, { ...defaultWorkload(), phase: "prefill", precision: "nvfp4" });
  const q = r.ops.find((o) => o.name === "gate_proj");
  assert(q.computeDtype === "bf16", "910C has no FP4, GEMM falls back to bf16");
  assert(q.flopsVector > 0, "910C NVFP4 fallback still dequants weights into BF16");
  assert(!supportsPrecision(chip, "nvfp4"), "910C should not enter NVFP4 compare chart");
  assert(!supportsPrecision(chip, "mxfp8"), "910C has no FP8 Tensor");
  assert(supportsPrecision(chip, "bf16"), "910C is valid in BF16");
  assert(supportsPrecision(chipById("h100-sxm"), "mxfp8"), "H100 has FP8");
  assert(!supportsPrecision(chipById("h100-sxm"), "nvfp4"), "H100 has no FP4");
  assert(supportsPrecision(chipById("b300-sxm"), "nvfp4"), "B300 has FP4");
  const rtx = chipById("rtx-5090");
  assert(supportsPrecision(rtx, "nvfp4"), "5090 has FP4");
  const p6 = chipById("rtx-pro-6000");
  assert(supportsPrecision(p6, "nvfp4"), "PRO 6000 has FP4");
  close(rtx.hbmBandwidthGBs, 1792, 0, "5090 GDDR7 1.792 TB/s");
  close(p6.hbmCapacityGB, 96, 0, "PRO 6000 96GB");
}

{
  const chip = chipById("b300-sxm");
  const r = simulate(model, chip, { ...defaultWorkload(), phase: "prefill", precision: "nvfp4" });
  const q = r.ops.find((o) => o.name === "gate_proj");
  assert(q.computeDtype === "fp4", "B300 NVFP4 GEMM uses fp4 peak");
  close(q.flopsVector, 2 * q.M * q.K, 0, "NVFP4 weight GEMM: Vec 2MK (global scale already known)");
  const wPayload = q.K * q.N * 0.5;
  const wScale = (q.K * q.N / 16) * 1;
  const expectB = q.M * q.K * 2 + wPayload + wScale + 4 + q.M * q.N * 2;
  close(q.bytes, expectB, 0, "NVFP4 GEMM bytes: BF16 act + W4 + scale/16 + BF16 out");
  assert(q.formula.includes("W4"), `formula explains traffic, got ${q.formula}`);
  const fa = r.ops.find((o) => o.name === "flash_attn");
  assert(fa.computeDtype === "bf16", "FA stays bf16");
  const gdn = r.ops.find((o) => o.name === "gdn_delta_rule");
  assert(gdn.unitHint === "bf16", "GDN matmul path is BF16 Cube");
  assert(gdn.vectorHint === "fp32", "GDN decay is FP32 Vector");
  assert(gdn.tCube > 0 && gdn.tVector > 0, "both Cube and Vector times");
}

{
  const chip = chipById("custom");
  const wl = { ...defaultWorkload(), phase: "decode", batch: 4, context: 8192, precision: "nvfp4" };
  const result = simulate(model, chip, wl);
  const ex = buildExplain(model, chip, wl, result);
  assert(ex.d.M === 4, `decode M=B*S got ${ex.d.M}`);
  assert(ex.d.C === 8192, "decode C");
  assert(ex.types.filter((t) => t === "gdn").length === 48, "explain types");
  const html = renderExplainHtml(ex);
  assert(html.strip.includes("L3") && html.strip.includes("cell-attn"), "layer 3 is attn");
  assert(html.config.includes("5120"), "config shows hidden");
  assert(html.walks.includes("2·M·K·N") || html.walks.includes("2·M"), "gemm formula present");
  assert(html.timing.includes("T_tensor"), "timing formula");
  const pre = simulate(model, chip, { ...wl, phase: "prefill", seq: 8192, batch: 4 });
  const exP = buildExplain(model, chip, { ...wl, phase: "prefill" }, pre);
  assert(exP.d.M === 4 * 8192, `prefill M got ${exP.d.M}`);
}

{
  const r = chipRatios(chipById("custom"));
  close(r.ridgeBf16, 125, 0.01, "custom ridge BF16 FLOP/byte");
  close(r.vectorToMatmul, 0.125, 0.01, "custom Cube:Vector BF16 8:1");
  close(r.bwPerMatmul, 8, 0.01, "custom 8:1 bandwidth");
}

{
  const a100 = chipById("a100-sxm");
  close(a100.bf16Tflops, 312, 0, "A100 Cube BF16");
  close(a100.vectorTflops, 39, 0, "A100 Vector BF16 (CUDA non-tensor)");
  close(a100.bf16Tflops / a100.vectorTflops, 8, 0.01, "A100 Cube:Vector BF16");
  assert(a100.sourceUrl && a100.sourceUrl.startsWith("http"), "A100 has source url");

  const b2 = chipById("910b");
  close(b2.bf16Tflops, 353.8944, 0.001, "910B2 Cube 24×1.8GHz×16³×2");
  close(b2.vectorTflops, 22.1184, 0.001, "910B2 Vector BF16 128×2×48×1.8");
  close(b2.bf16Tflops / b2.vectorTflops, 16, 0.02, "910B2 Cube:Vector BF16");
  assert(b2.bf16Tflops < 370, "910B2 is not the 400T marketing SKU");
  assert(b2.sourceUrl && b2.sourceUrl.startsWith("http"), "910B2 has source url");
}

{
  const t = pipelineTime(10, 3, 8, 0.2);
  close(t, 10 + 0.2 * 8, 1e-9, "γ=0: max(C,V) then overlap BW");
  const t2 = pipelineTime(2, 3, 10, 0.2);
  close(t2, 10 + 0.2 * 3, 1e-9, "BW wall: max(3,10)+0.2*3");
  const t3 = pipelineTime(10, 3, 8, 0.2, 0.4);
  close(t3, 11.2 + 0.2 * 8, 1e-9, "γ=0.4: Tcomp=10+0.4*3=11.2");
}

{
  const chip = chipById("custom");
  const B = 4;
  const C = 8192;
  const S = 1;
  const ops = expandOps(model, chip, {
    ...defaultWorkload(),
    phase: "decode",
    batch: B,
    context: C,
    decodeTokens: S,
    precision: "nvfp4",
  });
  const fa = ops.find((o) => o.name === "flash_attn");
  const kvB = PRECISIONS.nvfp4.kvBits / 8;
  const kvRead = B * 4 * C * 256 * 2 * kvB;
  const kvWrite = B * 4 * S * 256 * 2 * kvB;
  close(fa.kvReadBytes, kvRead, 0, "decode FA reads full KV cache");
  close(fa.kvWriteBytes, kvWrite, 0, "decode FA writes only new token KV");
  assert(fa.kvReadBytes > fa.kvWriteBytes * 1000, "cache read dwarfs new-token write at 8K");
  assert(fa.shape.includes("C=8192"), `decode FA shape shows context, got ${fa.shape}`);
  assert(fa.shape.includes("S=1"), `decode FA shape shows S=1, got ${fa.shape}`);
  close(fa.flopsTensor, 4 * B * 24 * S * C * 256, 0, "decode FA cube FLOPs attend over C");
}

{
  const chip = chipById("custom");
  const pair = simulatePair(model, chip, {
    ...defaultWorkload(),
    batch: 1,
    decodeBatch: 8,
    seq: 1024,
    context: 2048,
    precision: "bf16",
  });
  const pg = pair.prefill.ops.find((o) => o.name === "gate_proj");
  const dg = pair.decode.ops.find((o) => o.name === "gate_proj");
  assert(pg.shape.startsWith("1024×"), `prefill uses Bp=1, got ${pg.shape}`);
  assert(dg.shape.startsWith("8×"), `decode uses Bd=8, got ${dg.shape}`);
}

{
  const chip = chipById("custom");
  const B = 4;
  const S = 8192;
  const ops = expandOps(model, chip, {
    ...defaultWorkload(),
    phase: "prefill",
    batch: B,
    seq: S,
    precision: "nvfp4",
  });
  const fa = ops.find((o) => o.name === "flash_attn");
  const kvB = PRECISIONS.nvfp4.kvBits / 8;
  const kv = B * 4 * S * 256 * 2 * kvB;
  close(fa.kvReadBytes, kv, 0, "prefill FA KV read = full S");
  close(fa.kvWriteBytes, kv, 0, "prefill FA KV write = full S");
  const gate = ops.find((o) => o.name === "gate_proj");
  assert(gate.shape === `${B * S}×5120×17408`, `FFN GEMM shape, got ${gate.shape}`);
}

{
  const chip = chipById("custom");
  const d8k = simulate(model, chip, { ...defaultWorkload(), phase: "decode", context: 8192, batch: 4 });
  const d32k = simulate(model, chip, { ...defaultWorkload(), phase: "decode", context: 32768, batch: 4 });
  const fa8 = d8k.ops.filter((o) => o.name === "flash_attn").reduce((s, o) => s + o.bytes, 0);
  const fa32 = d32k.ops.filter((o) => o.name === "flash_attn").reduce((s, o) => s + o.bytes, 0);
  assert(fa32 > fa8 * 3, `FA bytes should grow with KV cache: 32k=${fa32} 8k=${fa8}`);
  const ffn8 = d8k.ops.filter((o) => o.family === "ffn").reduce((s, o) => s + o.bytes, 0);
  const ffn32 = d32k.ops.filter((o) => o.family === "ffn").reduce((s, o) => s + o.bytes, 0);
  close(ffn8, ffn32, 0.001, "FFN bytes independent of KV cache length");
}

{
  const chip = chipById("custom");
  const r = simulate(model, chip, { ...defaultWorkload(), phase: "decode", batch: 4, context: 8192 });
  const rows = opTable(r);
  const fa = rows.find((x) => x.name === "flash_attn");
  const gate = rows.find((x) => x.name === "gate_proj");
  assert(fa && fa.n === 16, `flash_attn aggregated over 16 layers, got ${fa && fa.n}`);
  assert(gate && gate.n === 64, `gate_proj aggregated over 64 layers, got ${gate && gate.n}`);
  assert(fa.shape.includes("C=8192"), "table keeps FA shape");
  assert(gate.shape.includes("×"), "table keeps GEMM shape");
  close(fa.t, fa.tPipe, 1e-12, "row tPipe is pipeline time");
  assert(fa.tCube >= 0 && fa.tVector >= 0 && fa.tBw >= 0, "cube/vector/bw times present");
  assert(gate.flopsCube > 0 && gate.bytes > 0, "table has cube FLOPs and bytes");
}

{
  const chip = chipById("custom");
  const r = simulate(model, chip, { ...defaultWorkload(), phase: "prefill", batch: 4, seq: 8192 });
  const tot = tableTotal(opTable(r));
  assert(tot.name === "total", "total row name");
  close(tot.tPipe, r.t, 1e-9, "total 流水 = simulate t");
  close(tot.tCube, r.tTensor, 1e-9, "total cube time");
  close(tot.tVector, r.tVector, 1e-9, "total vector time");
  close(tot.tBw, r.tBw, 1e-9, "total bw time");
  close(tot.flopsCube, r.flopsT, 1e-9, "total cube FLOPs");
  close(tot.flopsVector, r.flopsV, 1e-9, "total vector FLOPs");
  close(tot.bytes, r.bytes, 1e-9, "total bytes");
  assert(tot.n === r.ops.length, `total layers/ops ${tot.n} vs ${r.ops.length}`);
}

{
  const chip = chipById("custom");
  const wl = defaultWorkload();
  const vec = sweepVecRatio(model, chip, wl, [8, 2]);
  assert(vec.length === 2, "vec sweep points");
  close(vec[0].cube, chip.bf16Tflops, 1e-6, "fixed Cube at 8:1");
  close(vec[1].cube, chip.bf16Tflops, 1e-6, "fixed Cube at 2:1");
  const rel = Math.abs(vec[1].prefillMs - vec[0].prefillMs) / vec[0].prefillMs;
  assert(rel < 0.05, `fixed Cube: Prefill barely moves with Vector (${rel})`);
  const iso = sweepVecRatio(model, chip, wl, [8, 2], { isoBudget: true });
  assert(iso[1].prefillMs > iso[0].prefillMs, "iso-budget opt-in: more Vector share (less Cube) slows prefill");
  const bw = sweepBwRatio(model, chip, wl, [2, 8]);
  assert(bw[1].decodeTps > bw[0].decodeTps * 1.5, "more HBM per Cube lifts decode");
}

{
  const chip = chipById("custom");
  const wl = { ...defaultWorkload(), precision: "w4bf16" };
  const heat = sweepHeatmap(model, chip, wl, {
    phase: "decode",
    vecRatios: [1 / 8, 1 / 32],
    bwRatios: [4, 8],
  });
  const cell = (vr, br) => heat.cells.find((c) => c.vecRatio === vr && c.bwRatio === br);
  const tightSkinny = cell(1 / 32, 4);
  const defaultSkinny = cell(1 / 32, 8);
  const defaultFat = cell(1 / 8, 8);
  assert(tightSkinny.primary === "bandwidth", `W4 skinny Vector + tight HBM stays bandwidth (${tightSkinny.primary})`);
  assert(defaultSkinny.primary === "vector", `W4 skinny Vector + default HBM is vector (${defaultSkinny.primary})`);
  assert(defaultFat.primary === "bandwidth", `W4 8:1 at default HBM stays bandwidth (${defaultFat.primary})`);
  assert(defaultFat.tokPerSec > defaultSkinny.tokPerSec, "fatter Vector at same HBM lifts W4 decode once Vector is the wall");
}

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
