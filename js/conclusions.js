import { chipRatios } from "./chips.js";
import { PRECISIONS, peakTflops, gemmComputeFallback } from "./engine.js";

function pct(x) {
  return `${(x * 100).toFixed(0)}%`;
}

function fmtMs(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(1)} µs`;
  if (ms < 1000) return `${ms.toFixed(2)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtTps(x) {
  if (x >= 1000) return `${(x / 1000).toFixed(1)}k tok/s`;
  return `${x.toFixed(1)} tok/s`;
}

function boundLabel(b) {
  return { tensor: "Cube 算力墙", vector: "Vector 算力墙", bandwidth: "HBM 带宽墙", balanced: "接近配平" }[b] || b;
}

export function shortBound(b) {
  return { tensor: "Cube 墙", vector: "Vector 墙", bandwidth: "带宽墙", balanced: "配平" }[b] || b;
}

export function buildConclusions(model, chip, workload, pair) {
  const r = chipRatios(chip);
  const prec = PRECISIONS[workload.precision];
  const gemmDt = gemmComputeFallback(chip, prec.gemmDtype);
  const { prefill, decode } = pair;
  const lines = [];

  lines.push({
    title: "当前切片",
    body: `${chip.vendor} ${chip.name} · ${prec.label} · Prefill B=${workload.batch} S=${workload.seq} → ${fmtMs(prefill.ms)}（${fmtTps(prefill.tokPerSec)}）· Decode B=${workload.batch} C=${workload.context} → ${fmtMs(decode.ms)}/step（${fmtTps(decode.tokPerSec)}）。`,
  });

  lines.push({
    title: "芯片配比",
    body: (() => {
      let s = `Cube:Vector (BF16) = ${r.cubeToVector.toFixed(1)}:1；带宽(GB/s):Cube BF16 = ${r.bwPerMatmul.toFixed(2)}:1。Ridge：BF16 ${r.ridgeBf16.toFixed(1)} FLOP/byte`;
      if (chip.fp8Tflops > 0) s += `，FP8 ${r.ridgeFp8.toFixed(1)}`;
      if (chip.fp4Tflops > 0) s += `，FP4 ${r.ridgeFp4.toFixed(1)}`;
      else if (workload.precision === "nvfp4") s += "。无原生 FP4 时 NVFP4 投影按 BF16 峰值算、权重仍按 4.5 bit 搬运";
      return s + "。";
    })(),
  });

  if (gemmDt !== prec.gemmDtype) {
    lines.push({
      title: "精度回退",
      body: `这颗芯片没有原生 ${prec.gemmDtype.toUpperCase()} Tensor 峰值，投影 GEMM 回退到 ${gemmDt.toUpperCase()} 算力，只吃到存储位宽下降。Prefill 加速会明显小于 Decode。`,
    });
  }

  const pShare = {
    tensor: prefill.tTensor / prefill.t,
    vector: prefill.tVector / prefill.t,
    bw: prefill.tBw / prefill.t,
  };
  const dShare = {
    tensor: decode.tTensor / decode.t,
    vector: decode.tVector / decode.t,
    bw: decode.tBw / decode.t,
  };

  lines.push({
    title: "Prefill 瓶颈",
    body: `主瓶颈是${boundLabel(prefill.primary)}。三条墙相对端到端时间（可重叠）：MatMul ${pct(pShare.tensor)}、Vector ${pct(pShare.vector)}、带宽 ${pct(pShare.bw)}。FlashAttention 出现在 ${prefill.inv.nAttn}/${model.numLayers} 层，softmax 的 Vector 压力按 hybrid 比例缩放。`,
  });

  lines.push({
    title: "Decode 瓶颈",
    body: `主瓶颈是${boundLabel(decode.primary)}。三条墙相对端到端时间（可重叠）：MatMul ${pct(dShare.tensor)}、Vector ${pct(dShare.vector)}、带宽 ${pct(dShare.bw)}。Batch=${workload.batch} 时投影接近 GEMV，算术强度大约是「token 数 / 每权重字节」。`,
  });

  const vecRatioNeededPrefill = prefill.tVector > 0 && prefill.tTensor > 0
    ? (chip.vectorTflops * (prefill.tVector / prefill.tTensor)) / chip.bf16Tflops
    : r.vectorToMatmul;
  const vecAsOneTo = 1 / Math.max(r.vectorToMatmul, 1e-9);

  if (pShare.vector > 0.35) {
    lines.push({
      title: "Vector 配比偏紧（Prefill）",
      body: `Prefill 里 Vector 墙已经吃掉 ${pct(pShare.vector)} 的时间。当前 1:${vecAsOneTo.toFixed(1)} 在 S=${workload.seq} 下不够挡住 softmax / Norm / GDN gate。若希望 Vector 不再当主瓶颈，需要把 Vector:MatMul 抬到大约 1:${(1 / Math.max(vecRatioNeededPrefill, 1e-6)).toFixed(0)} 附近（或给 softmax 单独 SFU）。`,
    });
  } else {
    lines.push({
      title: "Vector 配比（Prefill）",
      body: `Prefill 中 Vector 约占 ${pct(pShare.vector)}，当前 1:${vecAsOneTo.toFixed(1)} 还没有把 RMSNorm/SiLU/softmax 推成主瓶颈。若把 Vector 降到约 1:${Math.round(vecAsOneTo * 2)}，长序列上 softmax 更可能翻成主瓶颈。`,
    });
  }

  if (dShare.bw > 0.55) {
    const ridgeNeed = 2 / bitsToBytesSafe(prec.weightBits);
    lines.push({
      title: "Decode 吃带宽",
      body: `Decode 时间的 ${pct(dShare.bw)} 花在搬运上。权重约 ${prec.weightBits} bit，要把 Decode 投影推上算力墙，需要有效 token 批宽大约 ≥ ${ridgeNeed.toFixed(0)} × ridge(${gemmDt}) / 2。当前 ridge≈${((peakTflops(chip, gemmDt) * 1e3) / Math.max(chip.hbmBandwidthGBs, 1e-9)).toFixed(0)} FLOP/byte，B=${workload.batch} 远低于这个点——加大 batch 或加带宽，加 MatMul 峰值几乎不动 Decode。`,
    });
  } else if (dShare.tensor > 0.5) {
    lines.push({
      title: "Decode 已经碰到算力墙",
      body: `当前 batch=${workload.batch} 已经让部分投影的强度够到 ${gemmDt.toUpperCase()} 峰值。再加带宽收益变小，加 MatMul 或提高精度峰值才会涨 tok/s。`,
    });
  }

  const ffnT = (prefill.byFamily.ffn?.t || 0) / prefill.t;
  const attnT = ((prefill.byFamily.attn_proj?.t || 0) + (prefill.byFamily.flashattn?.t || 0)) / prefill.t;
  const gdnT = ((prefill.byFamily.gdn_proj?.t || 0) + (prefill.byFamily.gdn_state?.t || 0) + (prefill.byFamily.gdn_conv?.t || 0)) / prefill.t;
  lines.push({
    title: "结构：不是「每层都是 Attention」",
    body: `Prefill 时间拆开：FFN ${pct(ffnT)}、GDN ${pct(gdnT)}、全注意力(投影+FA) ${pct(attnT)}。64 层里 48 层线性注意力状态与 seq 无关（约 ${decode.mem.stateGB.toFixed(2)} GB），16 层 KV 随 C 涨（当前 ${decode.mem.kvGB.toFixed(2)} GB）。带宽规划按 KV + GDN state 一起算。`,
  });

  if (decode.mem.oom || prefill.mem.oom) {
    lines.push({
      title: "容量不够",
      body: `工作集约 ${decode.mem.totalGB.toFixed(1)} GB，超过 ${chip.hbmCapacityGB} GB 显存。权重 ${decode.mem.weightGB.toFixed(1)} + KV ${decode.mem.kvGB.toFixed(1)} + GDN state ${decode.mem.stateGB.toFixed(2)}。需要降 batch/上下文、更低 KV 位宽，或多卡切分。`,
    });
  }

  const bwIdealDecode = suggestBwRatio(decode, chip, prec);
  const vecIdealPrefill = suggestVecRatio(prefill, chip);
  lines.push({
    title: "理论配平参考（本切片）",
    body: `Decode B=${workload.batch} C=${workload.context}：带宽:BF16-TFLOPS 大约 ${bwIdealDecode.toFixed(1)}:1 时三条墙比较接近。Prefill S=${workload.seq}：Vector:MatMul 大约 1:${vecIdealPrefill.toFixed(0)} 时 Vector 时间仍低于 MatMul。配平带随 batch/seq/精度移动，以热力图为准。`,
  });

  return {
    lines,
    kpis: {
      prefillMs: prefill.ms,
      decodeMs: decode.ms,
      decodeTps: decode.tokPerSec,
      prefillBound: prefill.primary,
      decodeBound: decode.primary,
      vecRatio: r.vectorToMatmul,
      bwRatio: r.bwPerMatmul,
      oom: decode.mem.oom,
      weightGB: decode.mem.weightGB,
      kvGB: decode.mem.kvGB,
      stateGB: decode.mem.stateGB,
    },
  };
}

function bitsToBytesSafe(bits) {
  return bits / 8;
}

function suggestBwRatio(result, chip, prec) {
  const bytesPerBf16FlopPeak = result.bytes / Math.max(chip.bf16Tflops * 1e12 * result.t, 1);
  const current = chip.hbmBandwidthGBs / chip.bf16Tflops;
  if (result.primary === "bandwidth") return Math.max(current * 1.4, current);
  if (result.primary === "tensor") return Math.max(0.5, current * 0.7);
  return current;
}

function suggestVecRatio(result, chip) {
  const currentOneTo = chip.bf16Tflops / Math.max(chip.vectorTflops, 1e-9);
  if (result.tVector > result.tTensor * 0.8) return Math.max(4, currentOneTo * 0.6);
  return Math.max(6, currentOneTo);
}

export { fmtMs, fmtTps, boundLabel, pct };
