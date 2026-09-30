import { layerTypes, gdnDims, attnDims } from "./models.js";

export const PRECISIONS = {
  nvfp4: {
    id: "nvfp4",
    label: "NVFP4 (W4A4)",
    weightBits: 4.5,
    actBits: 4.5,
    kvBits: 16,
    gemmDtype: "fp4",
    faCompute: "bf16",
    dequantFlopsPerWeight: 0,
    fallbackDequantPerWeight: 3,
    /**
     * 激活动态量化在 CUDA、FP32：
     *   s_block = (block_amax / 6) / s_global
     *   x_fp4 = round(x / (s_global * s_block))
     * 每 16 个输入：16 次 abs、15 次 max、2 次除法；每个输入再 1 次除法。
     * MatMul 走 FP4 Cube，块 scale 在 Tensor Core。累加输出按 FP32；
     * 张量级 s_global 在 FP32 Vector 上乘到 MN 个输出。
     */
    actQuantFlopsPerElem: 0,
    gemmActBytes: 2,
    gemmOutBytes: 2,
    gemmWeightBits: 4,
    weightScaleGroup: 16,
    weightScaleBytes: 1,
    actScaleBytes: 4,
  },
  mxfp8: {
    id: "mxfp8",
    label: "MXFP8 (W8A8)",
    weightBits: 8.25,
    actBits: 8.25,
    kvBits: 16,
    gemmDtype: "fp8",
    faCompute: "bf16",
    dequantFlopsPerWeight: 0,
    fallbackDequantPerWeight: 1,
    actQuantFlopsPerElem: 0,
    gemmActBytes: 2,
    gemmOutBytes: 2,
    gemmWeightBits: 8,
    weightScaleGroup: 32,
    weightScaleBytes: 1,
    actScaleBytes: 0,
  },
  w4bf16: {
    id: "w4bf16",
    label: "W4BF16 (权重量化)",
    weightBits: 4.125,
    actBits: 16,
    kvBits: 16,
    gemmDtype: "bf16",
    faCompute: "bf16",
    dequantFlopsPerWeight: 3,
  },
  bf16: {
    id: "bf16",
    label: "BF16 dense",
    weightBits: 16,
    actBits: 16,
    kvBits: 16,
    gemmDtype: "bf16",
    faCompute: "bf16",
    dequantFlopsPerWeight: 0,
  },
};

const VECTOR_COEFF = {
  rmsnorm: 8,
  silu: 4,
  mul: 1,
  add: 1,
  sigmoid: 4,
  rope: 6,
  l2norm: 6,
};

/** Online softmax 每个 score 的 FP32 Vector：running max、两次减、两次 exp、l 的乘加、最后除。exp 按 8 次 FMA 估。 */
const FA_SOFTMAX_PER_SCORE =
  1 + // running max
  2 + // s − m_new, m_old − m_new
  2 * 8 + // two exp
  1 + // l *= exp(Δm)
  1 + // l += exp(s−m)
  1; // p /= l
const FA_KV_BLOCK = 64;

export function defaultWorkload() {
  return {
    phase: "both",
    batch: 4,
    decodeBatch: 4,
    seq: 8192,
    context: 8192,
    decodeTokens: 1,
    precision: "nvfp4",
    overlapAlpha: 0.2,
    faOverlapAlpha: 0,
    /** Cube∥Vector: 0=完全并行(max)，1=完全串行(sum)。默认 0.4，不完全独立。 */
    cubeVectorGamma: 0.4,
    /** RMS/SiLU/残差等 epilogue 按这个比例计 HBM（片上为主）。 */
    epilogueHbmFrac: 0.2,
    gdnStateUnit: "vector",
    lmHeadTokens: "last",
    stateBytes: 2,
  };
}

export function peakTflops(chip, dtype) {
  if (dtype === "fp4") return chip.fp4Tflops > 0 ? chip.fp4Tflops : chip.bf16Tflops;
  if (dtype === "fp8") return chip.fp8Tflops > 0 ? chip.fp8Tflops : chip.bf16Tflops;
  if (dtype === "bf16") return chip.bf16Tflops;
  return chip.fp32Tflops || chip.vectorTflops;
}

export function gemmComputeFallback(chip, dtype) {
  if (dtype === "fp4" && !(chip.fp4Tflops > 0)) return "bf16";
  if (dtype === "fp8" && !(chip.fp8Tflops > 0)) return "bf16";
  return dtype;
}

/** Native Tensor peak for this precision's GEMM dtype (not a BF16 fallback). */
export function supportsPrecision(chip, precision) {
  const dtype = (PRECISIONS[precision] || PRECISIONS.bf16).gemmDtype;
  if (dtype === "fp4") return chip.fp4Tflops > 0;
  if (dtype === "fp8") return chip.fp8Tflops > 0;
  return true;
}

function bitsToBytes(bits) {
  return bits / 8;
}

function gemm(M, K, N, wBits, aBits, extra = {}) {
  const flops = 2 * M * K * N;
  const bytes =
    extra.bytes != null
      ? extra.bytes
      : M * K * bitsToBytes(aBits) +
        K * N * bitsToBytes(wBits) +
        M * N * bitsToBytes(extra.outBits ?? aBits);
  return {
    kind: "gemm",
    M,
    K,
    N,
    shape: `${M}×${K}×${N}`,
    flopsTensor: flops,
    flopsVector: extra.dequantFlops || 0,
    flopsFp32: extra.flopsFp32 || 0,
    bytes,
    weightBytes: extra.weightBytes != null ? extra.weightBytes : K * N * bitsToBytes(wBits),
    weightElems: K * N,
    formula: extra.formula || "",
    actBytes: extra.actBytes || 0,
    scaleBytes: extra.scaleBytes || 0,
    nTiles: extra.nTiles || Math.max(1, Math.ceil(M / 128) * Math.ceil(N / 128)),
  };
}

function vec(flops, bytes, extra = {}) {
  return {
    kind: "vector",
    flopsTensor: extra.flopsTensor || 0,
    flopsVector: flops,
    bytes,
    weightBytes: extra.weightBytes || 0,
    weightElems: extra.weightElems || 0,
    shape: extra.shape || "",
  };
}

function makeOp(layer, family, name, cost, extra = {}) {
  return {
    layer,
    family,
    name,
    ...cost,
    shape: extra.shape || cost.shape || "",
    formula: extra.formula || cost.formula || "",
    kvReadBytes: extra.kvReadBytes ?? cost.kvReadBytes ?? 0,
    kvWriteBytes: extra.kvWriteBytes ?? cost.kvWriteBytes ?? 0,
    unitHint: extra.unitHint || null,
    vectorHint: extra.vectorHint || null,
    fused: extra.fused || false,
  };
}

const EPILOGUE_OPS = new Set([
  "silu_mul", "res_mix", "res_ffn", "rms_pre_mix", "rms_pre_ffn", "final_rms",
  "qk_norm", "rope", "gdn_silu_l2", "gdn_out_gate", "attn_out_gate",
]);

/**
 * Tile 流水：稳态 max(C,V)，头尾气泡 min/nTiles。
 * γ 是即使很多 tile 也藏不住的份额（抢 SM / 依赖）。
 * nTiles=1 且 γ=1 → 完全串行 C+V。
 */
export function pipelineTime(tCube, tVector, tBw, alpha, gamma = 0, nTiles = Infinity) {
  const mn = Math.min(tCube, tVector);
  const mx = Math.max(tCube, tVector);
  const n = Number.isFinite(nTiles) && nTiles > 0 ? nTiles : 1e9;
  const tComp = mx + mn * (gamma + (1 - gamma) / n);
  return Math.max(tComp, tBw) + alpha * Math.min(tComp, tBw);
}

export function paramInventory(model) {
  const H = model.hiddenSize;
  const I = model.intermediateSize;
  const V = model.vocabSize;
  const g = gdnDims(model);
  const a = attnDims(model);
  const nGdn = model.blockRepeat * model.gdnPerBlock;
  const nAttn = model.blockRepeat * model.attnPerBlock;

  const ffnPer = 3 * H * I;
  const gdnPer =
    H * g.convDim +
    H * g.valueDim +
    H * g.baDim +
    g.valueDim * H +
    g.convDim * model.gdn.convKernel;
  const attnGate = model.attn.outputGate === "elementwise" ? H * a.qDim : H * model.attn.numQHeads;
  const attnPer = H * a.qDim + H * a.kvDim * 2 + a.qDim * H + attnGate;
  const embed = V * H;
  const lmHead = model.tiedEmbeddings ? 0 : V * H;

  return {
    ffn: ffnPer * model.numLayers,
    gdn: gdnPer * nGdn,
    attn: attnPer * nAttn,
    embed,
    lmHead,
    total: ffnPer * model.numLayers + gdnPer * nGdn + attnPer * nAttn + embed + lmHead,
    nGdn,
    nAttn,
    ffnPer,
    gdnPer,
    attnPer,
    kvElemsPerToken: nAttn * model.attn.numKVHeads * model.attn.headDim * 2,
    gdnStateElems: nGdn * g.stateElems,
    gdnConvStateElems: nGdn * g.convStateElems,
  };
}

function weightDequantFlops(prec, chip, K, N) {
  const native = gemmComputeFallback(chip, prec.gemmDtype);
  if (prec.gemmDtype === "fp4" || prec.gemmDtype === "fp8") {
    if (native === prec.gemmDtype) return 0;
    return (prec.fallbackDequantPerWeight || 0) * K * N;
  }
  return (prec.dequantFlopsPerWeight || 0) * K * N;
}

function actQuantFlops(prec, M, K) {
  return (prec.actQuantFlopsPerElem || 0) * M * K;
}

function expandLinear(layer, family, name, M, K, N, prec, chip) {
  const gemmDtype = gemmComputeFallback(chip, prec.gemmDtype);
  const dequant = weightDequantFlops(prec, chip, K, N);
  const aq = actQuantFlops(prec, M, K);
  if (prec.gemmActBytes) {
    const wPayload = K * N * bitsToBytes(prec.gemmWeightBits);
    const wScale = (K * N / prec.weightScaleGroup) * prec.weightScaleBytes;
    const aScale = prec.actScaleBytes;
    const aIn = M * K * prec.gemmActBytes;
    const aOut = M * N * prec.gemmOutBytes;
    const bytes = aIn + wPayload + wScale + aScale + aOut;
    const gsz = prec.weightScaleGroup;
    const blocks = (M * K) / gsz;
    const amaxOps = gsz === 16 ? 16 + 15 + 2 : gsz + (gsz - 1) + 1;
    const quantFp32 = blocks * amaxOps + M * K;
    const castIn = 2 * M * K + blocks;
    const outFp32 = M * N;
    const globalScale = prec.actScaleBytes ? M * N : 0;
    const castOut = M * N;
    const quantFormula =
      prec.id === "nvfp4"
        ? `独立算子。FP32 量化：每 16 个 abs+max+(amax/6)/s_global，每元素再除一次，cast BF16→FP32→E2M1。寄存器内完成，不另写 HBM`
        : `独立算子。FP32 量化：每 32 个 abs+max+E8M0，每元素再除一次，cast BF16→FP32→FP8。寄存器内完成，不另写 HBM`;
    const quantOp = makeOp(
      layer,
      family,
      `${name}_quant`,
      {
        kind: "vector",
        flopsTensor: 0,
        flopsVector: 0,
        flopsFp32: quantFp32 + castIn,
        bytes: 0,
        weightBytes: 0,
        weightElems: 0,
        shape: `${M}×${K}`,
        formula: quantFormula,
      },
      { vectorHint: "fp32", unitHint: "fp32" }
    );
    const formula =
      `Cube 2MKN ${prec.id === "nvfp4" ? "FP4" : "FP8"}（块 scale 在 Tensor Core）· 输出 FP32 MN` +
      (globalScale ? " · s_global 乘 MN 走 FP32 Vector，与 Cube 按 tile 流水（γ）" : " · 无张量级 s_global") +
      ` · 写出 FP32→BF16` +
      `${dequant ? " · W dequant" : ""}` +
      ` · 搬 act BF16 MK + W payload + block scale + out BF16 MN`;
    const g = gemm(M, K, N, prec.gemmWeightBits, 16, {
      dequantFlops: dequant + aq,
      flopsFp32: outFp32 + globalScale + castOut,
      bytes,
      weightBytes: wPayload + wScale,
      formula,
      actBytes: aIn,
      scaleBytes: wScale + aScale,
      outBits: 16,
    });
    return [quantOp, makeOp(layer, family, name, g, { unitHint: gemmDtype })];
  }
  const formula =
    prec.id === "bf16"
      ? `Cube 2MKN BF16 dense：权重、激活、写出全是 BF16，无量化、无 s_global`
      : `Cube 2MKN · Vec dequant/quant · 搬 MK·a + KN·w + MN·out`;
  const g = gemm(M, K, N, prec.weightBits, prec.actBits, {
    dequantFlops: dequant + aq,
    formula,
  });
  return [makeOp(layer, family, name, g, { unitHint: gemmDtype })];
}

function pushLinear(ops, layer, family, name, M, K, N, prec, chip) {
  ops.push(...expandLinear(layer, family, name, M, K, N, prec, chip));
}

/** 注意力投影：NVFP4 / MXFP8 下整段 dense BF16。 */
function attnProjPrec(prec) {
  if (prec.id === "nvfp4" || prec.id === "mxfp8") return PRECISIONS.bf16;
  return prec;
}

/**
 * 对齐 SGLang Qwen3.8-NVFP4 / RadixArk：MLP 才是 NVFP4；
 * GDN 的 qkv/z/out 走 FP8，in_proj_ba（衰减/写入门）留 BF16。
 * 复发核本身已是 BF16 Cube + FP32 衰减。
 */
function gdnProjPrec(name, prec) {
  if (prec.id !== "nvfp4" && prec.id !== "mxfp8") return prec;
  if (name === "gdn_in_ba") return PRECISIONS.bf16;
  return PRECISIONS.mxfp8;
}

export function phaseBatch(workload) {
  if (workload.phase === "decode") return Math.max(1, workload.decodeBatch ?? workload.batch ?? 1);
  return Math.max(1, workload.batch ?? 1);
}

export function expandOps(model, chip, workload) {
  const phase = workload.phase === "decode" ? "decode" : "prefill";
  const B = phaseBatch(workload);
  const S = phase === "prefill" ? workload.seq : workload.decodeTokens;
  const C = phase === "prefill" ? workload.seq : workload.context;
  const M = B * S;
  const H = model.hiddenSize;
  const I = model.intermediateSize;
  const prec = PRECISIONS[workload.precision];
  const types = layerTypes(model);
  const gdn = gdnDims(model);
  const att = attnDims(model);
  const bf16B = 2;
  const ops = [];

  const embedRead = B * S * H * bf16B;
  ops.push(
    makeOp(
      -1,
      "embed",
      "embed_lookup",
      vec(0, embedRead, {
        weightBytes: model.vocabSize * H * bf16B,
        weightElems: model.vocabSize * H,
        shape: `${B}×${S}×${H} gather`,
      })
    )
  );

  types.forEach((kind, li) => {
    ops.push(
      makeOp(
        li,
        "norm",
        "rms_pre_mix",
        vec(VECTOR_COEFF.rmsnorm * M * H, 2 * M * H * bf16B, { shape: `${M}×${H}` })
      )
    );

    if (kind === "gdn") {
      pushLinear(ops, li, "gdn_proj", "gdn_in_qkv", M, H, gdn.convDim, gdnProjPrec("gdn_in_qkv", prec), chip);
      pushLinear(ops, li, "gdn_proj", "gdn_in_z", M, H, gdn.valueDim, gdnProjPrec("gdn_in_z", prec), chip);
      pushLinear(ops, li, "gdn_proj", "gdn_in_ba", M, H, gdn.baDim, gdnProjPrec("gdn_in_ba", prec), chip);

      const convFlops = 2 * M * gdn.convDim * model.gdn.convKernel;
      const convBytes =
        M * gdn.convDim * bf16B +
        gdn.convDim * model.gdn.convKernel * bf16B +
        M * gdn.convDim * bf16B;
      ops.push(
        makeOp(li, "gdn_conv", "gdn_conv1d", vec(convFlops, convBytes, {
          weightBytes: gdn.convDim * model.gdn.convKernel * bf16B,
          weightElems: gdn.convDim * model.gdn.convKernel,
          shape: `${M}×${gdn.convDim}×${model.gdn.convKernel}`,
        }))
      );
      ops.push(
        makeOp(li, "gdn_conv", "gdn_silu_l2", vec(
          (VECTOR_COEFF.silu + VECTOR_COEFF.l2norm) * M * gdn.convDim,
          M * gdn.convDim * bf16B,
          { shape: `${M}×${gdn.convDim}` }
        ))
      );

      const d2 = model.gdn.numVHeads * model.gdn.headDim * model.gdn.headDim;
      const cubeFlops = 6 * d2 * M;
      const vecFlops = 2 * d2 * M;
      const recBytes = B * gdn.stateElems * 4;
      const convStateHbm = B * gdn.convStateElems * 2;
      const qkvoBytes = M * (gdn.keyDim * 2 + gdn.valueDim * 2) * 2;
      const hbmState = phase === "decode" ? 2 * (recBytes + convStateHbm) : recBytes + convStateHbm;
      ops.push(
        makeOp(
          li,
          "gdn_state",
          "gdn_delta_rule",
          {
            flopsTensor: cubeFlops,
            flopsVector: vecFlops,
            bytes: hbmState + qkvoBytes,
            weightBytes: 0,
            weightElems: 0,
            shape: `B=${B} Vh=${model.gdn.numVHeads} D=${model.gdn.headDim}²`,
            formula:
              "Cube BF16 6·M·Vh·D²（Sk、βvkᵀ、qS） · Vec FP32 2·M·Vh·D²（S⊙g 衰减，无 FP32 Cube） · 搬 S FP32 + qkvo BF16",
            nTiles: Math.max(1, M * model.gdn.numVHeads),
          },
          { unitHint: "bf16", vectorHint: "fp32" }
        )
      );

      const zElems = M * gdn.valueDim;
      ops.push(makeOp(li, "gdn_state", "gdn_out_gate", vec(
        (VECTOR_COEFF.sigmoid + VECTOR_COEFF.mul) * zElems,
        2 * zElems * bf16B,
        { shape: `${M}×${gdn.valueDim}` }
      )));
      pushLinear(ops, li, "gdn_proj", "gdn_out", M, gdn.valueDim, H, gdnProjPrec("gdn_out", prec), chip);
    } else {
      const ap = attnProjPrec(prec);
      pushLinear(ops, li, "attn_proj", "q_proj", M, H, att.qDim, ap, chip);
      pushLinear(ops, li, "attn_proj", "k_proj", M, H, att.kvDim, ap, chip);
      pushLinear(ops, li, "attn_proj", "v_proj", M, H, att.kvDim, ap, chip);
      if (model.attn.outputGate === "elementwise") {
        pushLinear(ops, li, "attn_proj", "attn_gate_proj", M, H, att.qDim, ap, chip);
      } else {
        pushLinear(ops, li, "attn_proj", "attn_gate_proj", M, H, model.attn.numQHeads, ap, chip);
      }

      if (model.attn.qkNorm) {
        ops.push(makeOp(li, "norm", "qk_norm", vec(
          VECTOR_COEFF.rmsnorm * M * (att.qDim + att.kvDim),
          M * (att.qDim + att.kvDim) * bf16B,
          { shape: `${M}×(Q ${att.qDim}+K ${att.kvDim})` }
        )));
      }
      ops.push(makeOp(li, "norm", "rope", vec(
        VECTOR_COEFF.rope * M * model.attn.ropeDim * (model.attn.numQHeads + model.attn.numKVHeads),
        M * model.attn.ropeDim * (model.attn.numQHeads + model.attn.numKVHeads) * bf16B,
        { shape: `${M}×rope ${model.attn.ropeDim}` }
      )));

      const causal = phase === "prefill" ? 0.5 : 1;
      const nq = model.attn.numQHeads;
      const hd = model.attn.headDim;
      const scores = causal * B * nq * S * C;
      const oElems = B * nq * S * hd;
      const faTensor = 4 * causal * B * nq * S * C * hd;
      // KV 是 BF16；QKᵀ / PV 是 BF16 Cube。online softmax（含动态 max / exp / 归一化）和 O 按块缩放走 FP32 Vector。
      const nKvBlocks = Math.max(1, Math.ceil(C / FA_KV_BLOCK));
      const faSoftmax = FA_SOFTMAX_PER_SCORE * scores;
      const faORescale = 2 * oElems * nKvBlocks;
      const faScoreCast = scores;
      const faOutFp32 = oElems + oElems;
      const qBytes = B * nq * S * hd * bf16B;
      const kvReadBytes = B * model.attn.numKVHeads * C * hd * 2 * bf16B;
      const oBytes = B * nq * S * hd * bf16B;
      const kvWriteBytes = B * model.attn.numKVHeads * S * hd * 2 * bf16B;
      ops.push(
        makeOp(li, "flashattn", "flash_attn", {
          flopsTensor: faTensor,
          flopsVector: 0,
          flopsFp32: faSoftmax + faORescale + faScoreCast + faOutFp32,
          bytes: qBytes + kvReadBytes + oBytes + kvWriteBytes,
          weightBytes: 0,
          weightElems: 0,
          kvReadBytes,
          kvWriteBytes,
          shape: `B=${B} Qh=${nq} KVh=${model.attn.numKVHeads} S=${S} C=${C} D=${hd}`,
          formula: "Cube 4·causal·B·Qh·S·C·D (QKᵀ+PV, BF16) · KV BF16 · FP32 online softmax（max/sub/exp/sum/div，每 score）· O 按 KV 块缩放 · P FP32→BF16 · O 累加 FP32 再写出 BF16 · 搬 Q+KV读+O+KV写",
          nTiles: Math.max(1, B * nq * Math.ceil(S / 64) * Math.ceil(C / 64)),
        }, { fused: true, unitHint: "bf16" })
      );

      const gateElems = model.attn.outputGate === "elementwise" ? M * att.qDim : M * model.attn.numQHeads;
      ops.push(makeOp(li, "flashattn", "attn_out_gate", vec(
        (VECTOR_COEFF.sigmoid + VECTOR_COEFF.mul) * gateElems,
        2 * gateElems * bf16B,
        { shape: `${M}×${model.attn.outputGate === "elementwise" ? att.qDim : model.attn.numQHeads}` }
      )));
      pushLinear(ops, li, "attn_proj", "o_proj", M, att.qDim, H, ap, chip);
    }

    ops.push(makeOp(li, "residual", "res_mix", vec(VECTOR_COEFF.add * M * H, 3 * M * H * bf16B, { shape: `${M}×${H}` })));
    ops.push(makeOp(li, "norm", "rms_pre_ffn", vec(VECTOR_COEFF.rmsnorm * M * H, 2 * M * H * bf16B, { shape: `${M}×${H}` })));
    pushLinear(ops, li, "ffn", "gate_proj", M, H, I, prec, chip);
    pushLinear(ops, li, "ffn", "up_proj", M, H, I, prec, chip);
    ops.push(makeOp(li, "ffn", "silu_mul", vec(
      (VECTOR_COEFF.silu + VECTOR_COEFF.mul) * M * I,
      3 * M * I * bf16B,
      { shape: `${M}×${I}` }
    )));
    pushLinear(ops, li, "ffn", "down_proj", M, I, H, prec, chip);
    ops.push(makeOp(li, "residual", "res_ffn", vec(VECTOR_COEFF.add * M * H, 3 * M * H * bf16B, { shape: `${M}×${H}` })));
  });

  ops.push(makeOp(-2, "norm", "final_rms", vec(VECTOR_COEFF.rmsnorm * M * H, 2 * M * H * bf16B, { shape: `${M}×${H}` })));

  const lmM = workload.lmHeadTokens === "all" ? M : B * (phase === "prefill" ? 1 : S);
  const lmOps = expandLinear(-2, "lm_head", "lm_head", lmM, H, model.vocabSize, prec, chip);
  const lm = lmOps[lmOps.length - 1];
  if (model.tiedEmbeddings) {
    lm.weightBytes = 0;
    lm.bytes -= model.vocabSize * H * bitsToBytes(prec.weightBits);
    lm.weightElems = 0;
  }
  ops.push(...lmOps);
  return ops;
}

export function timeOp(op, chip, workload) {
  const gemmDtype = op.unitHint || PRECISIONS[workload.precision].gemmDtype;
  const computeDtype = gemmComputeFallback(chip, gemmDtype);
  const tCube = op.flopsTensor / (peakTflops(chip, computeDtype) * 1e12);
  const vecPeak =
    op.vectorHint === "fp32" || op.unitHint === "fp32"
      ? Math.max(chip.fp32Tflops || chip.vectorTflops, 1e-9)
      : Math.max(chip.vectorTflops, 1e-9);
  const fp32Peak = Math.max(chip.fp32Tflops || chip.vectorTflops, 1e-9);
  const tVector = op.flopsVector / (vecPeak * 1e12) + (op.flopsFp32 || 0) / (fp32Peak * 1e12);
  const hbmFrac = EPILOGUE_OPS.has(op.name) ? (workload.epilogueHbmFrac ?? 1) : 1;
  const tBw = (op.bytes * hbmFrac) / (chip.hbmBandwidthGBs * 1e9);
  const alpha = op.fused ? workload.faOverlapAlpha : workload.overlapAlpha;
  const gamma = workload.cubeVectorGamma ?? 0;
  const nTiles = op.nTiles || 1;
  const tComp = pipelineTime(tCube, tVector, 0, 0, gamma, nTiles);
  const t = pipelineTime(tCube, tVector, tBw, alpha, gamma, nTiles);
  let bound = "balanced";
  const slack = 1.15;
  if (tBw > tComp * slack) bound = "bandwidth";
  else if (tVector > tCube * slack && tVector > tBw * slack) bound = "vector";
  else if (tCube > tBw * slack) bound = "tensor";
  else if (tBw >= tComp) bound = "bandwidth";
  else bound = tVector > tCube ? "vector" : "tensor";
  return {
    ...op,
    tTensor: tCube,
    tCube,
    tVector,
    tBw,
    tComp,
    tPipe: t,
    t,
    bound,
    computeDtype,
    intensity: op.bytes > 0 ? (op.flopsTensor + op.flopsVector + (op.flopsFp32 || 0)) / op.bytes : Infinity,
  };
}

export function opTable(result) {
  const map = new Map();
  for (const o of result.ops) {
    const cur = map.get(o.name) || {
      name: o.name,
      family: o.family,
      shape: o.shape || "",
      formula: o.formula || "",
      n: 0,
      flopsCube: 0,
      flopsVector: 0,
      bytes: 0,
      kvReadBytes: 0,
      kvWriteBytes: 0,
      tCube: 0,
      tVector: 0,
      tBw: 0,
      tPipe: 0,
      t: 0,
      bound: o.bound,
    };
    cur.n += 1;
    cur.flopsCube += o.flopsTensor;
    cur.flopsVector += o.flopsVector + (o.flopsFp32 || 0);
    cur.bytes += o.bytes;
    cur.kvReadBytes += o.kvReadBytes || 0;
    cur.kvWriteBytes += o.kvWriteBytes || 0;
    cur.tCube += o.tCube ?? o.tTensor;
    cur.tVector += o.tVector;
    cur.tBw += o.tBw;
    cur.tPipe += o.tPipe ?? o.t;
    cur.t += o.t;
    map.set(o.name, cur);
  }
  const rows = [...map.values()].sort((a, b) => b.t - a.t);
  for (const r of rows) {
    const slack = 1.15;
    if (r.tBw > r.tCube * slack && r.tBw > r.tVector * slack) r.bound = "bandwidth";
    else if (r.tVector > r.tCube * slack && r.tVector > r.tBw * slack) r.bound = "vector";
    else r.bound = "tensor";
  }
  return rows;
}

export function tableTotal(rows) {
  const tot = {
    name: "total",
    family: "",
    shape: "",
    formula: "各行合计",
    n: 0,
    flopsCube: 0,
    flopsVector: 0,
    bytes: 0,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    tCube: 0,
    tVector: 0,
    tBw: 0,
    tPipe: 0,
    t: 0,
    bound: "tensor",
  };
  for (const r of rows) {
    tot.n += r.n;
    tot.flopsCube += r.flopsCube;
    tot.flopsVector += r.flopsVector;
    tot.bytes += r.bytes;
    tot.kvReadBytes += r.kvReadBytes || 0;
    tot.kvWriteBytes += r.kvWriteBytes || 0;
    tot.tCube += r.tCube;
    tot.tVector += r.tVector;
    tot.tBw += r.tBw;
    tot.tPipe += r.tPipe;
    tot.t += r.t;
  }
  const slack = 1.15;
  if (tot.tBw > tot.tCube * slack && tot.tBw > tot.tVector * slack) tot.bound = "bandwidth";
  else if (tot.tVector > tot.tCube * slack && tot.tVector > tot.tBw * slack) tot.bound = "vector";
  else tot.bound = "tensor";
  return tot;
}

function sumBy(arr, keyFn, valFn) {
  const m = new Map();
  for (const x of arr) {
    const k = keyFn(x);
    m.set(k, (m.get(k) || 0) + valFn(x));
  }
  return m;
}

export function memoryFootprint(model, chip, workload) {
  const inv = paramInventory(model);
  const prec = PRECISIONS[workload.precision];
  const B = phaseBatch(workload);
  const C = workload.phase === "prefill" ? workload.seq : workload.context;
  const wB = bitsToBytes(prec.weightBits);
  const attnB = prec.id === "nvfp4" || prec.id === "mxfp8" ? 2 : wB;
  const g = gdnDims(model);
  const gdnBa = inv.nGdn * model.hiddenSize * g.baDim;
  const gdnConvW = inv.nGdn * g.convDim * model.gdn.convKernel;
  const gdnLin = inv.gdn - gdnBa - gdnConvW;
  const gdnBytes =
    prec.id === "nvfp4" || prec.id === "mxfp8"
      ? gdnLin * bitsToBytes(PRECISIONS.mxfp8.weightBits) + (gdnBa + gdnConvW) * 2
      : inv.gdn * wB;
  const weightBytes =
    inv.ffn * wB +
    gdnBytes +
    (model.tiedEmbeddings ? 0 : inv.lmHead) * wB +
    inv.attn * attnB +
    inv.embed * 2;
  const kvBytes = inv.kvElemsPerToken * C * B * bitsToBytes(prec.kvBits);
  const stateBytes = B * (inv.gdnStateElems * 4 + inv.gdnConvStateElems * 2);
  const actBytes = B * Math.max(workload.seq, workload.decodeTokens) * model.hiddenSize * 2;
  const total = weightBytes + kvBytes + stateBytes + actBytes;
  return {
    weightGB: weightBytes / 1e9,
    kvGB: kvBytes / 1e9,
    stateGB: stateBytes / 1e9,
    actGB: actBytes / 1e9,
    totalGB: total / 1e9,
    capGB: chip.hbmCapacityGB,
    oom: total / 1e9 > chip.hbmCapacityGB,
  };
}

export function simulate(model, chip, workload) {
  const wl = { ...defaultWorkload(), ...workload };
  const ops = expandOps(model, chip, wl).map((op) => timeOp(op, chip, wl));
  const t = ops.reduce((s, o) => s + o.t, 0);
  const tTensor = ops.reduce((s, o) => s + o.tTensor, 0);
  const tVector = ops.reduce((s, o) => s + o.tVector, 0);
  const tBw = ops.reduce((s, o) => s + o.tBw, 0);
  const flopsT = ops.reduce((s, o) => s + o.flopsTensor, 0);
  const flopsV = ops.reduce((s, o) => s + o.flopsVector + (o.flopsFp32 || 0), 0);
  const bytes = ops.reduce((s, o) => s + o.bytes, 0);
  const byFamily = {};
  for (const o of ops) {
    const f = byFamily[o.family] || { t: 0, tTensor: 0, tVector: 0, tBw: 0, flopsTensor: 0, flopsVector: 0, bytes: 0 };
    f.t += o.t;
    f.tTensor += o.tTensor;
    f.tVector += o.tVector;
    f.tBw += o.tBw;
    f.flopsTensor += o.flopsTensor;
    f.flopsVector += o.flopsVector;
    f.bytes += o.bytes;
    byFamily[o.family] = f;
  }
  const byName = {};
  for (const o of ops) {
    const f = byName[o.name] || { t: 0, tTensor: 0, tVector: 0, tBw: 0, flopsTensor: 0, flopsVector: 0, bytes: 0, bound: o.bound, family: o.family, intensity: 0, n: 0 };
    f.t += o.t;
    f.tTensor += o.tTensor;
    f.tVector += o.tVector;
    f.tBw += o.tBw;
    f.flopsTensor += o.flopsTensor;
    f.flopsVector += o.flopsVector;
    f.bytes += o.bytes;
    f.n += 1;
    byName[o.name] = f;
  }
  for (const f of Object.values(byName)) {
    f.intensity = f.bytes > 0 ? (f.flopsTensor + f.flopsVector) / f.bytes : Infinity;
    const slack = 1.15;
    if (f.tBw > f.tTensor * slack && f.tBw > f.tVector * slack) f.bound = "bandwidth";
    else if (f.tVector > f.tTensor * slack && f.tVector > f.tBw * slack) f.bound = "vector";
    else f.bound = "tensor";
  }

  const boundCounts = { tensor: 0, vector: 0, bandwidth: 0, balanced: 0 };
  for (const o of ops) boundCounts[o.bound] += 1;
  const primary =
    tBw > tTensor && tBw > tVector ? "bandwidth" :
    tVector > tTensor ? "vector" : "tensor";

  const tokens = wl.phase === "decode" ? phaseBatch(wl) * wl.decodeTokens : phaseBatch(wl) * wl.seq;
  const tokPerSec = t > 0 ? tokens / t : 0;
  const mem = memoryFootprint(model, chip, wl);
  const utilTensor = t > 0 ? Math.min(1, tTensor / t) : 0;
  const utilVector = t > 0 ? Math.min(1, tVector / t) : 0;
  const utilBw = t > 0 ? Math.min(1, tBw / t) : 0;

  return {
    phase: wl.phase === "decode" ? "decode" : "prefill",
    workload: wl,
    ops,
    t,
    tTensor,
    tCube: tTensor,
    tVector,
    tBw,
    flopsT,
    flopsV,
    bytes,
    byFamily,
    byName,
    boundCounts,
    primary,
    tokPerSec,
    ms: t * 1e3,
    mem,
    utilTensor,
    utilVector,
    utilBw,
    inv: paramInventory(model),
  };
}

export function simulatePair(model, chip, workload) {
  const prefill = simulate(model, chip, { ...workload, phase: "prefill" });
  const decode = simulate(model, chip, { ...workload, phase: "decode" });
  return { prefill, decode };
}

export function sweepSeq(model, chip, workload, seqs) {
  return seqs.map((seq) => {
    const wl = { ...workload, seq, context: seq };
    const pair = simulatePair(model, chip, wl);
    return { seq, prefill: pair.prefill, decode: pair.decode };
  });
}

/** Prefill 关键算子：FFN MatMul（含 *_quant）、FlashAttention、GDN 线性注意力（delta rule）。 */
export function keyOpTimes(result) {
  const sum = (pred) => {
    const ops = result.ops.filter(pred);
    return {
      t: ops.reduce((s, o) => s + o.t, 0),
      tCube: ops.reduce((s, o) => s + (o.tCube ?? o.tTensor), 0),
      tVector: ops.reduce((s, o) => s + o.tVector, 0),
      tBw: ops.reduce((s, o) => s + o.tBw, 0),
    };
  };
  const matmul = sum((o) => o.family === "ffn" && (o.kind === "gemm" || String(o.name).endsWith("_quant")));
  const attn = sum((o) => o.name === "flash_attn");
  const gdn = sum((o) => o.name === "gdn_delta_rule");
  const bound = (x) => {
    const slack = 1.15;
    if (x.tBw > x.tCube * slack && x.tBw > x.tVector * slack) return "bandwidth";
    if (x.tVector > x.tCube * slack && x.tVector > x.tBw * slack) return "vector";
    return "tensor";
  };
  return {
    matmulMs: matmul.t * 1e3,
    matmulBound: bound(matmul),
    matmul: matmul,
    attnMs: attn.t * 1e3,
    attnBound: bound(attn),
    attn: attn,
    gdnMs: gdn.t * 1e3,
    gdnBound: bound(gdn),
    gdn: gdn,
  };
}

export function sweepVecRatio(model, chip, workload, ratios, opts = {}) {
  const cube0 = Math.max(chip.bf16Tflops, 1e-9);
  const vec0 = Math.max(chip.vectorTflops, 1e-9);
  const fp32_0 = Math.max(chip.fp32Tflops || vec0, 1e-9);
  const budget = cube0 + vec0;
  const iso = opts.isoBudget === true;
  return ratios.map((cv) => {
    const r = Math.max(cv, 0.1);
    let cube;
    let vec;
    if (iso) {
      cube = budget * (r / (r + 1));
      vec = budget / (r + 1);
    } else {
      cube = cube0;
      vec = cube0 / r;
    }
    const c = {
      ...chip,
      bf16Tflops: cube,
      vectorTflops: vec,
      fp8Tflops: chip.fp8Tflops > 0 ? chip.fp8Tflops * (cube / cube0) : 0,
      fp4Tflops: chip.fp4Tflops > 0 ? chip.fp4Tflops * (cube / cube0) : 0,
      fp32Tflops: fp32_0 * (vec / vec0),
    };
    const pair = simulatePair(model, c, workload);
    const keys = keyOpTimes(pair.prefill);
    return {
      x: cv,
      cube,
      vec,
      prefillMs: pair.prefill.ms,
      decodeTps: pair.decode.tokPerSec,
      prefillBound: pair.prefill.primary,
      decodeBound: pair.decode.primary,
      matmulMs: keys.matmulMs,
      matmulBound: keys.matmulBound,
      attnMs: keys.attnMs,
      attnBound: keys.attnBound,
      gdnMs: keys.gdnMs,
      gdnBound: keys.gdnBound,
    };
  });
}

export function sweepBwRatio(model, chip, workload, ratios) {
  const bf16 = Math.max(chip.bf16Tflops, 1e-9);
  return ratios.map((br) => {
    const c = { ...chip, hbmBandwidthGBs: bf16 * br };
    const pair = simulatePair(model, c, workload);
    return {
      x: br,
      prefillMs: pair.prefill.ms,
      decodeTps: pair.decode.tokPerSec,
      prefillBound: pair.prefill.primary,
      decodeBound: pair.decode.primary,
    };
  });
}

export function sweepBatch(model, chip, workload, batches) {
  return batches.map((batch) => {
    const pair = simulatePair(model, chip, { ...workload, batch });
    return { batch, prefill: pair.prefill, decode: pair.decode };
  });
}

export function sweepPrecision(model, chip, workload) {
  return Object.keys(PRECISIONS).map((precision) => {
    const pair = simulatePair(model, chip, { ...workload, precision });
    return { precision, label: PRECISIONS[precision].label, prefill: pair.prefill, decode: pair.decode };
  });
}

export function sweepHeatmap(model, chip, workload, opts = {}) {
  const vecRatios = opts.vecRatios || [1 / 2, 1 / 4, 1 / 8, 1 / 12, 1 / 16, 1 / 24, 1 / 32, 1 / 64];
  const bwRatios = opts.bwRatios || [2, 4, 6, 8, 12, 16];
  const bf16 = Math.max(chip.bf16Tflops, 1e-9);
  const vec0 = Math.max(chip.vectorTflops, 1e-9);
  const fp32_0 = Math.max(chip.fp32Tflops || vec0, 1e-9);
  const phase = opts.phase || "decode";
  const cells = [];
  for (const vr of vecRatios) {
    for (const br of bwRatios) {
      const vec = bf16 * vr;
      const c = {
        ...chip,
        vectorTflops: vec,
        fp32Tflops: fp32_0 * (vec / vec0),
        hbmBandwidthGBs: bf16 * br,
      };
      const r = simulate(model, c, { ...workload, phase });
      cells.push({
        vecRatio: vr,
        bwRatio: br,
        t: r.t,
        ms: r.ms,
        tokPerSec: r.tokPerSec,
        primary: r.primary,
        utilTensor: r.utilTensor,
        utilVector: r.utilVector,
        utilBw: r.utilBw,
      });
    }
  }
  return { vecRatios, bwRatios, cells, phase };
}

export function compareChips(model, chips, workload) {
  return chips.map((chip) => {
    const pair = simulatePair(model, chip, workload);
    return {
      id: chip.id,
      vendor: chip.vendor,
      name: chip.name,
      prefillMs: pair.prefill.ms,
      decodeTps: pair.decode.tokPerSec,
      oom: pair.prefill.mem.oom || pair.decode.mem.oom,
      mem: pair.decode.mem,
      prefillBound: pair.prefill.primary,
      decodeBound: pair.decode.primary,
    };
  });
}

export function sensitivity(model, chip, workload, phase = "decode") {
  const keys = [
    ["hbmBandwidthGBs", "HBM 带宽"],
    ["vectorTflops", "Vector 算力"],
    ["bf16Tflops", "BF16 MatMul"],
    ["fp4Tflops", "FP4 MatMul"],
    ["fp8Tflops", "FP8 MatMul"],
  ];
  const base = simulate(model, chip, { ...workload, phase });
  return keys
    .filter(([, ], i) => {
      const k = keys[i][0];
      return chip[k] > 0;
    })
    .map(([k, label]) => {
      const up = { ...chip, [k]: chip[k] * 1.2 };
      const down = { ...chip, [k]: chip[k] * 0.8 };
      const tUp = simulate(model, up, { ...workload, phase }).t;
      const tDown = simulate(model, down, { ...workload, phase }).t;
      return {
        key: k,
        label,
        dUp: (base.t - tUp) / base.t,
        dDown: (tDown - base.t) / base.t,
      };
    });
}

export { sumBy };
