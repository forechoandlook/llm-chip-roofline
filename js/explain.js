import { layerTypes, gdnDims, attnDims } from "./models.js";
import { PRECISIONS, peakTflops, gemmComputeFallback, paramInventory } from "./engine.js";
import { boundLabel, fmtMs } from "./conclusions.js";
import { chipRatios } from "./chips.js";

function fmtN(x) {
  if (!Number.isFinite(x)) return "∞";
  const ax = Math.abs(x);
  if (Number.isInteger(x) && ax < 1e6) return String(x);
  if (ax >= 1e12) return `${(x / 1e12).toFixed(3)} T`;
  if (ax >= 1e9) return `${(x / 1e9).toFixed(3)} G`;
  if (ax >= 1e6) return `${(x / 1e6).toFixed(3)} M`;
  if (ax >= 1e3) return `${(x / 1e3).toFixed(3)} k`;
  if (ax >= 100) return x.toFixed(1);
  if (ax >= 1) return x.toFixed(3);
  return x.toExponential(2);
}

function fmtS(s) {
  if (s < 1e-9) return `${(s * 1e12).toFixed(2)} ps`;
  if (s < 1e-6) return `${(s * 1e9).toFixed(2)} ns`;
  if (s < 1e-3) return `${(s * 1e6).toFixed(2)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(3)} ms`;
  return `${s.toFixed(3)} s`;
}

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function dimsOf(model, workload, phase) {
  const B = workload.batch;
  const S = phase === "prefill" ? workload.seq : workload.decodeTokens;
  const C = phase === "prefill" ? workload.seq : workload.context;
  return { B, S, C, M: B * S, H: model.hiddenSize, I: model.intermediateSize };
}

function firstOp(result, name) {
  return result.ops.find((o) => o.name === name);
}

export function buildExplain(model, chip, workload, result) {
  const phase = result.phase;
  const d = dimsOf(model, workload, phase);
  const g = gdnDims(model);
  const a = attnDims(model);
  const prec = PRECISIONS[workload.precision];
  const types = layerTypes(model);
  const inv = paramInventory(model);
  const gemmDt = gemmComputeFallback(chip, prec.gemmDtype);
  const peak = peakTflops(chip, gemmDt);
  const r = chipRatios(chip);
  return { model, chip, workload, result, phase, d, g, a, prec, types, inv, gemmDt, peak, r };
}

export function renderLayerStrip(types) {
  const cells = types.map((t, i) => {
    const cls = t === "attn" ? "cell-attn" : "cell-gdn";
    const lab = t === "attn" ? "A" : "G";
    return `<span class="lcell ${cls}" title="L${i} ${t === "attn" ? "Gated Attention" : "Gated DeltaNet"}">L${i}<b>${lab}</b></span>`;
  }).join("");
  return `<div class="layer-strip">${cells}</div>
    <p class="legend"><span class="swatch cell-gdn"></span> G = Gated DeltaNet（48 层）
    <span class="swatch cell-attn"></span> A = Gated Attention + FlashAttention（16 层）
    排布：16 × (G G G A)</p>`;
}

export function renderConfig(ex) {
  const { model, g, a, inv, prec } = ex;
  const rows = [
    ["模型", model.name],
    ["范围", "语言模型主干（Vision Encoder 未计入）"],
    ["层数", `${model.numLayers} = ${model.blockRepeat} × ( ${model.gdnPerBlock} GDN + ${model.attnPerBlock} Attn )`],
    ["hiddenSize H", model.hiddenSize],
    ["FFN intermediate I", `${model.intermediateSize}（SwiGLU：gate / up / down）`],
    ["vocab V", model.vocabSize],
    ["tied embeddings", model.tiedEmbeddings ? "yes" : "no（embed 与 lm_head 各一份）"],
    ["GDN QK heads × dim", `${model.gdn.numKHeads} × ${model.gdn.headDim} → keyDim=${g.keyDim}`],
    ["GDN V heads × dim", `${model.gdn.numVHeads} × ${model.gdn.headDim} → valueDim=${g.valueDim}`],
    ["GDN conv", `depthwise conv1d, kernel=${model.gdn.convKernel}, convDim=${g.convDim}`],
    ["GDN state / 层", `[B, ${model.gdn.numVHeads}, ${model.gdn.headDim}, ${model.gdn.headDim}] = ${fmtN(g.stateElems)} elem`],
    ["Attn Q heads × dim", `${model.attn.numQHeads} × ${model.attn.headDim} → qDim=${a.qDim}（≠ H）`],
    ["Attn KV heads × dim", `${model.attn.numKVHeads} × ${model.attn.headDim} → kvDim=${a.kvDim}，GQA group=${a.groupSize}`],
    ["RoPE dim", model.attn.ropeDim],
    ["QK RMSNorm", model.attn.qkNorm ? "yes" : "no"],
    ["Attn output gate", model.attn.outputGate],
    ["参数量（语言塔）", `${fmtN(inv.total)} ≈ ${(inv.total / 1e9).toFixed(2)} B`],
    ["其中 FFN", `${fmtN(inv.ffn)}（${(100 * inv.ffn / inv.total).toFixed(1)}%）`],
    ["其中 GDN 投影", `${fmtN(inv.gdn)}`],
    ["其中 Attn 投影", `${fmtN(inv.attn)}`],
    ["其中 embed + lm_head", `${fmtN(inv.embed + inv.lmHead)}`],
    ["KV / token（16 层）", `${fmtN(inv.kvElemsPerToken)} elem`],
    ["当前精度存储", `FFN：${prec.label}。GDN qkv/z/out：FP8（NVFP4/MXFP8 页）；in_proj_ba + 复发核：BF16/FP32。QKVO + FA + KV：完整 BF16`],
  ];
  return `<table class="doc">
    <tbody>${rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("")}</tbody>
  </table>`;
}

export function renderShapes(ex) {
  const { d, g, a, model, phase, prec } = ex;
  const { B, S, C, M, H, I } = d;
  return `
    <p class="cap">当前视图 <b>${phase}</b>。符号：B=batch，S=本步 query 长度，C=KV/历史长度，M=B·S 是 GEMM 的 token 维。</p>
    <table class="doc">
      <tbody>
        <tr><th>B</th><td>${B}</td></tr>
        <tr><th>S</th><td>${S}　${phase === "prefill" ? "（= Prefill seq）" : "（= decode tokens / step）"}</td></tr>
        <tr><th>C</th><td>${C}　${phase === "prefill" ? "（因果自注意力，C=S）" : "（= Decode context）"}</td></tr>
        <tr><th>M = B·S</th><td>${M}</td></tr>
        <tr><th>hidden 激活</th><td>[${B}, ${S}, ${H}]</td></tr>
        <tr><th>FFN gate/up</th><td>输入 [${M}, ${H}] × 权重 [${H}, ${I}] → [${M}, ${I}]</td></tr>
        <tr><th>FFN down</th><td>[${M}, ${I}] × [${I}, ${H}] → [${M}, ${H}]</td></tr>
        <tr><th>GDN in_qkv</th><td>[${M}, ${H}] × [${H}, ${g.convDim}]　convDim = 2·${g.keyDim}+${g.valueDim}</td></tr>
        <tr><th>GDN state</th><td>[${B}, ${model.gdn.numVHeads}, ${model.gdn.headDim}, ${model.gdn.headDim}]，与 C 无关</td></tr>
        <tr><th>Attn Q</th><td>[${B}, ${model.attn.numQHeads}, ${S}, ${model.attn.headDim}]　投影 [${M}, ${H}]×[${H}, ${a.qDim}]</td></tr>
        <tr><th>Attn K/V</th><td>投影出 [${B}, ${model.attn.numKVHeads}, ${S}, ${model.attn.headDim}]；cache 为 [${B}, ${model.attn.numKVHeads}, ${C}, ${model.attn.headDim}]</td></tr>
        <tr><th>FlashAttention</th><td>Q [${B},${model.attn.numQHeads},${S},${model.attn.headDim}] · K/V cache [${B},${model.attn.numKVHeads},${C},${model.attn.headDim}]，计算 dtype ${prec.faCompute}</td></tr>
      </tbody>
    </table>
    <h4>一层里有哪些算子</h4>
    <div class="flow">
      <div class="flow-col">
        <h5>GDN 层（48）</h5>
        <ol>
          <li>RMSNorm(H)</li>
          <li>in_qkv、in_z、in_ba 三个 GEMM</li>
          <li>depthwise conv1d k=${model.gdn.convKernel} + SiLU + L2Norm</li>
          <li>Gated delta rule 更新 state</li>
          <li>sigmoid(z) ⊙ output → out_proj 回 H</li>
          <li>residual</li>
          <li>RMSNorm → SwiGLU FFN（gate、up、SiLU⊙、down）→ residual</li>
        </ol>
      </div>
      <div class="flow-col">
        <h5>Gated Attention 层（16）</h5>
        <ol>
          <li>RMSNorm(H)</li>
          <li>q / k / v / gate 四个 GEMM</li>
          <li>QK-RMSNorm、RoPE（dim=${model.attn.ropeDim}）</li>
          <li>FlashAttention（BF16 MMA + FP32 softmax）</li>
          <li>sigmoid(gate) ⊙ O → o_proj 回 H</li>
          <li>residual</li>
          <li>RMSNorm → 同一套 SwiGLU FFN → residual</li>
        </ol>
      </div>
    </div>
    <p class="cap">前后还有 embed_lookup、final RMSNorm、lm_head（默认只算最后 token）。</p>`;
}

function gemmWalk(op, label, M, K, N, prec, chip, peak, gemmDt, alpha) {
  if (!op) return "";
  const flop = 2 * M * K * N;
  return `<div class="walk">
    <h5>${esc(label)}</h5>
    <pre>GEMM  [${fmtN(M)}, ${fmtN(K)}] × [${fmtN(K)}, ${fmtN(N)}]
Tensor FLOP = 2·M·K·N
            = 2 · ${fmtN(M)} · ${fmtN(K)} · ${fmtN(N)}
            = ${fmtN(flop)}     （本层实测 ${fmtN(op.flopsTensor)}）
weight-dequant Vector = ${fmtN(op.flopsVector)}（原生 FP4/FP8 MMA 为 0）
FP32 epilogue（输出 + s_global）= ${fmtN(op.flopsFp32 || 0)}，与 Cube 按 tile 流水 γ
Bytes（算子实测，激活按 BF16 读）= ${fmtN(op.bytes)}
强度 I = FLOP / Bytes = ${fmtN(op.intensity)} FLOP/byte
计算峰值 = ${gemmDt.toUpperCase()} ${fmtN(peak)} TFLOPS
T_tensor = FLOP / (peak·1e12) = ${fmtS(op.tTensor)}
T_vector = ${fmtS(op.tVector)}
T_bw     = Bytes / (BW·1e9)   = ${fmtS(op.tBw)}
T_comp   = max(C,V) + γ·min（tile 流水）= ${fmtS(op.tComp)}
T        = max(T_comp, T_bw) + α·min(...)   α=${alpha}
         = ${fmtS(op.t)}
瓶颈     = ${boundLabel(op.bound)}</pre>
  </div>`;
}

function faWalk(op, d, model, phase, prec, alpha) {
  if (!op) return "";
  const { B, S, C } = d;
  const causal = phase === "prefill" ? 0.5 : 1;
  const nq = model.attn.numQHeads;
  const nkv = model.attn.numKVHeads;
  const hd = model.attn.headDim;
  return `<div class="walk">
    <h5>FlashAttention（单层，×16）</h5>
    <pre>因果因子 causal = ${causal}　（Prefill 三角 1/2，Decode 对 cache 为 1）
Tensor FLOP = 4 · causal · B · nq · S · C · d
            = 4 · ${causal} · ${B} · ${nq} · ${S} · ${C} · ${hd}
            = ${fmtN(op.flopsTensor)}
FP32 Vector = online softmax（running max、s−m、两次 exp、l 乘加、p/l）
            + O 按 KV 块缩放 2·B·nq·S·d·⌈C/64⌉
            + P FP32→BF16 + O 写出
            = ${fmtN(op.flopsFp32 || op.flopsVector)}
Bytes（不物化 S×S）= Q + K + V + O + KV写
  Q,O,K,V : BF16，2 字节/elem　nkv=${nkv}
实测 Bytes = ${fmtN(op.bytes)}
FA 计算墙走 BF16 Tensor，α_FA=${alpha}
T_tensor=${fmtS(op.tTensor)}  T_vector=${fmtS(op.tVector)}  T_bw=${fmtS(op.tBw)}
T=${fmtS(op.t)}  瓶颈=${boundLabel(op.bound)}</pre>
  </div>`;
}

function gdnWalk(op, d, model, g, phase, workload) {
  if (!op) return "";
  const { B, M } = d;
  const unit = workload.gdnStateUnit;
  return `<div class="walk">
    <h5>GDN delta rule（单层，×48）</h5>
    <pre>state = [B, nV, d, d] = [${B}, ${model.gdn.numVHeads}, ${model.gdn.headDim}, ${model.gdn.headDim}]
Cube 6·M·Vh·D² + Vector FP32 2·M·Vh·D²（衰减）
FLOP = ${fmtN(op.flopsTensor + op.flopsVector)}   Cube ${fmtN(op.flopsTensor)} / Vec ${fmtN(op.flopsVector)}
HBM：${phase === "decode" ? "读+写" : "写回"} state+conv
    = ${fmtN(op.bytes)}　（与上下文长度 C 无关）
T_tensor=${fmtS(op.tTensor)}  T_vector=${fmtS(op.tVector)}  T_bw=${fmtS(op.tBw)}
T=${fmtS(op.t)}  瓶颈=${boundLabel(op.bound)}</pre>
  </div>`;
}

export function renderTiming(ex) {
  const { chip, workload, result, peak, gemmDt, r, prec } = ex;
  const alpha = workload.overlapAlpha;
  const faA = workload.faOverlapAlpha;
  return `
    <h4>三条墙</h4>
    <pre>T_tensor = TensorFLOP / (MatMulPeak(dtype) · 1e12)
T_vector = VectorFLOP / (VectorPeak · 1e12)
T_bw     = Bytes / (HBM_GBs · 1e9)
T_comp   = max(T_tensor, T_vector)
T        = max(T_comp, T_bw) + α · min(T_comp, T_bw)

当前芯片：BF16 ${fmtN(chip.bf16Tflops)} TFLOPS，FP8 ${fmtN(chip.fp8Tflops)}，FP4 ${fmtN(chip.fp4Tflops)}
          Vector ${fmtN(chip.vectorTflops)} TFLOPS，HBM ${fmtN(chip.hbmBandwidthGBs)} GB/s
本精度 GEMM 走 ${gemmDt.toUpperCase()} 峰值 ${fmtN(peak)} TFLOPS
Vector:MatMul = 1:${(1 / r.vectorToMatmul).toFixed(2)}
带宽:BF16-TFLOPS = ${r.bwPerMatmul.toFixed(3)}:1
Ridge BF16 = peakFLOP/s / BW = ${r.ridgeBf16.toFixed(1)} FLOP/byte
α=${alpha}（普通算子），FlashAttention α=${faA}
端到端把每层每个算子的 T 相加（${result.ops.length} 个展开算子）
ΣT = ${fmtS(result.t)} = ${fmtMs(result.ms)}</pre>
    <h4>精度怎么进公式</h4>
    <pre>${prec.label}
FFN 权重 ${prec.weightBits} bit，GEMM ${prec.gemmDtype.toUpperCase()}（无该峰值则回退 BF16）
GDN qkv/z/out：FP8；in_proj_ba + 复发核：BF16 / FP32
QKVO + FA + KV：完整 BF16
动态量化只发生在走 FP4/FP8 的 FFN（以及 GDN 的 FP8 投影）之前
Cast 按较宽类型走 FP32 峰值。激活和逐元素算子按 BF16 搬</pre>
    <h4>显存工作集</h4>
    <pre>权重 = FFN·所选精度 + GDN(qkv/z/out FP8，ba/conv BF16) + Attn BF16 + lm_head + embed·2B
KV   = 16层 · 4 KV头 · 256 · 2(K,V) · C · B · kvBytes
GDN  = B · 48层 · (48·128·128 + conv state) · 2B
当前：权重 ${result.mem.weightGB.toFixed(2)} GB，KV ${result.mem.kvGB.toFixed(2)} GB，
      state ${result.mem.stateGB.toFixed(2)} GB，激活下限 ${result.mem.actGB.toFixed(2)} GB，
      合计 ${result.mem.totalGB.toFixed(2)} / ${result.mem.capGB} GB</pre>`;
}

export function renderWalks(ex) {
  const { model, chip, workload, result, d, g, prec, peak, gemmDt } = ex;
  const { M, H, I } = d;
  const alpha = workload.overlapAlpha;
  const gate = firstOp(result, "gate_proj");
  const qkv = firstOp(result, "gdn_in_qkv");
  const q = firstOp(result, "q_proj");
  const fa = firstOp(result, "flash_attn");
  const gdn = firstOp(result, "gdn_delta_rule");
  const lm = firstOp(result, "lm_head");
  const lmM = workload.lmHeadTokens === "all" ? M : d.B * (result.phase === "prefill" ? 1 : d.S);
  return [
    gemmWalk(gate, `FFN gate_proj（单层，×${model.numLayers}）`, M, H, I, prec, chip, peak, gemmDt, alpha),
    gemmWalk(qkv, `GDN in_qkv（单层，×${result.inv.nGdn}，FP8）`, M, H, g.convDim, PRECISIONS.mxfp8, chip, peakTflops(chip, gemmComputeFallback(chip, "fp8")), gemmComputeFallback(chip, "fp8"), alpha),
    gemmWalk(q, `Attn q_proj（单层，×${result.inv.nAttn}，固定 BF16）`, M, H, attnDims(model).qDim, PRECISIONS.bf16, chip, peakTflops(chip, "bf16"), "bf16", alpha),
    faWalk(fa, d, model, result.phase, prec, workload.faOverlapAlpha),
    gdnWalk(gdn, d, model, g, result.phase, workload),
    gemmWalk(lm, `lm_head（M=${lmM}，V=${model.vocabSize}）`, lmM, H, model.vocabSize, prec, chip, peak, gemmDt, alpha),
  ].join("");
}

export function renderExplainHtml(ex) {
  return {
    strip: renderLayerStrip(ex.types),
    config: renderConfig(ex),
    shapes: renderShapes(ex),
    timing: renderTiming(ex),
    walks: renderWalks(ex),
  };
}
