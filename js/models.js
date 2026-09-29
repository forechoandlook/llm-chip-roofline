/** Frozen architecture cards. Numbers come from public model cards / configs. */

export const MODELS = {
  "qwen3.8-27b": {
    id: "qwen3.8-27b",
    name: "Qwen3.8-27B",
    notes:
      "Dense hybrid: 16 × (3× Gated DeltaNet + 1× Gated Attention), language tower only. Vision encoder is excluded in v1.",
    hiddenSize: 5120,
    intermediateSize: 17408,
    vocabSize: 248320,
    numLayers: 64,
    blockRepeat: 16,
    gdnPerBlock: 3,
    attnPerBlock: 1,
    tiedEmbeddings: false,
    gdn: {
      numKHeads: 16,
      numVHeads: 48,
      headDim: 128,
      convKernel: 4,
    },
    attn: {
      numQHeads: 24,
      numKVHeads: 4,
      headDim: 256,
      ropeDim: 64,
      qkNorm: true,
      /** Elementwise output gate: extra GEMM hidden → q_dim, then sigmoid. */
      outputGate: "elementwise",
    },
    ffn: { type: "swiglu" },
  },
};

export function layerTypes(model) {
  const types = [];
  for (let b = 0; b < model.blockRepeat; b++) {
    for (let i = 0; i < model.gdnPerBlock; i++) types.push("gdn");
    for (let i = 0; i < model.attnPerBlock; i++) types.push("attn");
  }
  if (types.length !== model.numLayers) {
    throw new Error(`layer pattern ${types.length} != numLayers ${model.numLayers}`);
  }
  return types;
}

export function gdnDims(model) {
  const g = model.gdn;
  const keyDim = g.numKHeads * g.headDim;
  const valueDim = g.numVHeads * g.headDim;
  return {
    keyDim,
    valueDim,
    convDim: keyDim * 2 + valueDim,
    baDim: g.numVHeads * 2,
    stateElems: g.numVHeads * g.headDim * g.headDim,
    convStateElems: (keyDim * 2 + valueDim) * (g.convKernel - 1),
  };
}

export function attnDims(model) {
  const a = model.attn;
  return {
    qDim: a.numQHeads * a.headDim,
    kvDim: a.numKVHeads * a.headDim,
    groupSize: a.numQHeads / a.numKVHeads,
  };
}
