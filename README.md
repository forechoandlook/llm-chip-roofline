# LLM 芯片配比实验室

纯前端解析 Roofline：给定 **Qwen3.8-27B** 的算子与 shape，按最少 FLOP / 最少搬运估计 Prefill / Decode。

核心问题不是「1:8 好还是 1:16 好」，而是 **什么样的 Vector 配比不会拖慢 Cube**。结论见 [docs/配比结论.md](docs/配比结论.md)。不考虑面积时：Prefill 几乎全是 Cube，Decode 几乎全是带宽，只拧 Cube:Vector 没有意义。

## 运行

```bash
node tests/engine.test.js
node serve.mjs
```

打开 http://127.0.0.1:8765

## 怎么用

1. 左边选芯片仓库（NVIDIA / 华为 / AMD / 寒武纪 / Intel）或「自定义设计点」。
2. 所有峰值和配比都可以填。改 `Vector : MatMul = 1 : N` 或 `带宽 GB/s : BF16 TFLOPS` 会回写绝对值。
3. 精度默认 NVFP4：投影走 FP4 Tensor（芯片 FP4=0 则回退 BF16 算力、仍按 4.5 bit 搬权重），FlashAttention 保持 BF16。
4. 图和底部表格会说明当前切片的瓶颈；热力图扫配比。

## 模型

时间：`T = max(T_comp, T_bw) + α · min(T_comp, T_bw)`，FlashAttention 默认 α=0。更细的 tile 流水、SRAM、NVFP4、GDN 在 `~/tmp/chip-simulator`。

这是理论下限，不是 vLLM/SGLang 实测。
