# 模型测试优先级

目标：先测试公开视频测评里排名靠前、并且在 Mac 本地有现实可跑路径的模型；后续逐步覆盖对方测过的全量模型。

## P0：先测

这些模型优先进入第一轮 ready case 测评。

1. `Qwen3-ASR-1.7B`
   - 多语种视频里被推荐为当前最均衡基线。
   - 中文、日语、英语都在前列。
   - 需要测非量化和 4bit。

2. `Qwen3-ASR-0.6B`
   - 小模型对照，适合本地速度/内存基线。
   - 需要测非量化和 4bit。

3. `Whisper large-v3 / large-v3-turbo`
   - Mac 本地工程基线。
   - 重点看稳定性、长音频、安装成本、速度和内存。

4. `MiMo-V2.5-ASR / MiMo-V2.5-Pro`
   - 中文和英语视频中排名靠前。
   - 优先找 MLX 或 Apple Silicon 可跑版本。

5. `Cohere Transcribe`
   - 英语视频综合排名第一，多语种榜单争议较大。
   - 需要确认许可、本地运行路径和 Apple Silicon 支持。

## P1：第二批

6. `FunASR / Fun-ASR-Nano`
   - 中文实用场景重要，轻量可部署。

7. `SenseVoice / SenseVoiceSmall`
   - 中文和多语种候选，速度快，适合作为本地轻量方案。

8. `NVIDIA Parakeet-TDT-0.6B-v2`
   - 英文强对照，速度和准确率都值得测。

9. `Parakeet-TDT-0.6B-JA`
   - 日语视频里作者最终推荐的高性价比模型。

10. `GLM-ASR-Nano`
    - 中文/英语/日语视频都有出现，轻量对照。

## 专项跟进

- `Mega-ASR`
  - 基于 Qwen3-ASR 的困难声学场景鲁棒性增强方案，优先验证远场、混响、噪声、回声、设备底噪、带宽丢失等 case。
  - 不是默认通用 ASR 替代品；先按 `docs/todo/mega_asr_research_todo.md` 做分层 PoC，再决定是否进入正式模型队列或作为 `robust-mode` 后端。

## P2：补全作者测过的模型

11. `VibeVoice-ASR`
    - 多语种表现强，但模型偏大，本地 32GB Mac 上要重点看内存和速度。

12. `FireRed-ASR2 / FireRed-ASR2-CTC`
    - 中文视频里出现，作为补充覆盖。

13. `Granite Speech 4.1 2B / Plus`
    - 英语和日语视频里出现，作为补充覆盖。

14. `Canary-1B-v2`
    - 多语种视频里提到，后续作为多语种/英文对照。

## 变体测试原则

每个模型尽量记录：

- 权重：例如 0.6B、1.7B、2B。
- 精度：非量化、4bit、int8、GGUF/MLX 等。
- 后端：Transformers、MLX、whisper.cpp、FunASR runtime、NeMo 等。
- 本地资源：模型大小、峰值内存、RTF、安装耗时。
- 输出能力：标点、时间戳、语言识别、长音频稳定性。

## 第一轮建议顺序

1. `Qwen3-ASR-0.6B-4bit`
2. `Qwen3-ASR-0.6B`
3. `Qwen3-ASR-1.7B-4bit`
4. `Qwen3-ASR-1.7B`
5. `whisper.cpp large-v3-turbo`
6. `whisper.cpp large-v3`
7. `MiMo-V2.5-ASR`
8. `Cohere Transcribe`
9. `SenseVoiceSmall`
10. `FunASR / Fun-ASR-Nano`

实际执行时，如果某个模型在 Mac 本地安装明显受阻，先记录阻塞原因，然后跳到下一项。
