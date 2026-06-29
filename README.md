# 个人 ASR Gold 测评集

[English README](README_EN.md)

这是一个面向真实本地 ASR 使用场景的个人开源测评集。仓库只公开最终 Gold 数据集、人工复核或严格自动筛选后的 reference、评分脚本和方法文档；不包含模型权重、音视频文件、下载缓存、候选集、草稿字幕或中间结果。

## 包含内容

- `data/gold_manifest.v1.json`：Gold-only 数据清单，包含来源 URL、片段范围、场景、语言、主指标、权重、热词和 reference 元数据。
- `data/gold_references/`：每条 Gold case 的最终 reference 文本。
- `tools/validate_manifest.py`：检查 manifest 结构和 reference 文件完整性。
- `tools/summarize_dataset.py`：汇总数据集覆盖情况。
- `tools/materialize_gold_audio.py`：根据公开视频 URL 和时间段在本地重建音频片段，用于复现模型测试。
- `tools/score_transcript.py`：计算 CER/WER/hybrid TER、关键词、热词、数字、标点、幻觉、粤语繁简归一诊断、日语表记诊断等指标。
- `docs/`：测评原则、指标定义、数据卡和模型优先级说明。

## 数据集快照

截至 2026-06-29：

- Gold case：202 条
- 短音频 Gold：197 条
- 长音频 Gold：5 条
- 语言分布：英文 100、普通话中文 67、日语 25、粤语 10
- 主要场景：学生课堂、销售/客服通话、会议/圆桌、公开视频演讲、访谈/播客、课程、工具演示、医疗/咨询、新闻叙事和长音频。
- 来源：带平台字幕/transcript 或人工复核字幕草稿的公开 YouTube / Bilibili 视频。

Gold 确认层级：

- `user_confirmed_real_audio`：项目作者或 assistant-led QA 已对真实音频进行复核并明确接受。
- `auto_screened_public_subtitle`：reference 来自公开字幕/transcript，并通过严格自动筛选；本地 ASR 输出没有被用作 reference 文本。

## 快速开始

安装核心评分依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

校验公开 Gold 数据集：

```bash
python tools/validate_manifest.py
python tools/summarize_dataset.py
```

重建少量本地音频做 smoke test：

```bash
python tools/materialize_gold_audio.py --limit 3
```

脚本会在被 `.gitignore` 忽略的本地目录写入：

```text
data/audio/gold/audio_manifest.json
data/audio/gold/<case_id>.wav
```

部分 Bilibili 或 YouTube 来源可能需要地区访问或登录 cookies。可通过标准 `yt-dlp` 参数传入浏览器 cookies：

```bash
python tools/materialize_gold_audio.py --case <case_id> --cookies-from-browser chrome
```

## 测试一个 ASR 模型

仓库不绑定某一个 ASR 模型。任意模型只要为每个 case 输出一个 UTF-8 文本文件，就可以评分：

```text
predictions/my_model/<case_id>.txt
```

对一个预测目录评分：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model \
  --out results/my_model.score.json
```

对单条 case 评分：

```bash
python tools/score_transcript.py \
  --case youtube_bbc_ai_vocab_001 \
  --prediction predictions/my_model/youtube_bbc_ai_vocab_001.txt
```

如果模型使用了 case 中的 `hotwords` 或上下文术语，需要和 zero-shot 结果分开汇报：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_with_context \
  --hotword-mode with_context_terms \
  --out results/my_model_with_context.score.json
```

如果模型支持原生 decoder hotword 或 vocabulary bias，应使用：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_native_hotwords \
  --hotword-mode native_hotwords \
  --out results/my_model_native_hotwords.score.json
```

## 可复现约定

这个公开仓库的目标是让其他用户不依赖私有中间文件，也能复现完整测评流程：

1. 校验 `data/gold_manifest.v1.json`。
2. 用 `tools/materialize_gold_audio.py` 根据来源 URL 和时间段重建本地音频。
3. 用任意 ASR 模型处理 `data/audio/gold/audio_manifest.json`。
4. 每个 case 保存一份预测文本。
5. 用 `tools/score_transcript.py` 评分。

仓库有意不重新分发音视频媒体，因为这些内容属于上游发布者。仓库保留来源 URL 和片段元数据用于复现。

## 评分说明

- `weighted_text_score_90` 是本测评集自己的加权综合文本分，不是业界统一标准分。
- CER/WER/hybrid TER 会保留在输出中，便于和常见 ASR 评估方式对照。
- 粤语会额外输出繁简归一诊断分，用于区分“内容没听对”和“繁体/简体字形不一致”。
- 日语会额外输出振假名/表记差异诊断分；明显错误的 reference 仍然应该修正或降级，不能靠诊断规则掩盖。
- ITN/数学格式化单独观察；语义等价的数字或公式写法差异，不应直接解释为原始 ASR 声学识别失败。

详细说明见 [docs/metrics.md](docs/metrics.md) 和 [docs/benchmark_principles.md](docs/benchmark_principles.md)。

## 不公开的内容

- 模型权重和模型缓存。
- 下载的音频、视频和字幕文件。
- 浏览器 cookies 或账号状态。
- 非 Gold 候选集、backup case、合成压力测试 manifest、草稿 reference、本地 ASR 草稿。
- 完整模型预测目录和中间测评输出。

## License

代码使用 MIT License。数据集元数据和项目整理的 reference 文本用于研究和评估；上游媒体和来源字幕仍受原发布者条款约束。见 [DATA_LICENSE.md](DATA_LICENSE.md)。
