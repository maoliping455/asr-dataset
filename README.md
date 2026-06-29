# 个人 ASR 测评集

[English README](README_EN.md)

这是一个面向真实使用场景的本地 ASR 测评集。它关注的问题很直接：如果你要把课堂视频、销售通话、会议录音、公开视频演讲或播客转成文字，哪些 ASR 模型在本地真正好用，哪些问题会影响后续总结、检索和笔记整理。

这个项目不是公开排行榜复刻，也不是只看 WER/CER 的单一指标测试。它更像一套可复现的个人基准：样本来自真实公开视频和公开字幕/transcript，reference 经过人工复核或严格筛选，评分会同时看内容准确率、术语、数字、幻觉、标点，以及粤语/日语这类语言特定问题。

## 适合谁

- 想在 Mac 或本地机器上比较开源 ASR 模型的人。
- 需要课堂、会议、销售、访谈、播客转写质量基准的人。
- 关心中文、英文、日语、粤语多语言 ASR 表现的人。
- 希望用自己的模型输出跑一套可解释评分的人。

## 数据覆盖

截至 2026-06-29，正式测试集包含 202 条样本：

- 短音频样本：197 条
- 长音频样本：5 条
- 语言分布：英文 100、普通话中文 67、日语 25、粤语 10
- 场景覆盖：学生课堂、销售/客服通话、会议/圆桌、公开视频演讲、访谈/播客、在线课程、工具演示、医疗/咨询、新闻叙事和长音频。

数据来源主要是公开 YouTube / Bilibili 视频。仓库保留来源 URL、片段时间段、场景标签、语言标签、热词和最终 reference 文本，便于其他人复现同样的测试。

## 快速开始

安装依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

检查数据集：

```bash
python tools/validate_manifest.py
python tools/summarize_dataset.py
```

重建少量音频片段做 smoke test：

```bash
python tools/materialize_audio.py --limit 3
```

音频会写到本地忽略目录：

```text
data/audio/benchmark/audio_manifest.json
data/audio/benchmark/<case_id>.wav
```

部分 Bilibili 或 YouTube 来源可能需要地区访问或登录 cookies，可用：

```bash
python tools/materialize_audio.py --case <case_id> --cookies-from-browser chrome
```

## 测试你的模型

任意 ASR 模型都可以接入。只需要为每个 case 输出一个 UTF-8 文本文件：

```text
predictions/my_model/<case_id>.txt
```

批量评分：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model \
  --out results/my_model.score.json
```

单条评分：

```bash
python tools/score_transcript.py \
  --case youtube_bbc_ai_vocab_001 \
  --prediction predictions/my_model/youtube_bbc_ai_vocab_001.txt
```

如果模型使用了上下文术语或热词，请单独跑一份结果，不要和 zero-shot 混在一起：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_with_context \
  --hotword-mode with_context_terms \
  --out results/my_model_with_context.score.json
```

如果模型有原生热词表、decoder bias 或 vocabulary bias：

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_native_hotwords \
  --hotword-mode native_hotwords \
  --out results/my_model_native_hotwords.score.json
```

## 怎么看分数

核心输出是文本侧 90 分：

- `weighted_text_score_90`：加权综合文本分，适合做本项目内模型对比。
- `primary_error_rate`：每条样本的主错误率，中文/日语/粤语多用 CER，英文多用 WER。
- `keyword_f1`：专名、产品名、术语、人名等关键内容是否识别对。
- `number_f1`：数字、金额、时间、版本号等是否识别对。
- `hallucination_score`：是否有过长输出、重复、静音幻觉等问题。
- `script_normalized_text_score_90`：粤语繁简归一诊断，帮助区分内容错误和字形差异。
- `japanese_orthographic_alias_text_score_90`：日语表记差异诊断，帮助区分听错和写法差异。

`weighted_text_score_90` 是本项目自己的综合分，不是业界统一标准。正式比较时建议同时看分语言、分场景明细，不要只看一个总分。

## 复现说明

仓库不重新分发第三方音视频，因为这些内容属于原发布者。复现流程是：

1. 使用 `data/benchmark_manifest.v1.json` 读取来源 URL 和时间段。
2. 用 `tools/materialize_audio.py` 在本地重建音频片段。
3. 用任意 ASR 模型生成预测文本。
4. 用 `tools/score_transcript.py` 评分。

这样可以避免把大文件、模型权重、下载缓存和账号状态放进仓库，同时保留足够的信息让别人复现同一批测试。

## 目录

- `data/benchmark_manifest.v1.json`：正式测试集清单。
- `data/benchmark_references/`：最终 reference 文本。
- `tools/materialize_audio.py`：从公开来源重建本地音频。
- `tools/score_transcript.py`：评分脚本。
- `tools/validate_manifest.py`：数据校验。
- `tools/summarize_dataset.py`：数据分布汇总。
- `docs/metrics.md`：指标定义。
- `docs/benchmark_principles.md`：测评原则。
- `docs/dataset_card.md`：数据卡。

## License

代码使用 MIT License。数据集元数据和项目整理的 reference 文本用于研究和评估；上游媒体和来源字幕仍受原发布者条款约束。见 [DATA_LICENSE.md](DATA_LICENSE.md)。
