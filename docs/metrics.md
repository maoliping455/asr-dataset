# ASR 测评指标

目标不是只看排行榜上的 WER，而是衡量一个模型在本地字幕、笔记、会议和长音频工作流里的可用性。

## 一级指标

### 1. 内容准确率

- `CER`：字符错误率，主要用于中文、日语、粤语、阿拉伯语等不稳定分词场景。
- `WER`：词错误率，主要用于英文和空格分词语言。
- `Hybrid TER`：混合 token 错误率，把 CJK 字符、拉丁单词、数字、阿拉伯文字分别 token 化，用于中英混合、日英混合和技术口播。

评分脚本会按 case 的 `primary_error_metric` 选择主误差：

- `cer`：中文、日语、粤语、阿拉伯语。
- `wer`：纯英文。
- `hybrid_ter`：混合语言、技术词密集片段。

### 2. 专名和术语

每个 case 的 `keywords` 包含专名、产品名、缩写、地名、人名或领域术语。脚本计算：

- `keyword_precision`
- `keyword_recall`
- `keyword_f1`

这个指标会直接惩罚把 `Blackwell`、`CUDA`、`DHH`、`OpenAI`、`粵語` 这类词识别错的模型。

### 3. 术语提示 / 热词适配能力

每个 case 可以额外维护 `hotwords`，用于模拟真实业务里提前知道的词表，例如课程名、客户公司名、项目名、产品名、竞品名、人名和缩写。

`hotwords` 不是越多越好。只放少量高价值、容易错、错了影响业务理解的词；普通地名、常见词、临时昵称和模型裸跑通常能识别的词不放。

热词评测必须按实现机制分 track：

- `zero_shot`：不把 `hotwords` 传给模型，测模型裸识别能力。
- `with_context_terms`：把 `hotwords` 当作上下文术语提示传给 Qwen3-ASR 这类生成式 ASR，测 prompt/context 能否帮助写对术语。
- `native_hotwords`：把 `hotwords` 传给真正支持热词表、词表 ID 或 decoder bias 的模型，例如 FunASR / Paraformer。

`score_transcript.py` 会输出 `hotword_mode`、`term_prompt_mechanism`、`hotword_f1`、`hotword_matched_terms` 和 `hotword_missing_terms`。`hotword_f1` 不进入当前 `text_score_90`，避免把模型裸能力和词表适配能力混在一个分数里；同一个模型应分别汇报 `score_zero_shot`、`score_with_context_terms` 和必要时的 `score_native_hotwords`。

Qwen3-ASR 的本地 MLX 路径没有原生加权热词表；当前通过 `system_prompt` 注入上下文术语。它可能提升专名写法，也可能把提示词错误注入输出。因此 `with_context_terms` 结果必须同时看 `text_score_90`、`length_ratio`、`quality_flags` 和 `context_prompt_failure`，不能只看 `hotword_f1`。

### 4. 数字和单位

抽取阿拉伯数字、百分号、金额、时间、版本号等，计算 `number_f1`。数字错通常比普通虚词错更严重，尤其是会议纪要、财经、技术发布会。

评分时会把常见中文数字归一化，例如“一百五十”和 `150` 视为同一个数字，“五百”和 `500` 视为同一个数字，“三、二、一”和 `321` 视为同一个连续口播数字；但 `INSTA360` 被识别成“三零零”这类实质数字错误仍会扣分。

### 5. 数学 ITN / 公式格式化

数学课堂、理科讲解和公式密集音频单独评估 `math_formula_itn_evaluation`。这类 case 可以在 manifest 中标记：

```json
"formatting": {
  "itn_domain": "math_formula"
}
```

脚本会额外输出：

- `math_normalized_error_rate`：把中文数字读法和常见运算词归一后再计算的内容错误率。例如 `八十八拆成八乘十一` 和 `88拆成8×11` 会尽量按等价内容处理。
- `math_itn_format_gap`：`primary_error_rate - math_normalized_error_rate`，表示原始 CER 中有多少主要来自数学书面格式差异。
- `math_itn_format_score_10`：0 到 10 的格式化诊断分，只用于观察模型或后处理链路是否直接输出书面数学格式。

数学 ITN 不应混入核心 ASR 内容准确率。对于“学生课堂视频转文本再总结”的主场景，优先看 `math_normalized_error_rate` 判断是否听对，再把 `math_itn_format_*` 作为下游格式化能力单独比较。

### 6. 粤语繁简归一诊断

粤语 reference 常来自香港/粤语字幕，通常使用繁体和粤语正字；本地 ASR 模型常输出简体混合粤语字。原始 CER 会把 `歡迎/欢迎`、`電腦/电脑`、`觀眾/观众` 这类字形差异当成错误，因此粤语必须同时看两套口径：

- `text_score_90`：严格 raw 分，保留 reference 原始字形和粤语正字要求。
- `script_normalized_text_score_90`：把 reference 和 prediction 都用 OpenCC `t2s` 做繁转简后再评分，用于观察内容识别是否真的错。
- `script_normalized_delta_text_score_90`：繁简归一后的提升幅度。提升很大时，说明低分主要来自繁简/字形口径，而不是语音内容完全没听对。

当前 `script_normalized_*` 只对 `language == "yue"` 自动输出，不进入默认主排行榜；正式报告应同时展示粤语 raw 分和繁简归一分。

### 7. 日语表记ゆれ诊断

日语 ASR 评估必须区分“听错”和“表记不同”。同一个口播内容可能被写成汉字、平假名、片假名、罗马字或数字，例如 `Haru/春/ハル`、`YouTube/ユーチューブ`、`Bite Size Japanese/バイトサイズジャパニーズ`。这类差异会显著影响 raw CER，但不一定代表语音内容没听对。

日语主分仍使用严格 `text_score_90`，同时输出诊断字段：

- `japanese_furigana_stripped_text_score_90`：去掉 reference/prediction 中疑似非口播振假名括注后重算，例如 `制度（せいど） -> 制度`。
- `japanese_furigana_stripped_delta_text_score_90`：振假名括注对分数的影响。
- `japanese_orthographic_alias_text_score_90`：应用 case 级 `orthographic_aliases` 白名单后重算。
- `japanese_orthographic_alias_delta_text_score_90`：可接受表记差异对分数的影响。

`orthographic_aliases` 必须按 case 显式维护，不能全局猜测。推荐格式：

```json
"orthographic_aliases": [
  {
    "canonical": "Bite Size Japanese",
    "aliases": ["バイトサイズジャパニーズ", "バイトサイズジャパニーズポッドキャスト"]
  },
  {
    "canonical": "Haru",
    "aliases": ["春", "ハル"]
  }
]
```

日语别名诊断不进入默认主排行榜。它的作用是解释 raw CER 中有多少来自可接受表记差异；明显错误的 reference 应修复或降级，不能靠别名规则掩盖。

### 8. 幻觉和冗余

脚本目前用两个可解释信号做惩罚：

- `length_ratio`：预测文本长度相对 reference 的比例，过长或过短都会扣分。
- `repetition_rate`：重复 n-gram 比例，检测模型循环输出。

负样本会把目标文本设为空，模型若输出大段内容，综合分会大幅降低。

### 9. 标点和断句

`punctuation_f1` 衡量常见中英文标点集合的匹配程度。这个分数权重较低，因为标点可以二次处理，但字幕工作流里仍然有价值。

如果某个 ready reference 明确没有人工标点，manifest 中应标记 `reference.punctuation = "none"`。这类 case 仍可用于内容、专名、数字和幻觉评分，但 `score_transcript.py` 会把标点评估记为 `ignored_reference_has_no_punctuation`，不因 reference 无标点而扣分。

### 10. 性能指标

性能指标不由 `score_transcript.py` 从文本中推断，需要模型运行器记录：

- `rtf`：real-time factor，越低越快。
- `peak_memory_gb`：峰值内存。
- `model_disk_gb`：模型落盘大小。
- `first_token_latency_ms`：流式场景首字延迟。
- `install_friction`：人工评分，依赖冲突、是否支持 Apple Silicon/MLX/Metal。

## 综合分

默认综合分为 100 分：

- 主内容准确率：50%
- 术语/专名：15%
- 数字/单位：10%
- 幻觉/冗余：10%
- 标点/断句：5%
- 性能：10%

`score_transcript.py` 当前只计算文本侧 90 分，并把性能字段预留在输出 JSON 中。最终排行榜应把模型运行器记录的性能分并入。

### Case 权重层级

每个 case 的原始 `weight` 表示场景和样本重要性。正式汇总时还会叠加确认度权重：

- `user_confirmed_real_audio`：用户已听校确认的真实外部音频，`confidence_weight_multiplier = 3.0`。
- `auto_screened_public_subtitle`：来自公开字幕/transcript，并通过严格自动筛选的 Gold case，`confidence_weight_multiplier = 2.0`。
- `ready_reference`：内部使用的其他 ready reference，`confidence_weight_multiplier = 1.0`。公开 Gold-only 数据集默认不包含此层。
- `other`：未人工复核的候选、deferred case 或其他低可信样本，`confidence_weight_multiplier = 0.2`。

最终汇总使用 `effective_weight = weight * confidence_weight_multiplier`。这样用户确认过的真实音频优先级最高，其余 ready case 其次，其他候选最低。

## 场景权重建议

不同场景的重点不同：

- 技术演讲：术语、英文缩写、数字权重上调。
- 会议/播客：长音频稳定性、说话人、断句权重上调。
- 方言/口音：CER 与关键词召回优先。
- 歌曲/背景音乐：幻觉、漏词、重复输出优先。
- 负样本：幻觉惩罚优先。

## 排名规则

正式排名只使用满足以下条件的 case：

- `reference.status == "ready"`。
- `reference.review_level` 是 `user_confirmed_real_audio` 或 `auto_screened_public_subtitle`。
- reference 至少来自公开字幕/transcript 或人工校对文本；不能来自本地 ASR 模型输出。
- 片段时长建议 60 到 180 秒；长音频能力另设 10 到 30 分钟样本。
- 同一来源不要贡献超过 15% 的正式分数。
