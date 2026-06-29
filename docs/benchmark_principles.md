# ASR Benchmark Principles

目标是构建一个专业、可复现、但服务个人真实场景的本地 ASR 测评体系。它不是为了复刻公开排行榜，而是为了回答一个实际问题：在 Mac 本地运行时，哪个 ASR 模型最适合课堂、销售、会议、演讲、播客和多语言转录工作流。

## 1. 总体原则

- 以真实使用场景驱动测试集，而不是以模型排行榜驱动测试集。
- 不使用现成公开 ASR benchmark 的原始测试集，避免分数被训练集污染或公开榜单惯性影响。
- 第三方素材只保存必要片段、URL、元数据和人工 reference，不保存完整外部音频、完整字幕或完整 transcript。
- 每个正式 case 必须有人工确认 reference；公开视频字幕、B 站 AI 字幕、播客 transcript 和模型草稿都只能作为草稿来源。
- 用户亲自听校确认的真实音频优先级最高，其次是其他 ready reference，再其次是未确认候选。
- 评分要解释模型是否适合实际工作流，而不是只输出一个无法诊断的总分。

## 2. 数据分层

### Gold Set

正式结论只使用 Gold Set。Gold Set 可以来自两类确认路径：

- `reference.review_level = "user_confirmed_real_audio"`
- `reference.review_level = "auto_screened_public_subtitle"`

其中用户亲自听校确认的真实音频权重最高，严格自动筛选通过的公开字幕/transcript 用于扩大语言和场景覆盖。两者都必须有最终 reference 文件，且 reference 文本不能来自本地 ASR 模型输出。

### Backup / Ready Set

内部维护时可以保留 reference 已整理完成、但尚未进入 Gold 的 backup/ready case，用于候选复查、工程验证和错误分析。公开仓库默认不包含这部分数据。

Backup/ready 不能和 Gold 混为一类解释；正式公开报告必须说明使用的 scope。

### Candidate Set

已下载或记录音频、字幕、草稿和元数据，但还没完成听校。Candidate Set 只用于排队复查和准备工作，不进入正式模型排名。

### Synthetic / Self-Owned Set

项目自有脚本、自录、合成音频和负样本。它们适合做端到端跑通、噪声控制、边界行为和快速粗排，但不能替代真实外部音频。

## 3. 场景覆盖目标

第一阶段优先覆盖：

- `student_lecture`：课堂视频转文本，用于后续总结和笔记整理。
- `sales_call`：销售对客语音，关注姓名、机构、金额、时间、异议和承诺。
- `video_meeting`：多人会议，关注插话、弱麦克风、任务项和时间线。
- `online_talk`：B 站 / YouTube 演讲，关注专名、技术词、长句和中英混合。
- `podcast_interview`：自然口语、多人访谈、长音频稳定性。
- `news_or_narrative`：清晰但结构复杂的叙事音频。
- `noisy_audio`：背景音乐、弱人声、混响、噪声和负样本。

第二阶段再扩展：

- 粤语。
- 主流方言。
- 更长的 10 到 30 分钟音频。
- 真正流式麦克风场景。

## 4. Reference 标注纪律

reference 是测评质量的核心资产。修改 reference 的唯一正当理由是听音发现 reference 本身不准，而不是某个模型表现差。

正式入池前必须确认：

- 音频片段和 reference 时间范围对齐。
- 专名、人名、机构名、产品名、课程名、数字和公式已核对。
- 口误、重复、停顿和插话按实际语音保留，除非该 case 明确是整理型 reference。
- 如果 reference 没有句末标点，标记 `reference.punctuation = "none"`。
- 每个 case 都必须判断 `hotwords`；没有可用术语时写 `hotwords: []`。
- reference 变更后必须重新评分，并保留变更原因。

## 5. 评分维度

正式评分不只看 WER/CER。核心维度包括：

- 内容准确率：`CER`、`WER`、`Hybrid TER`。
- 归一化内容准确率：对数字、单位、数学公式、大小写、常见 ITN 差异做必要归一后判断是否听对。
- 专名和术语：`keyword_f1`。
- 数字和单位：`number_f1`。
- 数学 ITN / 公式格式化：`math_normalized_error_rate`、`math_itn_format_gap`、`math_itn_format_score_10`。
- 术语提示 / 热词适配：`zero_shot`、`with_context_terms`、`native_hotwords` 分开。
- 幻觉和稳定性：长度比、重复率、静音/噪声输出。
- 标点和断句：低权重独立诊断，不应压过内容准确率。
- 性能：RTF、峰值内存、模型大小、启动时间、流式延迟、安装复杂度。

## 6. ITN 和格式化原则

ASR 内容识别和文本格式化必须分开评价。

例如数学课堂中：

- `八十八拆成八乘十一`
- `88拆成8×11`

这两种写法对 ASR 内容来说通常等价。前者是听到的口语形式，后者是更适合笔记的书面形式。原始 CER 会把它们算成大量字符错误，因此需要单独记录：

- raw error：原始字符串差异。
- normalized content error：归一化后内容是否正确。
- ITN / formatting score：模型或后处理链路是否直接输出更好的书面格式。

对于后续由大模型总结、整理笔记或生成会议纪要的工作流，ITN 可以交给后处理模型完成，不应该被混同为 ASR 声学识别错误。

## 7. 热词和上下文术语原则

不同模型的“热词”机制不能混排。

- `zero_shot`：不传术语，测模型裸能力。
- `with_context_terms`：Qwen3-ASR 这类生成式 ASR 使用 prompt / context 注入术语。它不是原生加权热词表。
- `native_hotwords`：FunASR / Paraformer 这类真正支持词表 ID、decoder bias 或热词权重的模型。

case 的 `hotwords` 应少而准，优先放业务中可能提前知道且容易错的词：

- 人名、公司名、产品名、课程名、专有机构名、罕见术语、英文缩写。

避免放：

- 普通地名。
- 普通中文词。
- 语气词、拟声词、歌词哼唱词。
- 模型裸跑大概率能识别的词。
- 过长句子或整段 reference。

## 8. 公正性和可复现性

为了保持测评可信：

- 不因为某个模型分数低而修改 reference。
- 不把 prompt-context 结果和 native-hotword 结果混在同一排名。
- 不把 Gold Set、Ready Set、Candidate Set 混成一个无法解释的总分。
- 每次模型测试记录模型 ID、权重来源、量化方式、命令、语言参数、运行环境、运行日期和结果文件。
- 每次规则变化后，保留旧结果可解释；必要时建立新的结果版本。
- 评分脚本和 manifest 校验必须通过后，结果才可用于报告。

## 9. 报告结构

每个模型报告应至少包含：

- 模型版本和本地运行方式。
- 覆盖的 case 集合和确认度分层。
- Gold Set 分数。
- 非 Gold 分数必须单独标记 scope；公开默认不发布非 Gold 集合。
- zero-shot 与术语提示 / 热词结果分开。
- 文本侧分数和性能侧分数分开。
- 主要失败类型：漏转、幻觉、专名错误、数字错误、格式化问题、语言识别问题、长音频稳定性问题。
- 适合和不适合的实际场景。

## 10. 后续优化路线

短期目标：

- 为每个 Gold case 生成 `case_card.md`，记录来源、片段、reference 状态、热词、ITN 域、复查记录和回测摘要。
- 建立 Gold Set 专用 leaderboard。
- 把 `text_score_90` 拆出更清晰的 raw / normalized 诊断字段，避免数学和 ITN 差异误伤 ASR 内容分。
- 每新增或更新 ready case 自动运行 Qwen3-ASR-1.7B 4bit 回测。

中期目标：

- 扩充 Gold Set 到每个第一阶段主场景至少 5 条。
- 增加更稳定的人工复查工作流：候选、草稿、用户确认、入池、回测、报告。
- 对销售、会议、课堂分别建立更细的错误分类。
- 对英语、中文、日语分别维护语言特定归一化规则。

长期目标：

- 支持长音频、流式 ASR、说话人分离和时间戳质量评估。
- 建立模型版本历史，比较同一模型不同量化、不同解码参数、不同热词机制。
- 构建一个个人但专业的 ASR benchmark：数据可信、场景真实、评分可解释、结果可复现。
