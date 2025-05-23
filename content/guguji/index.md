+++
title = '咕咕记'
slug = 'guguji'
info = '通过播客音频的转写与说话人分离，结合LLM生成内容摘要与结构化笔记，实现从“声音”到“知识”的自动化转化流程。'
stack = ["ASR", "Speaker Diarization", "LLM Sumarize"]
date = 2025-04-21T10:59:58+08:00
draft = false
+++


![cover.png](/guguji/cover.png)

最近迷上了一档名为 All-In with Chamath, Jason, Sacks & Friedberg的播客节目，节目邀请行业资深人士，围绕经济、政治、科技、社会等多方面话题展开深度探讨。每期播客时长都超过一小时，对我来说既是享受，也是挑战。很多地方我听不太懂，尤其是俚语和玩笑部分。为了更好地理解播客内容，我希望能将播客转写成文本。

## 已有解决方案调研

市场上有很多音频转写和AI摘要工具，我主要调研了以下三个产品：通义听悟、Podwise、NotebookLM



![reserch.png](/guguji/1.png)
![reserch.png](/guguji/2.png)
![reserch.png](/guguji/3.png)
![reserch.png](/guguji/4.png)
![reserch.png](/guguji/5.png)



我想要有一个能帮助我渐进式理解播客内容的工具，理想的使用流程是：

**播客摘要 → 结构化笔记（重要观点 + 举例、数据等说明）→ 精听感兴趣的部分**。

目前市面上的工具还不够理想：

- **Podwise** 功能接近，但交互体验不便：Mindmap 和转录文本分在两个 tab，需要频繁切换查看；Mindmap节点默认收起，手动展开很麻烦；最关键的是免费版每月仅支持 4 个播客生成 AI 内容，且不支持导出。
- **通义听悟** AI生成的内容啰嗦、缺乏结构，阅读体验差。
- **NotebookLM** 的模型能力很强，摘要和结构化笔记重点明确且自带引用，但交互复杂，不支持音频与文本联动，不能边听边看，而且上传音频有 200M 限制，无法处理多数 2 小时以上的播客。

因此，我打算根据自己的使用习惯，自制一个播客小助手，打造更符合个人节奏和理解路径的工具，顺便学习一下相关技术。

我计划使用 Streamlit 快速搭建播客小助手的原型，聚焦以下核心功能：

**播客RSS解析 → 下载音频  → 自动转写 → 说话人分离 → 摘要生成 → 结构化笔记展示**。

## 技术栈调研与选择

### **音频转写方案**

- **Whisper**
    
    [Whisper](https://github.com/openai/whisper) 是 OpenAI 开源的语音识别模型，基于 Transformer 自回归架构，具备以下核心能力：
    
    - 支持多语言识别，包括中文和中英文混合语音；
    - 语言自动检测；
    - 支持将非英语语音直接翻译为英文文本；
    - 自动根据语速与停顿进行文本切分，并添加时间戳，可直接用于字幕生成。
    
    Whisper 提供 5 个模型尺寸（tiny、base、small、medium、large），以及优化推理速度的 large-turbo 版本，分别测试了tiny、base、small和turbo模型，结果：
    
    - tiny、base、small 在英文语音场景下识别准确率和速度表现良好但中文识别率相对偏低；
    - turbo 对中文及中英混合识别准确率显著提高，但存在两个问题：默认输出不带标点；部分输出为繁体字。上述问题可通过参数 initial_prompt="以下是普通话句子。" 解决，实测有效。
- **Whisper衍生版本**
    - **Faster Whisper**：Whisper 的高性能实现版本，基于 CTranslate2 推理引擎，在保证精度的前提下：推理速度提升约 4 倍，降低内存占用，适合部署在资源受限设备上。
    - **WhisperX**：扩展 Whisper 能力的增强版，结合 pyannote.audio 和 wav2vec 2.0，支持说话人分离和单词级时间戳，更适合需要精细对齐或语音分析任务的应用场景。
- **SenseVoice**
    
    [SenseVoice](https://github.com/alibaba-damo-academy/SenseVoice) 是阿里达摩院开源的多模态语音基础模型，具备：
    
    - 自动语音识别（ASR），支持多语言；
    - 语言识别
    - 情感识别（SER）；
    - 语音事件监测（AED）。
    
    SenseVoice 有两个尺寸，small和large，开源的是small模型，它采用的是非自回归端到端框架。测试了一下small模型，结果：
    
    - 转写速度非常快，官方数据称比Whisper-Large快约 15 倍；
    - 英文场景下，转写准确率仍略低于 Whisper-Base；
    - 输出不含时间戳，不能进行后续说话人分离工作。

在 MacBook M2（32GB 内存）上，我使用一段时长为 19 分 13 秒的音频对比了以下3个模型的转写速度，结果如下：

SenseVoice-Small： 37.28 秒 
Whisper-base： 43.04 秒 
FasterWhisper-base： 125.54 秒 

SenseVoice-Small最快，但转写结果没有时间戳；FasterWhisper在mac机子上不能发挥其优势。

**综合转写准确率、速度、语言支持及时间戳需求，采用 `Whisper-base` 作为默认模型。**

### **说话人分离方案**

- **pyannote.audio**
    
    [pyannote.audio](https://github.com/pyannote/pyannote-audio) 是一个基于 PyTorch 的开源工具包，专注于说话人识别任务，提供了多个高性能的预训练模型和模块，包括：
    
    - 语音活动检测
    - 说话人变化检测
    - 重叠语音检测
    - 说话人嵌入
    
    其完整的说话人分离管道 pyannote/speaker-diarization-3.1 包含上述模块，并集成成一条高精度的推理流程：
    
    ![图片](/guguji/6.png)
    
    该管道在实际测试中表现稳定、易于使用，社区活跃，支持本地部署，适合播客这类多说话人的语音内容。
    
- **3D speaker**
    
    [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) 是阿里达摩院开源的一套用于说话人验证、识别和分离的工具包，支持单模态和多模态（音频 + 视频）场景。
    
    其音频说话人分离流程主要包含：
    
    - 重叠检测（可选）
    - 语音活动检测
    - 语音段切分
    - 说话人嵌入提取
    - 聚类与分离
    
    代码中部分模块（如语音分割、重叠检测）依赖于 pyannote/segmentation-3.0。虽然提供了较完整的端到端说话人分离推理脚本 infer_diarization.py，便于命令行快速调用，但缺乏模块化封装和公开 API 文档，不适合直接在项目中灵活嵌入。
    
- **WhisperX**
    
    [WhisperX](https://github.com/m-bain/whisperx) 是基于 OpenAI Whisper 的增强版工具，集成了：
    
    - Whisper 的语音识别能力
    - pyannote.audio 的说话人分离能力
    - wav2vec2.0 的强力对齐模块，支持词级时间戳精度
    
    虽然 WhisperX 提供了较为完整的一体化体验，但存在以下限制：
    
    - 不支持在 macOS 上使用 pyannote.audio 的 MPS 加速
    - 词级时间戳不是我的刚需，反而提升了资源消耗
    - 模块间耦合较强，不便于灵活替换组件

**综合考虑可扩展性与调用便利性，最终选择使用 `pyannote/speaker-diarization-3.1` 作为说话人分离方案。**

## **内容生成方案（摘要 + 结构化笔记）**

播客文本长度 LLM在处理长文本时普遍面临两个挑战：

- **上下文依赖性缺失**：随着输入文本长度增加，模型对早期信息的记忆能力逐渐下降，容易造成关键信息的遗失或理解偏差。
- **计算复杂度高**：以 Transformer 为代表的主流架构，其注意力机制的计算复杂度为 O(n^2)，输入长度越长，计算成本和内存开销呈指数级增长。

LLM模型通常设有最大上下文长度限制（如 4K、8K、32K token）。为了避免超出限制、降低资源消耗，同时提升信息处理的准确性，需要对长文本进行合理分块。

### 文本分块

LangChain 的 `text_splitter` 模块提供多种文本分割器（TextSplitter），用于适配不同类型的文档结构与处理需求。所有分割器均支持 `chunk_size` 和 `chunk_overlap` 参数，可控制每个文本块的最大 token 长度与重叠部分，从而满足大语言模型上下文窗口的输入要求。

**按规则分割器**

- **CharacterTextSplitter**
    
    按指定字符（如换行符 `\\n`）拆分文本，可自定义分隔符。
    
- **RecursiveCharacterTextSplitter**
    
    多级字符分割器，优先按段落（如 `\\n\\n`）、句子（`\\n`）、符号（如 `.`、`,`）等逐层拆分。
    
- **TokenTextSplitter**
    
    使用编码器（如 OpenAI 的 `tiktoken`）将文本编码为 token，并按 token 数量进行分块，适用于需要精确控制模型输入长度的场景。
    
- **MarkdownHeaderTextSplitter**
    
    根据 Markdown 标题级别（如 `#`、`##`、`###`）拆分文档结构。
    
- **PythonCodeTextSplitter**
    
    按照 Python 语言的结构（如函数、类定义）拆分代码。
    
- **HTMLSectionSplitter**
    
    基于 HTML 层级结构进行拆分，识别常见 HTML 标签（如 `<div>`、`<p>`、`<h1>`）并进行结构化分块。
    
- **LatexTextSplitter**
    
    利用 LaTeX 中的结构指令（如 `\\documentclass{}`、`\\begin{}`）进行拆分，适合科学论文类文档。
    
- **RecursiveJsonSplitter**
    
    将嵌套的 JSON 对象递归拆分为较小片段，同时保留其层级结构和键值对应关系。
    

**模型辅助分割器**

- **SpacyTextSplitter**
    
    使用 spaCy 的依存句法分析器进行句子级分割，再根据设置拼接句子形成文本块。默认使用 `en_core_web_sm` 模型，适用于语法规范、标点清晰的标准书面语文本。
    
- **SemanticChunker**
    
    先将文本分割为句子，再使用嵌入模型（如 OpenAI 的 `text-embedding-3-large`）计算相邻句子之间的语义相似度，并在语义突变显著处插入断点，生成语义连贯的分块。可配置相似度阈值策略（如百分位数、标准差等）。
    

**语义分块 vs 固定分块**

论文《*Is Semantic Chunking Worth the Computational Cost?*》研究了语义分块在 RAG 系统中的实际效益。比较了三种方法在文档检索、证据检索和基于检索的问答任务中的表现：

- **固定大小分块（Fixed-size Chunker）**
    
    按句子数量固定划分文本，作为基线方法。
    
- **基于断点的语义分块（Breakpoint-based Semantic Chunker）**
    
    通过检测连续句子间的语义突变位置进行分割，强调连贯性。
    
- **基于聚类的语义分块（Clustering-based Semantic Chunker）**
    
    先生成句向量，再基于聚类算法对句子进行语义分组，允许非连续句子形成语义单元。
    

结果表明，语义分块的计算成本并没有通过一致的性能提升来证明其合理性。固定大小分块对于实际的 RAG 应用来说仍然是一个更有效和可靠的选择。

**其他分割方法**

除了 LangChain 提供的分割器外，论文 *《Sequence Model with Self-Adaptive Sliding Window for Efficient Spoken Document Segmentation》* 提出了一个模型：

- **SeqModel**
将文档分割任务建模为“句子级序列标注问题”，通过自适应滑动窗口动态构造上下文窗口，再使用 Transformer 对句子进行编码，并预测哪些句子是段落起始点。该方法推理效率高，参数量小，尤其适用于 ASR（自动语音识别）转录文本的段落恢复任务。

测试了一下`SeqModel` ，其分割效果和指定字符`.`分割效果近似。

**综合考虑效率与后续处理效果，我选择了以说话人切换为主、固定长度为辅的分块方式。该策略既能保持语义自然断点，又兼顾了上下文控制的灵活性。**

### 文本摘要

LangChain 提供了 `load_summarize_chain` 方法，用于构建摘要链，目前支持以下三种链式策略：

- **stuff**
    
    将全部文本直接塞入 prompt，一次性生成摘要，适用于上下文窗口足够大的模型，但测试开源的大语言模型时，即使输入量在范围内，总结质量仍较低。
    
- **map-reduce**
    
    首先将文本分块，对每个分块独立生成摘要，再将所有摘要合并，并进行二次摘要整合。
    
- **refine**
    
    采用逐步精炼（Refinement）策略。先对首个文本块生成初始摘要，随后每处理一个新块时，结合该块与已有摘要生成新版本，实现逐步增强的摘要质量。该方法具有上下文依赖性，处理过程是串行的，无法并行。
    

**综合考虑处理效率和摘要质量，我最终选择`map-reduce`方法。**

### **LLM 选择**

使用播客《Fixing the American Dream with Andrew Schulz》的转写文本，在摘要任务上对比几个了最近发布的开源模型：`Gemma 3（1B、4B、12B、27B）`，`Deepseek-R1:7B`**，**`qwq:32B` 

![图片](/guguji/7.png)


在本轮摘要任务中，模型表现如下：

- Gemma3: 1B 和 Deepseek-R1: 7B 存在明显幻觉问题，输出内容与原文偏差较大；
- Gemma3: 4B 表达空泛，未能抓住核心要点；
- Gemma3: 12B 能识别出关键主题，但内容组织较为松散，缺乏进一步的凝练；
- Gemma3: 27B 在主题识别、信息提取和结构组织方面表现优秀，具备一定深层理解能力，逻辑清晰、层次分明；
- QWQ: 32B 在信息覆盖和表达精准度上表现最佳，但存在少量内容重复的问题，且在内容深度理解方面略逊于 Gemma3: 27B，推理耗时为后者的约 3.4 倍。

**综合考虑准确性与推理效率，最终选择 `Gemma3: 27B` 作为摘要和后续结构化笔记模型。**

### 结构化笔记

我希望能够基于播客的 **转写文本** 和 **shownotes（时间段概要）**，自动提取每个时间段内的核心谈话主旨、分论点，以及用于支撑这些论点的事实、例子、数据等内容。

基于shownotes来梳理转写文本内容，我尝试了两种方法：

**方法一：时间驱动的结构化提炼**

根据 shownotes 的时间戳将转写文本进行切分，并对每一段文本进行结构化提炼：若分段长度未超过 chunk_size，则直接提炼主旨；若超过限制，则再细分为更小块分别提炼，并将结果整合；最终输出按时间顺序组织。

**方法二：语义驱动的内容问答（RAG 模式）**

将全部转写文本进行嵌入，并存入向量数据库。将每条 shownote作为 query 发起语义检索，获取相关文本片段；再使用 LLM 基于 shownote 与检索内容进行问答。

![图片](/guguji/8.png)

方法一按时间顺序组织内容，结构清晰，易于阅读，且可与摘要链共用分段摘要，处理效率高。

方法二更注重语义关联，适合跳跃式、结构松散的谈话内容，且可引用原始文本片段，快速验证原文，后续可扩展更为自由的问题提问。但方法二依赖嵌入质量与语义检索的准确性，调参与验证成本较高。我使用的Embedding模型是`BAAI/bge-base-en-v1.5` ， 向量数据库是`Chroma`，在不对搜索结果过滤的情况下容易引入不相关的文本内容，影响回答质量。

**综合考虑调试成本与当前目标，优先采用结构清晰、效率更高的第一种方法，将第二种方法作为后续功能扩展方向。**

### **前端 & 应用框架**

- **Streamlit**：简洁快速的 Python Web 框架，支持组件交互、Markdown 渲染、音频播放等。

## Demo 实现

### 功能清单

- 支持播客 RSS URL 解析与音频获取
- 音频自动转写（ASR）
- 说话人分离（Speaker Diarization）
- 结合 Shownotes 时间戳、说话人信息与固定长度策略，进行文本块切分
- 基于 LLM 自动生成摘要与结构化笔记

### **流程图**

![图片](/guguji/12.png)

### **名称和Logo**

**名称：** 结合工具的功能和动物/拟声词，让LLM帮我脑暴几个名称，最终选择了”咕咕记“，谐音”咕咕鸡“，好记又好玩。

**Logo：** 结合钢笔、小鸡脑袋和声波的形状，用Figma简单绘制了个Logo。

![图片](/guguji/11.png)

### **交互原型**

![图片](/guguji/9.png)

### 最终Demo

[Demo](/guguji/guguji.mp4)

### **功能扩展计划**
- 实现音频与文本的双向联动跳转（点击文本定位音频 & 音频定位文本）
- 支持基于播客内容的自由问答（RAG）
- 异步处理转写与AI内容生成，提高处理效率
- 引入数据库存储实现内容存储与增删改管理