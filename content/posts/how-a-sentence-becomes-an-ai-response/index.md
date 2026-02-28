---
title: "一句话是怎么变成 AI 回复的：LLM 的工作原理"
date: 2026-02-15
draft: false
summary: "从 Tokenization 到自回归生成，用一句「帮我查一下北京明天的天气」走完 LLM 处理全流程。面向 Agent 开发者的灰盒认知指南。"
description: "从 Tokenization 到自回归生成，用一句「帮我查一下北京明天的天气」走完 LLM 处理全流程。面向 Agent 开发者的灰盒认知指南。"
tags: ["LLM", "Agent", "Tokenization", "Embedding", "Attention", "RAG"]
categories: ["AI Agent Engineering"]
series: ["面向 Agent 开发者的 LLM"]
---

> **本文面向 Agent 开发者和 AI 应用工程师**，用一个真实的输入示例走完 LLM 处理全流程。你不需要懂线性代数，但读完后你会理解 LLM 的每一个“玄学行为”背后的技术原因。
>
> 📌 **本文是三篇系列的第一篇**。本篇讲模型如何处理输入并生成输出，第二篇[《模型的能力是怎么来的：从预训练到 RLHF》](/posts/how-model-capabilities-come-from/)讲模型能力的来源，第三篇[《推理服务是怎么影响你的 Agent 的：推理框架与架构决策》](/posts/inference-frameworks-for-agent-architects/)聚焦推理层的架构决策。

## 1. 引言：为什么 Agent 开发者需要理解 LLM 原理

你让 AI 比较 9.11 和 9.9 的大小，它自信地告诉你 “9.11 更大”。你把 Prompt 里的逗号改成句号，Agent 的输出就从 JSON 变成了自然语言，代码解析直接崩溃。你让它 “不要胡说八道”，它依然在不知道的问题上一本正经地编造答案。

这些看起来像玄学。但读完本文你会发现，每一个都有清晰的技术解释。

### 谁适合读这篇文章？

- 正在开发 Agent / AI 应用的工程师
- 用过 LLM API 但不了解底层原理的开发者
- 被 Agent 的“玄学问题”折磨过的人
- **不适合**：想深入研究 Transformer 数学推导的研究者（本文没有公式）

### 1.1 当 API 不再确定

对于应用层开发者来说，过去几十年的经验告诉我们：计算机是确定的。`if a > b`，结果永远是 True 或 False；API 只要参数对，返回结果永远一致。

但在大模型时代，我们面对的是一个概率性的黑盒。

当你把 LLM 仅仅视为一个神奇的 API（输入 Prompt，输出结果），你很快会在开发 Agent 系统时撞墙：

- **脆弱性**：Prompt 改了一个标点符号，输出格式就变了。
- **不可控**：你明确要求它不要编造，它依然幻觉。
- **非逻辑**：两位数乘法时对时错，9.11 和 9.9 都能比错大小。

脆弱性来自 Tokenization 的边界效应，幻觉来自模型的概率生成本质，算术错误来自分词对数字结构的破坏。

当你把模型当黑盒用时，你永远找不到这些答案。

### 1.2 从“黑盒”到“灰盒”

我们不需要成为 Transformer 的研究员，去推导 Attention 矩阵的梯度更新（那是科学家的工作）。但作为 Agent 开发者，我们需要构建一个**“灰盒”认知**。就像你不需要懂发动机的每一颗螺丝，但你得知道油门、刹车和方向盘分别控制什么。

这篇文章会带你走完一句话从键盘输入到 AI 回复的全部旅程。我们用一个 Agent 场景中最常见的用户输入作为主线：

> **“帮我查一下北京明天的天气”**

这句话进入模型后，会依次经过以下站点：

```
用户键盘                                              AI 回复
   │                                                    ▲
   ▼                                                    │
┌──────────────┐                                 ┌──────────────┐
│ Tokenization │ 把文字切碎成 Token               │  自回归生成   │ 一个字一个字
│  （第2章）    │ "北京" → ID: 19340              │  （第5章）    │ "挤牙膏"输出
└──────┬───────┘                                 └──────┬───────┘
       ▼                                                ▲
┌──────────────┐                                 ┌──────────────┐
│  Embedding   │ 数字变成语义向量                  │   采样策略    │ Temperature
│  （第3章）    │ ID → [0.23, -0.45, ...]         │  （第5章）    │ 控制"创造力"
└──────┬───────┘                                 └──────┬───────┘
       ▼                                                ▲
┌──────────────┐                                 ┌──────────────┐
│  Attention   │ 理解词与词的关系                  │   概率分布    │ 预测下一个词
│  （第4章）    │ "北京"+"天气" = 天气查询         │              │ 的可能性
└──────┬───────┘                                 └──────────────┘
       └──────────────→ 经过 N 层处理 ──────────────→┘
```

每一站，你都会看到：

1. **它到底在干什么**：用这句话的具体例子拆解
2. **它和 Agent 开发有什么关系**：对应哪些工程决策和常见坑

读完本文，你将能够：

- 解释 Agent 死循环、指令遗忘、格式错误的根本原因
- 理解 RAG 检索的底层原理，知道为什么有时候“搜不准”
- 设计更健壮的 Prompt 布局和上下文管理策略
- 在 Agent 出问题时，知道从哪里下手排查

## 2. 第一站：Tokenization，模型的”读写障碍”

在大众眼里，LLM 读的是字，写的是文。但在模型眼里，它根本不认识“字”。

你输入的 “帮我查一下北京明天的天气”，模型看到的是一串数字：`[58626, 7522, 13451, 97667, 19340, 11071, 867, 1616, 167823]`。Tokenization（分词）就是把人类语言切碎、变成数字的过程。它是模型处理一切信息的第一步，也是 Agent 开发中很多“灵异事件”的根源。

### 2.1 Token 既不是字符，也不是单词

你可能以为分词就是“一个汉字一个 Token”或“一个单词一个 Token”。实际上，目前主流 LLM 使用的 BPE（Byte Pair Encoding）算法，遵循的是另一套逻辑：**常见的词组合并成一个 Token，不常见的词拆开处理**。

BPE 的核心思路是：从最小单位（字节）开始，不断合并训练语料中出现频率最高的相邻字符对，直到达到预设的词表大小。GPT-4o 的 o200k_base 词表经过约 200,000 次合并构建而成[^1]。

我们用主线例子来看：

```
用户输入："帮我查一下北京明天的天气"

GPT-4o（o200k_base）的真实分词结果：

  "帮"   "我"   "查"   "一下"   "北京"   "明"   "天"   "的"   "天气"
    ↓      ↓      ↓      ↓       ↓       ↓      ↓     ↓      ↓
  58626   7522  13451  97667   19340   11071   867   1616  167823

→ 12 个汉字被切成 9 个 Token
→ "一下""北京""天气" 因高频被合并为单 Token
→ "明天" 没有被合并："明"(11071) 和 "天"(867) 是两个独立 Token
```

> 你可以用 [Tiktokenizer](https://tiktokenizer.vercel.app) 或 [OpenAI Tokenizer](https://platform.openai.com/tokenizer) 在线验证任意文本的分词结果。

注意几个反直觉的地方：

- “一下”“北京”“天气” 是两个汉字，但因为在训练语料中太常见，被合并成了 **1 个 Token**
- “明天” 看起来也是高频词组，但 o200k_base 词表中 “明” 和 “天” 是两个独立 Token，**分词边界并不总是符合人类直觉**
- 如果你输入一个生僻的专业术语，比如“胆囊切除术”，它会被拆成 6 个 Token（其中“囊”因为太生僻，被进一步拆成了字节级 Token）

**Token 是模型处理信息的最小原子单位，它和人类语言的“字”或“词”没有对应关系。**

英文的分词同样如此：

```
"apple"        →  1 个 Token（常见词，直接合并）
"unbelievable" →  多个 Token：拆成 ["un", "bel", "ievable"]（注意：BPE 是统计驱动，拆分边界不一定符合词根词缀）
"Compute"      →  1 个 Token
```

> 你可以把 BPE 想象成一个“高频短语词典”。这个词典是在训练阶段，通过统计几 TB 的文本语料中哪些字符组合出现频率最高来构建的。频率越高的组合，越可能被合并成一个 Token。

### 2.2 中英文 Token 效率差异：同样的窗口，装多少内容？

这个差异对 Agent 开发者来说，直接影响**成本**和**上下文容量**。

不同模型的 Tokenizer 词表大小和构成差异很大，对中文的压缩效率也截然不同：

| 模型 | 词表大小 | 中文效率 | 同样 1000 汉字消耗 Token | 数据来源 |
|------|---------|---------|------------------------|---------|
| GPT-4o | 200,000 | 比 GPT-4 有改善，中文效率已接近国产模型 | ~630-700 | GPT-4o 使用 o200k_base 编码，词表 200K[^1]；tiktoken 实测纯中文约 0.65 Token/字 |
| DeepSeek V3 | 128,000 | 1 汉字 ≈ 0.6 Token | ~600 | DeepSeek 官方 API 文档[^2] |
| Qwen 2.5 | 151,646 | 1 Token ≈ 1.5~1.8 个中文字符 | ~550-670 | Qwen 官方文档[^3]；Qwen2 技术报告[^4] |

为什么有差距？核心在于**词表大小和构成**。Qwen 系列的词表达 151,646 个 Token[^4]，并且在 BPE 训练时专门优化了多语言压缩效率。DeepSeek V3 的技术报告也明确指出，其 Tokenizer 的 pretokenizer 和训练数据经过了专门修改，以优化多语言压缩效率[^5]。值得注意的是，GPT-4o 的 o200k_base 词表相比早期的 cl100k_base 已大幅改善了中文效率，与国产模型的差距明显缩小。

**这对 Agent 开发意味着什么？**

假设你的 Agent 每轮对话需要携带 System Prompt + Tool 定义 + 历史对话，总计约 5000 个中文字符：

- 用 GPT-4o：消耗约 3200-3500 Tokens（按实测约 0.65 Token/字换算）
- 用 DeepSeek V3：消耗约 3000 Tokens（按官方 0.6 换算[^2]）

**国产模型在中文效率上仍有优势，尤其是 Qwen 系列**。当你的 Agent 在做多轮复杂任务时，更高的压缩效率意味着同样的上下文窗口能装更多历史对话，降低后期“遗忘”早期指令和上下文的风险。

### 2.3 为什么 AI 算数经常翻车

你可能见过这个经典翻车：问 AI “9.11 和 9.9 哪个大？”，它自信地回答 “9.11 大”。

这不是模型逻辑差，是 Tokenization 把它坑了。

```
人类看到的：
  9.11  vs  9.9
  → 数值比较：9.11 < 9.9 ✅

模型看到的：
  [9] [.] [11]  vs  [9] [.] [9]
  → Token 序列比较："11" 和 "9" 哪个大？
  → 在文本语料里，11 通常"大于" 9
  → 输出：9.11 更大 ❌
```

**根本原因**：Tokenization 破坏了数字的位值结构。模型根本不知道 “11” 是小数点后的两位，它只看到两个独立的 Token，然后用文本模式匹配的直觉（类似比较版本号 “v11” > “v9”）来判断大小。

同样的道理，一个长数字如 “1234567890” 会被切成多个 Token（如 [“123”, “456”, “789”, “0”]），模型看不到这是一个完整的十位数，它看到的是几个独立的文本片段。这就是为什么让 LLM 做精确计算几乎必然翻车。

> **Agent 启示**：永远不要让 LLM 直接做数值计算或比较。遇到算术需求，调用 Python 代码工具（Code Interpreter）才是正解。这不是模型“笨”，是它的输入格式天然不支持数学运算。

### 2.4 工程避坑备忘录

理解了 Tokenization，你就能避开 Agent 开发中的几个隐形大坑。

**坑 1：成本估算的陷阱**

```python
# ❌ 错误：用字符数估算成本
cost = len(text) * price_per_token  

# ✅ 正确：用官方 tokenizer 跑真实 Token 数
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
token_count = len(enc.encode(text))
cost = token_count * price_per_token
```

API 按 Token 计费，不是按字符计费。JSON 格式（Agent 大量使用的 Tool 定义和输出格式）因为花括号、引号、冒号密集，Token 消耗比你直觉估算的要高。务必用 `tiktoken`[^6]（OpenAI 系列）或模型官方提供的 tokenizer 工具实测。DeepSeek 也提供了官方离线 tokenizer 代码包[^2]。

**坑 2：Stop Sequence 的 Token 边界问题**

很多开发者用特殊符号分割 Prompt，比如 `=== Output ===`，然后设置 `stop_sequence = "==="`，期望模型输出到 `===` 时停止。

但实际可能发生这种情况：

```
你的设置：stop_sequence = "==="
预期：模型输出到 === 时停止
实际：模型输出了 "== ="（中间有空格），没停！

原因：
  "==="  → 可能是 1 个 Token
  "== =" → 可能是 2 个 Token
  在模型看来，这是完全不同的 ID 序列，无法触发停止条件
```

**对策**：优先使用 XML 标签（如 `<output>`、`</output>`）作为分隔符和停止符。XML 标签在训练语料中大量出现，分词行为稳定得多。这也是 Anthropic 官方推荐使用 XML 标签做 Prompt 结构化的原因之一[^7]。

**坑 3：空格敏感性**

在 BPE 分词中，空格通常会被编码为 Token 的一部分[^6]。这意味着：

```
" user"（前面有空格） 和 "user"（没有空格）
→ 是完全不同的 Token ID
```

这个坑在 Agent 开发中的典型表现：让模型输出 JSON 时，它偶尔会在 Key 前面多一个空格，变成 `{" name": "test"}`。虽然肉眼看几乎一样，但你的 JSON parser 可能因此报错，或者解析出来的 key 变成了 `" name"` 而不是 `"name"`，导致后续逻辑全部崩溃。

**对策**：使用支持 JSON Mode 的 API（如 OpenAI 的 `response_format={"type": "json_object"}`），或者在代码层面对 JSON key 做 `strip()` 处理。

---

> **本章小结**：Tokenization 是模型处理信息的第一步，也是最容易被忽视的一步。记住三个核心认知：
> 1. **Token ≠ 字符**：模型的最小处理单位和人类语言的单位不一致
> 2. **数字会被拆碎**：这是 LLM 算术差的根本原因，遇到计算请调工具
> 3. **看不见的字符差异**：空格、标点在 Token 层面可能导致完全不同的行为
>
> 现在，我们的那句 “帮我查一下北京明天的天气” 已经变成了一串 Token ID。但数字 ID 只是编号，模型还不理解它们的含义。下一站，我们来看这些 ID 是怎么变成模型能“理解”的语义向量的。

---

## 3. 第二站：Embedding，从编号到“语义坐标”

上一章，我们的那句话 “帮我查一下北京明天的天气” 已经被切成了一串 Token ID。但 Token ID 只是字典里的编号，就像学号 001 和 002 挨着，并不代表这两个学生关系好。

模型要理解语言，需要把每个 Token 从一个“编号”变成一个能表达含义的“坐标”。这个过程就是 **Embedding（嵌入）**。

### 3.1 什么是 Embedding：给每个 Token 一张“语义地图上的 GPS 坐标”

Embedding 的原理其实很朴素：模型内部有一张巨大的查找表（Embedding Table），每个 Token ID 对应表中的一行，每一行是一个高维向量。

以 Llama 3（8B）为例，它的 Embedding 表的形状是 `[128,256 × 4,096]`[^8]，也就是说，词表中的 128,256 个 Token，每个都对应一个 4,096 维的向量。

用我们的主线例子来看：

```
"北京" → Token ID: 17891 → 查 Embedding 表第 17891 行
         （注意：这里用的是 Llama 3 的词表，
          所以 ID 和前文 GPT-4o 的 19340 不同——
          不同模型的词表不同，同一个词的 ID 也不同）
                          → 得到一个 4096 维的向量
                          → [0.23, -0.45, 0.78, 0.12, ..., -0.33]
                                    ↑
                              4096 个浮点数
```

这 4,096 个数字就是“北京”在模型内部的“语义坐标”。

**这个向量从哪来？** 它不是人工设定的，而是在预训练阶段通过梯度下降学出来的[^9]。当模型在几万亿 Token 的语料上反复做 Next-Token Prediction 时，Embedding 表的参数会不断调整，最终让语义相关的词在向量空间中的距离更近。

打个比方：如果把所有词的向量画在一张地图上（当然是 4096 维的，我们想象成 2 维），你会发现：

- “北京” 和 “上海” 离得很近（都是中国城市）
- “北京” 和 “天气” 离得较远（一个是地点，一个是现象）
- “天气” 和 “温度” 又离得很近（语义相关）

维度越高，模型能编码的语义细节越丰富，但参数量和计算成本也越大。以 Llama 3（8B）为例，它的 Embedding 维度是 4,096[^8]，Embedding 表本身就有 128,256 × 4,096 ≈ **5.25 亿**个参数，占 8B 模型总参数的约 6.5%[^8]。更大的模型维度更高，DeepSeek-V3 达到 7,168 维[^12]。

### 3.2 为什么 Agent 开发者必须理解 Embedding

你可能觉得 Embedding 是模型内部的事，和开发 Agent 没关系。其实不然，RAG（检索增强生成）的整个底层逻辑都建立在 Embedding 之上。

**RAG 的检索原理，本质就是 Embedding 空间中的“距离比较”：**

```
用户问："北京明天会下雨吗？"

Step 1: 用 Embedding 模型把用户问题转成向量
        → query_vector = [0.21, -0.43, 0.80, ...]

Step 2: 在向量数据库中，找和 query_vector 距离最近的文档向量
        → 文档 A "北京未来三天天气预报" → distance = 0.15 ✅ 近！
        → 文档 B "上海房价走势分析"     → distance = 0.87 ❌ 远
        → 文档 C "北京市交通管理条例"   → distance = 0.62 ❌ 较远

Step 3: 把文档 A 作为上下文喂给 LLM，让它基于真实数据回答
```

理解了这个原理，你就能解释 RAG 开发中的常见困惑：

**困惑 1：“为什么检索不准？”**

用户问 “降温了穿什么”，你的知识库里有 “气温下降时的穿衣建议”。语义上完全对口，但如果 Embedding 模型没有很好地学到“降温”和“气温下降”的语义等价关系，两个向量在空间中的距离可能偏大，导致检索不到。

**对策**：选择在你的目标语言上训练充分的 Embedding 模型。比如处理中文场景，BGE 系列（BAAI）和 GTE 系列（阿里）通常比通用英文模型效果好。

**困惑 2：“为什么换了 Embedding 模型，之前的向量库就不能用了？”**

因为不同 Embedding 模型生成的向量空间完全不同。模型 A 觉得“北京”和“天气”距离 0.3，模型 B 可能觉得距离 0.7。向量数据库里的文档向量是用模型 A 生成的，你不能用模型 B 去做 query，就像用北京的 GPS 坐标去东京的地图上找路，找不到的。

**困惑 3：“维度越高越好吗？”**

OpenAI 提供了两个 Embedding 模型：`text-embedding-3-small`（1,536 维）和 `text-embedding-3-large`（3,072 维）[^13]。维度更高意味着语义表达更丰富，但也意味着：向量存储空间翻倍、检索计算量增大。对大多数 Agent 场景，1,536 维已经够用。

### 3.3 位置编码：模型怎么知道词序

Embedding 只管语义，不管顺序。“我打你” 和 “你打我” 如果只看每个词的 Embedding 向量，加起来是一样的。但这两句话的意思完全不同。

模型需要知道每个 Token 在序列中的位置。这就是**位置编码（Positional Encoding）**的作用。

早期的 Transformer 用固定的正弦函数来编码位置（原始 “Attention Is All You Need” 论文[^14]的做法）。但这种方式很难扩展到长上下文。

目前主流的方案是 **RoPE（Rotary Position Embedding，旋转位置编码）**，被 Llama 3[^8]、Qwen 2.5[^15]、DeepSeek V3[^12] 等主流模型广泛采用。RoPE 的核心思想是：根据 Token 在序列中的绝对位置，对它的向量做一个“旋转”操作。两个 Token 之间的相对位置信息，自然地被编码在了它们旋转后的向量的点积中[^16]。

**你不需要理解 RoPE 的数学细节**，但需要知道它对 Agent 开发的两个影响：

1. **上下文长度的上限由 RoPE 的配置决定**。Llama 3 通过调整 RoPE 的基频参数（`rope_theta: 500000.0`[^8]），将上下文窗口从 8K 扩展到了 128K。这就是为什么不同模型支持的最大上下文长度不同。

2. **同一段信息放在 Prompt 的不同位置，效果可能不同**。这不仅是因为 Attention 权重分布（第 4 章会讲），也因为位置编码本身会影响 Token 之间的关系计算。“系统指令”放在开头 vs 放在中间，对模型来说是不同的位置信号。

### 3.4 完成转换：从文字到“模型能理解的矩阵”

经过 Tokenization + Embedding + 位置编码三步，我们的那句话已经完成了从人类语言到数学表示的转换：

```
"帮我查一下北京明天的天气"

→ Tokenization: [Token₁, Token₂, Token₃, Token₄, Token₅, Token₆, Token₇, Token₈, Token₉]

→ Embedding:    每个 Token 查表得到 4096 维向量
                → 矩阵形状：[9 × 4096]

→ 位置编码:     对每个向量施加位置旋转（RoPE）
                → 矩阵形状仍然是 [9 × 4096]，但编码了位置信息

结果：一个 9 行 4096 列的数字矩阵
      每一行代表一个 Token 的"语义 + 位置"信息
      这就是送入 Attention 层的输入
```

现在，模型有了一张“语义地图”，每个词都有了坐标。但这些词之间的关系（“北京”和“天气”该怎么组合理解？）还需要下一站 Attention 机制来计算。

---

> **本章小结**：Embedding 是模型理解语义的基础，也是 RAG 检索的底层原理。记住三个核心认知：
> 1. **Embedding 表是学出来的**：Token ID 通过查表变成高维向量，这个表在预训练中学习
> 2. **RAG = 向量空间中的距离搜索**：理解这个，你就理解了为什么检索有时不准，以及为什么不能混用不同的 Embedding 模型
> 3. **位置编码让模型知道词序**：RoPE 决定了模型能支持多长的上下文

---

## 4. 第三站：Attention，模型的“即时检索引擎”

上一章，我们的那句话已经变成了一个 `[9 × 4096]` 的数字矩阵，9 个 Token，每个 Token 有一个 4096 维的语义+位置向量。但此刻，每个 Token 的向量仍然是“孤立”的。“北京”不知道旁边有“天气”，“明天”也不知道自己修饰的是“天气”而不是“查”。

Attention（注意力）机制的工作，就是让每个 Token 去“看”所有其他 Token，计算它们之间的关联度，然后把最相关的信息“融合”到自己身上。它是 Transformer 架构的灵魂[^17]，也是 LLM 能理解复杂语义的基础。

### 4.1 直觉理解：不是“记忆”，而是“即时检索”

不要把 Context Window（上下文窗口）想象成人类的“大脑记忆”。人类读完一本书，记住的是梗概（压缩的信息）。而 LLM 预测下一个 Token 时，会**重新扫描一遍上下文里的每一个词**，并计算它们与当前位置的关联度。

用我们的主线例子来看：

```
输入："帮我查一下北京明天的天气"

当模型处理到"天气"这个位置时，Attention 机制在做什么？

它对上文每个词计算一个"关注度分数"（Attention Weight）：

  "帮我"   → 0.03  （动作词，但和"天气"关联不大）
  "查"     → 0.05  （动作词）
  "一下"   → 0.02  （语气词，几乎忽略）
  "北京"   → 0.38  （关键！天气查询的地点）
  "明天"   → 0.35  （关键！天气查询的时间）
  "的"     → 0.02  （虚词）
  "天气"   → 0.15  （自身参考）

模型通过这些权重理解了：
这是一个关于【北京】（地点）【明天】（时间）的【天气查询】。
```

> 以上权重为示意值，用于展示 Attention 的直觉逻辑。实际的 Attention 权重分布因模型、层数和 head 而异。

这个过程的本质是：**对于每个位置，Attention 机制都像一个搜索引擎，在上下文中检索与当前位置最相关的信息，然后把这些信息加权融合进来。**

形式上，这个计算涉及三个角色[^17]：

- **Query（查询）**：“天气”在问：“谁和我最相关？”
- **Key（键）**：每个上文词都举起一个“标签”，表明自己“是关于什么的”
- **Value（值）**：每个上文词携带的“实际信息内容”

Attention 分数 = Query 和 Key 的点积（衡量匹配度）。分数越高，该词的 Value 对当前位置的贡献越大。

你不需要记住 Q/K/V 的数学细节，只需要记住这个直觉：**Attention 是一个即时的、加权的信息检索过程。**

### 4.2 多头注意力：同时从不同角度“检索”

模型不是只有一束探照灯，而是有**几十束同时扫描**。这就是 Multi-Head Attention（多头注意力）[^17]。

以 Llama 3 (8B) 为例，它有 32 个 Attention Head[^18]。每个 Head 独立地做一次 Attention 计算，但关注的“维度”不同：

- Head 1 可能在关注**语法结构**：“明天的”→“天气”（修饰关系）
- Head 5 可能在关注**语义关联**：“北京”+“天气”= 天气查询
- Head 12 可能在关注**位置关系**：哪些词离当前位置近
- Head 28 可能在关注**共指关系**：如果前文出现过“首都”，它会关联到“北京”

**比喻**：就像一组侦探同时从不同角度调查同一个案件：有人查物证，有人问证人，有人看监控。最后把所有线索汇总，得出全面的结论。

32 个 Head 的结果会被拼接（Concatenate）在一起，再通过一个线性投影压缩回原始维度[^17]。这个过程在 Llama 3 中会重复 **32 层**[^18]，也就是说，信息会经过 32 次这样的“多角度检索与融合”，每一层都在前一层理解的基础上进一步深化。

### 4.3 Lost in the Middle：注意力的物理极限

既然 Attention 能扫描全文，是不是上下文越长越好？

**并不是。** 学术界发现了一个令人尴尬的现象：**U 型曲线（U-shaped Performance Curve）**。

斯坦福大学 Liu et al. 的论文 “Lost in the Middle: How Language Models Use Long Contexts”[^19] 通过系统实验发现：

> 当关键信息放在输入上下文的**开头或结尾**时，模型准确率最高；当关键信息放在**中间**时，准确率显著下降。

这个 U 型曲线在多种模型（GPT-3.5-Turbo、GPT-4、Claude、Llama 2 等）上都被观察到[^19]。甚至 GPT-4 虽然绝对准确率更高，但仍然展现出同样的 U 型趋势[^20]。更值得注意的是，这个现象在 base model（未经指令微调的模型）上同样存在，说明它是模型架构的固有特性，而非训练方式的副产品[^19]。

后续研究进一步发现，这种 U 型曲线与人类记忆中的“序列位置效应”（Serial Position Effect）高度相似：人类回忆列表时也倾向于更好地记住开头（首因效应）和结尾（近因效应）的元素[^21]。

**这意味着什么？** 上下文的中间部分是一个“注意力黑洞”。

**Agent 开发的核心启示：“三明治”布局法**

```
┌──────────────────────────────────────────────────┐
│  ★ 最顶层：System Prompt                          │
│     人设、核心指令、输出格式约束                     │
│     → Attention 的首因效应（Primacy Bias）          │
├──────────────────────────────────────────────────┤
│  · 中间层：次要信息                                │
│     背景资料、历史对话摘要、                        │
│     "可能有用但不是最关键"的检索结果                 │
│     → 注意力黑洞区，放不太重要的东西                 │
├──────────────────────────────────────────────────┤
│  ★ 最底层：用户最新问题 + 关键知识                  │
│     最新的 Query、最重要的 RAG 检索结果、            │
│     格式 Refresher 提醒                            │
│     → Attention 的近因效应（Recency Bias）          │
└──────────────────────────────────────────────────┘
```

### 4.4 KV Cache：为什么 Input 便宜 Output 贵

理解 Attention 还能帮你理解一个关键的成本问题：**为什么 API 计费中 Input Token 通常比 Output Token 便宜？**

在自回归生成中，每生成一个新 Token，模型理论上都要对之前所有的上下文重新计算 Attention。如果已经生成了 1000 个词，生成第 1001 个词时，难道要重新算前 1000 个词的 Q/K/V？

不需要。**KV Cache（键值缓存）**就是解决这个问题的。

```
Prefill 阶段（处理输入 Prompt）：
  - 模型第一次看到整个 Prompt
  - 计算每个 Token 的 Key 和 Value，存入显存
  - 这一步没有缓存可用，必须完整计算
  - → 首字延迟（TTFT）高

Decode 阶段（逐个生成输出）：
  - 生成每个新 Token 时，只需要计算新 Token 的 Query
  - 然后和缓存中已有的 Key/Value 做 Attention
  - 不需要重新计算旧 Token 之间的关系
  - → 生成速度快
```

这就是为什么：

- **TTFT（首字延迟）通常较高**：因为 Prefill 阶段没有缓存，需要对整个 Prompt 做完整计算
- **后续 Token 生成很快**：利用 KV Cache，每步只需增量计算
- **Input Token 便宜**：Prefill 阶段是并行计算（GPU 擅长），效率高
- **Output Token 贵**：Decode 阶段是逐个串行生成，且每一步都要读取整个 KV Cache

**Prompt Caching：更进一步的省钱利器**

如果你的 Agent 每次请求都带着同样的 System Prompt + 5000 字的 Tool 定义，那每次都从头算一遍 Prefill 是很浪费的。

目前主流模型厂商都支持了 Prompt Caching 机制：如果连续请求的 Prompt 有相同的前缀，服务端会缓存这部分的 KV Cache，下次请求直接复用，节省计算时间和费用。DeepSeek 的上下文缓存在 Prompt 前缀匹配且缓存命中时，输入价格降低为原价的约 1/10[^22]。Claude 也提供了类似的 Prompt Caching 功能[^23]。

### 4.5 工程实战备忘录

| 原则 | 原理依据 | 做法 |
|------|---------|------|
| **位置即权重** | Lost in the Middle U 型曲线[^19] | 最重要的指令放最前面（System Prompt），最新的问题和关键信息放最后面，中间放次要内容 |
| **上下文有代价** | KV Cache 占用显存 | 上下文越长，推理越慢、成本越高。精简、结构化的上下文往往比冗长的上下文效果更好 |
| **利用 Prompt Caching 省钱** | KV Cache 复用 | 把静态内容（Tool 定义、Few-shot 例子）固定在 Prompt 的头部，利用缓存机制降低成本[^22][^23] |
| **长上下文不是银弹** | Attention 衰减 | 不要因为模型支持 128K 就把所有信息塞进去。信息过载反而降低准确率[^19] |
| **重要指令要重申** | 近因效应 | 如果对话超过 10 轮，在 Prompt 最后动态插入 Refresher（格式提醒、核心约束重申） |

---

> **本章小结**：Attention 是模型理解词间关系的核心引擎。记住三个核心认知：
> 1. **Attention 是即时检索，不是记忆**：模型每次生成都在重新扫描全文，而不是“记住”了什么
> 2. **中间位置是黑洞**：Lost in the Middle 是经过严格实验验证的现象，用“三明治”布局应对
> 3. **KV Cache 决定了成本结构**：理解 Prefill/Decode 的区别，利用 Prompt Caching 省钱
>
> 现在，模型已经通过 Attention 理解了这句话的含义：这是一个关于“北京明天天气”的查询。下一站，模型要开始逐个 Token 地生成回复了。

---

## 5. 第四站：自回归生成，一个字一个字“挤牙膏”

经过 Tokenization、Embedding 和 Attention 三站，模型已经理解了 “帮我查一下北京明天的天气” 这句话的完整语义。现在，轮到它开始生成回复了。

很多人把 LLM 想象成一个“无所不知的智者”，问它一个问题，它在脑子里检索答案，然后一次性告诉你。但实际上，LLM 更像一个**超级版的输入法自动联想**：它并不“知道”整个答案，它只知道**下一个字是什么**。

### 5.1 Next-Token Prediction：模型的“思考”方式

“自回归”（Auto-regressive）的意思是：**根据上文预测下文，然后把生成的下文再喂给自己，继续预测再下一个**。

用 Agent 场景来看。假设模型（经过训练后）判断应该调用天气工具，它生成回复的过程是这样的：

```
第1步：基于整个输入，预测下一个 Token
       → 概率分布：{ "{": 0.82, "I": 0.05, "The": 0.03, ... }
       → 选择 "{"
       → 当前输出：{

第2步：基于输入 + "{"，预测下一个 Token
       → 概率分布：{ ""name"": 0.76, ""tool"": 0.15, ... }
       → 选择 ""name""
       → 当前输出：{"name"

第3步：基于输入 + "{"name""，预测下一个 Token
       → 选择 ":"
       → 当前输出：{"name":

第4步：...
       → 选择 ""get_weather""
       → 当前输出：{"name":"get_weather"

...逐个 Token 继续，直到生成完整的 JSON
```

**关键认知**：模型不是一次性“想好”了整个 JSON 再输出，而是一个 Token 一个 Token 挤出来的。在输出第一个 `{` 的时候，它并不确定最后一个 `}` 里会写什么。

### 5.2 采样策略：Temperature 与 Top-P

在每一步预测中，模型算出的是一个概率分布，词表中每个 Token 的出现概率。但模型并不是永远选概率最高的那个（除非你强制它）。控制这个选择过程的核心参数，就是 **Temperature（温度）**。

**Temperature 的本质：改变概率分布的“形状”**

想象概率分布像一个高低不平的柱状图，最高的柱子是“最可能的词”，矮的柱子是“有创意但可能跑偏的词”：

- **低温（Temp < 1，如 0.1）**：柱状图被拉尖。高的柱子更高，矮的几乎消失。模型几乎只能看见概率最高的词。**表现**：保守、准确、重复。
- **高温（Temp > 1，如 1.5）**：柱状图被拍扁。原本概率很小的词现在也有机会被选中。**表现**：发散、有创意、容易胡说。
- **Temp = 0**：**贪婪解码（Greedy Decoding）**，永远只选概率最大的 Token，不做任何随机采样。

**Top-P（Nucleus Sampling）**：如果说 Temperature 是调节分布的形状，Top-P 是一个“截断刀”。Top-P = 0.9 意味着：把概率从高到低排列，累加到 90% 为止，剩下 10% 的长尾低概率词直接剔除，不给它们被选中的机会。

**工程实战备忘录：**

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| 工具调用 / JSON 生成 | Temp = 0 | 语法必须严谨，不需要创造力 |
| CoT 推理 / 数学计算 | Temp = 0 ~ 0.3 | 逻辑推理需要严密 |
| 文案润色 / 闲聊 | Temp = 0.7 ~ 0.9 | 需要词汇多样性 |
| 创意风暴 / 角色扮演 | Temp = 1.0+ | 鼓励探索罕见的词汇组合 |

### 5.3 深度解析：为什么 Temperature=0 依然有随机性？

在 Agent 开发中，我们通常被建议：“涉及工具调用时，请把 Temperature 设为 0。” 理论上，Temp=0 是贪婪解码，应该完全确定。但在工程实践中，同样的 Prompt 跑多次，偶尔还是会出现微小的输出差异。

**这不是玄学**。Thinking Machines Lab（Mira Murati 创办的公司）2025 年的一项研究系统地揭示了真正的原因[^24]：

他们用 Qwen3-235B 在 Temperature=0 下对同一个 Prompt 采样 1000 次，得到了 **80 个不同的输出**，从第 103 个 Token 开始分叉。

**根因并不是大多数人以为的“GPU 浮点随机性”**。实际上，单个 GPU kernel 的前向计算是确定性的，同样的矩阵乘法跑 1000 次，结果完全一致[^24]。真正的罪魁祸首是**批次不变性（Batch Invariance）失败**：

当推理服务器（如 vLLM、SGLang）同时处理多个用户请求时，会将它们打包成不同大小的 batch。而某些关键计算操作（归一化、矩阵乘法、注意力计算）在不同 batch size 下，浮点数的累加顺序会改变[^24]。由于浮点数运算不满足结合律（`(a + b) + c ≠ a + (b + c)`[^25]），这会导致最终的 logits（预测分数）产生微小差异。

当两个候选 Token 的概率极度接近时（例如 0.5000001 vs 0.5000000），这个微小差异就足以翻转 argmax 的结果。一旦第一个 Token 变了，根据自回归原理，后续所有输出都会走上完全不同的路径。

此外，使用 MoE（混合专家）架构的模型（如 GPT-4、DeepSeek V3）还有一个额外的非确定性来源：不同 batch 中的 Token 会竞争专家路由的槽位，导致同一个 Token 可能被分配给不同的专家[^26]。

> **Agent 开发警告**：即使 Temp=0，也不要认为你的 Agent 具有 100% 的确定性。如果你的代码逻辑依赖于模型输出“一个字都不能差”，那你的架构就是脆弱的。必须加上容错机制：JSON 修复、重试逻辑、模糊匹配。

### 5.4 “线性诅咒”：为什么模型不能回头修改

因为一个字接一个字的生成方式，LLM 有一个巨大的先天缺陷：**它没有“橡皮擦”**。

人类思考问题时，通常是“先构思整体，再填细节”，写错了可以在脑海里修正。但 LLM 不行：

- 一旦一个 Token 被生成出来，它就变成了历史上下文的一部分，成为**不可更改的事实**
- 后续所有的生成，都要基于这个已经生成的 Token 继续

这就是为什么模型会“一本正经地胡说八道”：如果模型一开始因为概率偏差生成了一个错误的词（比如把“李白”说成了“宋朝诗人”），为了保证语言的流畅性，它后续的所有内容都会被迫围绕“宋朝”这个错误前提去编造。因为它无法回头把“宋朝”改成“唐朝”。

这也解释了 **Streaming（流式输出）** 的特性：当你用 ChatGPT 时，看到文字一个个蹦出来，已经蹦出来的字不会突然变，因为 Token 一旦生成就“离开了”模型，变成了历史的一部分。

### 5.5 CoT（思维链）：用“空间”换“智力”

既然模型不能回头修改，也不能“默默思考”（它的思考过程必须体现为生成的 Token），那么处理复杂任务时，直接让它给答案往往会出错。

**Chain of Thought（CoT）的本质，是用生成更多的 Token 来换取更强的推理能力。**

```
没有 CoT（直接回答）：

  用户：23 × 18 等于多少？
  模型：[内部概率计算...] → 424 ❌
  失败原因：计算步骤太复杂，无法在一个 Token 的预测步骤里完成

有 CoT（让子弹飞一会儿）：

  用户：23 × 18 等于多少？请一步步计算。
  模型：20 × 18 = 360，3 × 18 = 54，360 + 54 = 414 ✅
```

为什么这在原理上有效？当模型输出了 “20 × 18 = 360” 这串字符后，这些字符就进入了上下文。当模型预测最终结果时，它不再是凭空计算 23 × 18，而是基于上下文里已有的中间步骤 `360 + 54` 做一个简单的加法预测。

**Agent 开发避坑指南：**

- **强制思考**：在 System Prompt 中，永远要求 Agent 先输出 Reasoning/Thought，再输出 Action/Answer
- **避免抢答**：如果你的 Agent 表现出逻辑混乱，检查一下是不是让它直接输出 JSON 结果了。让它先用自然语言“碎碎念”几句，正确率通常大幅提升
- **生成越多 = 思考越深**：生成的 Token 越多，消耗的推理算力越多，实际上就是赋予了模型更多的“思考时间”。这也是 OpenAI o1/o3、DeepSeek R1 等推理模型的核心思路[^27]

---

> **本章小结**：自回归生成是 LLM 产出文本的方式，也是很多“诡异行为”的根源。记住三个核心认知：
> 1. **模型是“挤牙膏”式生成**：一个 Token 一个 Token 输出，生成第一个字时不知道最后一个字写什么
> 2. **Temperature=0 不等于确定性**：批次不变性失败、MoE 路由等系统级因素都会引入非确定性，代码必须有容错
> 3. **CoT 的本质是用空间换智力**：让模型“说出思考过程”，等价于给它更多的推理时间
>
> 至此，我们已经走完了一句话从输入到输出的完整旅程。下一章，我们把所有站点串起来，画出全景图。

---

## 6. 全流程回顾与 Agent 工程启示

### 6.1 一图看懂：一句话的完整旅程

让我们回到那句 “帮我查一下北京明天的天气”，画出它从你的键盘到 AI 回复的完整路径：

```
用户输入                     模型内部处理                        输出
─────────────────────────────────────────────────────────────────────────

"帮我查一下          ① Tokenization（第2章）
 北京明天的天气"        把文字切碎成 Token ID
                       → [Token₁, Token₂, ..., Token₉]
                            │
                            ▼
                     ② Embedding（第3章）
                        每个 ID 查表得到 4096 维向量
                        → 矩阵 [9 × 4096]
                            │
                            ▼
                     ③ 位置编码（第3章）
                        用 RoPE 给每个向量注入位置信息
                        → 矩阵仍为 [9 × 4096]，但编码了词序
                            │
                            ▼
                     ④ Attention × 32 层（第4章）
                        每层 32 个 Head 同时检索词间关系
                        "北京" + "天气" → 天气查询
                        "明天" + "天气" → 时间限定
                        → 每层输出仍为 [9 × 4096]
                            │
                            ▼
                     ⑤ 概率分布（第5章）
                        最后一个位置的向量 → 投影到词表维度
                        → 每个 Token 的出现概率
                        → { "{": 0.82, "I": 0.05, ... }
                            │
                            ▼
                     ⑥ 采样（第5章）
                        根据 Temperature 策略选词
                        Temp=0 → 选概率最高的 "{"
                            │
                            ▼
                     ⑦ 自回归循环（第5章）               ──→ 逐个 Token
                        把 "{" 加入上下文，回到 ④               输出给用户
                        继续预测下一个 Token...
                        → {"name":"get_weather","city":"北京"}
```

整个过程中，模型做的事情始终只有一件：**预测下一个 Token**。无论是理解你的问题、决定调用什么工具、还是生成 JSON 参数，本质上都是在做 Next-Token Prediction。区别只在于，经过训练后，模型学会了在不同的上下文下，预测出不同“风格”的 Token 序列。

### 6.2 Agent 开发的“物理学定律”

把全文的工程启示汇总为一张表。这些不是经验之谈，每一条都有本文讲过的原理支撑：

| 你遇到的问题 | 背后的原理（章节） | 正确的做法 |
|---|---|---|
| Token 计费超预期 | Tokenization 效率差异（第2章） | 用 tiktoken 等官方工具实测，选择中文效率高的模型 |
| Agent 算术出错 | Tokenization 破坏数字位值结构（第2章） | 调用代码工具计算，永远不让 LLM 口算 |
| RAG 检索不准 | Embedding 空间中向量距离偏大（第3章） | 选对 Embedding 模型，优化 query 表述，不混用不同模型的向量 |
| 长 Prompt 指令遗忘 | Lost in the Middle U 型曲线（第4章） | 三明治布局：关键指令放最前和最后，次要信息放中间 |
| Agent 调用成本高 | KV Cache + Prefill/Decode 定价差异（第4章） | 静态内容（Tool 定义、Few-shot）前置，利用 Prompt Caching |
| 模型幻觉 / 胡编 | 线性诅咒，无法回头修改（第5章） | CoT 强制模型先推理再回答；RAG 提供外挂知识 |
| JSON 格式偶尔出错 | 概率生成本质 + 批次非确定性（第5章） | Temp=0 + JSON Mode + 代码层面兜底修复 |
| 同样 Prompt 偶尔输出不同 | 批次不变性失败 / MoE 路由（第5章） | 不依赖 100% 确定性，架构层面加容错（重试、模糊匹配） |
| CoT 推理效果不好 | 模型被要求直接输出结果，没有“思考空间”（第5章） | System Prompt 中强制先输出 Thought 再输出 Action |

### 6.3 一个思维转换

传统软件工程的核心信念是**确定性**：代码对了，结果就对。

Agent 开发需要你接受一个新的现实：**你的“代码”（Prompt）可能完全正确，但输出仍然可能出错**，因为你面对的不是一个计算器，而是一个概率性的文本生成器。

理解 LLM 原理不是为了消除这些不确定性，而是为了**理解并管理它们**：

- 因为理解了 **Tokenization**，你学会了避开特殊字符的坑，用正确的工具估算成本
- 因为理解了 **Embedding**，你知道了 RAG 检索不准时该去哪里排查
- 因为理解了 **Attention**，你学会了把关键指令放在上下文的“黄金位置”
- 因为理解了 **自回归生成**，你学会了用 CoT 让模型慢下来思考，用容错机制兜住概率性错误

Agent 开发本质上是一种“人机协作的工程”。你不再是指挥官，你更像是一个“牧羊人”：你知道羊群（模型）大体往哪里走，但你无法控制每一只羊（Token）的具体步伐。你需要做的，是建好围栏（Prompt 约束）、准备好草场（RAG 知识库）、并时刻盯着那只可能掉队的羊（错误监控与兜底）。

---

## 7. 推荐资源

以下是本文涉及的核心概念的高质量学习资源。不追求全面，只推荐信噪比最高的。

### 视频

| 资源 | 时长 | 推荐理由 |
|------|------|---------|
| Andrej Karpathy: [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | 1h | 全网最好的 LLM 入门课，把预训练、SFT、RLHF 讲得极其通透 |
| 3Blue1Brown: [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) | 27min | 用动画解释 Token、Embedding、Softmax，零基础友好 |
| 3Blue1Brown: [Attention in Transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 25min | 可视化 Attention 的 Q/K/V 机制，让你“看见”矩阵乘法的物理意义 |

### 工具

| 工具 | 用途 |
|------|------|
| [Tiktokenizer](https://tiktokenizer.vercel.app) | 在线可视化文本如何被切分成 Token，支持多种 Tokenizer，发布前验证文章数据必备 |
| [OpenAI Tokenizer](https://platform.openai.com/tokenizer) | OpenAI 官方 Token 可视化工具 |
| [LLM Visualization](https://bbycroft.net/llm) | 3D 可视化 LLM 推理过程（Embedding → Attention → 输出），非常直观 |

### 文章

| 文章 | 推荐理由 |
|------|---------|
| Jay Alammar: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解 Transformer 的经典之作，最直观的 Attention 可视化解释 |
| Dugas: [The GPT-3 Architecture, on a Napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html) | 用手绘草图拆解 GPT-3 每一层的数据流和维度变化，适合想理解“数字怎么流动”的人 |

---

### 动手验证：配套实验 Notebook

本文的每一个关键结论都可以用代码验证。配套的 Jupyter Notebook 包含 11 个实验：

- **Tokenization（实验 1-4）**：只需 `pip install tiktoken`，无需 API Key，本地即可运行。亲手看到 “9.11” 被拆成 [“9”, “.”, “11”]，比读十遍文章都深刻。
- **Embedding（实验 5-6）**：调用 OpenAI Embedding API，计算词语间的余弦相似度矩阵，验证“语义近的词向量近”。
- **Attention（实验 7-8）**：复现 Lost in the Middle 效应，观测 Prompt Caching 的 `cached_tokens` 字段。
- **自回归生成（实验 9-11）**：对比不同 Temperature 的输出多样性，检测 Temp=0 的非确定性，测试 CoT 对数学推理的提升。

所有需要 API 的实验都预置了真实输出，没有 Key 也能看到完整结果。

> Notebook 地址：[article1-experiments.ipynb](article1-experiments.ipynb)

---

> **本文是三篇系列的第一篇**，聚焦于“模型怎么工作”，从一句话走完 LLM 的完整处理流程。
>
> **第二篇**[《模型的能力是怎么来的：从预训练到 RLHF》](/posts/how-model-capabilities-come-from/)将聚焦模型能力的来源：预训练如何压缩互联网知识、SFT 如何让模型学会对话、RLHF 如何对齐人类偏好、以及 Function Calling 的训练本质。
>
> **第三篇**[《推理服务是怎么影响你的 Agent 的：推理框架与架构决策》](/posts/inference-frameworks-for-agent-architects/)将聚焦推理层如何影响 Agent 架构师的每一个设计决策。
>
> Build, fail, iterate. Good luck!


---

### 参考资料

[^1]: GPT-4o 使用 o200k_base tokenizer，词表约 200,000 tokens。来源：OpenAI tiktoken 库 & 社区分析。https://github.com/openai/tiktoken ；详细分析见 https://www.njkumar.com/gpt-o-multilingual-token-compression/

[^2]: DeepSeek 官方 API 文档 “Token & Token Usage”：1 English character ≈ 0.3 token，1 Chinese character ≈ 0.6 token。官方还提供了离线 tokenizer 代码包。https://api-docs.deepseek.com/quick_start/token_usage

[^3]: Qwen 官方文档 “Key Concepts”：Qwen 使用 byte-level BPE，词表 151,646 tokens。1 token ≈ 3–4 characters (English)，1 token ≈ 1.5–1.8 characters (Chinese)。https://qwen.readthedocs.io/en/latest/getting_started/concepts.html

[^4]: Qwen2 Technical Report (Yang et al., 2024)：词表由 151,643 regular tokens + 3 control tokens 组成，tokenizer 展现出更优的压缩率（better compression rate）。https://arxiv.org/abs/2407.10671

[^5]: DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)：Tokenizer 采用 Byte-level BPE，词表 128K tokens，pretokenizer 和训练数据经过修改以优化多语言压缩效率。https://arxiv.org/abs/2412.19437

[^6]: OpenAI tiktoken 库：Python 实现的 BPE tokenizer，支持 o200k_base、cl100k_base 等编码。使用教程见 OpenAI Cookbook “How to count tokens with tiktoken”。https://github.com/openai/tiktoken ；教程：https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken

[^7]: Anthropic Prompt Engineering 文档推荐使用 XML 标签进行 Prompt 结构化。https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags

[^8]: Llama 3 (8B) 架构配置：`dim: 4096, n_layers: 32, n_heads: 32, vocab_size: 128256, rope_theta: 500000.0`。Embedding 层形状为 `Embedding(128256, 4096)`。来源：Meta Llama 3 源码及模型配置。https://github.com/meta-llama/llama3 ；架构详解见 https://github.com/FareedKhan-dev/Building-llama3-from-scratch

[^9]: Embedding 通过梯度下降在预训练中学习，Token 嵌入矩阵作为模型的可训练参数之一。来源：Karpathy, “Let's build GPT from scratch”；GPT 架构解析 https://dugas.ch/artificial_curiosity/GPT_architecture.html

[^10]: GPT-2 架构：12 层，12 attention heads，embedding 维度 768，词表 50,257。来源：Radford et al., “Language Models are Unsupervised Multitask Learners” (2019)；参数计算详见 https://blog.ando.ai/posts/ai-transformer-sizes/

[^11]: GPT-3 架构：96 层，96 attention heads，embedding 维度 12,288。来源：Brown et al., “Language Models are Few-Shot Learners” (2020). https://arxiv.org/abs/2005.14165 ；图解版 https://dugas.ch/artificial_curiosity/GPT_architecture.html

[^12]: DeepSeek-V3 架构：embedding 维度 7,168，128K 词表，采用 RoPE 位置编码。来源：DeepSeek-V3 Technical Report (2024). https://arxiv.org/abs/2412.19437

[^13]: OpenAI Embedding 模型：text-embedding-3-small（1,536 维）和 text-embedding-3-large（3,072 维），支持维度缩减。来源：OpenAI 官方公告 (2024年1月)。https://openai.com/index/new-embedding-models-and-api-updates/

[^14]: 原始 Transformer 使用固定正弦位置编码。来源：Vaswani et al., “Attention Is All You Need” (2017). https://arxiv.org/abs/1706.03762

[^15]: Qwen 2.5 采用 RoPE 位置编码，支持 128K 上下文。来源：Qwen2.5 Technical Report (2024). https://arxiv.org/abs/2412.15115

[^16]: RoPE（旋转位置编码）通过对 Query 和 Key 向量施加基于位置的旋转，使得点积自然编码相对位置信息。来源：Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding” (2021). https://arxiv.org/abs/2104.09864

[^17]: Vaswani et al., “Attention Is All You Need” (2017)。提出了 Transformer 架构，定义了 Scaled Dot-Product Attention 和 Multi-Head Attention 机制，以及 Query/Key/Value 的概念。https://arxiv.org/abs/1706.03762

[^18]: Llama 3 (8B) 架构配置：`n_heads: 32, n_layers: 32, n_kv_heads: 8`（使用 Grouped Query Attention）。来源：Meta Llama 3 源码。https://github.com/meta-llama/llama3 ；架构详解见 Towards Data Science, “Deep Dive into LlaMA 3 by Hand” https://towardsdatascience.com/deep-dive-into-llama-3-by-hand-%EF%B8%8F-6c6b23dc92b2/

[^19]: Liu et al., “Lost in the Middle: How Language Models Use Long Contexts” (2024), Transactions of the Association for Computational Linguistics, 12:157–173。核心发现：模型表现出 U 型性能曲线，关键信息在开头或结尾时准确率最高，在中间时显著下降。该现象在 GPT-3.5-Turbo、Claude、Llama 2 等多种模型上被观察到，且在 base model 上同样存在。https://arxiv.org/abs/2307.03172 ；正式发表版 https://aclanthology.org/2024.tacl-1.9/

[^20]: 同 [^19] 论文 Figure 15：GPT-4 虽然绝对准确率高于其他模型，但仍展现 U 型性能曲线。

[^21]: U 型曲线与人类记忆中的“序列位置效应”（Serial Position Effect）的关联。原始心理学研究：Murdock, “The serial position effect of free recall” (1962)。LLM 与人类记忆的类比分析见：Hu et al., “Lost in the Middle: An Emergent Property from Information Retrieval Demands in LLMs” (arXiv preprint, 2025). https://arxiv.org/abs/2510.10276

[^22]: DeepSeek 上下文缓存（Context Caching）：当 Prompt 前缀匹配时，缓存命中的 token 价格大幅降低。来源：DeepSeek API 文档 “Context Caching”。https://api-docs.deepseek.com/guides/kv_cache

[^23]: Anthropic Claude Prompt Caching：支持缓存 Prompt 的静态前缀部分，复用 KV Cache 以降低延迟和成本。来源：Anthropic 官方文档。https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

[^24]: Thinking Machines Lab, “Defeating Nondeterminism in LLM Inference” (2025)。核心发现：Temperature=0 下的非确定性并非来自 GPU 浮点随机性（单个 kernel 是确定性的），而是来自批次不变性（Batch Invariance）失败，不同 batch size 导致浮点累加顺序不同，进而影响 logits。实验：Qwen3-235B 在 Temp=0 下对同一 Prompt 采样 1000 次，产生 80 个不同输出。https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

[^25]: 浮点数非结合律是计算机科学的基础知识。经典参考：Goldberg, “What Every Computer Scientist Should Know about Floating-Point Arithmetic” (1991), ACM Computing Surveys。在 LLM 推理上下文中的讨论见 Šubonis, “Zero Temperature Randomness in LLMs” (2025). https://martynassubonis.substack.com/p/zero-temperature-randomness-in-llms

[^26]: MoE 架构中的非确定性来源：Sparse MoE 的 Token 路由在不同 batch 组成下可能将同一 Token 分配给不同专家。来源：Chann, “Non-determinism in GPT-4 is caused by Sparse MoE” (2023)；Puigcerver et al., “From Sparse to Soft Mixtures of Experts” (2023)。综述见 Schmalbach, “Does Temperature 0 Guarantee Deterministic LLM Outputs?” (2025). https://www.vincentschmalbach.com/does-temperature-0-guarantee-deterministic-llm-outputs/

[^27]: 推理模型通过生成更多 Token（思维链）来换取更强推理能力的思路，被 OpenAI o1/o3 和 DeepSeek R1 等模型验证。DeepSeek-R1 技术报告：https://arxiv.org/abs/2501.12948
