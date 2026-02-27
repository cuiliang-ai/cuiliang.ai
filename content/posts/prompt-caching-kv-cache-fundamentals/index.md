---
title: "KV Cache 原理：LLM 推理的底层机制"
date: 2026-01-10
draft: false
summary: "从 Token、Embedding、Attention 到 Prefill/Decode，一次讲透 KV Cache 的底层原理。面向 Agent 工程师的推理基础指南。"
description: "从 Token、Embedding、Attention 到 Prefill/Decode，一次讲透 KV Cache 的底层原理。面向 Agent 工程师的推理基础指南。"
tags: ["Prompt Caching", "KV Cache", "Agent", "LLM", "Transformer"]
categories: ["AI Agent Engineering"]
series: ["Agent 工程师的 Prompt Caching"]
---

> 📌 **本文是「Agent 工程师的 Prompt Caching」系列的第 1/4 篇**。第一篇讲 KV Cache 底层原理，第二篇讲 Prompt Cache 核心约束，第三篇讲实战踩坑与行业全景，第四篇讲 Context Engineering 架构手艺。

在讨论 Agent 系统如何优化缓存之前，我们需要先理解底层机制——KV Cache 是什么，为什么需要它，以及它如何影响 LLM 的推理性能和成本。

---

## 1.0 基础概念：Token, Embedding, Attention 与 Logits

讨论 KV Cache 之前，先对几个核心概念做一个简明解释。如果你已经熟悉 Transformer 架构，可以跳过本节。

### Token：模型的最小处理单元

LLM 不直接处理文字，而是先把文本切分成 token——模型能识别的最小单元。Token 既不完全等于字，也不完全等于词，而是介于两者之间的一种"子词"单元。

```
输入文本: "今天天气真好"
Token 序列: ["今天", "天气", "真", "好"]  (中文大致按词切分)

输入文本: "Understanding KV Cache"
Token 序列: ["Under", "standing", "KV", " Cache"]  (英文可能按子词切分)
```

一个粗略的经验：英文中 1 token ≈ 0.75 个单词，中文中 1 个汉字通常对应 1-2 个 token。当我们说"100K token 的 context"，大约相当于一本 7-8 万字的中文书。

**不同模型的词表不同**

每个模型有自己的 tokenizer 和词表。词表大小、切分规则各不相同：

| 模型 | 词表大小 | Tokenizer |
|------|---------|-----------|
| GPT-4 / GPT-4o | ~100,000 | o200k_base (tiktoken) |
| Claude 3/4 系列 | ~100,000 | Anthropic 自有 tokenizer |
| Llama 3 | ~128,000 | 基于 tiktoken 扩展 |
| DeepSeek V3 | ~128,000 | 自研 tokenizer |
| Gemini | ~256,000 | SentencePiece |
| Qwen 2.5 | ~152,000 | 自研 tokenizer |

同一段文本被不同模型切分出的 token 序列可能完全不同：

```
输入: "KV Cache 是一种优化技术"

GPT-4 可能切分为: ["KV", " Cache", " 是", "一种", "优化", "技术"]
Llama 3 可能切分为: ["K", "V", " Cache", " ", "是一种", "优化", "技术"]
Qwen 可能切分为: ["KV", " Cache", "是", "一种", "优化技术"]
```

不仅模型参数不同，连 token 粒度都不一样。

这意味着：**不同模型的 KV Cache 完全不能互用。**

**同一模型中，所有语言共享一个词表**

现代主流 LLM 使用统一的多语言词表——中文、英文、日文、德文全在同一个词表里。词表的底层 256 个 UTF-8 基础 byte token 作为兜底，确保世界上任何语言的任何字符都能被编码：

```
词表 (~100,000+ token)

├── 256 个基础 byte token (0x00 - 0xFF)      ← 兜底
├── 英文高频子词和词 (~40,000-50,000)        ← 最多
├── 中文高频字和词 (~10,000-20,000)
├── 日文、韩文、德文等 (若干)
├── 代码相关 (~5,000-10,000)
└── 特殊 token (<|begin_of_text|> 等)
```

常见字符会被直接收录为单 token，罕见字符回退到 UTF-8 字节编码：

```
常见中文字 "的": → 1 个 token       (词表直接收录)
罕见汉字 "龘":  → 可能 2-3 个 byte token  (回退到字节编码)
```

**不同语言的编码效率差异**

虽然共用词表，但不同语言的编码效率差异很大。词表通过 BPE (Byte Pair Encoding) 算法从训练数据中构建——训练数据中哪种语言多，该语言的高频组合就被合并得越充分。大部分模型的训练数据中英文占 60-80%，所以：

```
英文: "The quick brown fox jumps over the lazy dog"
→ 大约 9-10 个 token (很多常见词是单 token)

中文: "敏捷的棕色狐狸跳过了那条懒狗"
→ 大约 12-15 个 token (很多汉字需要单独编码)
```

同样的语义内容，中文通常比英文多消耗 30-50% 的 token。这直接影响：

- **成本**：中文用户为相同语义的对话付更多钱（按 token 计费）
- **Context 容量**：同样 200K 的 context window，能装的中文内容比英文少
- **KV Cache 大小**：中文的 KV Cache 更大（token 数更多）

专门针对中文优化过 tokenizer 的模型（如 Qwen、DeepSeek）会增加中文语料在 tokenizer 训练中的比重，缩小这个差距。

> **实践建议**：如果你在开发 Agent 系统，System prompt 和 tool definitions 建议用英文编写——这部分是每次请求都要加载的固定成本，用英文可以节省 30-50% 的 token 消耗，同时模型对英文指令的遵循也更精确。用户交互层面则跟随用户语言。

### Embedding：把 Token 变成向量

模型无法直接"理解"文字符号。每个 token 通过一个**嵌入层 (Embedding Layer)** 被转换成一个高维向量（比如 4096 维的浮点数数组）。这个向量就是 token 在模型内部的"数学表示"——语义相近的 token，向量在空间中距离更近。

```
"猫"  → [0.12, -0.85, 0.33, …, 0.07]  (4096 维向量)
"狗"  → [0.15, -0.79, 0.31, …, 0.09]  (与"猫"方向相近)
"经济" → [-0.67, 0.42, -0.11, …, 0.55] (与"猫"方向相远)
```

后续所有计算都在这些向量上进行。

### Attention：让每个 Token "看到"其他 Token

Attention（注意力机制）是 Transformer 的核心。它解决的问题是：一个 token 的含义往往取决于它周围的 token。比如"苹果"在"吃苹果"和"苹果发布会"中含义完全不同。

Attention 的工作方式可以用一个比喻理解：

> 想象一个会议室里坐了 10 个人（10 个 token）。每个人想更新自己的理解时，会"环顾"其他所有人，根据相关程度决定重点听谁的，然后综合大家的信息来更新自己的认知。

技术上，Attention 通过三个矩阵实现——Q (Query), K (Key), V (Value)：

- **Q（查询）**：代表"我在找什么信息"——当前 token 的需求
- **K（键）**：代表"我能提供什么信息"——每个历史 token 的标签
- **V（值）**：代表"我实际携带的信息"——每个历史 token 的内容

计算过程：

1. 每个 token 的 embedding 分别乘以三个权重矩阵，得到 Q、K、V 向量
2. 当前 token 的 Q 和所有 token 的 K 做点积 → 得到"相关性分数"
3. 分数经过 softmax 归一化 → 变成注意力权重（加起来等于 1）
4. 用注意力权重对所有 token 的 V 做加权求和 → 得到该 token 的新表示

写成公式：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

其中 √d 是一个缩放因子，防止点积值过大。

**为什么 Q、K、V 是理解 KV Cache 的关键？** 因为 KV Cache 的核心就是：在生成过程中，历史 token 的 K 和 V 不会变，可以缓存起来复用。这一点我们在 1.2 节详细展开。

**用 "Understanding KV Cache" 走一遍 Attention 计算**

抽象的 Q、K、V 概念需要一个具体例子来落地。我们用文章开头的四个 token，完整走一遍 Attention 的计算过程。

假设模型正在处理这四个 token，当前要为 Token 3 ("Cache") 计算新的表示：

```
Token 0: "Under"
Token 1: "standing"
Token 2: "KV"
Token 3: " Cache"   ← 当前正在处理
```

**第一步：每个 token 生成自己的 Q、K、V。** 每个 token 的 embedding 向量分别乘以三个权重矩阵（W_Q、W_K、W_V），得到三个不同"角色"的向量：

```
Token 0 "Under":    → Q₀, K₀, V₀
Token 1 "standing": → Q₁, K₁, V₁
Token 2 "KV":       → Q₂, K₂, V₂
Token 3 " Cache":   → Q₃, K₃, V₃
```

这三个向量是同一个 token 的三种"投影"，各有分工：

- **Q₃**（" Cache" 的查询）："我是 Cache，我前面是什么类型的 Cache？数据库 Cache？浏览器 Cache？还是某个技术术语的一部分？"
- **K₂**（"KV" 的标签）："我是 KV，一个技术缩写，通常指 Key-Value。"
- **V₂**（"KV" 的内容）：KV 实际携带的语义信息——关于键值对、关于存储结构的含义。

**第二步：Q₃ 和所有 K 做点积，计算相关性分数。**

```
Q₃·K₀ = 0.3    "Under" 和 "Cache" 关系不大
Q₃·K₁ = 0.5    "standing" 和 "Cache" 有一点关系
Q₃·K₂ = 4.2    "KV" 和 "Cache" 高度相关！KV Cache 是固定搭配
Q₃·K₃ = 1.0    自己和自己
```

Q₃·K₂ 的点积衡量的是"查询"和"标签"之间的匹配度。" Cache" 在找"我是什么类型的 Cache"，而"KV"的标签说"我是 Key-Value 的缩写"——高度匹配，分数最高。

**第三步：Softmax 归一化为注意力权重。**

```
原始分数:   [0.3,  0.5,  4.2,  1.0]
              ↓ Softmax
注意力权重: [0.02, 0.03, 0.88, 0.07]
              ↑      ↑      ↑      ↑
           "Under" 几乎  "KV"    自身
            忽略   忽略  重点关注 略微关注
```

softmax 把分数变成了"注意力预算分配"：总共 100% 的注意力，88% 分给了 "KV"。

**第四步：用注意力权重对 V 做加权求和。**

```
" Cache" 的新表示 = 0.02 × V₀ + 0.03 × V₁ + 0.88 × V₂ + 0.07 × V₃
                     忽略        忽略        主要来源       少量
                   "Under"    "standing"   "KV"的语义
```

经过 Attention 之后，Token 3 " Cache" 的表示被 "KV" 的语义深度浸染。模型现在"理解"了这不是通用的 cache 概念，而是 KV Cache 这个技术术语。这个新的表示（hidden state）会传入下一层 Transformer 继续处理。

**为什么 K 和 V 要分开？**

一个自然的疑问：K 和 V 来自同一个 token，为什么不用一个向量？

因为它们承担的角色不同：

- **K 决定"该不该被关注"**——它是匹配标签。"KV" 的 K 向量被优化成能和 " Cache" 的 Q 高度匹配的方向。
- **V 决定"关注后传递什么信息"**——它是实际内容。"KV" 的 V 向量携带的是关于键值对存储的具体语义。

类比：K 像图书馆书架上的分类标签（"计算机科学-缓存技术"），V 像书里面的实际内容。你通过标签（K）定位相关的书，但读的是书的内容（V）。如果 K 和 V 是同一个向量，"善于被找到"和"善于传递信息"两个目标就耦合了，模型的表达能力会受限。分开之后，模型可以独立优化这两个能力。

### Logits：模型的"原始预测"

模型处理完所有 token 后，最后一步是预测"下一个 token 是什么"。模型并不直接输出一个 token，而是输出一个巨大的向量——logits——长度等于词表大小（比如 Claude 的词表约 10 万 token）。

```
logits = [2.1, -0.5, 0.8, 3.7, ..., -1.2]  (长度 = 词表大小，比如 100,000)
          ↑                   ↑
        token_0          token_3 得分最高
```

每个位置的值代表模型认为对应 token 是"下一个 token"的未归一化分数。分数越高，模型越"觉得"这个 token 应该出现在下一个位置。

**注意**：logits 不是概率分布。logits 是 softmax 之前的原始分数——值可以是任意实数（正数、负数、零都行），加起来不等于 1。logits 经过 softmax 之后才变成概率分布：

```
logits:        [2.1, -0.5, 0.8, 3.7, ..., -1.2]  ← 原始分数，任意实数
                          | softmax
probabilities: [0.04, 0.003, 0.01, 0.82, ..., 0.001]  ← 概率分布，和 = 1
```

之所以保留 logits 而不直接输出概率，是因为很多场景下不需要真的算 softmax——比如取 argmax（找最高分的 token）时，logits 最大的和概率最大的一定是同一个，省掉 softmax 的指数运算。

然后通过采样策略（temperature、top-p 等）从概率分布中选出下一个 token。

**为什么理解 logits 重要？** 后面讨论 Manus 的工具管理方案时，会提到两种基于 logits 的技术：

- **Logits masking**：在模型输出 logits 之后、softmax 之前，把某些 token 的 logits 设为负无穷（-∞）。经过 softmax 后，这些 token 的概率变为 0，永远不会被选中。这是一种在不改变 prompt 的情况下精确控制模型输出的技术。
- **Response prefill**：在 assistant 的回复开头预填充一段文本（比如 `<tool_call>{"name": "browser_`），模型从预填充位置之后续写。物理上只能输出以此开头的工具名，不需要修改 logits。

### Transformer Decoder 的整体流程

把上面的概念串起来，一次 LLM 推理的完整流程是：

```
输入文本
    ↓
Tokenizer (分词器)
    ↓
Token 序列: [token_1, token_2, ..., token_n]
    ↓
Embedding Layer
    ↓
向量序列: [vec_1, vec_2, ..., vec_n]
    ↓
Transformer Layer 1 (Attention + FFN)
    ↓
Transformer Layer 2
    ↓
... 
    ↓
Transformer Layer L (比如 80 层)
    ↓
最终隐藏状态: [h_1, h_2, ..., h_n]
    ↓
取最后一个位置 h_n，乘以输出矩阵
    ↓
Logits: [score_1, score_2, ..., score_vocab_size]
    ↓
Softmax + 采样
    ↓
下一个 Token
```

每一层 Transformer 都包含一个 Attention 模块（用到 Q、K、V）和一个前馈网络（FFN）。模型有几十层这样的结构堆叠。KV Cache 就是在每一层缓存 K 和 V，所以缓存的总量 = 层数 × 2（K 和 V）× 序列长度 × 向量维度。

有了这些概念基础，我们来看自回归生成为什么会产生重复计算问题。

---

## 1.1 自回归生成与重复计算问题

大语言模型（LLM）的文本生成是**自回归（Autoregressive）**的：每次只生成一个 token，然后把这个 token 拼到已有序列后面，再预测下一个。

用伪代码表示：

```python
# 自回归生成的朴素实现
output_tokens = []
for step in range(max_new_tokens):
    # 每一步都要把完整序列送进模型
    logits = model(input_tokens + output_tokens)  # logits: 词表大小的分数向量
    next_token = sample(logits[-1])  # 取最后一个位置的 logits，采样得到下一个 token
    output_tokens.append(next_token)
```

**问题出在哪？**

每一步生成，模型都要对所有历史 token 重新做 Attention 计算——包括 Q、K、V 矩阵乘法。但对于已经出现过的 token，它们的 K 和 V 不会变（因为模型参数没变、token 没变）。唯一在变的，只有"最新生成的那个 token"对应的 Q、K、V。

举一个直观的例子。假设模型已经生成了"今天天气"四个 token，现在要预测第五个 token：

```
步骤 1: model("今")       → 预测"天"     计算 1 个 token 的 Attention
步骤 2: model("今天")     → 预测"天"     计算 2 个 token 的 Attention
步骤 3: model("今天天")   → 预测"气"     计算 3 个 token 的 Attention
步骤 4: model("今天天气") → 预测"真"     计算 4 个 token 的 Attention
步骤 5: model("今天天气真") → 预测"好"   计算 5 个 token 的 Attention
```

到步骤 5 时，"今天天气"这四个 token 的 K 和 V 已经在步骤 1-4 中各被计算过多次了——完全是重复劳动。如果要生成一段 1000 token 的回复，前面的 token 会被反复计算上百次。

这就引出了一个自然的优化思路：能不能把已经算过的 K 和 V 缓存起来，下次直接用？

---

## 1.2 KV Cache 核心思想

KV Cache 的核心思想非常直接：

> 把每一层 Attention 中、每个已处理 token 对应的 K 向量和 V 向量缓存下来。后续生成新 token 时，只需要计算新 token 自己的 Q、K、V，然后将新的 K、V 追加到缓存中，用缓存里的完整 K、V 序列做 Attention。

这里需要理解为什么只缓存 K 和 V，不缓存 Q。在 Attention 计算中（回顾 1.0 节的概念）：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- **K 和 V** 来自所有历史 token，是"被查询"的对象——新 token 需要和所有历史 token 的 K 做匹配，然后用对应的 V 加权求和。历史 token 的 K、V 不会因为新 token 的到来而改变。
- **Q** 来自当前正在处理的 token，是"发起查询"的主体——我们只需要最新 token 的 Q 去查询所有历史的 K、V。

所以：K 和 V 需要积累（每个历史 token 贡献一份），Q 只需要当前一个。缓存 K 和 V 可以避免重复计算，缓存 Q 没有意义（每步的 Q 都是新的）。

带 KV Cache 的伪代码：

```python
# 带 KV Cache 的生成
kv_cache = {}  # 每一层缓存 K, V
for step in range(max_new_tokens):
    if step == 0:
        # 第一步: 处理所有 input tokens, 填充 cache
        logits, kv_cache = model(input_tokens, kv_cache=None)
    else:
        # 后续步: 只送入上一步生成的 1 个 token
        logits, kv_cache = model([last_token], kv_cache=kv_cache)
    next_token = sample(logits[-1])
    last_token = next_token
```

效果对比：

| | 不用 KV Cache | 使用 KV Cache |
|---|---|---|
| 第 1 步 | 计算 N 个 token 的 Q、K、V | 计算 N 个 token 的 Q、K、V（相同）|
| 第 100 步 | 计算 N+99 个 token 的 Q、K、V | 只计算 1 个 token 的 Q、K、V |
| 第 1000 步 | 计算 N+999 个 token 的 Q、K、V | 只计算 1 个 token 的 Q、K、V |
| 总计算量 | O(N × T + T²) | O(N + T)（T 为生成长度）|

KV Cache 将生成阶段的计算量从二次方降到了线性，是 LLM 推理中最基础也最重要的优化。

### KV Cache 的代价

KV Cache 不是免费午餐——它用显存（GPU Memory）换计算（FLOPs）。随着序列变长，KV Cache 占用的显存线性增长：

```
KV Cache 显存 = 2 × batch_size × num_layers × hidden_size × sequence_length
              (K和V)                                          (float16)
```

以一个典型的 70B 模型为例，处理 100K token 的输入，KV Cache 可能占用**数十 GB 显存**。这也是为什么后来出现了 GQA（Grouped Query Attention）、DeepSeek MLA 等 KV Cache 压缩技术——它们的核心目标就是在不损失太多精度的前提下，减少 K、V 的存储量。

### 为什么 KV Cache 能成立：W_Q、W_K、W_V 是模型参数

前面多次提到"历史 token 的 K 和 V 不会变"——这不是一个假设，而是由模型的物理结构保证的。Q、K、V 向量是这样算出来的：

```
Q = embedding(token) × W_Q
K = embedding(token) × W_K
V = embedding(token) × W_V
```

这里的 W_Q、W_K、W_V 就是模型参数——训练过程中通过大量数据学出来的权重矩阵，保存在模型文件里，推理时固定不变。

以 "KV" 这个 token 为例：

```
K₂ = embedding("KV") × W_K
```

计算的两个输入——embedding("KV") 由 token 本身决定，W_K 是固定的模型参数——都不会变。所以 K₂ 永远相同，算一次缓存起来就够了。V 同理。如果 W_K 在推理过程中会变化（比如边推理边更新参数），KV Cache 就失效了。但标准推理不更新参数，所以 KV Cache 是安全的。

一个 Transformer Decoder 每一层包含的可训练参数：

```
每一层 Transformer:
├── Attention 模块
│   ├── W_Q (Query 投影矩阵)  ← 参数
│   ├── W_K (Key 投影矩阵)    ← 参数
│   ├── W_V (Value 投影矩阵)  ← 参数
│   └── W_O (Output 投影矩阵) ← 参数
└── FFN 模块 (前馈网络)
    ├── W_1 (第一层线性变换)   ← 参数
    └── W_2 (第二层线性变换)   ← 参数
```

这套参数在每一层都有独立的一份。一个 80 层的模型就有 80 套。当我们说"70B 模型"，指的就是这些权重矩阵加起来总共约 700 亿个浮点数。每个浮点数在 float16 精度下占 2 个字节，所以 70B 模型的参数占用 = 700 亿 × 2 bytes = 140 GB 显存。

### GPU 显存的三大块：不只是 KV Cache

了解了模型参数和 KV Cache 之后，可以看一下推理时 GPU 显存的完整占用情况。它主要由三部分构成：

```
GPU 显存总占用 = 模型参数 + KV Cache + 激活值 (Activations)
```

模型参数是最大的固定开销，和序列长度无关：

```
70B 模型 × float16 (2 bytes/参数)  = 140 GB
70B 模型 × int8 量化 (1 byte/参数) = 70 GB
70B 模型 × int4 量化 (0.5 byte/参数) = 35 GB
```

KV Cache 随序列长度线性增长，前面已经详细讨论。

**激活值（Activations）**是前向计算过程中的临时中间结果。Prefill 阶段激活值不小（同时处理大量 token），Decode 阶段几乎可以忽略（每步只有 1 个 token）。

三者的比例在不同场景下差异很大：

**场景 1：70B 模型，短对话（1K tokens）**

```
模型参数: 140 GB  ████████████████████████████████  (~97%)
KV Cache: 2.5 GB  █                                 (~2%)
激活值:   1.5 GB  █                                 (~1%)
→ 模型参数是绝对大头
```

**场景 2：70B 模型，长对话（100K tokens）**

```
模型参数: 140 GB  █████████████████                  (~36%)
KV Cache: 250 GB  █████████████████████████████████  (~63%)
激活值:   4 GB    █                                  (~1%)
→ KV Cache 反超模型参数
```

短序列时模型参数是大头，长序列时 KV Cache 反超。这也解释了为什么业界在两个方向同时优化：模型参数压缩（量化、蒸馏）减少固定开销，KV Cache 压缩（GQA、MLA、稀疏注意力）减少动态开销。

### LLM 的 KV Cache vs Redis 等应用层 KV Cache

如果你有后端开发背景，看到"KV Cache"可能第一反应是 Redis、Memcached 这类键值缓存。它们名字相同，但本质上是完全不同的东西：

| | LLM 的 KV Cache | Redis / Memcached |
|---|---|---|
| K 和 V 是什么 | Attention 中的数学向量（高维浮点数组）| 业务数据的键值对（string → object）|
| 存在哪里 | GPU 显存（HBM，带宽 ~3 TB/s）| CPU 内存（DDR，带宽 ~50 GB/s）|
| 数据内容 | 对人类不可读的中间计算结果 | 对人类可读的业务数据 |
| 缓存目的 | 避免重复的矩阵乘法运算 | 避免重复的数据库查询 |

LLM 的 KV Cache 必须在 GPU 显存里，原因是 Decode 阶段每一步都要读取整个 KV Cache。存储介质的带宽差异决定了一切：

```
从 GPU 显存读 10GB: 10 GB ÷ 3 TB/s  = ~3 ms      ✅
从 CPU 内存读 10GB: 10 GB ÷ 50 GB/s = ~200 ms     ❌ 慢 60 倍
从 SSD 读 10GB:     10 GB ÷ 7 GB/s  = ~1.4 s      ❌ 慢 400 倍
```

如果把 KV Cache 放在 CPU 内存（像 Redis 那样），生成 500 token 的回复需要 500 × 200ms = 100 秒——显然不可接受。

不过在超长序列场景下，业界确实在探索**分层缓存**——把 KV Cache 按"热/温/冷"分层存储：

```
热数据（最近的 token）:     GPU 显存  ← 频繁访问，3 TB/s
温数据（中间的 token）:     CPU 内存  ← 偶尔访问，50 GB/s
冷数据（很久之前的 token）: NVMe SSD  ← 几乎不访问，7 GB/s
```

这种分层策略和 Redis 的多级缓存在**架构思想上确实相似**——都是用数据冷热程度来平衡速度和容量。支持 1M token context window 的模型（如 Claude Opus 4.6），很可能在内部就使用了类似的分层 KV Cache 策略。

---

## 1.3 Prefill 与 Decode：推理的两个阶段

理解了 KV Cache 之后，我们可以把 LLM 推理清晰地分成两个阶段。这两个阶段的计算特性截然不同，理解它们对后面理解 Prompt Cache 至关重要。

### Prefill 阶段：一次性处理所有输入

Prefill 就是上面伪代码中 `step == 0` 的那一步：模型一次性处理所有输入 token（system prompt + user message），为每一层、每个 token 计算出 K 和 V 并存入 cache。

关键特点：

- **所有输入 token 可以并行处理**——虽然 Attention 是 causal 的（每个 token 只能看到它之前的 token），但 GPU 可以用矩阵乘法一次完成所有位置的计算
- **计算量大**：N 个 token × 所有层 × Q/K/V 矩阵运算
- **Compute Bound（计算密集型）**：GPU 的算力是瓶颈，数据搬运不是瓶颈

### Decode 阶段：逐个 token 生成

Decode 就是后续的 `step > 0`：每一步只输入 1 个 token，利用 KV Cache 做 Attention，生成下一个 token。

关键特点：

- **每步只处理 1 个 token**——因为下一个 token 依赖上一个的输出，无法并行
- 每步的计算量不大——只需要 1 个 token 的 Q 和 cache 中所有 K/V
- 但每步都要从显存读取整个 KV Cache
- **Memory Bound（内存带宽密集型）**：GPU 的显存带宽是瓶颈，计算单元在等数据搬运

### 用 "Understanding KV Cache" 走一遍完整推理过程

我们用一个具体场景把 Prefill 和 Decode 完整走一遍：

```
用户输入: "Understanding KV Cache"
模型生成: " is a key optimization technique"

输入 token: ["Under", "standing", " KV", " Cache"]                    → 4 token
输出 token: [" is", " a", " key", " optimization", " technique"]      → 5 token
```

**Prefill：并行处理 4 个输入 token**

模型一次性处理全部 4 个输入 token，填充 KV Cache，并预测第一个输出 token：

```
输入: ["Under", "standing", "KV", " Cache"] → 同时送入模型

┌──────────────────────────────────────────────────┐
│          Transformer Layer 1                      │
│                                                   │
│  "Under"    → Q₀, K₀, V₀                        │
│  "standing" → Q₁, K₁, V₁  → Attention 计算       │
│  "KV"       → Q₂, K₂, V₂    (4×4 的矩阵运算)    │
│  " Cache"   → Q₃, K₃, V₃                        │
│                                                   │
│  写入 KV Cache: [K₀, K₁, K₂, K₃]               │
│                 [V₀, V₁, V₂, V₃]               │
│                                                   │
│  Transformer Layer 2 ... Layer L                  │
│  (每层重复上述过程，每层有独立 KV Cache)           │
└──────────────────────────────────────────────────┘
                    ↓
取最后位置 (" Cache") 的输出 → logits → softmax
                    ↓
预测出第一个输出 token: " is"
```

Prefill 阶段的 Attention 是一个 4×4 的矩阵运算（causal mask，下三角形式——每个 token 只能看到自己和之前的 token）：

```
        K₀   K₁   K₂   K₃
Q₀ [ 0.8   -    -    -  ]   "Under" 只能看到自己
Q₁ [ 0.3  0.9   -    -  ]   "standing" 看到前两个
Q₂ [ 0.1  0.1  0.7   -  ]   "KV" 看到前三个
Q₃ [ 0.3  0.5  4.2  1.0 ]   "Cache" 看到全部
```

4 个 token 并行计算，GPU 用矩阵乘法一步完成。计算完成后，每一层都存下了 4 组 K、V 向量。这一步的耗时 = TTFT（用户等待第一个字出现的时间）。

**Decode 步骤 1：生成 " a"**

Prefill 完成后，KV Cache 已有 4 组 K、V。现在送入刚生成的 " is"：

```
送入 1 个 token: "is"

┌──────────────────────────────────────────────────┐
│          Transformer Layer 1                      │
│                                                   │
│  "is" → 计算: Q₄, K₄, V₄ (只有 1 组)            │
│                                                   │
│  从 Cache 读取: [K₀, K₁, K₂, K₃]               │
│                                                   │
│  Q₄ · [K₀,K₁,K₂,K₃,K₄] → 注意力权重            │
│  权重 × [V₀,V₁,V₂,V₃,V₄] → 输出                │
│                                                   │
│  追加到 Cache: [K₀,K₁,K₂,K₃,K₄]                │
│               [V₀,V₁,V₂,V₃,V₄]                │
│                                                   │
│  → "is" 的新表示                                  │
└──────────────────────────────────────────────────┘
                    ↓
… 后续层同样处理 …
                    ↓
logits → softmax → "a"
```

和 Prefill 的关键区别：

- **Prefill**：4 个 token 同时计算，Attention 是 4×4 矩阵乘法
- **Decode**：1 个 token 计算，Attention 是 1×5 向量点积

计算量从矩阵运算降到了向量运算，但每步都要从显存读取整个 KV Cache。

**Decode 步骤 2-5：依次生成后续 token**

```
步骤 2: 送入 "a"
  ├── 计算 Q₅, K₅, V₅
  ├── Q₅ 和 Cache 中 6 个 K 做点积
  ├── 追加 K₅, V₅ → Cache 现在有 6 组
  └── 输出: " key"

步骤 3: 送入 " key"
  ├── 计算 Q₆, K₆, V₆
  ├── Q₆ 和 Cache 中 7 个 K 做点积
  ├── 追加 K₆, V₆ → Cache 现在有 7 组
  └── 输出: " optimization"

步骤 4: 送入 " optimization"
  ├── 计算 Q₇, K₇, V₇
  ├── Q₇ 和 Cache 中 8 个 K 做点积
  ├── 追加 K₇, V₇ → Cache 现在有 8 组
  └── 输出: " technique"

步骤 5: 送入 " technique"
  └── 输出: <end_of_text> → 生成结束
```

每一步 KV Cache 都在增长，读取量也在增长：

```
步骤 1: 读 4 组 KV，新算 1 组 → Cache: 5 组
步骤 2: 读 5 组 KV，新算 1 组 → Cache: 6 组
步骤 3: 读 6 组 KV，新算 1 组 → Cache: 7 组
步骤 4: 读 7 组 KV，新算 1 组 → Cache: 8 组
步骤 5: 读 8 组 KV，新算 1 组 → Cache: 9 组
```

每步计算量很小（1 个 token），但要读的 KV Cache 越来越大——这就是 Memory Bound 的本质。

**没有 KV Cache 会怎样？**

```
有 KV Cache:
步骤 4: 送入 "optimization" (1 个 token)
        读 Cache 的 7 组 KV，做 1×8 的 Attention → 输出 "technique"

没有 KV Cache:
步骤 4: 送入全部 8 个 token ["Under","standing"," KV"," Cache"," is","a"," key"," optimization"]
        重新计算所有 8 个 token 的 K 和 V
        做完整的 8×8 Attention → 输出 "technique"
```

在这个 8 token 的小例子里差距不太明显。但在 Agent 场景——累积了 100K token 的对话历史——每一步重算 100K token 的 KV vs 只算 1 个 token，差距是灾难性的。

**完整时间线**

```
时间轴:
│
│<── Prefill ──>│<──────────── Decode ─────────────────>│
│               │                                       │
│  处理 4 个    │  "is" "a" "key" "optimization" "technique" │
│  输入 token   │  (逐个生成，每步都用 KV Cache)         │
│  (并行)       │                                       │
│<── TTFT ──>│                                          │
│ 用户在等待    │  用户开始看到文字逐个出现              │
│               │  ←TPOT→ 每个 token 之间的间隔          │
```

### Compute Bound vs Memory Bound 直觉

要理解这两种瓶颈，需要引入一个概念——**算术强度（Arithmetic Intensity）**：

```
算术强度 = 计算量 (FLOPs) / 数据搬运量 (Bytes)
```

- **Compute Bound**：算术强度高。大量的计算操作对应少量的数据搬运。GPU 的计算单元忙不过来，数据搬运不是瓶颈。Prefill 就是这种情况——大矩阵乘法，计算密集。
- **Memory Bound**：算术强度低。少量的计算操作需要大量的数据搬运。GPU 的计算单元在等数据从显存搬过来。Decode 就是这种情况——每步只有 1 个 token 的小矩阵运算，但要读取整个 KV Cache。

用一个现实比喻：

- **Prefill** 像工厂流水线批量处理 1000 个订单——机器满负荷运转，原材料提前备好了
- **Decode** 像每次只来 1 个快递——打包机器大部分时间空闲，瓶颈在于从仓库取货的速度

**一个值得思考的问题：Prefill 为什么要计算所有 token 的 Q？**

这个问题第一次看到会觉得很有道理：既然我们只需要预测 next token，Prefill 阶段不是只需要最后一个 token 的 Q 吗？K 和 V 确实需要全算（因为最后一个 Q 要和所有 K 做 attention），但 Q 为什么不能只算最后一个？

答案在于：**Transformer Decoder 有很多层。**

如果只有一层，确实只需要最后一个 token 的 Q。但实际的 Decoder 有几十层，上一层所有位置的输出是下一层所有位置的输入：

1. **第 1 层**：输入是 token embedding。为了得到所有位置的 K、V（要存入 cache），需要完整的 Attention 计算——包括所有位置的 Q。
2. **第 2 层**：输入是第 1 层的输出。第 1 层输出取决于 Attention 的完整计算，所以第 2 层的 K、V 依赖第 1 层所有位置的 Q 参与计算。
3. **第 N 层**：同理递推，依赖前面所有层的完整输出。

所以结论是：

> Prefill 必须计算所有 token 的 Q，不是因为最终预测需要，而是因为每一层的 KV Cache 依赖于上一层所有位置的完整输出，而完整输出需要所有位置的 Q 参与计算。

这也从另一个角度解释了为什么 Prefill 是 Compute Bound——它确实需要做大量计算，不是在浪费。

---

## 1.4 TTFT 与 TPOT：两个核心延迟指标

两个推理阶段对应两个不同的用户体验指标：

### TTFT — Time To First Token（首 token 延迟）

**定义**：用户发送请求到看到第一个输出 token 的时间。

**由什么决定**：Prefill 阶段的耗时决定。模型需要处理完所有输入 token，才能开始生成第一个输出 token。

**影响因素**：

- 输入 token 数——输入越长（大 prompt），Prefill 时间越长
- 模型大小——参数越多，每层计算越慢
- GPU 算力——Prefill 是 Compute Bound

**直观感受**：用户按下回车后，等待光标开始闪烁的时间。如果输入很长（比如长文档问答、Agent 的多轮对话累积了大量历史），这个等待时间会很明显。

### TPOT — Time Per Output Token（每 token 生成延迟）

**定义**：生成每个后续 token 的平均时间。

**由什么决定**：主要由 Decode 阶段的每步耗时决定。

**影响因素**：

- KV Cache 大小（序列长度）——Cache 越大，每步读取数据越多
- GPU 显存带宽——Decode 是 Memory Bound
- 模型大小——影响每步的计算和数据量

**直观感受**：光标开始闪烁后，文字流出的速度。这个通常比 TTFT 感受弱一些，因为一旦开始输出，用户就能边看边等。

### 两个指标的关系

```
总响应时间 = TTFT + (输出 token 数 × TPOT)
```

示例（假设 10K token 输入，生成 500 token 输出）：

```
不优化: TTFT = 3.0s + 500 × 0.02s = 3.0s + 10.0s = 13.0s
优化后: TTFT = 0.5s + 500 × 0.02s = 0.5s + 10.0s = 10.5s
                ↑
          Prompt Cache 主要优化这里
```

**注意：Prompt Cache 主要优化的是 TTFT，而非 TPOT。** 因为 Prompt Cache 跳过了已缓存前缀的 Prefill 计算，直接使用已有的 KV Cache。Decode 阶段不受影响——不论前缀是否缓存，生成每个新 token 时都要读取完整的 KV Cache。

对于 Agent 场景，这个优化尤为重要。一个 Agent 可能累积了 100K+ token 的对话历史，每一步都要重新发送。没有 Prompt Cache 时，每一步的 TTFT 都很长（需要 Prefill 整个 100K+ token 的序列）。有了 Prompt Cache，前缀部分的 Prefill 被跳过，TTFT 大幅降低。

---


## 本节小结

| 概念 | 核心要点 |
|------|---------|
| Token 与 Tokenizer | 模型的最小处理单元，不同模型词表不同，同一模型多语言共享词表但编码效率有差异 |
| Embedding | 把离散的 token 转换为高维向量，语义相近的 token 在向量空间中距离更近 |
| Attention (Q/K/V) | Q 发起查询，K 提供匹配标签，V 提供实际内容；三者来自同一个 token 的不同投影 |
| Logits | 模型输出的未归一化分数向量，经 softmax 后变为概率分布 |
| 模型参数 | W_Q、W_K、W_V 等权重矩阵，训练时学出，推理时固定——这是 KV Cache 成立的根本保证 |
| 自回归生成 | 每步只生成一个 token，导致历史 token 被反复计算 |
| KV Cache | 缓存历史 token 的 K、V 向量，避免 Decode 阶段重复计算，用显存换计算 |
| GPU 显存占用 | 模型参数（固定）+ KV Cache（随序列长度增长）+ 激活值（临时），短序列参数为主，长序列 KV Cache 为主 |
| Prefill | 并行处理所有输入 token，Compute Bound，决定 TTFT |
| Decode | 逐个生成输出 token，Memory Bound，决定 TPOT |

这些是理解后续内容的基础。下一篇我们将进入 Prompt Cache 的世界——当 KV Cache 从单次请求扩展到跨请求的维度，它如何成为 Agent 系统架构设计的核心约束。

---

**「Agent 工程师的 Prompt Caching」系列导航**

1. **KV Cache 原理：LLM 推理的底层机制**（本文）
2. [Prompt Cache：Agent 成本控制的核心约束](/posts/prompt-caching-core-constraints/)
3. [Cache 杀手与行业实战：从踩坑到最佳实践](/posts/prompt-caching-killers-and-industry/)
4. [Context Engineering：Agent 架构师的核心手艺](/posts/prompt-caching-context-engineering/)
