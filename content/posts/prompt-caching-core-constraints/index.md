---
title: "Prompt Cache：Agent 成本控制的核心约束"
date: 2026-01-17
draft: false
summary: "从 KV Cache 到 Prompt Cache 的认知跃迁：前缀精确匹配的铁律、Agent 的 I/O 100:1 失衡、以及为什么 Cache-Safe 是 Client 侧工程责任。"
description: "从 KV Cache 到 Prompt Cache 的认知跃迁：前缀精确匹配的铁律、Agent 的 I/O 100:1 失衡、以及为什么 Cache-Safe 是 Client 侧工程责任。"
tags: ["Prompt Caching", "KV Cache", "Agent", "LLM", "Cost Optimization"]
categories: ["AI Agent Engineering"]
series: ["Agent 工程师的 Prompt Caching"]
ShowToc: true
TocOpen: true
---

> 📌 **本文是「Agent 工程师的 Prompt Caching」系列的第 2/4 篇**。第一篇讲 KV Cache 底层原理，第二篇讲 Prompt Cache 核心约束，第三篇讲实战踩坑与行业全景，第四篇讲 Context Engineering 架构手艺。

## 从 KV Cache 到 Prompt Cache：认知跃迁

上一篇我们搞清楚了 KV Cache 的本质——**用显存换计算**，在单次请求内缓存已算过的 K/V 矩阵，避免自回归生成时的重复计算。

但如果你只理解到这一层，你还停留在"LLM 工程"的视角。

**真正改变 Agent 架构师思维方式的，是下面这个问题：**

> 如果用户每次对话，前面 80% 的 Prompt 都一模一样（System Prompt + 工具定义 + 历史对话），凭什么每次都要重新算一遍 Prefill？

这就是 **Prompt Cache（前缀缓存）** 要解决的问题。它把 KV Cache 的思想从单次请求扩展到了跨请求的维度：

> 如果两次 API 调用的 prompt 共享相同的前缀，第二次调用可以直接复用第一次 Prefill 算出来的 KV Cache，跳过前缀部分的 Prefill 计算。

### 从"单场考试草稿纸"到"跨科目公式表"

先用一个类比帮你建立直觉：

**KV Cache** 像是考试时的草稿纸。你在解一道大题，每算一步就把中间结果记在草稿纸上，下一步直接引用，不用从头算。但考完这场试，草稿纸就扔了。下一场考试，哪怕遇到一模一样的前几步，你还得重新算。

**Prompt Cache** 像是考试时允许带入的公式表。数学考完，物理还能用。只要公式表的内容没变，每场考试都可以直接引用，不用重新推导。

用技术语言说：

| 维度 | KV Cache | Prompt Cache |
|------|----------|--------------|
| 作用范围 | 单次请求**内部** | **跨请求** |
| 缓存什么 | 当前请求中已生成 token 的 K/V | 之前请求中已计算的前缀 K/V |
| 省的是什么 | Decode 阶段的重复计算 | **Prefill 阶段**的重复计算 |
| 对谁有价值 | 所有 LLM 推理 | **输入重复率高的场景**（Agent、多轮对话） |

这对以下场景特别有价值：

- **多轮对话**：每轮对话的 prompt 都以之前的对话历史作为前缀
- **相同 System Prompt**：同一个应用的所有请求共享相同的 system prompt
- **Agent 系统**：Agent 每一步的 prompt = 上一步的 prompt + 新的 action/observation

这意味着第二次请求的 TTFT（首字延迟）会大幅降低——因为大部分 Prefill 计算被跳过了。

---

## 前缀精确匹配：一条铁律

Prompt Cache 的匹配规则极其严格：

> **必须从第一个 token 开始完全一致，一个 token 的差异就会导致该位置之后的 cache 全部失效。**

```
请求 1: [System Prompt] [User: Hello]                → 无 cache，全部计算
请求 2: [System Prompt] [User: Hello] [Assistant: Hi] [User: 你好] → [System Prompt][User: Hello] 命中 cache
请求 3: [Modified System Prompt] [User: Hello]       → 完全 miss，第一个 token 就不一样
```

这不是"大致相似就行"的模糊匹配，而是 **byte-level exact match**。这条约束看似简单，但它对 Agent 系统的架构设计产生了深远影响——如何组织 prompt 结构、如何管理工具列表、如何更新状态、如何压缩历史，所有这些设计决策都必须围绕"不破坏前缀"这条铁律展开。

### 用例子理解前缀匹配的严格性

继续用 "Understanding KV Cache" 的例子。假设用户发了两次请求：

```
请求 1: "Understanding KV Cache"
→ Prefill 4 个 token: ["Under", "standing", " KV", " Cache"]
→ KV Cache 写入: K₀K₁K₂K₃ / V₀V₁V₂V₃

请求 2: "Understanding KV Cache is important because"
→ 前 4 个 token 和请求 1 完全一致 → 命中 Cache！
→ 只需 Prefill 新增的 3 个 token: [" is", " important", " because"]
```

但如果请求 2 有一个微小的拼写变化——"UnderstandingKV"（少了空格）：

```
请求 2 (错误): "UnderstandingKV Cache is important because"
→ tokenizer 切分结果完全不同: ["Understanding", "KV", " Cache", " is", ...]
→ 第一个 token "Understanding" ≠ "Under" → 从第一个位置就不匹配
→ Cache 完全 miss，需要全量 Prefill
```

即使人眼看起来"几乎一样"，一个空格的差异就导致 tokenizer 切分出完全不同的 token 序列，Cache 100% miss。

这个例子揭示了一个更深层的问题：在 Agent 系统中，Cache miss 往往不是因为人打错字，而是代码层面的非确定性。比如 JSON 序列化时 key 的排序不一致：

```python
# 请求 1: {"a": 1, "b": 2} → 某组 token 序列
# 请求 2: {"b": 2, "a": 1} → 不同的 token 序列
# 语义完全相同，但 Cache 完全 miss
```

---

## 经济账：不是"省一点"，而是"能不能用得起"

很多人觉得 Prompt Cache 就是"省点钱"。不对。在 Agent 场景下，这是**能不能用得起**的问题。

### Agent 的 I/O 比例严重失衡

普通聊天场景，用户输入和模型输出差不多长。但 Agent 不一样——每一步都要把完整的对话历史（System Prompt + 工具定义 + 之前所有的 action/observation）全部发给模型，模型只输出一小段（一个工具调用或一段回复）。

Manus 披露过一个数据：**input : output ≈ 100 : 1**。

如果没有 Prompt Cache，每一步都要重新 Prefill 所有历史 token。随着对话轮次增加，输入 token 数线性增长，但你要为**每一步**都付全价。总成本？**二次方增长**。

### 三家的实际数据

| 场景 | 不缓存 | 缓存后 | 节约 |
|------|--------|--------|------|
| Claude（正常 vs cached input） | $3/MTok | $0.30/MTok | **90%** |
| OpenAI GPT-5（正常 vs cached） | $10/MTok | $2.50/MTok | **75%** |
| Claude Code 单任务（约 2M tokens） | ~$6.00 | ~$1.15 | **81%** |

Claude Code 团队的 Thariq 说过一句被广泛引用的话：

> "Coding agents would be cost prohibitive without prompt caching."
> （没有 Prompt Caching，编程 Agent 在成本上根本不可行。）

### 延迟同样关键

不只是钱的问题。当你的 Agent 一个任务要跑 50 步，每步的 Prompt 有 150K+ tokens，TTFT 的累积效应非常显著。

- OpenAI 数据：150K+ tokens 时，cached 请求 TTFT 快 **67%**
- Manus：把 KV cache hit rate 视为 "**the single most important metric**"（最重要的单一指标）
- Claude Code 团队：Cache 命中率下降 → 当作**线上事故（SEV）**处理

你没看错，Cache 命中率掉了，不是发个告警，是当 SEV 处理。这说明了什么？说明在生产级 Agent 系统中，Prompt Cache 已经是基础设施级别的依赖。

---

## Cache 的存储管理：服务端的复杂工程

理解了 Cache 的成本影响之后，一个自然的问题是：这些 Cache 在服务端是怎么管理的？

### 多条目共存与 TTL 机制

Cache 条目不会在新请求到来时被覆盖，而是共存。Anthropic 的 Cache TTL（生存时间）约为 5 分钟，命中后刷新。

在 Agent 场景下，每一步都产生一个新的（更长的）Cache 条目：

```
Agent 步骤 1:  [System][Tools][Msg1]              → Cache 条目 A
Agent 步骤 2:  [System][Tools][Msg1][Msg2]        → Cache 条目 B (A 仍存在)
Agent 步骤 3:  [System][Tools][Msg1][Msg2][Msg3]  → Cache 条目 C (A, B 仍存在)
...
Agent 步骤 20: [System][Tools][Msg1]...[Msg20]    → Cache 条目 T (A-S 可能仍存在)
```

如果每步间隔不到 5 分钟，20 条 Cache 条目可能同时存在于服务端显存中。再乘以成千上万的并发用户，服务端的 Cache 存储系统承受着巨大的压力。

如果因为非确定性序列化（如前面 JSON key 排序不一致的例子），同一个用户的同一个对话还可能产生多条语义几乎相同但无法互相复用的 Cache 条目——既浪费计算，又浪费存储。

### 开源推理引擎的 Cache 管理

商业 API 的 Cache 管理对开发者是黑盒。但开源推理引擎提供了透明的实现，可以帮助理解原理：

**vLLM 的 Block 级管理**：vLLM 把 KV Cache 按固定大小的 block 存储（比如每 16 个 token 一个 block），用 hash 做索引。不同请求如果有共同前缀，可以在 block 粒度上共享，比"整条前缀要么全命中要么全 miss"更高效：

```
请求 1: [Block_A: "Under","standing",...] [Block_B: "KV"," Cache",...] [Block_C: ...]
请求 2: [Block_A: "Under","standing",...] [Block_B: "KV"," Cache",...] [Block_D: ...]
                                                                        ↑ 这里开始不同
Block_A 和 Block_B 被两个请求共享，Block_C 和 Block_D 各自独立
```

**SGLang 的 Radix Tree**：用基数树管理前缀，共同前缀只存一份，分叉点之后各存各的：

```
         [System Prompt]
            /        \
   [用户A的对话]    [用户B的对话]
      / \                |
  [步骤2] [步骤2变体]   [步骤2]
```

两种方案都使用 LRU（最近最少使用）策略，在显存不足时淘汰最久没被命中的 Cache 条目。

这对 Agent 开发者的启示是：**做 Cache-Safe 设计，本质上是在帮服务端减少无效的 Cache 条目，提高整体资源利用效率。** Cache 命中率越高，服务端的显存浪费越少，你支付的 Cache Write 溢价也越值得。

---

## 这条约束如何改变 Agent 架构

**前缀精确匹配这条约束，直接决定了 Agent 的整个架构设计。**

Claude Code 团队的 Thariq 说得很直白：Prompt Cache 不只是一个省钱技巧，它是 Agent 系统架构设计的核心约束。就像数据库的 Schema 设计会影响整个应用架构一样，前缀匹配约束深刻地影响了 Agent 的每一个设计决策。

具体来说：

- **Prompt 怎么组织？** → 稳定内容放前面，变化内容放后面
- **工具怎么管理？** → 工具列表固定不变，通过其他机制限制可用范围
- **动态信息怎么更新？** → 永远追加新消息，永远不修改旧消息
- **模型怎么选？** → 单次会话内不切换模型，用子代理隔离

这些设计原则，不是某个团队的"最佳实践"，而是被前缀匹配这条物理约束**逼出来的**。Claude Code、Manus、OpenAI Codex 三家独立演化，最终殊途同归。

---

## Prompt Caching 到底是 Agent 工程还是 LLM 工程？

答案是：**两者都是，但重心在 Agent 工程。**

| 层面 | 归属 | 谁负责 |
|------|------|--------|
| KV Cache 的实现 | LLM 工程 | 推理引擎开发者（vLLM、SGLang） |
| Prefix Caching 的服务端实现 | LLM 工程 | 模型服务提供商（Anthropic、OpenAI、DeepSeek） |
| **Prompt 结构设计（稳定前缀 + 可变后缀）** | **Agent 工程** | **你，Agent 架构师** |
| **工具列表的 Cache-Safe 管理** | **Agent 工程** | **你** |
| **上下文压缩策略** | **Agent 工程** | **你** |
| **子代理的 Cache 友好设计** | **Agent 工程** | **你** |
| **序列化的确定性保证** | **Agent 工程** | **你** |

开源推理引擎也已经是标配——vLLM 通过 `--enable-prefix-caching` 开启，SGLang 默认开启 RadixAttention。无论你用开源引擎自部署还是调用商业 API（Claude、OpenAI、DeepSeek），都可以获得这个优化。

一句话总结：

> **LLM 工程师把 Prompt Cache 的能力造好了，但能不能用上、用好，是 Agent 架构师的责任。Cache-Safe 是 Client 侧工程。**

很多做 Agent 开发的同学，用着 Claude API 或 OpenAI API，Prompt Cache 默认就开着，但因为 Prompt 结构设计不当（时间戳放开头、动态增删工具、非确定性序列化），Cache 命中率可能低得可怜——**你以为在省钱，其实在烧钱**。

到底是什么在破坏你的 Cache？下一篇我们来拆解 Cache 命中率的六大杀手，以及 Claude Code、Manus、OpenAI Codex 各自的应对方案，并做一次全行业扫描。

---

> 1. **KV Cache 是单次请求内的优化，Prompt Cache 是跨请求的优化**——这个跃迁是理解 Agent 成本控制的关键。
>
> 2. **前缀精确匹配是物理约束**，不是建议。任何位置的任何 token 差异都会破坏后续 Cache。三家头部 Agent 的架构设计都围绕这条约束展开。
>
> 3. **Agent 场景下，没有 Prompt Cache 就没有可行的商业模型**。100:1 的 I/O 比 + 二次方成本增长 = 不可持续。
>
> 4. **Cache-Safe 是 Client 侧工程责任**。推理引擎已经准备好了，问题在于你的 Agent 代码有没有为它做设计。

---

**「Agent 工程师的 Prompt Caching」系列导航**

1. [KV Cache 原理：LLM 推理的底层机制](/posts/prompt-caching-kv-cache-fundamentals/)
2. **Prompt Cache：Agent 成本控制的核心约束**（本文）
3. [Cache 杀手与行业实战：从踩坑到最佳实践](/posts/prompt-caching-killers-and-industry/)
4. [Context Engineering：Agent 架构师的核心手艺](/posts/prompt-caching-context-engineering/)
