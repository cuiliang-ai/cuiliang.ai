---
title: "Context Engineering：Agent 架构师的核心手艺"
date: 2026-01-31
draft: false
summary: "Context Rot 的物理现实、三层压缩策略、文件系统作为延展记忆、CLAUDE.md 最佳实践，以及从 Prompt Engineering 到 Context Engineering 的范式转变。"
description: "Context Rot 的物理现实、三层压缩策略、文件系统作为延展记忆、CLAUDE.md 最佳实践，以及从 Prompt Engineering 到 Context Engineering 的范式转变。"
tags: ["Prompt Caching", "Context Engineering", "Agent", "LLM", "CLAUDE.md"]
categories: ["AI Agent Engineering"]
series: ["Agent 工程师的 Prompt Caching"]
---

> 📌 **本文是「Agent 工程师的 Prompt Caching」系列的第 4/4 篇**（完结篇）。第一篇讲 KV Cache 底层原理，第二篇讲 Prompt Cache 核心约束，第三篇讲实战踩坑与行业全景，第四篇讲 Context Engineering 架构手艺。

## Context 管理的完整策略

前面几节聚焦于 Prompt Cache 这一个核心约束。但 Agent 面临的 context 挑战远不止 Cache——随着任务执行，context 会不断膨胀、腐烂、失焦。

这一节我们拉高视角，看 context 管理的完整策略：为什么要主动管理、怎么压缩、往哪里卸载、怎么渐进式加载。

---

### Context Rot：为什么"更大的 context window"不是解药

你可能想：模型现在都支持 128K、200K 甚至更长的 context window 了，把所有东西塞进去不就行了？

不行。

Anthropic 在 2025 年 9 月发布的 *Effective Context Engineering for AI Agents* 中，引用了 Chroma 团队的研究，正式提出了 **Context Rot（上下文腐烂）** 的概念：

> 随着 context window 中 token 数量增加，模型准确回忆信息的能力会下降。

这不是某个模型的 bug，而是 Transformer 架构的固有特性。每个 token 要和所有其他 token 建立注意力关系，token 数为 n 时，就有 n² 组 pairwise 关系。context 越长，每个 token 分到的"注意力预算"越少。

Anthropic 用了一个很精准的类比：

> LLMs, like humans, have an "attention budget" that they draw on when parsing large volumes of context. Every new token introduced depletes this budget.
> （LLM 和人类一样，有一个解析大量上下文时使用的"注意力预算"。每引入一个新 token，都在消耗这个预算。）

具体来说，Context Rot 会导致三个问题：

1. **关键指令被"稀释"**：System Prompt 中的核心规则被大量 tool output 淹没，模型开始"忘记"早期指令
2. **远程依赖断裂**：第 5 步做的决策到第 40 步时，模型可能已经"看不到"当时的推理过程
3. **噪声累积**：大量过时的 observation（比如早期读取的网页内容）仍然占据 context，干扰当前决策

所以 Anthropic 的结论很明确：

> Context must be treated as a finite resource with diminishing marginal returns.
> （Context 必须被视为一种边际收益递减的有限资源。）

**能塞进去，不代表模型能有效利用。主动管理 context，是 Agent 架构师的核心职责。**

---

### Compaction 三层策略：Raw → Compact → Summarize

既然 context 会腐烂，就需要主动压缩。但压缩不能一刀切——压得太狠会丢失关键信息，压得太轻又解决不了问题。

综合 Manus（Lance Martin 的 webinar 笔记）、Claude Code（Anthropic 工程博客）和 OpenAI Codex 的实践，可以提炼出一个三层递进的压缩策略：

### 第一层：Raw（完整保留）

最新的 tool 调用结果保留完整内容，不做任何压缩。

为什么？因为模型的**下一步决策**高度依赖最近的 observation。你刚读了一个文件的内容，模型需要看到完整内容才能决定下一步做什么。

### 第二层：Compact（压缩引用）

较早的 tool 调用结果，用**引用替换完整内容**。

Manus 的做法（来自 Lance Martin webinar 笔记）：

> Tool calls in Manus have a "full" and "compact" representation. The full version contains the raw content from tool invocation. The compact version stores a reference to the full result (e.g., a file path).
> （Manus 的工具调用有"完整"和"紧凑"两种表示。完整版包含原始内容，紧凑版存储引用（如文件路径）。）

比如第 5 步读取了一个网页，完整内容有 8000 token。到了第 20 步，这个内容已经不太可能影响下一步决策了，就把 context 中的完整内容替换为：

```
[Tool Result - Compact] 网页内容已保存至 /sandbox/web_page_step5.html
```

如果后续需要，模型可以重新读取文件恢复完整内容。**信息没有丢失，只是从 context 移到了文件系统。**

Anthropic 也在平台层面推出了类似功能（Context Editing）：

> Context editing automatically clears stale tool calls and results from within the context window when approaching token limits.
> （Context editing 在接近 token 限制时，自动清除 context window 中过时的工具调用和结果。）

Claude Code 的做法更精巧——它的压缩请求使用**完全相同的 system prompt + tools + 对话前缀**，只在末尾追加 compaction 指令。这样压缩请求本身就能**复用父会话的 Cache**。连压缩都是 Cache-Safe 的。

### 第三层：Summarize（语义摘要）

当 Compact 也不够时，对整个对话轨迹做语义摘要。

Manus 的三层策略图示：

```
Context 使用量
│
│  ┌──────────────────────────────────────┐
│  │         Full (Raw)                   │ ← 最新的 tool 结果，完整保留
│  ├──────────────────────────────────────┤
│  │      Compact (引用替换)               │ ← 较早的结果，替换为文件路径
│  ├──────────────────────────────────────┤
│  │    Summarize (语义摘要)               │ ← Compact 达到极限后，整体摘要
│  └──────────────────────────────────────┘
│
└─→ 随着任务步骤增加，从 Raw 逐层降级
```

Lance Martin 的笔记中记录了 Manus 的关键设计：

> When compaction reaches diminishing returns, Manus applies summarization to the trajectory. Summaries are generated using full tool results and Manus uses a schema to define the summary fields.
> （当 compaction 达到边际递减时，Manus 对整个轨迹做摘要。摘要使用完整的工具结果生成，并用 schema 定义摘要字段。）

注意两个细节：
1. **摘要使用完整结果生成**——在摘要之前，先从文件系统恢复完整内容，确保摘要质量
2. **用 schema 定义摘要字段**——不是自由格式的"帮我总结一下"，而是结构化的摘要模板，保证关键信息不遗漏

OpenAI Codex 的做法类似，提供了专门的 `/responses/compact` API 端点，超过 `auto_compact_limit` 自动触发。

### 压缩的核心原则

Anthropic 在 context engineering 博客中给出了压缩的指导原则：

> The art of compaction lies in the selection of what to keep versus what to discard, as overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later.
> （压缩的艺术在于选择保留什么、丢弃什么。过于激进的压缩可能丢失微妙但关键的上下文——这些上下文的重要性可能到后来才显现。）

他们的建议是：先最大化 recall（确保捕获所有关键信息），再迭代优化 precision（去除冗余）。最安全的"轻触"压缩方式是清除 tool calls 和 results——因为一旦工具在很早之前被调用过，模型通常不需要再看到原始结果。

---

### 文件系统作为延展记忆

上面提到 Compact 层把内容"卸载"到文件系统。这不是一个临时方案，而是一种设计范式——**文件系统就是 Agent 的外部记忆。**

### Manus 的文件系统哲学

Manus 博客（纪一超）对此有一段非常精彩的论述：

> We treat the file system as the ultimate context in Manus: unlimited in size, persistent by nature, and directly operable by the agent itself. The model learns to write to and read from files on demand—using the file system not just as storage, but as structured, externalized memory.
> （我们把文件系统视为 Manus 的终极 context：大小无限、天然持久、Agent 可直接操作。模型学会按需读写文件——文件系统不只是存储，而是结构化的外部记忆。）

关键洞察：**压缩策略必须是可恢复的**。

> Our compression strategies are always designed to be restorable. For instance, the content of a web page can be dropped from the context as long as the URL is preserved.
> （我们的压缩策略始终被设计为可恢复的。例如，只要保留了 URL，网页内容就可以从 context 中移除。）

这与传统的"有损压缩"（比如直接截断或丢弃）有本质区别。Manus 的做法是：context 中只保留"指针"（文件路径、URL），完整数据存在文件系统中，需要时随时恢复。

### 文件系统 vs 向量数据库

一个有趣的共识是：三家头部 Agent 都**没有**在运行时使用向量数据库做检索，而是直接用文件系统 + 基础工具：

| Agent | 搜索方式 | 说明 |
|-------|----------|------|
| Claude Code | `glob` + `grep` | 文件名模式匹配 + 全文搜索 |
| Manus | `glob` + `grep` + 沙盒工具 | 类似 Claude Code |
| OpenAI Codex | `shell` 工具 | 通过 shell 命令探索文件系统 |

Lance Martin 在笔记中总结：

> Similar to Claude Code, Manus uses basic utilities (e.g., glob and grep) to search the filesystem without the need for indexing (e.g., vectorstores).
> （与 Claude Code 类似，Manus 使用基础工具搜索文件系统，不需要索引。）

为什么不用向量数据库？因为对于 Agent 场景，文件系统有几个天然优势：

1. **零索引延迟**：不需要 embedding + 入库的过程，写入即可用
2. **精确匹配**：`grep` 做精确文本搜索，不会有语义检索的"近似偏差"
3. **结构信号**：文件名、目录层级、时间戳本身就携带大量语义信息
4. **Agent 原生**：模型天然理解文件系统操作，不需要额外的"检索工具"

### Structured Note-taking：Agent 的"笔记本"

Anthropic 在博客中还介绍了另一种文件系统记忆模式——**结构化笔记**：

> Structured note-taking is a technique where the agent regularly writes notes persisted to memory outside of the context window. These notes get pulled back into the context window at later times.
> （结构化笔记是一种技术，Agent 定期将笔记写入 context window 之外的持久化存储，后续需要时再拉回 context window。）

他们举了 Claude 打宝可梦的例子：Agent 维护了精确的进度统计（"过去 1234 步一直在路线 1 训练宝可梦，皮卡丘已升了 8 级，目标 10 级"），还自主开发了地图、战斗策略笔记。Context reset 后，Agent 读取自己的笔记就能无缝继续。

这和 Manus 早期用 `todo.md` 做任务规划是同一个思路——只不过 Manus 后来发现约 1/3 的 actions 浪费在更新 todo 上，于是改用专门的 planner agent。

---

### CLAUDE.md 最佳实践与 Progressive Disclosure

### CLAUDE.md / AGENTS.md：Agent 的"说明书"

Claude Code 和 OpenAI Codex 都采用了一个简单但强大的设计：在项目根目录放一个配置文件（`CLAUDE.md` 或 `AGENTS.md`），作为 Agent 的项目级上下文。

这些文件会在会话开始时直接加载到 context 中（Claude Code 四层架构的 Layer 2），包含：

- 项目背景和架构概述
- 代码规范和命名约定
- 常用命令和工作流
- 需要特别注意的坑和约束

这是一种**预加载策略**——把最高频、最通用的项目知识提前注入，避免 Agent 每次都要从头探索。

### Progressive Disclosure：渐进式发现

但 CLAUDE.md 不能写成百科全书。把所有信息一股脑塞进去，又回到了 Context Rot 的问题。

Anthropic 在博客中提出了 **Progressive Disclosure（渐进式披露）** 原则：

> Progressive disclosure allows agents to incrementally discover relevant context through exploration. Each interaction yields context that informs the next decision: file sizes suggest complexity; naming conventions hint at purpose; timestamps can be a proxy for relevance.
> （渐进式披露让 Agent 通过探索逐步发现相关上下文。每次交互产生的上下文为下一次决策提供信息：文件大小暗示复杂度，命名约定暗示用途，时间戳可以作为相关性的代理。）

Anthropic 的 Skills 系统就是这个原则的典范实现：

> Skills are stored in the filesystem, not as bound tools, and Claude only needs a few simple function calls (Bash, file system) to progressively discover and use them.
> （Skills 存储在文件系统中，而不是作为绑定的工具。Claude 只需要几个简单的函数调用就能渐进式地发现和使用它们。）

类比一本设计良好的技术手册：先看目录（CLAUDE.md 提供概览），再翻到具体章节（glob/grep 搜索相关文件），最后读附录（按需加载完整内容）。Agent 每一步只加载当前需要的信息，而不是把整本手册塞进工作记忆。

### 最佳实践：CLAUDE.md 怎么写

结合实际经验，CLAUDE.md 的写法有几条原则：

**1. 分层组织，最重要的放最前面**

```markdown
# 项目概述（必读，~200 token）
一句话说清楚这是什么项目、用什么技术栈。

# 关键约束（必读，~300 token）
部署环境、安全要求、不可触碰的底线。

# 开发规范（按需参考，~500 token）
命名规范、分支策略、测试要求。

# 常见问题（按需参考，~500 token）
已知的坑、workaround、历史决策原因。
```

前两节每次都会被读取（占 context 较小），后两节 Agent 可以按需引用。

**2. 面向 Agent 写，不是面向人写**

CLAUDE.md 的读者是 LLM，不是你的同事。要：
- 用明确的指令式语言，而不是解释性叙述
- 给出具体的命令和路径，而不是"参考相关文档"
- 标注哪些文件/目录是关键的，哪些可以忽略

**3. 保持简洁，避免 Context Rot**

CLAUDE.md 本身也占 context 预算。经验法则：**控制在 1000-2000 token 以内**。如果内容太多，把详细信息拆分到子目录的 README 中，让 Agent 按需探索。

**4. 定期更新，但不要频繁变动**

CLAUDE.md 在 Claude Code 的四层架构中位于 Layer 2（项目级不变）。如果你每天都改 CLAUDE.md，它在 context 中的位置变化会影响后续所有对话历史的 Cache 命中。合理的更新频率是随项目里程碑变更，而不是随每次代码提交变更。

### Just-in-Time vs Pre-loading：混合策略

Anthropic 最终推荐的是**混合策略**：

> The most effective agents might employ a hybrid strategy, retrieving some data up front for speed, and pursuing further autonomous exploration at its discretion.
> （最有效的 Agent 可能采用混合策略：预先检索一些数据以提高速度，同时自主探索更多信息。）

Claude Code 就是这个混合模型的典范：
- **Pre-loading**：CLAUDE.md 在会话开始时直接注入
- **Just-in-Time**：`glob` 和 `grep` 让 Agent 运行时按需搜索文件系统

这两者的平衡取决于任务特性。代码库迁移这种需要大量背景知识的任务，可以多 pre-load；实时调试这种快速迭代的任务，更适合 just-in-time 探索。

---

### 本节小结：Context 管理的完整框架

把以上策略整合起来，可以得到一个三维框架（借用 Manus 的 Reduce / Isolate / Offload 体系）：

| 维度 | 策略 | 实现方式 | 代表方案 |
|------|------|----------|----------|
| **Reduce** | 压缩 context | Raw → Compact → Summarize 三层降级 | Manus compaction, Claude Code Cache-Safe compaction, Codex /responses/compact |
| **Offload** | 卸载到外部 | 文件系统作为延展记忆 + 结构化笔记 | Manus 文件系统记忆, Claude Code CLAUDE.md + glob/grep |
| **Isolate** | 隔离 context | 子代理各自独立 context window | Manus planner→executor, Claude Code sub-agent handoff |

贯穿三个维度的核心原则：

1. **Context 是有限资源**，有边际收益递减（Context Rot）
2. **压缩必须可恢复**——保留指针，不做有损丢弃
3. **渐进式披露**——先概览后详情，Agent 按需探索
4. **所有操作必须 Cache-Safe**——压缩、卸载、隔离都不能破坏前缀稳定性

---

> 1. **Context Rot 是物理现实**：128K context window 不是"能塞 128K 有效信息"。越长的 context，每个 token 分到的注意力越少。主动管理是必须的。
>
> 2. **三层压缩策略**：最新结果保留完整（Raw），较早结果替换为文件引用（Compact），极限情况做结构化摘要（Summarize）。关键是**可恢复**——信息不丢失，只是从 context 移到文件系统。
>
> 3. **文件系统是 Agent 的"第二大脑"**：三家头部 Agent 都用 glob/grep 搜索文件系统，而不是向量数据库。零索引延迟 + 精确匹配 + 结构信号，对 Agent 场景更实用。
>
> 4. **CLAUDE.md 写法有讲究**：分层组织、面向 Agent 写、控制在 1000-2000 token、不要频繁变更。它是四层缓存架构的 Layer 2，稳定性直接影响 Cache 命中率。
>
> 5. **混合策略最优**：Pre-loading（CLAUDE.md 预加载高频知识）+ Just-in-Time（运行时按需检索）。不要试图把所有信息都塞进 context，也不要让 Agent 从零开始探索。

---

## 范式转变：从 Prompt Engineering 到 Context Engineering

写到这里，我们已经从 KV Cache 的底层原理，一路走到了三家头部 Agent 的完整架构方案。

最后一节，我想拉到最高的视角，聊一个更本质的问题：**我们正在经历一场什么样的范式转变？这场变化中，什么会被淘汰，什么会留下来？**

---

### 从"写好一段话"到"管好一整个系统"

2023 年，我们谈论最多的是 Prompt Engineering——怎么写 system prompt、怎么用 few-shot、怎么设计 chain-of-thought。核心关注点是：**怎么写一段话，让模型表现更好。**

2025 年，行业的关注点已经明显转移。Andrej Karpathy 在 X 上的一条帖子被广泛引用：

> Context engineering is the delicate art and science of filling the context window with just the right information for the next step.
> （Context Engineering 是一门精妙的艺术与科学——为 Agent 的下一步行动，在 context window 中填入恰好正确的信息。）

Anthropic 在 *Effective Context Engineering for AI Agents* 中给出了更系统的定义：

> Prompt engineering refers to methods for writing and organizing LLM instructions for optimal outcomes. Context engineering refers to the set of strategies for curating and maintaining the optimal set of tokens during LLM inference, including all the other information that may land there outside of the prompts.
> （Prompt Engineering 关注如何编写和组织 LLM 指令以获得最优结果。Context Engineering 关注在 LLM 推理过程中，策展和维护最优 token 集合的策略——包括 prompt 之外的所有信息。）

区别在哪？

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|---------------------|
| 关注对象 | 一段 prompt 文本 | 整个 context state（prompt + tools + 历史 + 外部数据） |
| 时间范围 | 单次推理 | 跨多轮、长时间任务 |
| 核心问题 | "怎么措辞" | "什么信息、什么时候、以什么形式进入 context" |
| 优化目标 | 单次输出质量 | 系统级效率（成本、延迟、Cache 命中、长期一致性） |
| 工程复杂度 | 写好一段文本 | 设计一套动态信息管理架构 |

Anthropic 说得很准确——Prompt Engineering 是一个**离散的任务**（写好一段 prompt），而 Context Engineering 是一个**迭代的过程**（每次推理前都要决定 context 中放什么）。

这不是说 Prompt Engineering 不重要了。System Prompt 的质量仍然关键。但在 Agent 场景下，它只是 Context Engineering 的一个子集——你的 system prompt 写得再好，如果 context 管理不当（Cache 命中率低、context 腐烂、工具列表不稳定），Agent 的表现一样会很差。

---

### Reduce / Isolate / Offload：三维框架

纪一超（Manus CEO）在 Lance Martin 的 webinar 中，提出了 Context Engineering 的三维操作框架。这个框架和 Anthropic 博客中讨论的策略高度吻合，可以作为 Agent 架构师的通用思维模型：

### 维度一：Reduce（缩减）

**核心问题**：context 中有哪些 token 可以去掉或压缩，而不影响 Agent 的决策质量？

| 策略 | 做法 | 代表实现 |
|------|------|----------|
| Compact 降级 | 旧 tool 结果从完整内容替换为文件路径引用 | Manus full→compact 表示 |
| 语义摘要 | Compact 到极限后，对整个轨迹做结构化摘要 | Manus schema-based summary |
| Cache-Safe Compaction | 压缩请求复用父会话 prefix | Claude Code compaction |
| Tool Result Clearing | 清除深层历史中的 tool 调用原始结果 | Anthropic Context Editing API |
| Auto Compaction | 超过阈值自动压缩 | Codex /responses/compact |

**判断标准**：这个 token 对 Agent 的**下一步决策**还有影响吗？如果没有，要么去掉，要么降级为引用。

### 维度二：Isolate（隔离）

**核心问题**：哪些任务可以在独立的 context window 中完成，避免污染主 context？

| 策略 | 做法 | 代表实现 |
|------|------|----------|
| 子代理隔离 | 分配独立 context window 做子任务 | Manus planner→executor |
| Sub-Agent Handoff | Fork 子代理，共享 prefix，独立执行 | Claude Code Explore/Plan 子代理 |
| 模型隔离 | 不同任务用不同模型，但单会话不切换 | Manus 任务级路由 |
| 结果压缩回传 | 子代理返回 1-2K token 的压缩摘要 | Anthropic 多代理研究系统 |

Anthropic 在博客中描述了子代理架构的本质：

> Rather than one agent attempting to maintain state across an entire project, specialized sub-agents can handle focused tasks with clean context windows. Each subagent might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work.
> （不是让一个 Agent 维护整个项目的状态，而是让专门的子代理在干净的 context window 中处理聚焦任务。每个子代理可能大量探索（使用数万 token），但只返回精炼的摘要。）

**关键洞察**：Manus 对子代理的定位不是"模拟人类分工"（设计师、工程师、项目经理），而是**隔离 context**。Lance Martin 的笔记记录了纪一超的观点：

> Manus takes a pragmatic approach to multi-agent, avoiding anthropomorphized divisions of labor. While humans organize by role due to cognitive limitations, LLMs don't necessarily share these same constraints. The primary goal of sub-agents in Manus is to isolate context.
> （Manus 对多代理采取务实的方法，避免拟人化的分工。人类因为认知限制按角色分工，但 LLM 不一定有同样的限制。Manus 子代理的首要目标是隔离 context。）

### 维度三：Offload（卸载）

**核心问题**：哪些信息可以从 context 移到外部存储，需要时再按需加载？

| 策略 | 做法 | 代表实现 |
|------|------|----------|
| 文件系统记忆 | tool 结果写入文件，context 只保留路径 | Manus 沙盒文件系统 |
| 结构化笔记 | Agent 定期写笔记到外部，后续拉回 | Claude Pokémon、NOTES.md |
| Progressive Disclosure | 先加载概览，按需探索详情 | CLAUDE.md + glob/grep |
| CLI 工具替代绑定工具 | MCP 工具通过 Bash CLI 执行，不绑定到 tools 数组 | Manus MCP via CLI |
| Skills 文件化 | 能力描述存在文件系统，不作为 tools 注入 | Anthropic Skills |

Lance Martin 总结了 Manus 的 Offload 哲学：

> Rather than bloating the function calling layer, Manus offloads most actions to the sandbox layer. Manus can execute many utilities directly in the sandbox with its Bash tool and MCP tools are exposed through a CLI.
> （Manus 不是在函数调用层膨胀工具数量，而是把大部分 action 卸载到沙盒层。Manus 用 Bash 工具在沙盒中直接执行大量实用程序，MCP 工具也通过 CLI 暴露。）

这个设计非常聪明：tools 数组保持精简（< 20 个原子工具），复杂操作通过 Bash 在沙盒中完成。工具列表稳定（保 Cache），能力范围无限（沙盒中什么都能做）。

### 三维框架全景图

```
                    Context Engineering
                    ┌──────────────────┐
                    │   Agent Context  │
                    │   Window (有限)   │
                    └──────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │  Reduce   │   │  Isolate  │   │  Offload  │
    │  缩减     │   │  隔离     │   │  卸载     │
    ├───────────┤   ├───────────┤   ├───────────┤
    │ Compact   │   │ 子代理    │   │ 文件系统  │
    │ Summarize │   │ 模型隔离  │   │ 笔记系统  │
    │ Tool清理  │   │ 结果压缩  │   │ CLI执行   │
    │ Compaction│   │ 回传摘要  │   │ Skills    │
    └───────────┘   └───────────┘   └───────────┘

    让 context        让不同任务       让信息住在
    中的 token        在独立空间       context 外面
    变少/变小         中执行           需要时再拉进来
```

---

### The Bitter Lesson：什么会被淘汰，什么会留下来

Rich Sutton 在 2019 年写的 *The Bitter Lesson* 是 AI 领域最有影响力的短文之一。核心观点：**依赖人类知识和手工设计的方法，最终总会被利用更多计算资源的通用方法击败。**

这个教训对 Agent 开发者意味着什么？

Lance Martin 在 webinar 笔记中记录了纪一超的警告：

> Peak warned that the agent's harness can limit performance as models advance; this is exactly the challenge called out by the Bitter Lesson. We add structure to improve performance at a point in time, but this structure can limit performance as compute (models) grows.
> （Peak 警告说，Agent 的框架可能会在模型进步时限制性能——这正是 Bitter Lesson 指出的挑战。我们在某个时间点添加结构来提升性能，但这些结构可能在计算（模型）增长时反而成为瓶颈。）

Manus 自 2025 年 3 月发布以来已经**重构了五次**。每次模型能力提升，之前的某些 workaround 就变得不必要。

Boris Cherny（Claude Code 创建者）也提到 The Bitter Lesson 影响了他的设计决策——让 Claude Code 保持 unopinionated（不预设立场），以便更好地适应模型改进。

### 但 Cache 不是 workaround

这里要做一个关键区分：**不是所有的工程设计都会被模型进步淘汰。**

有些设计是在弥补模型的不足（比如复杂的 few-shot 模板、手工设计的推理链、格式纠错逻辑）。随着模型变强，这些 workaround 确实会变得不必要。

但 Prompt Cache 的架构约束不是 workaround。它源于 Transformer 推理的物理现实：

- **Prefill 阶段是 Compute Bound** → 重复计算是真实的算力浪费
- **前缀匹配是 KV Cache 的数学性质** → 不是某个实现的限制，而是注意力机制的本质
- **Agent 的 I/O 100:1 失衡** → 只要 Agent 还需要发送对话历史，这个比例就不会改变

即使模型的 context window 从 200K 扩展到 2M，即使推理速度再提升 10 倍：

- 你仍然不想为已经算过的 token 重新付费
- 你仍然不想让用户等待已经可以跳过的 Prefill
- Context Rot 仍然会在更长的 context 中发生

chaofa 在他的系列文章结尾总结得很好：

> Cache 是物理约束，不是工程 hack。只要 Prefill 还是 Compute Bound，Prompt Cache 就会继续是 Agent 架构的核心考量。

### 怎么判断你的设计是"持久的"还是"临时的"？

纪一超给了一个实用的检验方法（来自 Lance Martin 笔记）：

> Run agent evaluations across varying model strengths. If performance doesn't improve with stronger models, your harness may be hobbling the agent.
> （在不同强度的模型上跑 Agent 评估。如果换了更强的模型性能没有提升，你的框架可能在拖后腿。）

华盛顿大学 Hyung Won Chung（曾在 OpenAI/MSL 工作）在他的演讲中进一步强调：

> Add structures needed for the given level of compute and data available. Remove them later, because these shortcuts will bottleneck further improvement.
> （根据当前可用的计算和数据水平添加结构。后续要移除它们，因为这些捷径会成为进一步提升的瓶颈。）

用这个标准衡量，围绕 Prompt Cache 的设计决策属于**持久层**：

| 设计 | 持久/临时 | 原因 |
|------|-----------|------|
| 前缀稳定 + append-only | **持久** | 源于 KV Cache 的数学性质 |
| 工具定义不变，控制可选范围 | **持久** | 源于前缀匹配约束 |
| 确定性序列化 | **持久** | 源于 token 级匹配要求 |
| 三层压缩策略（Raw→Compact→Summarize） | **持久** | 源于 Context Rot 的物理现实 |
| 文件系统作为延展记忆 | **持久** | 源于 context window 的有限性 |
| 复杂的 few-shot 模板 | **临时** | 模型变强后不再需要 |
| 格式纠错和重试逻辑 | **临时** | 模型输出质量提升后冗余 |
| 手工设计的推理链 | **临时** | 模型推理能力提升后冗余 |
| todo.md 任务管理 | **临时** | Manus 已经用 planner agent 替代 |

---

### 给 Agent 架构师的建议

最后，结合全文内容，给出几条可操作的建议：

**1. 立即做：审计你的 Cache 命中率**

如果你在用 Claude API 或 OpenAI API 构建 Agent，检查你的 API 响应中的 cache 命中数据。如果 cached token 占比低于 70%，你大概率有上面六大杀手中的某几个问题。

**2. 本周做：重构 Prompt 布局**

把 System Prompt 和工具定义稳定化。时间戳、环境状态等动态信息移到 user message 末尾。确保 JSON 序列化使用 sort_keys。

**3. 本月做：设计压缩策略**

实现 Raw → Compact → Summarize 三层压缩。确保压缩操作是 Cache-Safe 的（复用父会话 prefix）。

**4. 持续做：用 Bitter Lesson 检验你的设计**

每次升级模型后，跑一遍 Agent 评估。如果新模型没有带来性能提升，检查你的框架是否在限制模型能力。大胆移除不再需要的 workaround。

**5. 始终记住：Context 是有限资源**

不要因为模型支持 200K 就把所有信息塞进去。始终追求**最小高信号 token 集合**。这是 Anthropic 反复强调的核心原则：

> Find the smallest possible set of high-signal tokens that maximize the likelihood of your desired outcome.
> （找到最小的高信号 token 集合，最大化期望结果的可能性。）

---

> 1. **从 Prompt Engineering 到 Context Engineering**，不是术语升级，是工程复杂度的跃迁。从"写好一段文本"变成"设计一套动态信息管理架构"。
>
> 2. **Reduce / Isolate / Offload 三维框架**是 Agent 架构师的通用思维模型。任何 context 管理问题都可以拆解到这三个维度去解决。
>
> 3. **The Bitter Lesson 的双面性**：一方面，模型进步会淘汰很多 workaround，不要过度设计；另一方面，围绕物理约束（Cache、Context Rot、有限 context window）的设计是持久的，值得投入。
>
> 4. **判断持久 vs 临时的标准**：换更强的模型后性能有没有提升？如果没有，你的框架可能在拖后腿。
>
> 5. **一句话总结全文**：Prompt Cache 是 LLM 推理引擎提供的能力，但围绕它做的架构设计——前缀稳定、append-only、工具不变、确定性序列化、三层压缩、文件系统记忆——这些是 Agent 工程师的核心手艺。这手艺不会过时，因为它适配的是计算的物理现实，而不是某个模型的临时局限。



---

**「Agent 工程师的 Prompt Caching」系列导航**

1. [KV Cache 原理：LLM 推理的底层机制](/posts/prompt-caching-kv-cache-fundamentals/)
2. [Prompt Cache：Agent 成本控制的核心约束](/posts/prompt-caching-core-constraints/)
3. [Cache 杀手与行业实战：从踩坑到最佳实践](/posts/prompt-caching-killers-and-industry/)
4. **Context Engineering：Agent 架构师的核心手艺**（本文）
