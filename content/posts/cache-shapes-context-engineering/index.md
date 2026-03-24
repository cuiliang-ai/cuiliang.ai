---
title: "当缓存成为架构：Prompt Caching 如何串联上下文工程的五个模式"
date: 2026-03-24
draft: false
summary: "前缀匹配——Prompt Caching 的一条刚性规则——如何系统性地塑造上下文工程的五个模式，成为 Agent 架构中看不见的骨架。"
description: "前缀匹配——Prompt Caching 的一条刚性规则——如何系统性地塑造上下文工程的五个模式，成为 Agent 架构中看不见的骨架。"
tags:
  - Context Engineering
  - Prompt Caching
  - Agent
  - Claude Code
categories:
  - AI Agent Engineering
ShowToc: true
TocOpen: true
---

Claude Code 的 Plan Mode 有一个反直觉的设计：当用户进入规划模式时，模型仍然可以看到所有工具，包括写文件、执行命令这些在规划阶段完全用不到的能力。

直觉上，规划阶段应该只保留只读工具。去掉写入类工具不仅能减少模型的选择负担，还能从机制上防止规划阶段的误操作。Claude Code 团队最初也是这么想的。但最终的实现恰恰相反：工具集始终不变，规划状态通过调用 `EnterPlanMode` 和 `ExitPlanMode` 这两个工具来切换。

为什么？这个问题的答案，牵出了一条贯穿整个 Agent 上下文工程的暗线。

## 五个模式：一张看似完整的地图

Aurimas Griciūnas 在 2026 年 3 月的综述中，将上下文工程的主流实践归纳为五个模式：

- **渐进式披露（Progressive Disclosure）**：信息分层加载——发现层（名称和描述）、激活层（完整指令）、执行层（脚本和参考材料），而非一次性灌入。Agent Skills 是标准实现。
- **压缩（Compression）**：对累积的行动历史做摘要，防止 system prompt 和早期上下文被挤出窗口。Claude Code 称之为 Compaction。
- **路由（Routing）**：查询进入上下文前先分类，导向正确的知识源或子 Agent，避免全量加载。
- **检索演进（Retrieval Evolution）**：从固定 RAG 管道演进为 Agent 自主控制的检索循环——模型自己决定搜索策略并迭代优化。
- **工具管理（Tool Management）**：控制哪些工具 schema 进入上下文。单个复杂 schema 可超 500 token，多个 MCP server 叠加后工具定义开销轻松过 50,000 token。

五个模式分别解决上下文窗口的不同维度：什么时候加载、加载多少、从哪里加载、加载什么能力。Aurimas 对每个模式给出了 tradeoffs 分析，框架清晰，分类合理。

但这张地图有一个盲区：它把五个模式画成了正交的维度，好像可以独立设计、独立优化。Claude Code 的工程实践讲述的是一个不同的故事。

## 一条硬约束：前缀匹配

要理解这条暗线，需要先理解 Prompt Caching（提示缓存）的工作原理。

LLM 推理时，输入 token 经过 transformer 的注意力层，生成 Key 和 Value 向量（合称 KV 缓存）。对于一次多轮对话，如果前后两次请求共享相同的开头部分，那么这段共享前缀对应的 KV 缓存可以直接复用，不需要重新计算。这就是 Prompt Caching 的本质：复用已经计算过的前缀，跳过重复的前向传播。

这里有一条刚性规则：**缓存命中要求精确的前缀匹配**。不是"大致相似"，不是"语义接近"，而是逐 token 完全一致。前缀中任何位置的任何改动，包括多一个空格、调换两个工具定义的顺序、在 system prompt 里更新一个时间戳，都会导致从改动位置开始的所有缓存失效。

这条规则不是某家厂商的实现细节，而是当前所有主流 LLM API 的共同约束。

OpenAI 的 Prompt Caching 是自动触发的：请求超过 1,024 token 后，系统基于前缀 hash 路由到可能持有缓存的机器，然后做前缀匹配。OpenAI API 文档（Prompt Caching 指南）明确要求静态内容放在 prompt 开头，动态内容放在末尾，工具定义和图片在请求间必须保持一致。OpenAI Cookbook 的 Prompt Caching 201 一文进一步披露了 Codex 团队的实践：system instructions、tool definitions、sandbox configuration 始终保持一致的内容和顺序，agent loop 只追加新消息而不修改已有内容，以保持前缀稳定。

Google Gemini 同样如此。Google AI for Developers 的 Context Caching 文档将缓存内容定义为"a prefix to the prompt"。2025 年 5 月，Google Developers Blog 宣布 Gemini 2.5 模型默认启用隐式缓存（Implicit Caching），机制同样是基于公共前缀匹配来判定缓存命中。显式缓存（Explicit Caching）允许开发者手动创建缓存对象并设置 TTL，但本质上仍是前缀复用。

Anthropic 的实现需要开发者在请求中显式标记 `cache_control` 断点来指定缓存位置，这给了更精细的控制，但底层约束不变：断点之前的内容必须逐 token 一致才能命中。

三家的实现细节不同（自动 vs 显式、TTL 策略、计费模型各异），但共享同一条硬规则。DigitalOcean 的 "Prompt Caching Explained" 一文在对比三家实现后将其总结为一句话："Exact prefix matching is non-negotiable."

现在可以回答开头的问题了。Claude Code 的 Plan Mode 为什么不换工具集？因为工具定义在缓存前缀里。换一个工具，从那个位置开始的所有 KV 缓存全部失效。对于一个已经积累了数万 token 对话历史的会话来说，这意味着下一次请求需要重新计算整个前缀，延迟和成本都会大幅上升。保留全部工具、用工具调用来切换状态，是在缓存约束下的理性选择。

这不只是 Plan Mode 一个功能的设计取舍。当你沿着这条线索往下看，会发现缓存的前缀匹配约束渗透进了上下文工程的每一个模式。

## 约束如何入侵每个模式

### 工具管理：永远不增删，只隐藏

Aurimas 的框架指出了工具管理的核心难题：每个工具 schema 占数百 token，连接多个 MCP server 后，工具定义本身就能吃掉 50,000+ token。他提出的应对方向是控制哪些工具进入上下文、优化描述质量。

Claude Code 面对同样的问题，给出的解法受到了缓存约束的直接限制。

Claude Code 可以加载数十个 MCP 工具。把所有完整 schema 放进每次请求显然太贵，但在会话中途移除工具会破坏缓存前缀。他们的方案是 `defer_loading`：不移除任何工具，而是把不常用的工具替换为轻量级的 stub（只保留工具名称，标记 `defer_loading: true`）。模型通过一个 `ToolSearch` 工具按需发现并加载完整 schema。这样做保证了缓存前缀的稳定：同样的 stub 始终以同样的顺序出现在每次请求中。

这个设计的关键不在于"按需加载"这个思路本身（渐进式披露早就提出了这个方向），而在于它选择了"stub 占位 + 搜索发现"而非"动态增删"。两种方式都能实现按需加载，但只有前者兼容前缀匹配约束。

Claude Code 团队在 Prompt Caching 那篇博客中用粗体强调了一条规则："Never Add or Remove Tools Mid-Session." 这条规则表面上是关于工具管理的，但驱动它的是缓存机制。

OpenAI 的工程实践也印证了这一点。Codex 的 agent loop 保持工具定义和顺序在请求间完全一致，把运行时配置变化（比如切换工作目录、变更审批模式）放进追加的消息里，而不是修改前缀中的工具参数。

### 渐进式披露：不改 system prompt，只追加消息

渐进式披露的核心主张是"按需加载信息"。在 Aurimas 的框架中，这主要描述的是 Agent Skills 的三层加载机制。但"加载到哪里"这个问题，框架没有深入讨论。

直觉上的做法是把激活的信息写进 system prompt。技能被触发了，就把完整指令插入 system prompt 的相应位置。这正是缓存约束不允许的：system prompt 是缓存前缀的核心组成部分，任何修改都会导致失效。

Claude Code 的做法是把需要更新的信息放进下一轮的 user message 里。他们使用 `<system-reminder>` 标签在用户消息或工具结果中注入更新内容（比如"现在是星期三"、"用户刚修改了这个文件"）。从模型的视角看，这些信息和 system prompt 里的指令效果类似，但它们出现在对话历史的末尾，不影响前缀。

这意味着渐进式披露在实践中有两种形态：一种是修改前缀来加载新信息（缓存不友好），一种是在对话流中追加新信息（缓存友好）。理论框架没有区分这两种形态，但在工程实现中，选择哪一种直接影响成本和延迟。

同一篇博客中还给出了一个通用建议："Consider if you can pass in this information via messages in the next turn instead." 这句话读起来像是一个可选的优化技巧，实际上在他们的系统中是一条强制约束。

同样的逻辑也解释了为什么 Claude Code 的 skill 激活不修改前缀。skill 的发现层信息（名称和描述）在会话开始时就存在于前缀中，保持不变。当某个 skill 被激活，其完整指令通过对话消息的形式进入上下文，而不是插入 system prompt。skill 执行阶段需要的脚本和参考文件，通过模型主动读取文件系统来获取，同样不修改前缀。

整个三层加载机制（发现 → 激活 → 执行）在缓存约束下被重新塑造了：第一层是前缀的一部分（静态），第二层和第三层是对话流的一部分（追加）。

### 压缩：不只是缩减历史，还要复用前缀

Aurimas 对压缩模式的描述集中在"用什么策略缩减历史"：保留最近 N 轮、滑动窗口 + 摘要混合、长期记忆存储。他引用了 Manus 的两个实践细节（保留最近工具调用的原始格式以维持模型的"节奏"、不压缩错误堆栈以避免重复犯错），这些都是有价值的经验。

但 Claude Code 的 Compaction 实现揭示了一个 Aurimas 完全没有涉及的维度：**压缩操作本身的缓存感知**。

Compaction 发生在上下文窗口接近用尽时。Claude Code 需要把当前对话发送给模型生成摘要，然后用这个摘要开启新会话。最简单的实现方式是：用一个独立的 API 调用、一个专用的 system prompt（比如"你是一个摘要生成器"）、不带任何工具定义，把整段对话发过去生成摘要。

问题在于，这个摘要请求的前缀和主对话的前缀完全不同：不同的 system prompt、没有工具定义、没有用户上下文。主对话积累的所有缓存在这个请求上完全无法复用。对于一个已经有 100k+ token 历史的对话来说，这次摘要生成需要全价处理所有输入 token。

Claude Code 的解法是：Compaction 请求使用和父会话完全相同的 system prompt、用户上下文、工具定义，并且把父会话的对话历史原样放在前面，只在最后追加一条 compaction 指令作为新的 user message。从 API 的视角看，这个请求和父会话的最后一次请求几乎一模一样，只是末尾多了一条消息。缓存前缀被完整复用，只有 compaction 指令本身是新增的 token。

这个设计带来一个实际的工程约束：必须在上下文窗口中预留一个"compaction buffer"，为 compaction 指令和摘要输出预留空间。不能等到窗口完全用尽才触发 compaction，否则没有空间放 compaction 指令。

这个案例说明了一件事：在缓存约束下，压缩不只是"怎么缩减历史"的问题，还是"怎么在缩减的同时不浪费已有缓存"的问题。一个不感知缓存的压缩实现，可能在压缩过程本身消耗的成本比它节省的还多。

### 路由：单 Agent 切换身份 vs 多 Agent 各管一域

路由模式在 Aurimas 的框架中有多种实现方式：LLM 分类路由、层级式 lead agent 分发、规则路由、混合路由。这些方式隐含了一个假设：查询会被导向不同的处理单元，每个单元有自己的上下文窗口。

Claude Code 选择了一条不同的路径。Aurimas 自己在文章中也观察到了这一点：Claude Code 不会为 PDF 处理和电子表格处理分别启动独立的子 Agent，而是用同一个 Agent 按需激活不同的 skill，在任务期间切换身份（指令集、约束条件、行为模式），任务完成后回到基础状态。

缓存约束至少部分解释了这个选择。多 Agent 路由意味着每个子 Agent 有自己的 system prompt 和工具集，也就是各自独立的缓存前缀。主 Agent 积累的缓存在子 Agent 那里无法复用。如果一个会话在多个 Agent 之间频繁切换，缓存命中率会显著下降。

单 Agent + skill 切换的模型在缓存层面更高效：system prompt 和工具定义始终不变（前缀稳定），身份切换通过对话消息完成（追加而非修改）。代价是单个 Agent 的上下文窗口需要承载所有领域的信息，这又回到了渐进式披露和压缩需要解决的问题。

这里需要一个限定：Claude Code 选择单 Agent 模型不完全是因为缓存约束。减少 Agent 间通信的开销、避免上下文在传递中丢失、降低编排复杂度，这些都是独立的理由。但缓存约束和这些理由指向了同一个方向，强化了这个架构选择。

Claude Code 并非完全不使用子 Agent。他们的 Explore agent 使用 Haiku 模型执行轻量探索任务，Guide agent 处理关于 Claude Code 自身的问题。但这些子 Agent 的使用模式是"主 Agent 准备好 handoff 消息，子 Agent 独立完成任务后返回结果"，而不是"在主 Agent 的上下文窗口中频繁切换"。子 Agent 有自己的缓存生命周期，不干扰主会话的缓存前缀。

### 检索：约束最间接，但仍然可见

在五个模式中，缓存约束对检索的影响最为间接。检索的核心问题是"从外部获取什么信息"，而不是"上下文窗口的结构如何组织"。但影响仍然存在。

第一个影响点是搜索工具的定义。无论用向量搜索还是 Grep，搜索工具的 schema 定义是缓存前缀的一部分。如果因为某些原因需要在会话中途更换搜索工具（比如从通用搜索切换到特定数据库的查询工具），同样会触发缓存失效。Claude Code 的做法和工具管理一致：所有可能用到的搜索能力在会话开始时就以 stub 形式存在，按需激活完整 schema。

第二个影响点是搜索结果的注入方式。检索到的内容必须追加到对话历史的末尾（作为工具调用的结果），而不能插入到对话中间去"纠正"之前的上下文。这是前缀匹配的自然推论：对话历史是前缀的一部分，在中间插入内容会导致插入点之后的所有缓存失效。

第三个影响点来自 Claude Code 从 RAG 到 Grep 的演进。早期的 RAG 方案需要预先索引代码库，检索结果由系统注入到上下文中，模型是被动接收者。后来切换到 Grep 工具后，模型自己决定搜索什么、读取哪些文件。这个转变有一个缓存层面的附带好处：模型主动搜索产生的是工具调用和工具结果（追加到对话末尾），而系统预注入的 RAG 结果可能需要修改 system prompt 或在对话中间插入内容。前者对缓存更友好。

这不是说缓存约束是 Claude Code 从 RAG 转向 Grep 的主要原因。Claude Code 团队明确表示，主要原因是 RAG 需要索引和配置、在不同环境下容易出问题，而且更根本的问题是"Claude was given this context instead of finding the context itself"。但缓存友好性是这个转变的一个附加收益。

## 框架没覆盖的问题：信息何时卸载

渐进式披露解决了"信息何时加载"的问题，但它的逆问题同样重要：已经加载到上下文中的信息，何时应该被移除？

Aurimas 在讨论渐进式披露的 tradeoffs 时指出了这个问题："The key unsolved question: when does an activated skill get deactivated? Without explicit pruning logic, multiple activated skills destroy the token advantage over time." 这个观察很准确，但他没有给出解法，只是把它列为了一个开放问题。

Claude Code 三篇博客同样没有回答这个问题。

缓存约束让这个问题更难处理。已经通过对话消息进入上下文的 skill 指令，是对话历史的一部分，也就是缓存前缀的一部分。要移除它们，有两个选择：从对话历史中间删除这些消息（直接破坏前缀），或者等到 Compaction 时把它们压缩掉（被动移除，但时间不可控）。

第一个选择在缓存约束下不可行。第二个选择意味着在 Compaction 触发之前，所有曾经激活过的 skill 内容都留在上下文里。如果一个会话在 Compaction 之前激活了五六个 skill，每个 skill 的完整指令有数千 token，渐进式披露在发现层节省的 token 就被激活层的累积吃掉了。

一个可能的方向是在 Compaction 的摘要策略中加入 skill 感知：识别哪些 skill 的指令已经执行完毕，在摘要时只保留结果而丢弃指令。但这需要 Compaction 理解 skill 的生命周期，增加了压缩逻辑的复杂度。另一个方向是在 skill 设计层面控制指令的 token 量，让激活成本足够低，使得累积问题不至于太严重。Claude Code 的 skill 体系中，17 个内置 skill 在发现层总共约 1,700 token，Aurimas 引用的数据显示单个 skill 激活后的完整指令在 275 到 8,000 token 之间。如果控制在低端，累积压力可控；如果多个高端 skill 同时激活，问题就会显现。

这不是一个已经被解决的问题，更像是当前 Agent 上下文工程中一个真实的设计空白。

## 工程决策速查

前文的分析散布在五个模式中。下表把每个模式的核心决策压缩为一行，方便在设计 Agent 系统时直接对照：

| 模式 | 缓存不友好 | 缓存友好 | 关键机制 |
|------|-------------|------------|---------|
| 工具管理 | 会话中途增删工具定义 | stub 占位 + ToolSearch 按需加载 | 工具列表是前缀的一部分，增删即失效 |
| 渐进式披露 | 将激活信息插入 system prompt | 通过 user message / 工具结果追加 | system prompt 是前缀核心，修改即失效 |
| 压缩 | 用独立 system prompt 发摘要请求 | 复用父会话前缀，只追加 compaction 指令 | 前缀不同 = 缓存全部浪费 |
| 路由 | 频繁切换子 Agent（各有独立前缀） | 单 Agent + skill 切换身份 | 每个子 Agent 的缓存独立，无法复用 |
| 检索 | 系统预注入 RAG 结果到 system prompt | 模型主动搜索，结果作为工具返回值追加 | 工具调用结果追加在末尾，不破坏前缀 |

**通用原则：前缀只写一次，此后只追加、不修改。**

## 从缓存到更一般的规律

回顾这条线索：Plan Mode 不换工具集、Tool Search 用 stub 占位、system-reminder 追加而非修改、Compaction 复用父前缀、单 Agent 优先于多 Agent 路由。这些设计决策分布在上下文工程的不同模式里，但它们被同一个底层约束驱动。

这种"基础设施约束反向塑造应用架构"的现象，在 AI 工程中不止出现一次。

速率限制塑造了批处理和队列管理的策略。Token 计费粒度影响了 prompt 的长度和信息密度的取舍。多模态输入的序列化顺序（图片在前还是文字在前）影响了模型的注意力分配和响应质量。这些都是基础设施层的约束向上传导到应用设计的例子。

Prompt Caching 的前缀匹配约束之所以特别值得关注，是因为它的影响面特别广。它不只影响某个具体功能的实现方式，而是系统性地塑造了上下文窗口内所有内容的组织方式：什么放前面、什么放后面、什么可以改、什么不能动。这些决策叠加起来，就构成了一个 Agent 系统的骨架。

Aurimas 的五模式框架是一张有用的地图。它帮助工程师理解上下文工程需要解决哪些维度的问题。但如果只按地图上的分区逐个实现，可能会发现各个模式之间存在意料之外的耦合。因为地图上没有标注的那条约束线，在地形中真实存在。

在搭建 Agent 系统时，在选择具体的模式和实现方案之前，先搞清楚你的推理基础设施有哪些硬约束：缓存机制是什么、token 预算怎么分配、速率限制如何影响并发。这些约束会帮你过滤掉很多理论上可行但实践中不经济的方案，把设计空间缩小到一个可操作的范围。

找到约束，围绕约束设计。这个方法论比任何具体的模式分类都更耐久。因为模式会随着模型能力和 API 设计的迭代而变化，但"先识别约束再做设计"这个习惯，在约束本身更换之后仍然适用。

---

**参考来源**

1. Thariq, "[Lessons from Building Claude Code: Prompt Caching Is Everything](https://x.com/trq212/status/2024574133011673516)," X, 2026
2. Thariq, "[Lessons from Building Claude Code: Seeing like an Agent](https://x.com/trq212/status/2027463795355095314)," X, 2026
3. Thariq, "[Lessons from Building Claude Code: How We Use Skills](https://x.com/trq212/status/2033949937936085378)," X, 2026
4. Aurimas Griciūnas, "[State of Context Engineering in 2026](https://www.newsletter.swirlai.com/p/state-of-context-engineering-in-2026)," SwirlAI Newsletter, March 2026
5. OpenAI, "[Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)," OpenAI API Documentation
6. OpenAI, "[Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201)," OpenAI Cookbook
7. Google, "[Context caching](https://ai.google.dev/gemini-api/docs/caching)," Gemini API Documentation
8. Google Developers Blog, "[Gemini 2.5 Models now support implicit caching](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/)," May 2025
9. DigitalOcean, "[Prompt Caching Explained](https://www.digitalocean.com/community/tutorials/prompt-caching-explained)," December 2025
