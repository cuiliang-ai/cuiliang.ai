---
title: "Harness Engineering：Agent 工程的第三次范式跃迁"
date: 2026-03-09
draft: false
summary: "从 Prompt Engineering 到 Context Engineering，再到 Harness Engineering——Agent 时代的工程方法论正在经历第三次范式跃迁。这篇文章帮你搞清楚它是什么、为什么突然火了、以及怎么上手。"
description: "从 Prompt Engineering 到 Context Engineering，再到 Harness Engineering——Agent 时代的工程方法论正在经历第三次范式跃迁。这篇文章帮你搞清楚它是什么、为什么突然火了、以及怎么上手。"
tags:
  - Harness Engineering
  - AI Agent
  - Context Engineering
  - Code Agent
  - Software Engineering
categories:
  - AI Agent Engineering
ShowToc: true
TocOpen: true
---

> 从 Prompt Engineering 到 Context Engineering，再到 Harness Engineering。Agent 时代的工程方法论正在经历第三次范式跃迁。这篇文章帮你搞清楚它是什么、为什么突然火了、以及怎么上手。

---

## 三组出人意料的数据

2026 年 2 月，LangChain 做了一个实验：他们的编码 Agent 在 Terminal Bench 2.0 排行榜上排第 30 名。然后他们没有换模型，没有加数据，只是优化了 Agent 外围的约束系统、反馈回路和验证机制——也就是后来被称为 harness 的东西。结果排名从第 30 跳到了第 5。

同期，安全研究者 Can Bölük 在[个人博客](https://blog.can.ac/2026/02/12/the-harness-problem/)上发表了一个更极端的实验：同一个模型（Grok Code Fast 1），仅改变编辑工具的格式（从 patch 换成他设计的 hashline），表现从 6.7% 飙到 68.3%——10 倍差距，模型一行没动。

Vercel 的经验方向一致但策略相反：[削减了 80% 的 Agent 工具](https://www.philschmid.de/agent-harness-2026)，换来的是更少的步骤、更少的 token 消耗和更快的响应。

这三组数据指向同一个结论：**在 Agent 系统中，模型不是瓶颈，模型之外的一切才是。**

这个"模型之外的一切"，现在有了一个名字——**Harness Engineering**。

---

## 从一篇博客到行业共识：两周的加速度

Harness Engineering 不是凭空蹦出来的。回看时间线，这个概念的酝酿和引爆有一条清晰的脉络：

**2026 年 1 月 5 日**，Hugging Face AI 总监 [Phil Schmid](https://www.philschmid.de/agent-harness-2026) 发表《The importance of Agent Harness in 2026》，提出了一组精准的类比：模型是 CPU，Harness 是操作系统，Agent 是应用程序。这是 "harness" 一词在 Agent 语境下较早的系统性论述。

**2026 年 2 月 5 日**，关键节点来了。HashiCorp 联合创始人 **Mitchell Hashimoto**（Terraform 和 Vagrant 的创造者）在个人博客发表[《My AI Adoption Journey》](https://mitchellh.com/writing/my-ai-adoption-journey)，描述了 AI 采用的六个阶段。其中第五阶段被正式命名为 **"Engineer the Harness"**——每次 Agent 犯错，你不是去修 Agent 的输出，而是花时间去改善环境，确保 Agent 不会再犯同样的错。

他描述的六个阶段值得展开看，因为这可能是目前对 AI 工程化采用路径最清晰的一份个人叙事：

| 阶段 | 名称 | 核心行为 |
|:---|:---|:---|
| Stage 1 | 丢掉聊天机器人 | 从 ChatGPT 网页界面转向真正的 Agent（能读文件、执行程序、自主循环） |
| Stage 2 | 复现自己的工作 | 每项任务做两遍：先手动，再用 Agent，建立能力边界直觉 |
| Stage 3 | 下班前的 Agent | 利用每天最后 30 分钟启动 Agent 做深度调研、并行探索 |
| Stage 4 | 外包"稳赢"任务 | 能预测 Agent 胜任的任务就交给它后台跑 |
| **Stage 5** | **Engineer the Harness** | **每当 Agent 犯错，就工程化地修复系统，确保不再犯同样的错** |
| Stage 6 | 始终有 Agent 在运行 | 持续问自己"现在有什么可以让 Agent 做的？" |

**2026 年 2 月 11 日**，OpenAI 发表了 [**"Harness engineering: leveraging Codex in an agent-first world"**](https://openai.com/index/harness-engineering/)。这篇文章以一个极端实验为载体——3 人团队从空仓库出发，5 个月用 Codex Agent 写了约 100 万行产品代码，零人类手写——提供了迄今最详尽的 Harness Engineering 工程实践记录。

紧随其后，[Martin Fowler 网站](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)（Birgitta Böckeler 撰写，2026-02-17）和 [LangChain 博客](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)（2026-02-17）密集跟进。**Harness Engineering 在两周内完成了从个人博客术语到行业级工程概念的跃迁。**

---

## Harness Engineering 到底是什么

### 一个比喻

理解 Harness Engineering 最直觉的方式是想象你在骑马：

- **Prompt Engineering** 是你对马发出的语音指令："向右转"、"加速"。你在优化指令本身的表述。
- **Context Engineering** 是你展示给马看的地图和路标。你在优化马做决策时能看到的所有信息。
- **Harness Engineering** 是缰绳、马鞍、围栏和道路维护。你在设计整套物理基础设施，确保马在正确的道路上跑，跑偏了有围栏挡回来，跑累了有反馈告诉你。

三者是嵌套关系，每一层包含前一层并向外扩展：

```
Prompt Engineering  ⊂  Context Engineering  ⊂  Harness Engineering
  (优化指令)              (优化输入)              (优化整个系统)
```

Phil Schmid 给了另一个精准的[类比](https://www.philschmid.de/agent-harness-2026)：**模型是 CPU，Harness 是操作系统，Agent 是应用程序。** 无论 CPU 多强大，操作系统糟糕的话性能依然低下。

用一句话区分：**Context Engineering 问的是"给 Agent 看什么"，Harness Engineering 问的是"系统如何防止、测量和修复 Agent 的行为"。**

### 要解决的核心问题

Harness Engineering 要解决的问题很具体：**当 Agent 从 demo 走向生产，单靠优化 prompt 和 context 已经不够了。** 你需要回答一系列 prompt 层面根本无法触达的问题：

- Agent 犯了错，怎么保证它不会再犯同样的错？
- Agent 生成的代码/内容在积累技术债务，谁来清理？
- 多个 Agent 并行工作时，怎么保持架构一致性？
- Agent 运行了两个小时，怎么知道它有没有在兜圈子？
- 团队的知识散落在 Slack、Google Docs、口头共识里，Agent 怎么访问？

这些问题的答案不在 prompt 里，不在 context window 里，而在 Agent 外围的约束系统、反馈回路、知识管理和可观测性设计里。**这就是 harness。**

### 如何诊断你的问题在哪一层？

这是一个实用的判断框架：

- **Context 层信号**：单次输出偏离目标；必要信息未被引用；工具定义过于简略
- **Harness 层信号**：单次输出看起来没问题，但重复使用时质量参差不齐；架构一致性逐渐退化；前一个任务的修复在后续任务中被忽略

如果是后者，仅改进 prompt 或 CLAUDE.md 是不够的——你需要 Hooks、标准化 Commands 或 CI 质量门禁。

> **本节小结**：Harness Engineering 是 Context Engineering 的超集。Context 管的是"单次推理的输入质量"，Harness 管的是"跨数千次推理的系统级质量"。当你发现问题不在某一次对话里，而在系统的长期行为中，你就进入了 Harness Engineering 的领地。

---

## 为什么是 2026 年的焦点

Harness Engineering 在 2026 年 2 月集中爆发不是偶然，背后有三个结构性原因。

### 原因一：量化证据积累到了引爆点

除了开篇提到的三组实验，更多独立数据在持续验证同一个结论——harness 设计对 Agent 性能的影响超过模型本身：

| 实验方 | 做了什么 | 结果 |
|:---|:---|:---|
| Can Bölük（扩展数据） | 测试 16 个模型的 patch 失败率 | Grok 4 patch 失败率高达 50.7%，换 hashline 后输出 token 下降 61% |
| Aider Benchmark | 仅改变编辑格式（非模型） | GPT-4 Turbo 从 26% 跳到 59%，格式 > 模型代差 |
| LangChain（详细拆解） | 基线 vs 全套 harness 优化 | 52.8% → 66.5%，其中全程 xhigh 推理反而只有 53.9% |
| [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | 多次重构 Agent 框架 | 每次都是因为发现了更好的 context 塑造方式 |

LangChain 那行数据尤其值得注意——全程最高推理等级反而不如 harness 优化，具体拆解见后文"LangChain 实战拆解"一节。当这样的反直觉证据从多个独立团队同时涌现，行业共识就会突然形成。

### 原因二：OpenAI 提供了完整的工程范本

之前大家知道 harness 重要，但没有一个足够详尽的工程案例说清楚"具体怎么做"。OpenAI 那篇文章填补了这个空白——不是理论文章，而是一份工程操作手册：

1. **设计环境而非写代码**：Agent 遇到困难时追问"环境里缺什么"而不是让 Agent 更努力
2. **仓库即唯一真相来源**：所有知识推入 Git，AGENTS.md 作索引，渐进式披露
3. **机械式架构约束**：linter 的报错信息本身就是修复指南——工具在纠正 Agent 的同时也在教 Agent
4. **可观测性接入**：让 Agent 能捕获 DOM 快照、查询日志和指标，把目标变成可测量的
5. **对抗熵增**：自动化"垃圾回收 Agent"持续扫描偏离黄金原则的代码

有了这份范本，从业者第一次能照着做，而不只是"我知道应该这样但不知道怎么落地"。

### 原因三：模型趋同，竞争焦点转移

2025 年下半年，前沿模型之间的能力差距在快速缩小。当模型不再是差异化的核心变量，竞争焦点自然转移到模型之外的系统设计——也就是 harness。多方实验数据已经表明：基础设施配置对编码 benchmark 的影响，有时超过排行榜上顶尖模型之间的差距。

**模型是大家都能买到的 GPU，harness 是你自己写的操作系统。前者趋同，后者才是护城河。**

> **本节小结**：三个力量汇聚——量化证据打破了"模型至上"的信仰，OpenAI 的范本解决了"怎么做"的问题，模型趋同让"做不做"变成了竞争生死线。2026 年 2 月不是 Harness Engineering 被发明的时刻，而是行业集体承认它的时刻。

---

## LangChain 的实战拆解：三个中间件的设计

LangChain 的 Terminal Bench 实验值得单独展开，因为他们不仅给出了数据，还公开了具体的 harness 组件设计，可操作性很强。

### 三个中间件

**PreCompletionChecklistMiddleware（完成前检查清单）**

在 Agent 准备退出任务时拦截它，注入一条提醒：必须对照任务规格做一次验证。这强制了一个"构建→验证"循环，防止 Agent 在"代码看起来没问题"后就停下来。

**LoopDetectionMiddleware（循环检测）**

通过 tool call hooks 追踪每个文件的编辑次数。当对同一文件编辑超过 N 次时，注入上下文建议 Agent 重新考虑方案。他们在 trace 中观察到某些情况下，Agent 会对同一个错误方案反复微调 10+ 次——这个中间件就是用来打破这种"末日循环"的。

**LocalContextMiddleware（本地上下文）**

Agent 启动时自动映射当前工作目录和周边目录，发现可用工具（如 Python 安装路径）。减少 Agent 在环境导航上浪费的时间和 token。

### "推理三明治"策略

不是全程使用最高推理等级，而是采用 **xhigh → high → xhigh** 的三段式：

| 阶段 | 推理等级 | 用途 |
|:---|:---|:---|
| 初始规划 | xhigh | 高质量的方案设计 |
| 实现阶段 | high | 快速执行，避免超时 |
| 最终验证 | xhigh | 严格检查正确性 |

### 量化结果

| 配置 | 得分 | 备注 |
|:---|:---|:---|
| 基线（默认 Harness） | 52.8% | — |
| 全程 xhigh 推理 | 53.9% | 因超时反而效果不佳 |
| 全程 high 推理 | 63.6% | — |
| **完整 Harness 优化** | **66.5%** | **三明治策略 + 三个中间件** |

注意看：全程 xhigh 推理（53.9%）甚至不如全程 high 推理（63.6%）——因为过度思考导致超时。Harness 层面的优化（66.5%）比单纯提高推理等级的效果大得多。**这再次验证了"系统设计 > 模型能力"的核心论点。**

> **本节小结**：LangChain 的三个中间件分别解决了三个常见的 Agent 失败模式：不验证就收工、死循环、环境迷路。它们都不复杂，但都命中了真实痛点。这是 Harness Engineering 最有说服力的地方——不是大架构，而是对失败模式的工程化应对。

---

## 映射到 Claude Code：你手边的 Harness 组件

如果你正在用 Claude Code，好消息是你已经有了一套现成的 harness 工具箱。把理论映射到具体组件上：

| Claude Code 组件 | Harness 角色 | 所属层 |
|:---|:---|:---|
| CLAUDE.md | 仓库知识聚合、架构约束声明 | Context（结构化） |
| Commands | 可复现的常规任务执行 | Harness（工作流约束） |
| Hooks | 自动化事件触发处理 | Harness（反馈循环） |
| Skills | 最佳实践注入 | Context（结构化） |
| MCP Servers | 外部工具/数据连接 | Context 或反馈循环 |
| Permissions | 自动批准范围定义 | Harness（架构约束） |

注意这张表揭示了一个关键区分：**CLAUDE.md 和 Skills 属于 Context 层**（它们优化的是单次推理的输入），而 **Commands、Hooks、Permissions 属于 Harness 层**（它们约束的是系统的长期行为）。很多人把所有东西都塞进 CLAUDE.md 里试图解决问题，但如果你的问题在 Harness 层，再大的 CLAUDE.md 也不够。

> **本节小结**：不需要从零搭建 harness。Claude Code 的 Commands、Hooks、Permissions 就是现成的 harness 组件。关键是识别你的问题在哪一层，然后用对工具。

---

## 五个核心组件：Harness 的完整拼图

Harness 不是一个单一的东西，它是五个组件的组合。下表给出全貌，具体怎么落地在后面的"成熟度阶梯"中展开。

| 组件 | 解决的问题 | 一句话 | 落地层级 |
|:---|:---|:---|:---|
| 知识管理 | Agent 不了解项目 | 把团队知识变成 Agent 可发现的结构化文档 | L1 指令层 |
| 约束系统 | 好的指令被忽略，坏模式被复制 | 用机械化规则替代口头约定，报错即教学 | L2 约束层 |
| 反馈回路 | Agent 不知道自己做得对不对 | 让目标可测量，让 Agent 能自我验证 | L2-L3 |
| 熵管理 | 技术债务以 Agent 的速度积累 | 自动化"垃圾回收"，持续对抗架构退化 | L3 工作流层 |
| 状态与记忆 | 跨会话、跨 context window 的连续性 | 结构化进度文件 + 版本化存储 | L4 委托层 |

---

## 批判性视角：冷水时间

任何概念在爆发期都值得泼一盆冷水。以下几个问题是真实存在的：

### "做得对"不等于"做了对的事"

Martin Fowler 网站的 Birgitta Böckeler 提出了最有力的批评：OpenAI 的实践**完全聚焦于内部代码质量和可维护性**，却**忽略了对功能和行为的验证**。Harness 保证了 Agent 写出的代码是"好代码"，但谁来保证它做了正确的事情？这是一个显著的空白。

### 过度工程的陷阱

Manus 团队[多次重构 Agent 框架](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)，每次都是因为发现了更好的 context 塑造方式——Phil Schmid 将其总结为"移除僵化假设"。这暗示了一个反直觉的规律：**好的 harness 应该越来越简单，而非越来越复杂。** 如果你的 harness 在持续膨胀，可能是过度工程的信号。Rich Sutton 的 Bitter Lesson 在这里同样适用——简单的、可扩展的方案通常胜过精巧的、复杂的方案。

### 信源的利益冲突

OpenAI 有商业动机让你相信"AI 能写所有代码"——他们卖的就是 Codex。LangChain 卖的是 Agent 框架，harness 越重要他们的产品价值越高。读这些文章时需要意识到信源的利益结构。

### 遗留代码的现实

为新项目从零开始设计 harness 相对容易，但大多数团队面对的是已有的、非标准化的代码库。Böckeler 的警告很实在：往遗留代码库上套 harness，就像在遗留代码上跑静态分析——"你会被告警淹没"。改造旧项目的 ROI 可能远低于预期。

### 术语本身的稳定性

"Harness Engineering" 这个词 2026 年 2 月才结晶，不同作者的定义仍有差异。它能不能作为一个独立的工程学科立住，还是会被吸收进更广义的 "Agent Engineering"？时间会给答案。

> **本节小结**：Harness Engineering 指向的问题域是真实的，但概念本身还在早期。保持对核心洞察的信心（系统设计 > 模型能力），同时对具体实践保持批判性审视。

---

## 主要玩家的立场速览

不同公司对 Harness Engineering 的态度和实践：

| 公司 | 立场 / 实践 |
|:---|:---|
| **OpenAI** | 概念的工程实践标杆。3 人团队 + Codex 的[百万行实验](https://openai.com/index/harness-engineering/) |
| **Anthropic** | 不用 "Harness Engineering" 术语，但在 Agent 工具链设计（Claude Code 的 Hooks / Skills / Subagents）上做了最扎实的工程落地 |
| **LangChain** | 最系统化的第三方验证。公开了[中间件设计和量化数据](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) |
| **Manus** | [多次重构 Agent 框架](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)，实战经验证明"少即是多" |

---

## 落地框架：Harness 成熟度阶梯

理论讲完了，接下来回答最实际的问题：**我该从哪里开始？**

下面是一个分层递进的采用模型。每一层解决一类问题，向上叠加。你不需要一步到位——找到自己当前所在的层级，往上走一层就好。以 Claude Code 为主线，但思路适用于任何 Agent 工具链。

| 层级 | 名称 | 一句话 | 你在这层的信号 |
|:---|:---|:---|:---|
| L0 | 裸用 | 每次对话从零开始 | Agent 像每天换一个新实习生 |
| L1 | 指令层 | 把项目知识写下来 | 单次输出稳了，但跨任务还是乱 |
| L2 | 约束层 | 让机器替你执法 | Agent 反复犯同一类错 |
| L3 | 工作流层 | 把重复动作标准化 | 你在反复下达同一套指令序列 |
| L4 | 委托层 | 多 Agent 分工协作 | 单 Agent 上下文不够用了 |
| L5 | 治理层 | 权限、审计、沙箱 | 从个人工具变成团队基础设施 |

### L0 → L1：从裸用到指令层

**诊断信号**：你发现自己在每个新会话里重复说同样的话——"这个项目用 Hugo"、"文章放在 content/posts/ 下"、"front matter 用 YAML 格式"。Agent 每次都像新人报到，你是那个永远在做 onboarding 的 leader。

**最小可行实践**：写一个 CLAUDE.md（或 AGENTS.md），控制在 200 行以内。不是写小说，而是写给一个聪明但对你项目一无所知的工程师看的速查手册。核心内容：

- 项目是什么、技术栈是什么
- 常用命令（构建、测试、部署）
- 代码组织结构和命名规范
- 几条最重要的架构约束（"绝对不要做 X"比"尽量做 Y"有效得多）

OpenAI Codex 团队的 AGENTS.md 是一个好参考——它不是抽象的指导原则，而是极其具体的编码规范：crate 命名规则、clippy 配置、snapshot 测试策略、API 命名约定。**具体到 Agent 可以机械执行的程度，才算合格。**

**写法上的关键细节**：

- 用 Markdown 标题和列表，不要写长段落——Agent 解析结构化文本比自然语言段落更可靠
- "用 2 空格缩进"比"保持代码格式整洁"有效 10 倍
- 用 `IMPORTANT`、`YOU MUST`、`NEVER` 标记关键规则——权重信号真的有用
- 像对待代码一样对待它：出了问题就改，定期 prune，别让它膨胀成没人读的文档

**常见陷阱**：**塞太多。** CLAUDE.md 超过 200 行后，Agent 对后面内容的遵从度会明显下降。如果你发现规则越加越多但 Agent 表现没有提升，大概率是信息过载。这时候需要的不是更多指令，而是下一层——机械化约束。

### L1 → L2：从指令层到约束层

**诊断信号**：CLAUDE.md 里写了"所有组件必须放在 components/ 目录下"，但 Agent 偶尔还是把组件散落在别的地方。单次看没问题，跑十次任务发现三次违规。**指令是建议，约束才是法律。**

**最小可行实践**：用 Hooks 把最重要的规则从"写在文档里"变成"跑在流程中"。Claude Code 的 Hooks 系统提供了几个关键的生命周期钩点：

- **PreToolUse**：Agent 调用工具前拦截。比如在写文件前检查路径是否符合项目结构
- **PostToolUse**：Agent 调用工具后触发。比如每次编辑代码后自动跑 formatter
- **Stop**：Agent 准备结束任务时拦截。强制做一次完成前检查——"你跑测试了吗？"

一个具体的例子：LangChain 的 PreCompletionChecklistMiddleware 就是 Stop 钩子的典型应用——在 Agent 认为任务完成时拦截它，注入一条"对照规格做验证"的提醒。就这么一个简单的钩子，对他们的 benchmark 分数贡献显著。

**报错信息的设计是这一层最精巧的部分。** 不要只说"违规了"，要说"违规了，应该这样修"。OpenAI 团队的实践是让 linter 的报错信息本身就是修复指南——这样约束在纠正 Agent 的同时也在教 Agent。一条好的报错信息长这样：

```
ERROR: Component file found outside components/ directory.
FIX: Move this file to src/components/ and update imports.
```

而不是：

```
ERROR: Linting failed.
```

**常见陷阱**：**约束太多太细。** Manus 团队多次重构 Agent 框架的经验是每次都在简化而非增加复杂度。如果你发现自己在写第 20 条 Hook 规则，停下来问问：是不是应该简化架构本身，而不是用约束去弥补架构的复杂度？

### L2 → L3：从约束层到工作流层

**诊断信号**：你发现自己在反复给 Agent 下达同一套指令序列——"先读 X 文件，然后按 Y 模板生成，最后跑 Z 检查"。每次都手动编排，每次都怕漏一步。

**最小可行实践**：把重复的指令序列封装成 Skills 或 Custom Commands。在 Claude Code 中，Skills 是存放在 `.claude/skills/` 下的 Markdown 文件，可以通过斜杠命令触发，也可以被 Agent 自动识别和调用。

一个 Skill 本质上是**一段标准化的工作流程描述**——输入是什么、步骤是什么、输出应该是什么、质量标准是什么。它把你脑子里的隐性知识变成了 Agent 可以反复执行的显性流程。

这一层还有一个关键动作：**给 Agent 接反馈信号。** 选一个可量化的目标（测试通过率、构建时间、bundle size），让 Agent 执行完任务后能自己验证结果。一旦有了"做完→自检→确认达标"的闭环，你就拥有了一个自我纠正的工作流，而不只是一个执行指令的工具。

LangChain 的"推理三明治"策略也属于这一层的优化：不是全程最高推理等级，而是规划阶段用 xhigh、执行阶段用 high、验证阶段再切回 xhigh。这种资源分配策略只有在标准化的工作流中才可能实现。

**常见陷阱**：**过早标准化。** 一个流程至少手动跑过 5 次以上再考虑封装成 Skill。过早标准化意味着你把一个还没想清楚的流程固化了，后面改起来比从头写更痛苦。

### L4-L5：委托层与治理层

当你的 harness 稳定运行在 L3 之后，后面的两层才值得考虑。简要概述：

**L4 委托层**解决的是单 Agent 的能力边界问题。当任务复杂度超出单次会话的承载力，或者上下文互相污染时，你需要 Subagents——隔离的、有明确职责边界的子 Agent。典型模式是 Writer/Reviewer：一个 Agent 写代码，另一个在干净的上下文中审代码。Claude Code 已经内置了 Explore（只读快速搜索）、Plan（架构设计）、General-purpose（通用执行）三种 Subagent 类型。

**L5 治理层**解决的是"从个人工具变成团队基础设施"的问题。当多人共享同一套 harness，你需要回答：谁能让 Agent 做什么？哪些操作需要审批？Agent 的行为日志存在哪里？这涉及权限模型（Permissions）、沙箱隔离（Sandboxing）、主体层级（Anthropic 的 Principal Hierarchy：平台 → 开发者 → 用户）和 CI/CD 集成。大多数团队不需要现在就到这一层——但如果你在做平台级的 Agent Infra，这是你绕不开的终局。

> **本节小结**：L1 解决"Agent 不了解项目"，L2 解决"Agent 了解但不遵守"，L3 解决"Agent 遵守但需要人工编排"。大多数团队当前在 L0-L1，把 L2-L3 做扎实就已经能显著改善 Agent 的可靠性。不要追求一步到位——Mitchell Hashimoto 自己也是经过六个阶段才走到 Engineer the Harness。

---

## 写在最后

Harness Engineering 的核心洞察已经被多组独立实验验证：**在 Agent 系统中，系统设计对最终结果的影响超过模型能力本身。** 这不是说模型不重要，而是说当模型能力趋同后，你的竞争力取决于模型之外的一切——约束、反馈、知识管理、熵治理。

从工程师的角度看，这是一次角色进化。你的工作从"写代码"变成了"设计让 Agent 可靠地写代码的系统"。OpenAI 那篇文章的核心标语说得很直白：**Humans steer. Agents execute.** 你不再是执行者，你是基础设施的建设者。

这个术语本身能活多久？不确定。也许半年后会被新词取代或吸收。但它指向的问题域——**如何设计约束、反馈和改进循环来驾驭日益强大的 AI Agent**——会是接下来很长一段时间里，Agent 工程师最核心的工作内容。

至于这个词本身能不能立住，时间会给答案。但底下的问题是真的。

---

*本文综合整理自以下文献：*

1. *Mitchell Hashimoto — [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey)（2026-02-05）*
2. *OpenAI — [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)（2026-02-11）*
3. *Birgitta Böckeler / Martin Fowler 网站 — [Harness Engineering](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)（2026-02-17）*
4. *LangChain — [Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)（2026-02-17）*
5. *Phil Schmid — [The importance of Agent Harness in 2026](https://www.philschmid.de/agent-harness-2026)（2026-01-05）*
6. *Can Bölük — [I Improved 15 LLMs at Coding in One Afternoon. Only the Harness Changed.](https://blog.can.ac/2026/02/12/the-harness-problem/)（2026-02-12）*
7. *Yichao 'Peak' Ji / Manus — [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)（2025-07-18）*
