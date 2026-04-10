---
title: "Claude Managed Agents：模型厂商开始吞噬 Agent 基建"
date: 2026-04-09
draft: false
tags: ["Agent Engineering", "Claude", "Managed Agents", "Architecture"]
description: "Anthropic 发布 Claude Managed Agents，把 Harness 和 Sandbox 打包进平台。这不只是一个新产品，而是 Agent 基建层的一次结构性重组。"
---

上周你可能还在纠结选 E2B 还是 Daytona 做 Agent 沙箱，今天 Anthropic 告诉你：不用选了，我全包了。

2026 年 4 月，Anthropic 发布了 [Claude Managed Agents](https://www.anthropic.com/engineering/managed-agents)[^1]——一套云端托管的 Agent API。表面上看是又一个 Agent 平台产品，但如果拆开它的架构看，会发现一个更大的变化：模型厂商正在从「卖推理能力」扩张为「卖 Agent 运行时」，而这个扩张正在重新定义 Agent 开发者需要做什么、不需要做什么。

## Managed Agents 做了什么

先说产品本身。Managed Agents 提供三个核心 API 资源：Agent（定义模型、system prompt、工具）、Environment（定义容器环境、预装包、网络策略）、Session（启动一个实际运行的 Agent 实例）[^2]。应用与 Agent 之间通过 Events 交互——用户消息、工具结果、状态更新都以事件形式在 Session 内流转。

开发者创建一个 Agent 定义和一个 Environment 配置，然后按需启动 Session。每个 Session 会拿到一个独立的容器实例——Ubuntu 22.04，预装了 Python 3.12+、Node.js 20+、Rust 1.77+、Go 1.22+ 等八种语言运行时，最高 8GB 内存，10GB 磁盘[^3]。Session 之间完全隔离，文件系统不共享。同一个 Agent + Environment 可以并发启动多个 Session，Agent 和 Environment 是可复用的配置模板，Session 是运行实例。这个关系和 Docker 的 image/container 模型一致。

容器内，Agent 可以执行 bash 命令、读写文件、搜索网页、调用 MCP server。整个执行过程通过 Server-Sent Events (SSE) 流式返回。用户可以中途发送新的 event 来引导或打断 Agent 的执行方向。

用一句话概括：Managed Agents 为每个 Session 提供了一个具备 Claude Code 能力、但面向无人值守场景优化的云端 VM。

## 四层模型：拆解 Agent 系统的组成

要理解 Managed Agents 的战略意义，先需要一个框架来描述 Agent 系统的通用架构。任何 Agent 系统，不论用什么模型、什么框架，都可以拆成四层：

**Model** — 推理引擎。接受 prompt，输出文本和 tool calls。Claude、GPT、Gemini、开源模型，都在这一层。

**Harness** — 编排控制层。包含 agent loop（循环调用模型直到任务完成）、context 管理（prompt caching、compaction、context editing）、tool dispatch（接收 tool call、路由到执行层、返回结果）、错误恢复、自评估（outcomes）、多 Agent 协调。Harness 是 Agent 系统的中枢，它调用 Model 做推理，通过 MCP/Skills 获取外部能力，把执行任务派发到 Sandbox。

**MCP + Skills** — 可插拔的外部能力模块。MCP 提供运行时工具调用（连接 Notion、Gmail、Slack 等外部服务），Skills 提供领域知识和工作流模板（按需加载到 context）。这一层独立于 Harness 开发和分发，有自己的生态和 marketplace。

**Sandbox** — 执行环境。容器、文件系统、网络策略、预装运行时。代码在这里跑，文件在这里读写。

Anthropic 官方工程博客中的架构描述也印证了这个分层[^1]。Harness 位于中心，Session（事件日志）、Sandbox（执行环境）、Tools 三个模块向外辐射。Model、MCP+Skills、Sandbox 三者之间没有直接交互，全部通过 Harness 中转。

这四层各自有独立的替换边界和竞争维度：Model 层拼模型能力，Harness 层拼编排质量，MCP + Skills 层拼生态丰富度，Sandbox 层拼基础设施。

## 模型厂商的产品边界在扩张

有了四层模型，Managed Agents 的战略意图就清晰了。

之前的分工是：Anthropic 只提供 Model 这一层。开发者自己写 agent loop，自己选沙箱（E2B、Daytona），自己接工具（MCP），自己做 context 管理。四层里模型厂商只卖一层，其余三层是开发者的工作，也是中间件厂商的生存空间。

现在 Managed Agents 把 Harness 和 Sandbox 都包了。而且在 MCP + Skills 层，Anthropic 也在做基础供给：Claude.ai 内置了 Google Drive、Gmail、Notion 等常用 MCP connector，预建了 PowerPoint、Excel、PDF 等通用 Skills。

画成一个从「Anthropic 提供」到「用户自己做」的光谱：

- Model → Anthropic 提供
- Harness → Anthropic 提供
- Sandbox → Anthropic 提供
- 通用 MCP + Skills → Anthropic 提供基础盘
- 业务 MCP + Skills → 用户自己做

开发者的工作被压缩到了最右端：定义自己业务独有的工具接入和领域知识。成功标准是什么（Outcomes），连接哪些内部系统（MCP），需要什么领域工作流（Skills）。这些是最贴近业务价值的部分，也是模型厂商不可能替用户做的部分。

## Agent 领域的 PaaS 时刻

云计算的演进路径是 IaaS → PaaS：IaaS 给你虚拟机，你自己装运行时、配部署流程、管理进程；PaaS 把这些全部打包，你只管写应用代码，平台负责剩下的事。代价是你必须接受平台预设的构建方式和固定的运行规格。

Managed Agents 正在 Agent 领域复现这个过程。之前用 Messages API + E2B + 自写 agent loop 构建 Agent，相当于在 IaaS 上搭应用——每一层都要自己选型和集成。现在 Managed Agents 把 harness、sandbox、context 管理全部打包，开发者只需要定义 Agent 配置，平台负责 agent loop 怎么跑、context 怎么压缩、容器怎么隔离、错误怎么恢复。代价是你必须接受 Anthropic 的 harness 设计和容器规格（最高 8GB 内存、10GB 磁盘、无 GPU）。

用 PaaS / IaaS 的框架来看，当前 Agent 开发的工具生态可以分成三个层次：

**Managed Agents ≈ PaaS。** 不用管基础设施，按平台约定开发。快，但受平台约束——不能换模型、不能自定义 agent loop、容器规格固定。

**Agent 框架（LangGraph、CrewAI）≈ 开源中间件。** 给你一套现成的编排逻辑，但你自己部署和运维，也可以按需修改。

**自建 Harness + 独立 Sandbox ≈ IaaS。** 完全的架构自主权，可以选模型、定制 loop 逻辑、控制容器规格，但所有集成和运维都是你的事。

和云计算的历史一样，大多数开发者会选 PaaS 层——因为他们的核心竞争力不在基础设施，而在业务逻辑。只有少数有特殊需求的团队会留在 IaaS 层。

## 对 Agent 沙箱赛道的冲击

这个产品边界的扩张，直接挤压了独立 Agent 沙箱厂商的生存空间。

受冲击最大的是核心价值就是「给 AI Agent 提供安全代码执行环境」的产品：

| 产品 | 核心定位 | 与 Managed Agents 重叠度 |
|------|---------|------------------------|
| [E2B](https://e2b.dev/) | Firecracker microVM 沙箱，模型无关 | 高——代码执行 + 隔离是直接替代 |
| [Daytona](https://www.daytona.io/) | Docker 容器沙箱，主打持久化和快速冷启动 | 高——标准容器场景被覆盖 |
| [Blaxel](https://blaxel.ai/) | 永久待机沙箱，超低恢复延迟 | 中——低延迟场景仍有差异化 |
| [Runloop](https://www.runloop.ai/) | 企业级编码 Agent 沙箱，合规导向 | 中——合规需求是差异化空间 |

对于原本使用 Claude + E2B 的用户来说，Managed Agents 几乎是直接替代——减少集成维护、降低架构复杂度，还能拿到 Anthropic 内置的 prompt caching 和 compaction 优化。

模型厂商做这件事有几个结构性优势。首先是**信息不对称**：Anthropic 知道 Claude 的行为特征和工具调用模式，它的 harness 可以针对模型的具体版本做优化。官方工程博客披露，Managed Agents 的 harness 将 p50 首 token 延迟降低了约 60%，p95 延迟降低超过 90%[^1]。E2B 只能把模型当黑盒。其次是**定价权**：session-hour 仅 0.08 美元，token 费用不变，执行环境的成本被吸收进平台定价[^4]。最后是**分发渠道**：用户已经在 Claude Console 里了，开通 Managed Agents 就是加一个 beta header 的事，摩擦接近零。

但不是所有沙箱厂商都会受同等冲击。[Fly.io Sprites](https://fly.io/) 主打持久化大存储（100GB NVMe），可以跨 session 保持状态，这是 Managed Agents 10GB 临时磁盘覆盖不了的。[Modal](https://modal.com/) 的 GPU 推理和训练能力不受影响。Northflank 的 BYOC 部署满足数据合规需求。这些差异化能力目前在 Managed Agents 的产品范围之外。

而且这个冲击不只来自 Anthropic。OpenAI 的 Responses API + Code Interpreter、Google 的 Vertex AI Agent Builder 也在做类似的内置执行层。每个模型厂商都在吞噬自己生态里的沙箱中间件。

## 基建壁垒在分化，不是在消失

一个容易得出的结论是「基建不再是壁垒，业务理解才是」。这个判断对大多数 Agent 开发者成立，但需要限定条件。

它成立的前提是：你在模型厂商的托管平台上构建 Agent。在这个前提下，基建确实被抹平了——大家用的是同一个 Harness、同一个 Sandbox、同一个 Model。差异化只能来自 MCP + Skills 这一层：谁能定义更好的业务工具接入，谁能积累更精准的领域知识。

但在几种场景下，基建仍然是硬门槛。数据合规场景（金融、医疗、政府），客户不能把数据跑在模型厂商的基础设施上。多模型架构，Agent 需要在不同任务上切换不同模型，不能绑定单一平台。极端性能需求，Managed Agents 的容器是固定规格（最高 8GB 内存、无 GPU），覆盖不了大规模数据处理。平台锁定风险，把 Harness 和 Sandbox 全部交给模型厂商意味着核心运行逻辑完全依赖对方。

更准确的归纳是：基建壁垒的门槛在分化。对标准场景（SaaS 内嵌 Agent、内部工具自动化、编码助手），基建被托管平台抹平，竞争力转向业务理解。对高合规、多模型、高性能、反锁定的场景，基建仍然是稀缺能力，而且因为标准场景的基建被抹平，这些高端场景的基建反而更有价值。

E2B 们如果要找到可持续的位置，大概需要瞄准后者——服务那些需要「Agent 领域的 IaaS」的客户，而不是和模型厂商在 PaaS 层正面竞争。

## 对 Agent 开发者意味着什么

如果你正在构建 Agent 系统，Managed Agents 提供了一个新的决策维度。选型时先问自己一个问题：**你的竞争力在哪一层？**

如果在 MCP + Skills 层（业务工具和领域知识），那么把 Harness 和 Sandbox 交给托管平台是值得关注的方向。如果在 Harness 层（独特的编排逻辑、多模型路由、自定义 eval），或者你有合规和反锁定的约束，自建仍然是更稳妥的选择。

Managed Agents 目前仍处于 beta 阶段，产品形态和能力边界还在演进中，现在就做选型决策为时尚早。但不管最终选哪条路，四层模型（Model / Harness / MCP+Skills / Sandbox）可以作为一个持久的分析框架。模型厂商的产品边界会继续扩张，但这四层的分类逻辑不会变——变的只是每一层由谁提供。

---

[^1]: Anthropic Engineering, [*Managed Agents*](https://www.anthropic.com/engineering/managed-agents), 2026.
[^2]: Anthropic, [*Claude Managed Agents Overview*](https://platform.claude.com/docs/en/managed-agents/overview), 2026.
[^3]: Anthropic, [*Cloud Container Reference*](https://platform.claude.com/docs/en/managed-agents/cloud-containers), 2026.
[^4]: Anthropic, [*Introducing Claude Managed Agents*](https://claude.com/blog/claude-managed-agents), 2026.
