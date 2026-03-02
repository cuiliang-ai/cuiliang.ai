---
title: "[HITL 番外] 微软的 Agent 框架三国演义"
date: 2026-03-01T15:00:00+08:00
draft: false
summary: "AutoGen、Semantic Kernel 与 Agent Framework 的分裂与统一。从 2023 年两条平行线，到 2024 年 AutoGen 一分为二，再到 2025 年合并为 Microsoft Agent Framework。"
description: "AutoGen、Semantic Kernel 与 Agent Framework 的分裂与统一。从 2023 年两条平行线，到 2024 年 AutoGen 一分为二，再到 2025 年合并为 Microsoft Agent Framework。"
tags: ["AI Agent", "AutoGen", "Semantic Kernel", "Microsoft Agent Framework", "AG2"]
categories: ["AI Agent Engineering"]
series: ["AI Agent 的人类控制权"]
ShowToc: true
TocOpen: true
---

> 📌 **本文是「AI Agent 的人类控制权」系列的番外篇**。正文系列：[第一篇](/posts/agent-hitl-why-steering-wheel/)建立问题意识，[第二篇](/posts/agent-hitl-three-layer-model/)给出三层分析框架，[第三篇](/posts/agent-hitl-six-frameworks/)用框架做六大框架体检，[第四篇](/posts/agent-hitl-demo-to-production/)深入工程实现，[第五篇](/posts/agent-hitl-automation-bias/)讨论前端设计与 Automation Bias。

*——AutoGen、Semantic Kernel 与 Agent Framework 的分裂与统一*

> **声明：** 本文纯属个人技术好奇，与任何公司或组织无关。所有信息均来自互联网公开资料（官方博客、GitHub 仓库、PyPI 页面、社区讨论等），关键事件附有原始出处链接。如有事实偏差，欢迎指正。

---

正文系列里我们把 AutoGen、Semantic Kernel 和 Microsoft Agent Framework 作为三个独立框架来评估。但如果你深入了解它们的历史，会发现这三个框架其实是同一棵树上的三根枝，而且这棵树经历了一场相当戏剧性的分裂与合并。

理解这段历史，对你做技术选型有直接帮助：它解释了为什么这三个框架的 API 风格如此不同、为什么某些能力在某个框架中缺失，以及接下来你应该押注哪条线。

## 起源：两条平行线

故事要从 2023 年讲起。当时微软内部同时存在两个面向 LLM 应用开发的项目，它们诞生于不同的部门，服务于不同的目标群体。

### Semantic Kernel：企业 SDK 路线

Semantic Kernel 最早由微软的产品工程团队发起，定位是**企业级 LLM 应用开发 SDK**。它的设计哲学是"把 LLM 当作一个可编排的函数来调用"，你定义 Kernel、注册 Plugin（包含多个 Function）、通过 Planner 编排执行。

SK 从一开始就是 .NET First 的，Python 支持是后来追加的。这反映了它的目标用户：微软生态下的企业 .NET 开发者。它强调的是稳定性、可观测性、与 Azure 服务的深度集成、企业级安全合规。

SK 的 HITL 能力通过 Filter 模式实现，包括 `FunctionInvocationFilter` 和 `AutoFunctionInvocationFilter`。这是一种非常"中间件"思维的设计：在函数调用的管道中插入拦截器，灵活但需要开发者自己搭建上层逻辑。没有内建的 Checkpoint 机制，因为 SK 最初的设计场景是单次请求-响应，不是长时间运行的有状态 Agent。

### AutoGen：多 Agent 研究路线

AutoGen 诞生于微软研究院（Microsoft Research），由 Chi Wang 和 Qingyun Wu 领导。它的定位是**多 Agent 对话框架**——多个 AI Agent 之间通过对话来协作完成任务。

AutoGen 0.2 的核心抽象是 `ConversableAgent`，一个能参与对话的 Agent 基类。它的 HITL 设计简单直接：`UserProxyAgent` 是一个特殊的 Agent，代表人类参与多 Agent 对话。通过 `human_input_mode` 参数控制人类介入的频率：

- `ALWAYS`：每一轮都等人类输入
- `TERMINATE`：在对话结束时请求人类反馈
- `NEVER`：完全自动

这个设计在研究和原型阶段极其好用，几行代码就能搭一个有人类参与的多 Agent 系统。但它本质上是同步阻塞的（`input()` 等待终端输入），没有考虑生产部署场景。

## 分裂：AutoGen 一分为二

2024 年下半年发生了一件让整个 Agent 社区困惑的事：**AutoGen 的核心创建者离开了微软的 AutoGen 项目，fork 了代码，成立了社区驱动的 AG2 组织。**

事情分两步发生。2024 年 9 月，Chi Wang 和 Qingyun Wu 先在 GitHub 上 fork 了 AutoGen 仓库（[DEV Community 的事后梳理](https://dev.to/maximsaplin/microsoft-autogen-has-split-in-2-wait-3-no-4-parts-2p58)）。随后在 2024 年 11 月中旬，他们正式宣布 AutoGen "进化为 AG2"（[Qingyun Wu 的推文, Nov 14, 2024](https://x.com/qingyun_wu/status/1857169701140377852)），在新的 GitHub 组织 `ag2ai/ag2` 下继续开发。他们保留了 PyPI 包的控制权（`pyautogen`、`autogen`、`ag2` 这些包名都指向 AG2，[可在 PyPI 验证](https://pypi.org/project/autogen/)），以及原有的 Discord 社区。

值得注意的是，Chi Wang 后来加入了 Google DeepMind（[LinkedIn 资料](https://www.linkedin.com/in/chi-wang-autogen/)），Qingyun Wu 继续留在学术界。AG2 并非一家公司，而是一个由志愿者维护的社区开源项目。

微软方面随后强调 `github.com/microsoft/autogen` 仍然是官方维护的 AutoGen 仓库，并表示会继续加大对 AutoGen 的投资。在 [DEV Community 的讨论](https://dev.to/maximsaplin/microsoft-autogen-has-split-in-2-wait-3-no-4-parts-2p58)中，微软团队成员明确表示："the project is going from strength to strength，there are plenty of Microsoft teams that are really invested in it."

这导致了一个非常混乱的局面：

```
AG2（ag2ai/ag2）：
  - 由 AutoGen 原始创建者维护（社区志愿者组织）
  - 延续 AutoGen 0.2 的架构
  - 控制了 PyPI 包名和 Discord 社区
  - Apache 2.0 许可证
  - 定位：社区驱动的开源 Agent OS

Microsoft AutoGen（microsoft/autogen）：
  - 由微软团队维护
  - 全新架构重写为 0.4 版本
  - 基于 Actor 模型的分布式设计
  - MIT 许可证
  - 定位：企业级可扩展 Agent 框架
```

如果你在 2024 年底到 2025 年初这段时间 `pip install autogen`，你装到的其实是 AG2 的版本，而不是微软的版本。微软的 AutoGen 0.4 需要通过 `pip install autogen-agentchat` 等拆分后的包名安装。对于不了解这段历史的开发者来说，这几乎是一个陷阱。

## 重构：AutoGen 0.4 的断裂式升级

分裂之后，微软团队对 AutoGen 进行了一次**完全重写**。2025 年 1 月 17 日，AutoGen 0.4 正式发布（[Microsoft Research Blog](https://devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4/)）。

这不是一次渐进式升级，而是从理念到架构的全面推翻：

**0.2 → 0.4 的核心变化：**

- **从对话模型到 Actor 模型**。0.2 的核心是 Agent 之间的对话（message passing in conversation）；0.4 采用了 Actor 模型（message passing between independent actors），每个 Agent 是一个独立的计算单元，通过消息队列通信。
- **从单体到分层架构**。0.4 分为三层：Core（底层消息传递和 Actor 运行时）、AgentChat（高层多 Agent 对话抽象）、Extensions（第三方集成）。
- **从同步到异步优先**。0.4 全面拥抱 `async/await`，所有 Agent 交互都是异步的。
- **HITL 从 `human_input_mode` 到 `Handoff`**。0.2 的人类参与是通过 UserProxyAgent 的配置参数；0.4 通过 Handoff 机制，Agent 主动将控制权"交接"给人类或其他 Agent。

这次重写的代价是**与 0.2 完全不兼容**。所有基于 0.2 编写的代码都需要重写。微软提供了迁移指南，但社区中相当一部分用户选择留在 AG2（0.2 架构的延续）而不是迁移到 0.4。

微软研究院的博客坦诚地解释了重写的原因（[原文](https://devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4/)）：0.2 在实际使用中暴露了架构局限性、API 膨胀和调试工具不足等问题。社区反馈强调了对更好的可观测性、更灵活的多 Agent 协作框架和可复用组件的需求。

## 合并：Microsoft Agent Framework 的诞生

故事的高潮发生在 2025 年 10 月 1 日。微软宣布了 **Microsoft Agent Framework**，将 AutoGen 和 Semantic Kernel 统一为一个框架，以公开预览（Public Preview）形态发布（[Microsoft Foundry Blog](https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/)）。

这次合并的逻辑很清晰。AutoGen 的强项是多 Agent 编排和创新性的交互模式，但生产基础设施薄弱。Semantic Kernel 的强项是企业级稳定性、Azure 集成和 .NET 支持，但 Agent 编排能力有限。**两者的优势互补，短板互补。**

微软 Foundry 博客的原话：

> "Developers asked us: why can't we have both — the innovation of AutoGen and the trust and stability of Semantic Kernel — in one unified framework? That's exactly why we built the Microsoft Agent Framework."

Semantic Kernel 团队在官方博客中进一步明确了定位（[原文](https://devblogs.microsoft.com/semantic-kernel/semantic-kernel-and-microsoft-agent-framework/)）："Think of Microsoft Agent Framework as Semantic Kernel v2.0 (it's built by the same team!)."

合并后的架构：

```
Microsoft Agent Framework
├── Agent 抽象（来自 AutoGen 0.4 的 Actor 模型）
├── 工具系统（来自 SK 的 Plugin/Function 体系，简化为装饰器风格）
├── 编排模式（来自 AutoGen 的多 Agent 编排 + SK 的 Workflow）
│   ├── Sequential / Concurrent（确定性工作流）
│   ├── Group Chat / Magentic（LLM 驱动的动态编排）
│   └── Handoff（人类参与的交接模式）
├── HITL 系统
│   ├── 声明式工具审批（SK 的 Filter 思想简化版）
│   ├── 统一的人类交互协议
│   └── 编排级 HITL 配置
├── 运行时与基础设施
│   ├── Session State（状态管理，支持 Checkpoint 和 pause/resume）
│   ├── OpenTelemetry（可观测性）
│   └── Azure AI Foundry 集成
└── 多语言支持（Python + .NET）
```

**对于之前的 SK 用户：** 需要从 `Microsoft.SemanticKernel.*` 命名空间迁移到 `Microsoft.Extensions.AI.*`。Agent 不再需要通过 Kernel 创建，而是直接从 Provider 实例化。Plugin 和 Function 的概念简化为装饰器风格的工具定义。SK 官方明确表示会"至少在 Agent Framework 正式发布后一年内继续支持 Semantic Kernel"。

**对于之前的 AutoGen 0.4 用户：** 迁移成本相对较低，因为 Agent Framework 本身就是在 AutoGen 0.4 的架构基础上构建的。`AssistantAgent` 映射为新的 `ChatAgent`，消息类型统一为 `ChatMessage`，编排从事件驱动模型转为图（Graph）API。

**对于 AG2 用户：** AG2 和 Microsoft Agent Framework 是两条完全独立的路线。AG2 延续 0.2 的架构，Microsoft Agent Framework 基于 0.4 的重写。两者不兼容，不会合并。

## 当前状态：2026 年初的格局

截至本文写作时（2026 年 2 月），微软 Agent 生态的状态是：

**Microsoft Agent Framework**：公开预览阶段。微软官方表示"expect it to be in Preview for several months"，行业分析普遍预期 2026 年内正式发布（GA）。这是微软的主力方向，所有新的投资和功能开发集中在这里。已有 KPMG、BMW、Fujitsu 等企业在其上部署生产工作负载。

**Semantic Kernel**：继续维护，微软明确表示"the majority of new features will be built for Microsoft Agent Framework"（[SK Blog](https://devblogs.microsoft.com/semantic-kernel/semantic-kernel-and-microsoft-agent-framework/)）。已有的 SK 用户可以继续使用，微软承诺在 Agent Framework GA 后至少再支持一年。SK 不会被废弃，但会逐渐成为 Agent Framework 的底层组件而非独立产品。

**AutoGen 0.4（microsoft/autogen）**：作为 Agent Framework 的前身，其核心概念（Actor 模型、AgentChat、事件驱动运行时）已经被吸收进 Agent Framework。微软官方文档明确将 Agent Framework 定位为"the next generation of both Semantic Kernel and AutoGen"（[Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)）。

**AG2（ag2ai/ag2）**：独立发展，与微软无关。延续 0.2 架构，社区驱动。适合偏好 0.2 风格 API 的用户，以及不想绑定微软生态的团队。

> **[图1：建议插入时间线图]**
>
> ```
> 2023         2024              2024.9-11       2025.01         2025.10         2026
>   │            │                  │               │               │              │
>   │  SK 发布    │  AutoGen 0.2    │  分裂：        │  AutoGen 0.4  │  合并：       │  Agent FW
>   │  AutoGen   │  快速增长       │  核心创建者    │  全面重写     │  SK + AG 0.4  │  预计 GA
>   │  各自发展   │                 │  fork → AG2   │  Actor 模型   │  = Agent FW   │
> ```

## 这段历史对你意味着什么

如果你正在做技术选型，这段历史给出的信号很明确：

**如果你在微软生态内（Azure、.NET、企业环境）**，答案是 Microsoft Agent Framework。不需要纠结 SK 还是 AutoGen，它们的未来都在 Agent Framework 里。现在是预览阶段，正式发布在即。如果你等不及，可以先用 SK 搭建，后续迁移到 Agent Framework 的路径是清晰的（微软提供了 .NET 和 Python 的详细迁移文档）。

**如果你偏好 AutoGen 0.2 的 API 风格**（`ConversableAgent`、`UserProxyAgent`、`human_input_mode`），并且不想绑定微软生态，AG2 是延续这条线的选择。但要清楚，AG2 和微软的方向已经彻底分道扬镳，未来不会重新合并。

**如果你在评估框架的 HITL 能力**（这个系列的主题），Microsoft Agent Framework 的 HITL 设计是三者中最系统化的。声明式审批、统一交互协议、编排级配置，这些是吸取了 SK 和 AutoGen 两边教训后的设计。它同时解决了 SK 的"灵活但什么都要自己写"和 AutoGen 0.2 的"简单但只能在终端跑"这两个问题。微软 Foundry 博客明确提到 Agent Framework 支持"checkpointing, pause/resume, and human-in-the-loop flows"。

**一条容易踩的坑：** 如果你在网上搜索 AutoGen 教程，大量内容还是基于 0.2 的。而 `pip install autogen` 安装的是 AG2。微软的 AutoGen 0.4 / Agent Framework 需要用不同的包名（如 `pip install agent-framework` 或 `pip install autogen-agentchat`）。在开始之前，先确认你安装的是哪个版本。正如一位开发者在 Medium 上的吐槽："Ask 'How to create an Autogen agent' and you might be told to install PyAutogen (unsupported) by ChatGPT ... I once lost an entire day trying to figure out why published messages weren't being picked up by agents."

## 更大的图景

微软的这次"分裂→重写→合并"不是偶然事件。它反映了整个 Agent 框架领域在 2024-2025 年经历的**从实验到生产的转型阵痛**。

2023 年，Agent 框架的核心竞争力是"能让 demo 跑起来"，谁的 API 最简单、谁的 demo 最酷、谁支持的模型最多。AutoGen 0.2 凭借简洁的多 Agent 对话抽象赢得了这个阶段。

2025 年，竞争力变成了"能让生产跑起来"，谁的状态管理最可靠、谁的可观测性最好、谁的安全机制最完善、谁的 HITL 最适合 Web 部署。这正是 AutoGen 0.2 力不从心的地方，也是微软选择全面重写的原因。

这个转型也解释了为什么 LangGraph 在同一时期异军突起，它从一开始就把 Checkpoint 和持久化作为核心设计，恰好踩中了"从实验到生产"的需求窗口。

Agent 框架的下一个竞争战场，很可能是本系列讨论的核心议题：**谁能把 HITL 做得既安全又不碍事，既满足合规要求，又不让用户觉得 Agent 是个需要保姆陪着的笨蛋。** 微软的三国演义，就是这场竞争的缩影。

---

*本文信息截至 2026 年 2 月。Agent 框架生态迭代极快，具体 API 和产品状态请以各框架官方文档为准。*

*主要参考来源：*
- *[Microsoft Foundry Blog: Introducing Microsoft Agent Framework](https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/) (2025.10)*
- *[Semantic Kernel Blog: SK and Microsoft Agent Framework](https://devblogs.microsoft.com/semantic-kernel/semantic-kernel-and-microsoft-agent-framework/) (2025.10)*
- *[Microsoft Research Blog: AutoGen 0.4](https://devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4/) (2025.01)*
- *[Microsoft Learn: Agent Framework Overview](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)*

---

*这是「AI Agent 的人类控制权」系列的番外篇。*
*完整系列：*
- *[第一篇：Agent 为什么需要方向盘](/posts/agent-hitl-why-steering-wheel/)*
- *[第二篇：HITL 的三层解剖——谁介入、怎么介入、靠什么介入](/posts/agent-hitl-three-layer-model/)*
- *[第三篇：六个框架，六种答卷——用三层模型做框架体检](/posts/agent-hitl-six-frameworks/)*
- *[第四篇：从 Demo 到生产的距离——Agent 状态管理的深水区](/posts/agent-hitl-demo-to-production/)*
- *[第五篇：别让人类盲审——Agent HITL 的前端设计](/posts/agent-hitl-automation-bias/)*
- *番外篇：微软的 Agent 框架三国演义（本文）*
