---
title: "[HITL 3/5] 六个框架，六种答卷"
date: 2026-01-21
draft: false
summary: "用三层模型对 LangGraph、AutoGen、CrewAI、Semantic Kernel、MS Agent Framework、Claude Code 做 HITL 体检。结论：Checkpoint 是分水岭。"
description: "用三层模型对 LangGraph、AutoGen、CrewAI、Semantic Kernel、MS Agent Framework、Claude Code 做 HITL 体检。结论：Checkpoint 是分水岭。"
tags: ["AI Agent", "HITL", "LangGraph", "AutoGen", "CrewAI", "Semantic Kernel", "Agent Framework"]
categories: ["AI Agent Engineering"]
series: ["AI Agent 的人类控制权"]
ShowToc: true
TocOpen: true
---

> 📌 **本文是「AI Agent 的人类控制权」系列的第 3/5 篇**。[第一篇](/posts/agent-hitl-why-steering-wheel/)建立问题意识，[第二篇](/posts/agent-hitl-three-layer-model/)给出三层分析框架，本篇用框架做六大框架体检，[第四篇](/posts/agent-hitl-demo-to-production/)深入工程实现，[第五篇](/posts/agent-hitl-automation-bias/)讨论前端设计与 Automation Bias。另有[番外篇](/posts/agent-hitl-microsoft-three-kingdoms/)讲微软 Agent 框架的分裂与统一。

*——用三层模型给主流 Agent 框架做一次 HITL 体检*

---

上一篇我们建立了 HITL 三层模型：Layer 1 定义人类角色（把关人/协作者/纠正者），Layer 2 定义工程机制（工具级审批/计划审批/升级模式/反馈循环），Layer 3 定义基础设施（检查点/滑动自主性）。

模型的价值不在于它本身，而在于它能不能用。这篇我们就用它作为统一坐标系，对六个主流 Agent 框架做一次横向体检：**LangGraph、AutoGen（0.4）、CrewAI、Semantic Kernel、Microsoft Agent Framework、Claude Code**。

体检的方法是：从每个框架的官方文档和 API 中提取所有 HITL 相关 feature，然后逐一标注它属于三层模型的哪个位置——支持了哪种角色，实现了哪种机制，是否具备必要的基础设施。

---

## 逐个过堂

### LangGraph：标杆选手

LangGraph 是目前 HITL 能力最完整的框架，也是唯一一个**把持久化作为一等公民**设计的 Agent 框架。

**核心 API：**

- `interrupt(payload)` ——在任意节点内暂停执行，将 payload 呈现给人类。这是 2025 年初引入的统一 HITL 原语，取代了早期的 `interrupt_before` / `interrupt_after`。
- `Command(resume=value)` ——人类提供输入后恢复执行。
- `update_state(config, values)` ——直接修改 Agent 的状态（State Surgery）。人类可以在暂停期间改掉 Agent 记忆中的任何值，Agent 从修改后的状态继续。

**三层覆盖：**

- Layer 1：三种角色全覆盖。`interrupt` + 规则判断 = 把关人；`interrupt` + 等待用户输入 = 协作者；`update_state` = 纠正者。
- Layer 2：四种机制中覆盖三种。工具级审批（`interrupt` 嵌入工具节点）、升级模式（Agent 自主调用 `ask_human` 工具触发 `interrupt`）、反馈循环（`interrupt_after` + `update_state`）。计划审批没有原生支持，但可以用 `interrupt` 在计划节点后暂停来模拟。
- Layer 3：**这是 LangGraph 的决定性优势。** Checkpoint 是框架内建的核心能力，开箱即用。开发环境用 `InMemorySaver`，生产环境换成 `PostgresSaver`，只改一行代码，图逻辑完全不变。每一步执行都会自动写入检查点，`interrupt` 暂停时线程可以释放，任意时间后通过 `thread_id` 精确恢复。

**一句话评价：** HITL 能力最完整，且是唯一把 Checkpoint 真正做成"不需要你操心"的框架。如果你今天就要上线一个需要人类审批的 Agent 服务，LangGraph 是阻力最小的选择。

---

### AutoGen 0.4：架构大重构后的半成品

AutoGen 0.4 相对 0.2 做了彻底的架构重构。0.2 的 `human_input_mode` 简单粗暴但好用；0.4 转向了更灵活的 Handoff 机制，但 HITL 能力在重构过程中出现了明显的空窗。

**核心 API：**

- `Handoff(target="user")` ——Agent 将对话控制权交给用户。
- `HandoffTermination` ——检测到 Handoff 消息时终止当前 Agent 循环。
- 用户通过 `HandoffMessage` 将控制权交回 Agent。

**三层覆盖：**

- Layer 1：协作者是主要支持的角色（通过 Handoff 机制）。把关人支持有限——`HandoffTermination` 可以在特定条件下暂停，但没有工具级别的细粒度拦截。纠正者支持较弱——用户返回的 `HandoffMessage` 可以携带反馈，但没有 State Surgery 能力。
- Layer 2：升级模式是最自然的实现（Handoff 本质上就是 Agent 将控制权升级给人类）。工具级审批需要自己在 Handoff 逻辑中实现。反馈循环需要通过多轮 Handoff 来模拟。
- Layer 3：**这是 AutoGen 0.4 的最大短板。** 没有内建的 Checkpoint 机制。`save_state` / `load_state` 存在但需要开发者自己接持久化存储。这意味着在 Web 服务中部署时，暂停→恢复的流程需要从头搭建。

**一句话评价：** Handoff 机制在多 Agent 协作场景中有独特优势，但 HITL 的基础设施（特别是 Checkpoint）需要大量自建工作。适合研究和原型，生产部署的 HITL 门槛较高。

---

### CrewAI：最低门槛，最浅深度

CrewAI 的 HITL 设计哲学是极简——用最少的 API 让你最快跑通。

**核心 API：**

- `human_input=True` ——在 Task 级别开启，任务完成后自动请求人类反馈。
- `HumanTool` ——一个预定义的工具，Agent 可以在执行过程中调用它来问人类问题。

**三层覆盖：**

- Layer 1：纠正者通过 `human_input=True`（任务后反馈）。协作者通过 `HumanTool`（执行中提问）。把关人**缺失**——没有工具级的前置审批机制。
- Layer 2：反馈循环（`human_input=True`）和升级模式（`HumanTool`）。工具级审批和计划审批均不支持。
- Layer 3：**没有 Checkpoint，没有状态持久化。** 所有 HITL 交互都是同步阻塞的——`human_input=True` 触发时，进程在 `input()` 处阻塞等待终端输入。这意味着 CrewAI 的 HITL 只能在 CLI 或 Notebook 环境中使用，无法部署为 Web 服务。

**一句话评价：** 上手最快，5 分钟就能跑通一个有人类反馈的 Agent。但缺少把关人角色和 Checkpoint 基础设施，意味着它只适合原型验证和简单场景，无法满足需要安全审批的生产需求。

---

### Semantic Kernel：企业级过滤器，差一个持久层

Semantic Kernel 来自微软，走的是企业级中间件路线。它的 HITL 实现基于 Filter 模式——在函数调用的管道中插入过滤器。

**核心 API：**

- `FunctionInvocationFilter` ——在任意 Kernel Function 调用前/后插入拦截逻辑。
- `AutoFunctionInvocationFilter` ——专门针对 LLM Function Calling 的过滤器，可以在 AI 发起的工具调用前拦截。
- 过滤器内可以修改 `FunctionResult`，实现对工具返回值的篡改/修正。

**三层覆盖：**

- Layer 1：把关人是最强项——`AutoFunctionInvocationFilter` 可以在任何 AI 工具调用前拦截，检查参数，决定是否放行。纠正者部分支持——过滤器可以在工具执行后修改返回值。协作者较弱——没有原生的"暂停等人类回答"机制。
- Layer 2：工具级审批是核心能力（通过 Filter 实现）。反馈循环有限度支持（修改 FunctionResult）。升级模式和计划审批需要自建。
- Layer 3：**没有内建 Checkpoint。** Filter 本身是同步管道中的拦截器，要实现异步审批（暂停→持久化→恢复），需要开发者自己搭建状态管理。

**一句话评价：** Filter 模式在架构上很优雅，给了开发者极大的灵活性——你可以拦截任何函数调用并注入任意逻辑。但"灵活"也意味着"什么都要自己写"。适合已有成熟基础设施的企业团队，不适合从零开始搭建 HITL 系统。

---

### Microsoft Agent Framework：后来者的完整答卷

Microsoft Agent Framework（2025 年发布）是微软 Agent 生态的最新框架，也是六个框架中 HITL 设计最系统化的。它的官方文档明确声明："所有编排模式都支持 HITL。"

**核心 API：**

- `@tool(approval_mode="always_require")` ——装饰器声明工具需要审批。
- `AgentRequestInfoResponse.approve()` ——批准 Agent 的请求。
- `AgentRequestInfoResponse.from_messages()` ——以消息形式提供反馈。
- `with_request_info(agents=[...])` ——在编排层面选择性地为特定 Agent 启用 HITL。

**三层覆盖：**

- Layer 1：三种角色全覆盖。`approval_mode` = 把关人；`AgentRequestInfoResponse.from_messages()` = 纠正者；编排中的人类参与 = 协作者。
- Layer 2：工具级审批（`approval_mode`）、反馈循环（`from_messages`）、升级模式（`with_request_info`）。计划审批通过 Sequential/Concurrent Workflow 的阶段性审查实现。四种机制全覆盖。
- Layer 3：有 Session State 管理，支持异步审批流程。Checkpoint 能力存在但文档较少，需要进一步验证生产环境的成熟度。

**一句话评价：** 设计最系统化，API 最声明式（`approval_mode="always_require"` 这种装饰器方式对开发者非常友好）。但作为 2025 年才发布的框架，生态成熟度和社区实践还需要时间积累。如果你在微软技术栈内，这是优先选择。

---

### Claude Code：产品级 HITL，但不是框架

Claude Code 不是一个通用 Agent 框架，而是 Anthropic 的编码 Agent 产品。把它放在这里是因为它展示了一种独特的 HITL 设计思路——**以产品体验为中心，而非以 API 为中心**。

**核心机制：**

- **Allowlist / Denylist**——工具按风险分三级：允许自主执行的（如读文件）、需要审批的（如写文件）、完全禁止的。用户可以自定义分类。
- **模型侧升级**——Claude 在推理过程中自主判断何时需要问用户。这不是通过框架 API 实现的，而是通过模型训练和 system prompt 引导的。

**三层覆盖：**

- Layer 1：把关人通过 allowlist/denylist 实现。协作者通过模型侧升级（Claude 主动提问）。纠正者通过对话中的自然语言反馈。
- Layer 2：工具级审批是核心机制。升级模式依赖模型能力。反馈循环通过多轮对话实现。
- Layer 3：作为产品，Checkpoint 和状态管理由 Anthropic 内部处理，用户无感。滑动自主性通过 allowlist 的自定义实现——用户可以逐步扩大自动批准的工具范围。

**一句话评价：** 不是通用框架，不能直接拿来搭建自己的 Agent 系统。但它展示了"HITL 应该是什么体验"——用户几乎不需要理解底层机制，权限配置简单直觉，审批流程嵌入自然交互。其他框架在设计 HITL 的 UX 时，可以参考 Claude Code 的产品决策。

---

## 总览矩阵

把六个框架的能力放到一张表里：

### Layer 1 覆盖：角色支持

| 框架 | 把关人 | 协作者 | 纠正者 |
|------|--------|--------|--------|
| LangGraph | ✅ interrupt + 规则 | ✅ interrupt + 等待输入 | ✅ update_state |
| AutoGen 0.4 | ⚠️ 需自建 | ✅ Handoff | ⚠️ 有限（无 State Surgery） |
| CrewAI | ❌ 缺失 | ✅ HumanTool | ✅ human_input |
| Semantic Kernel | ✅ Filter 拦截 | ⚠️ 需自建 | ⚠️ 有限（修改返回值） |
| MS Agent FW | ✅ approval_mode | ✅ with_request_info | ✅ from_messages |
| Claude Code | ✅ allowlist/denylist | ✅ 模型侧升级 | ✅ 对话反馈 |

### Layer 2 覆盖：机制实现

| 框架 | 工具级审批 | 计划审批 | 升级模式 | 反馈循环 |
|------|-----------|---------|---------|---------
| LangGraph | ✅ | ⚠️ 可模拟 | ✅ | ✅ |
| AutoGen 0.4 | ⚠️ 需自建 | ❌ | ✅ | ⚠️ 多轮 Handoff |
| CrewAI | ❌ | ❌ | ✅ | ✅ |
| Semantic Kernel | ✅ | ❌ | ⚠️ 需自建 | ⚠️ 有限 |
| MS Agent FW | ✅ | ✅ Workflow | ✅ | ✅ |
| Claude Code | ✅ | ❌ | ✅ | ✅ |

### Layer 3 覆盖：基础设施

| 框架 | Checkpoint | 滑动自主性 |
|------|------------|-----------
| LangGraph | ✅ 内建（Postgres/SQLite/Memory） | ⚠️ 需自建策略 |
| AutoGen 0.4 | ⚠️ 需自建 | ❌ |
| CrewAI | ❌ | ❌ |
| Semantic Kernel | ❌ 需自建 | ❌ |
| MS Agent FW | ✅ Session State | ⚠️ 需自建策略 |
| Claude Code | ✅ 产品内建 | ✅ Allowlist 配置 |

---

## 读表：几个不应忽略的发现

**发现一：Checkpoint 是分水岭。**

六个框架在 Layer 1 和 Layer 2 的差异不大——基本都能以某种方式支持三种角色和大部分机制。真正拉开差距的是 Layer 3。只有 LangGraph 和 MS Agent Framework 有内建的 Checkpoint，其余框架要么需要自建，要么完全不支持。这直接决定了你的 HITL 能不能从"demo 里跑通"走到"生产环境上线"。

如果你选了一个没有 Checkpoint 的框架，你面临的不是"少一个 feature"的问题，而是"需要自己搭建一整套状态持久化和异步恢复机制"的问题。这个工作量可能比你的 Agent 业务逻辑本身还大。

**发现二：把关人是最被低估的角色。**

CrewAI 完全缺少把关人角色——没有工具级审批机制。这意味着如果 Agent 要调用一个高风险工具，框架层面没有任何手段在执行前拦截。你只能依赖 system prompt（软控制）来让模型"自觉"地先问用户。我们在[第一篇](/posts/agent-hitl-why-steering-wheel/)已经论证过，这不够。

**发现三：计划审批是目前最薄弱的环节。**

六个框架中，只有 MS Agent Framework 通过 Workflow 编排原生支持计划审批。其余框架要么完全不支持，要么需要用 `interrupt` 在特定节点后暂停来手动模拟。这可能反映了当前 Agent 生态的一个现实——大多数 Agent 还在做"接到指令就开始执行"的单步或少步任务，"先规划再执行"的模式尚未成熟。随着 Agent 承担的任务越来越复杂，计划审批会成为刚需。

**发现四：Claude Code 展示了终局体验。**

Claude Code 作为产品，在 HITL 体验上比所有框架都好。原因很简单——它不需要开发者做任何配置，所有 HITL 逻辑（allowlist、模型侧升级、对话反馈）都作为产品特性内建了。这提示了一个方向：**最好的 HITL 是用户感知不到"HITL"的存在的。** Agent 该问就问，该等就等，该自主就自主，过渡自然到用户根本不会意识到背后有一套复杂的控制机制。框架开发者和产品设计者都应该以此为目标。

---

## 选型建议

基于以上分析，针对不同场景的选型建议：

**"我需要今天就上线一个有人类审批的 Agent Web 服务"**
→ **LangGraph**。Checkpoint 开箱即用，interrupt API 成熟，社区实践最多。

**"我在微软技术栈内，需要企业级的 HITL"**
→ **MS Agent Framework**。API 设计最声明式，HITL 与编排深度集成。但要留意框架还比较新，生产案例积累中。

**"我只是想快速验证一个 Agent 原型，看看加人类反馈效果如何"**
→ **CrewAI**。5 分钟跑通，但请清楚地知道它的天花板在哪里——没有 Checkpoint，没有把关人。

**"我有成熟的基础设施团队，想在已有系统中嵌入 Agent HITL"**
→ **Semantic Kernel**。Filter 模式足够灵活，但 Checkpoint、异步审批等都需要自建。

**"我的场景是多 Agent 协作，人类参与的主要方式是在 Agent 之间传递信息"**
→ **AutoGen 0.4**。Handoff 机制天然适合这个场景。但要为 Checkpoint 做好自建准备。

---

## 小结

用三层模型做完这次体检，结论可以提炼为一句话：

**大多数框架在 Layer 1（角色支持）和 Layer 2（机制实现）上差距不大，真正的分水岭在 Layer 3（基础设施）。Checkpoint 的有无决定了你的 HITL 是一个 demo 还是一个可上线的系统。**

如果你从这个系列中只带走一个选型标准，那就是：**在评估一个 Agent 框架的 HITL 能力时，第一个问题不是"它支持哪种审批模式"，而是"它怎么处理暂停和恢复"。**

---

*这是「AI Agent 的人类控制权」系列的第三篇。*
*完整系列：*
- *[第一篇：Agent 为什么需要方向盘](/posts/agent-hitl-why-steering-wheel/)*
- *[第二篇：HITL 的三层解剖——谁介入、怎么介入、靠什么介入](/posts/agent-hitl-three-layer-model/)*
- *第三篇：六个框架，六种答卷（本文）*
- *[第四篇：从 Demo 到生产的距离——Agent 状态管理的深水区](/posts/agent-hitl-demo-to-production/)*
- *[第五篇：别让人类盲审——Agent HITL 的前端设计](/posts/agent-hitl-automation-bias/)*
- *[番外篇：微软的 Agent 框架三国演义](/posts/agent-hitl-microsoft-three-kingdoms/)*
