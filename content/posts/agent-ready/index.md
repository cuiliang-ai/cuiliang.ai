---
title: "Agent Ready：软件的下一场自我革命"
date: 2026-03-09
draft: false
summary: "有 API 不等于 Agent Ready。从可发现性、交互模式、状态管理、权限模型、容错边界五个维度，拆解产品怎么做才算真正 For Agent。"
description: "有 API 不等于 Agent Ready。从可发现性、交互模式、状态管理、权限模型、容错边界五个维度，拆解产品怎么做才算真正 For Agent。"
tags: ["AI Agent", "MCP", "A2A", "Agent Ready", "API Design"]
categories: ["AI Agent Engineering"]
ShowToc: true
TocOpen: true
---

> 软件的调用者正在悄悄换人。

---

## 一、有人注意到了一件事

Stripe 的 API 文档最近悄悄多了一页：专门写给 Agent 看的集成指南。不是给开发者读的教程，是给 LLM 理解的工具描述。一家成立于 2010 年的支付公司，在 2025 年把 Agent 当成了第一类 API 用户。

这件事本身说明了一些问题。

最近一段时间，有几篇文章陆续触到了同一个命题：软件的用户，正在从人变成 Agent。其中有一篇传播很广——作者「有机大橘子」写于 2026 年 2 月，题目叫《互联网已死，Agent 永生》。文章的核心判断是：软件公司不会消失，但会从面向人类的产品变成面向 Agent 的基础设施，从 2B、2C 走向 2A（to Agent）。

这个判断方向没有问题。但文章在提出命题后就结束了——它回答了"为什么要变"，没有回答"怎么变"。

这篇文章想接着往下谈：**产品要怎么做，才算真正 Agent Ready？**

---

## 二、三阶演进：软件调用者的迭代史

先建一个坐标系。

### 第一阶：For Human（GUI 时代）

产品设计的核心假设是：使用者是人。

- 交互方式：视觉反馈、点击、拖拽、表单填写
- 信息呈现：富文本、图表、颜色编码、Toast 提示
- 错误处理：弹窗告诉用户"哪里填错了"

这是过去 30 年软件产品的默认范式。

### 第二阶：For API（集成时代）

随着 SaaS 爆发，软件开始有了第二类调用者：另一个程序。

- REST/GraphQL/gRPC 接口、SDK、Webhook
- 结构化输入输出、认证鉴权、Rate Limiting
- 大多数成熟产品已经走到了这一步

For API 解决的是"能不能调通"的问题。文档给开发者读，开发者理解业务逻辑，把确定性的调用流程硬编码进代码。

### 第三阶：For Agent（智能体时代）

这是正在发生的跃迁。

很多人以为"有 API 就够了"。但这就像说"我们有楼梯，轮椅也能用"——技术上也许勉强走通，体验和可靠性完全不是一回事。

根本原因在于：**Agent 和传统程序调用 API，有本质区别。**

传统 API 调用者是**确定性程序**——开发者提前写死了调用逻辑，知道自己要什么、传什么参数、怎么处理返回值。API 只需要"能调通"。

Agent 是**概率性决策者**——它需要在运行时自己理解能做什么、判断该做什么、决定怎么做、处理没做好的情况。API 需要"能用好"。

> **For API 解决的是"能不能调通"的问题，For Agent 解决的是"能不能用好"的问题。**

---

## 三、Agent Ready 的两个场景

到这里需要引入一个关键区分，否则"For Agent"这个概念本身就不够准确。

"Agent 调用你的软件"其实包含两种完全不同的拓扑：

**场景一：Agent 把你的软件当工具（A2T，Agent to Tool）**

Agent 调用你提供的 API 或 MCP 工具来完成某个子任务。你的软件是工具，Agent 是使用者。这是目前讨论最多的场景，MCP 协议主要解决的就是这个问题。

**场景二：Agent 把你当另一个 Agent（A2A，Agent to Agent）**

调用方不是在调用一个工具，而是在委托另一个 Agent 完成一项任务。被调用方有自己的推理能力、状态管理和决策逻辑。双方是协作关系，不是主从关系。这是 Google 在 2025 年 4 月提出的 A2A 协议[^1]所针对的问题，发布时即获得超过 50 家技术合作伙伴支持[^2]，此后已捐献给 Linux Foundation 进行开源治理[^3]。

IBM 在其技术文档里对两者的定位做了简洁区分[^4]：MCP 是 AI 应用与外部服务（API、数据源、工具）之间的标准化通信层，A2A 则聚焦于 Agent 之间的协作通信。

两种场景的需求有交集，但核心挑战不同。下面分别拆解。

---

## 四、Agent Ready 的五个维度

A2T 和 A2A 要解决的核心问题是一样的，只是复杂度不同。用同一套维度来看两个场景，差异会更清楚。

两个场景的基础差异先说明白：A2T 里，被调用方是工具——无自主决策，执行指令；A2A 里，被调用方是 Agent——有自己的推理能力、状态管理和决策逻辑，双方是协作关系，不是主从关系。同一个维度在两个场景下的要求，随着这个差异逐级升高。

---

### 维度一：可发现性

**For API 的做法**：写一份 Swagger 文档，给开发者在开发阶段阅读。能力发现发生在写代码时，由人完成一次，然后硬编码。

**A2T**：Agent 需要在运行时动态理解"这个工具能做什么"。MCP 的 Tool 定义里有 `description` 字段，这不是给人看的文档，而是给 LLM 理解用的 prompt——**写好一个 Tool description 本身就是一种 prompt engineering**。

两种写法放在一起，差距一目了然：

**传统 OpenAPI 写法（写给开发者看）**

```yaml
/pipelines/{id}/run:
  post:
    summary: Run pipeline
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
    responses:
      '200':
        description: Pipeline started
```

**Agent-ready MCP Tool 写法（写给 LLM 理解）**

```json
{
  "name": "run_pipeline",
  "description": "触发一次 CI/CD 构建流水线。适用于代码合并后需要自动部署的场景。执行前请确认目标分支已通过代码审查，且当前没有同一流水线的运行中实例——重复触发会导致部署冲突。成功后返回 run_id，可用于查询构建进度。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "pipeline_id": {
        "type": "string",
        "description": "流水线唯一标识符，可从 list_pipelines 工具获取"
      },
      "branch": {
        "type": "string",
        "description": "要构建的 Git 分支名，默认为 main"
      }
    },
    "required": ["pipeline_id"]
  }
}
```

前者告诉开发者"接口长什么样"，后者告诉 LLM"什么时候该用、用了会发生什么、用之前要注意什么"。参数的 `description` 字段同样关键——`"可从 list_pipelines 工具获取"` 这句话让 Agent 知道找不到 ID 时下一步该做什么，而不是原地报错。

Stripe 的 `cancel_subscription` 工具描述是个好例子（Composio 整合页 [mcp.composio.dev/stripe](https://mcp.composio.dev/stripe)）：

> "Cancels a customer's active stripe subscription at the end of the current billing period, with options to invoice immediately for metered usage and prorate charges for unused time."

适用场景、决策点、操作后果一句话说清。对比 "DELETE /v1/subscriptions/{id}"——前者让 Agent 能判断"该不该调"，后者只告诉 Agent "怎么调"。

**A2A**：单个工具的 `description` 不够用了。被委托的对象不是执行一个动作，而是承担一类任务，所需的能力声明维度也完全不同。A2A 协议引入了 **Agent Card**——一个部署时发布的 JSON 文件，承载的信息远比 Tool description 丰富：这个 Agent 能处理哪类任务、支持哪些交互方式、需要什么授权、遵从哪个安全方案。其他 Agent 通过读取 Agent Card 来判断"要不要把这个任务委托给你"。

还是用 CI/CD 的例子。前面的 MCP Tool 描述的是一个动作——"触发一次构建"。下面这个 Agent Card 描述的是一个协作方——"一个能处理整套 CI/CD 任务的 Agent"：

```json
{
  "name": "CI/CD Pipeline Agent",
  "description": "管理 CI/CD 全流程的自治 Agent。可接受构建、测试、部署类任务委托，自主完成从代码检出到环境部署的完整链路。支持多分支并行构建，自动处理依赖冲突和回滚。执行过程中会主动汇报进度，遇到需要人工确认的变更（如生产环境部署、破坏性迁移）会暂停并请求授权。",
  "version": "1.2.0",
  "provider": {
    "organization": "Acme DevOps",
    "url": "https://acme.dev"
  },
  "supported_interfaces": [
    { "url": "https://cicd-agent.acme.dev/a2a", "protocol": "JSONRPC" }
  ],
  "capabilities": {
    "streaming": true,
    "push_notifications": true
  },
  "default_input_modes": ["application/json", "text/plain"],
  "default_output_modes": ["application/json", "text/plain"],
  "skills": [
    {
      "id": "build-and-test",
      "name": "构建与测试",
      "description": "检出指定分支代码，执行构建和全量测试。自动检测语言和构建工具，支持单体仓库和 monorepo。构建失败时返回结构化的错误诊断和修复建议。",
      "tags": ["ci", "build", "test"],
      "examples": [
        "帮我构建 feature/auth 分支并跑一遍测试",
        "main 分支最新提交的测试挂了，帮我排查"
      ]
    },
    {
      "id": "deploy",
      "name": "环境部署",
      "description": "将构建产物部署到指定环境。staging 环境自动执行，生产环境需要调用方确认后才会继续。部署失败自动回滚到上一个稳定版本。",
      "tags": ["cd", "deploy", "rollback"],
      "examples": [
        "把 v2.1.0 部署到 staging",
        "上一次生产部署有问题，回滚到前一个版本"
      ]
    }
  ],
  "security_schemes": {
    "oauth2": {
      "type": "oauth2",
      "flows": { "clientCredentials": { "tokenUrl": "https://auth.acme.dev/token", "scopes": {} } }
    }
  },
  "security_requirements": [{ "oauth2": [] }]
}
```

对比前面的 MCP Tool 定义，差异一目了然：Tool description 回答的是"这个动作做什么"，Agent Card 回答的是"这个协作方是谁、擅长什么、怎么沟通、凭什么信任它"。`skills` 里的 `examples` 字段甚至提供了自然语言任务示例，让调用方 Agent 能直接判断"这件事该不该交给它"。`capabilities` 声明了支持流式推送和进度通知——这意味着委托方不需要反复轮询，而是等着被推送进展。

与 Tool description 最大的区别不在于动态性，而在于**描述粒度**——它描述的不是一个操作，而是一个具备完整能力边界的协作方。

---

### 维度二：交互模式

**For API 的做法**：请求-响应，一问一答，同步等待。调用方发出请求，被调用方返回结果，交互结束。

**A2T**：单次同步调用仍是主流，但 Agent 执行的是多步骤任务，工具需要支持长时操作的进度查询和部分结果返回。不是全有或全无，而是"已完成 3/5 步，第 4 步需要你确认"。Dry-run 模式在这里也是关键需求：让 Agent 在不产生副作用的情况下先预演，再决定是否真正执行。Dry-run 在人工操作时代几乎不存在，但对需要"先想清楚再出手"的 Agent 来说是刚需。

**A2A**：交互模式发生了质变。委托的任务可能是长时运行的，涉及多个决策节点，期间双方需要来回沟通。A2A 协议定义了 **Task 对象和生命周期**：任务有状态（pending / working / completed / failed），支持进度推送，支持中途补充上下文。更关键的是，被委托的 Agent 可以在执行过程中反问、请求澄清、返回中间结果请求确认——这是对话，不是调用。

---

### 维度三：状态管理

**For API 的做法**：无状态设计。每次请求独立，上下文由调用方自己维护，被调用方不记得"上次做了什么"。

**A2T**：Agent 执行跨多步骤的复合任务时，需要工具侧提供任务级的上下文概念——Session（一系列操作属于同一个"任务"）和进度可查询。在 Agent Sandbox 的实际工程中，中间状态管理是最容易出问题的地方：不是因为单个 API 不好用，而是没有任何一个 API 设计了"任务级"的上下文概念，Agent 执行到一半断掉，没有任何恢复手段。

**A2A**：状态管理的边界从"工具侧维护"变成了"协议层定义"。A2A 的 Task 对象是协议级的一等公民，不是各家自己实现的 Session，而是有标准生命周期、标准状态机、标准进度推送接口的任务容器。本质上，它在协议层面标准化了 Agent 的"挂起/恢复（Suspend/Resume）"机制——任务可以在任意节点暂停、等待外部输入、再从断点继续，而不是从头重来。这让跨系统的任务委托成为可能：委托方和被委托方用同一套语言描述任务状态，不需要各自约定私有协议，也不需要靠外置数据库拼凑出一个脆弱的任务追踪系统。

---

### 维度四：权限模型

**For API 的做法**：OAuth scope，基于角色的固定权限集合。用户登录，拿到一组权限，所有调用都在这组权限下进行，权限边界在开发时确定。

**A2T**：Agent 代表用户行动，但不应该拥有用户的全部权限。需要的是任务级的最小授权——这次任务只允许操作这个资源，高风险操作（删除、发布、支付）走 human-in-the-loop，每个操作都能审计追溯到是哪个 Agent、代表哪个用户、基于什么决策执行的。Stripe 把这个原则直接写进了官方文档[^5]：

> "We strongly recommend using restricted API keys to limit access to the functionality your agent requires."

Restricted API Key（`rk_` 开头）让权限边界显式配置在 Dashboard 里，而不是藏在代码逻辑里。涉及资金操作的 human-in-the-loop 也是产品层面的明确设计，而不是留给调用方自己判断：

> "We recommend enabling human confirmation of tools and exercising caution when using the Stripe MCP with other servers to avoid prompt injection attacks."

**A2A**：信任模型变得更复杂。A2T 里"谁拿着 Key 谁就是调用方"的逻辑在 A2A 里失效了——调用方是一个 Agent，这个 Agent 是另一个系统派出来的，代表某个用户在行动，信任链是多层的。A2A 协议在 v0.3（2025 年 7 月 30 日）中为 Agent Card 新增了 `signatures` 字段[^6]，支持对 Agent Card 进行签名验证，解决跨系统 Agent 的身份可信问题。微软在 Azure AI Foundry 中集成了 A2A 支持[^7]，通过 Microsoft Entra 体系管理 Agent 身份和授权，让"这个 Agent 是谁、被授权做什么"在企业体系内可追溯。

---

### 维度五：容错与能力边界

**For API 的做法**：错误码 + 开发者预写的异常处理分支。`400 Bad Request`，程序按预设逻辑处理。能做什么不能做什么写在文档里，开发者自己判断，判断发生在写代码时。

**A2T**：Agent 需要从错误信息中自主理解发生了什么，然后决定重试、换参数还是换策略。错误信息要语义丰富：不是 `400 Bad Request`，而是"参数 `start_date` 不能晚于 `end_date`，请调整时间范围"，最好还附上修复建议。能力边界同样需要在运行时声明：工具需要告诉 Agent "在当前权限和环境下，你能用我做什么"，而不是让 Agent 盲目尝试所有工具。社区开发者开源的 [stripe-testing-mcp-tools](https://github.com/hideokamoto/stripe-testing-mcp-tools) 提供了基于 Stripe 测试模式的隔离环境，让 Agent 在不产生真实资金影响的情况下完整验证操作链路，是 dry-run 理念的工程化落地。

**A2A**：被委托的 Agent 是一个黑盒——它内部用什么框架、调用了哪些工具、推理过程如何，对外不可见。A2A 协议的官方定义里特意用了"opaque"这个词，不是偶然。这意味着容错和边界的责任全部压在接口契约上：能力声明要足够完整，任务失败时的错误传播要让上游 Agent 能感知并做出决策，不能依赖调用方了解任何内部细节。这比 A2T 的语义错误要求更高——A2T 里工具失败是一次调用失败，A2A 里 Agent 失败可能是一段已经进行了的复杂协作的中途崩溃。

---

**五个维度之间有一条隐线**：A2T 和 A2A 在每个维度上面对的挑战不同，但变化的方向是一致的——从"静态约定"走向"动态协商"，从"单次交互"走向"持续协作"，从"调用方承担理解成本"走向"双方共享语义契约"。

Stripe 目前是 A2T 方向的标杆，五个维度都有工程化落地，且每个设计都有文档出处可查（[stripe/ai](https://github.com/stripe/ai) 仓库提供远程 MCP Server、Agent Toolkit 和本地 CLI 三条接入路径[^8]）。A2A 层面尚无公开动作——这说明两个场景的成熟度本来就不在同一阶段，不是 Stripe 落后，而是整个行业的 A2A 基础设施还在建设中。

---

## 五、现实在哪里

绝大多数产品目前停留在 For API 阶段，A2T 能力尚不完整，A2A 能力几乎空白。

A2T 的补课有明确的优先级。最容易入手、收益最直接的是**工具描述质量**——不需要架构改造，只需要换一种写文档的思维，把 API 文档从"给人读的参数说明"改成"给 LLM 理解的语义描述"。其次是**语义化错误信息**，成本不高但经常被忽视，现有的 `400 Bad Request` 对 Agent 基本没有利用价值。最难做的是**任务级状态管理**，需要引入 Session 概念和进度查询机制，涉及架构改动，不是一次小改动能解决的。

A2A 的前置准备，当前阶段真正值得投入的只有一件事：**输出 Agent Card，让自己可被发现**。Task 生命周期管理和身份验证体系可以等协议进一步稳定再认真跟进——A2A 截至 2025 年 9 月已迭代到 v0.4[^9]，协议仍在快速演进中，过早投入有被变更打断的风险。

窗口期有多长，A2A 的扩散速度给了一个参考：2025 年 4 月发布时即有 50 余家技术合作伙伴[^2]，Microsoft、SAP、Salesforce、PayPal 等均在首批支持者之列，扩散速度不亚于 MCP 早期。

**最终被淘汰的，不一定是功能最弱的产品，而是最难被 Agent 调用、最难与 Agent 协作的产品。**

---

## 六、结尾

软件花了 30 年从 For Human 走到 For API。

从 For API 到 For Agent，不是一条路，而是两种定位：你的产品是被 Agent 调用的工具（A2T），还是被 Agent 委托的协作方（A2A）——这取决于系统的角色，不取决于成熟度。Stripe 把支付 API 做成顶级的 A2T，这是终点，不是起点。一个企业级智能助理从第一天起就需要 A2A，跟它有没有先做好 A2T 无关。

两种定位的要求不同，但出发点是一样的：**承认 Agent 是真正的用户，然后从这个假设出发重新设计一切。**

最容易入手的起点是 Tool description——成本最低，收益最直接，不需要架构改造，只需要换一种写文档的思维。从这里开始。

---

[^1]: Google Developers Blog, *"A2A: A new era of agent interoperability"*, April 9, 2025. https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
[^2]: 同上。原文表述为 "more than 50 technology partners"，合作伙伴包括 Atlassian、Salesforce、SAP、PayPal、ServiceNow、LangChain 等，另有 Accenture、Deloitte、McKinsey 等服务商。
[^3]: A2A 项目 GitHub 主页：*"A2A is an open source project under the Linux Foundation, contributed by Google."* https://github.com/a2aproject/A2A ；Google Cloud Blog 亦发布了开源治理公告。
[^4]: IBM, *"What is the Agent2Agent (A2A) protocol?"*. 原文对比：MCP "serves as a standardization layer for AI applications to communicate effectively with external services, such as APIs, data sources, predefined functions and other tools"；A2A "focuses on agent collaboration, facilitating communication between AI agents"。 https://www.ibm.com/think/topics/agent2agent-protocol
[^5]: Stripe, *"Model Context Protocol (MCP)"*, "Building autonomous agents" 一节。 https://docs.stripe.com/mcp
[^6]: A2A Protocol CHANGELOG, v0.3.0 (2025-07-30)：新增 `signatures` 字段到 AgentCard，同时引入 mTLS SecurityScheme 和 OAuth2 metadata URL。 https://github.com/a2aproject/A2A/blob/main/CHANGELOG.md
[^7]: Microsoft Tech Community, *"Embrace the future of AI with multi-agent systems and the A2A protocol"*. https://techcommunity.microsoft.com/blog/azure-ai-services-blog/embrace-the-future-of-ai-with-multi-agent-systems-and-the-a2a-protocol/4404538
[^8]: Stripe AI GitHub 仓库，包含 Agent Toolkit（LangChain / CrewAI / OpenAI Agent SDK）、远程 MCP Server（mcp.stripe.com）和本地 CLI（@stripe/mcp）。MIT 许可。 https://github.com/stripe/ai
[^9]: A2A Protocol CHANGELOG, v0.4.0 (2025-09-15)：新增 Task 列表查询与分页。协议从 2025 年 4 月发布至 9 月已迭代四个大版本（v0.1–v0.4），处于快速演进期。 https://github.com/a2aproject/A2A/blob/main/CHANGELOG.md

---
