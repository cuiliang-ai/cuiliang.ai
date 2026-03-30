---
title: "CLI → GUI → CLI：当命令行不再是给人用的"
date: 2026-03-30
draft: false
summary: "Google Workspace、飞书、钉钉同月开源 CLI。CLI 的回归不是技术倒退，而是软件消费者结构变化的必然——GUI 服务人类，CLI 服务 AI Agent。"
description: "Google Workspace、飞书、钉钉同月开源 CLI。CLI 的回归不是技术倒退，而是软件消费者结构变化的必然——GUI 服务人类，CLI 服务 AI Agent。"
tags:
  - AI Agent
  - CLI
  - MCP
  - Claude Code
  - Skill
categories:
  - AI Agent Engineering
ShowToc: true
TocOpen: true
---

2026 年 3 月，三件事几乎同时发生：Google 开源了 Google Workspace CLI[^3]，飞书开源了 lark-cli[^4]，钉钉开源了 dingtalk-workspace-cli[^5]。三家办公效率软件——Google Workspace、飞书、钉钉——在同一个月做了同一个决定：把自己的全部能力压缩成一套命令行接口。

第一反应是困惑：MCP（Model Context Protocol）不是已经解决了"Agent 怎么调用外部服务"的问题吗？为什么还要再做一层 CLI？

## MCP 的代价

MCP 是 Anthropic 推出的标准协议[^6]，让 AI Agent 以结构化的方式调用外部工具。它确实解决了工具接入的标准化问题，但带来了一个叫"上下文税"（Context Tax）的工程代价。

MCP 的工作方式是：Agent 启动时，把所有连接的 MCP server 暴露的每一个方法（tool）的 schema 全量加载到上下文窗口里。每个方法的定义——名字、描述、参数的 JSON Schema、字段说明、枚举值——都要各占一份 token。一个 Notion MCP server 就有超过 20 个方法定义[^1]，一个 GitHub MCP server 可能有 40 多个[^2]，每个方法定义占几百到上千 token。接三四个 MCP server，估计有数万 token 就在 Agent 还没开始推理之前被消耗掉了。

这不只是费用问题。上下文窗口是 Agent 的工作记忆。schema 占掉的部分，就是 Agent 用来理解用户意图、推理下一步行动、记住对话历史的空间被压缩了。工具多了之后，Agent 的表现会明显下降——不是因为模型变笨了，而是因为它的"工作台"被工具说明书堆满了。

CLI + Skill 的组合换了一种策略：Agent 不加载任何 tool schema，只读一个几百 token 的 Skill 文件，知道"飞书 CLI 能操作消息、文档、日历，常用命令是这些"。需要具体参数时，Agent 执行 `lark-cli calendar --help`，按需获取信息。上下文开销从几万 token 降到几百 token，差距是数量级的。

不过这个差距有前提：Skill 写得足够好，Agent 不需要频繁 `--help` 探测。每次探测都会消耗对话上下文并增加一轮 LLM 调用，复杂任务中累积的动态开销可能并不比一次性加载 schema 少。而且 MCP 的静态税本身也在被 Prompt Caching（上下文缓存）和更大的上下文窗口缓解。

所以上下文税是 CLI 出现的触发因素，但不是它最持久的优势。CLI 更持久的优势在别处：

**容错机制更符合模型的自我修正习惯。** 模型的训练数据里有海量的终端交互——StackOverflow 回答、GitHub README、技术教程——所以模型对 CLI 模式非常熟悉。但这并不意味着 CLI 的调用成功率一定高于 MCP：CLI 参数的变体（`-u` vs `--user`，位置参数的顺序）同样容易引发模型幻觉，而 MCP 的 JSON Schema 通过严格的类型约束和枚举边界，在语法正确率上往往更高。CLI 真正的优势在于出错之后的恢复：当 CLI 报错时，stderr 返回的是人类可读的错误信息（比如 `Unknown flag: --usr. Did you mean --user?`），模型可以直接理解错误原因并修正命令。MCP 的结构化错误码虽然稳定，但对模型的自主修复缺少直接的指导信息。

**不依赖客户端实现 MCP 协议。** 任何能执行 shell 命令的 Agent 都能用 CLI——Claude Code、Codex CLI、甚至一个简单的 `subprocess.run()`。MCP 要求宿主环境实现完整的协议栈，对很多轻量场景是不必要的门槛。

**可组合性。** CLI 天然支持 pipe 和脚本编排。Agent 可以在一次 tool call 里写一个 for 循环批量处理请求，而 MCP 需要逐次独立 tool call，每次都经过 LLM 推理。

**调试透明。** CLI 的输入输出都是人类可读的文本。开发者可以直接在终端里手动跑同一条命令来复现 Agent 的行为。MCP 的调试要穿透 JSON-RPC 协议层，排查成本更高。

这些优势解释了 CLI 为什么值得做，但还没有解释一个更根本的问题：今天的 CLI 到底是什么？

## 这不是你记忆中的 CLI

我最早接触电脑时用的还是 DOS 5.22，所有操作都在命令行里完成。后来换到 Windows 95，从 CLI 到 GUI 是体验上质的飞跃——不用再记命令，点鼠标就能操作。对大多数用户来说，CLI 从那时起就变成了过时的东西。它之所以留存下来，是因为开发者需要自动化和脚本编排能力，GUI 做不到。CLI 变成了"GUI 覆盖不了的角落"，一个防守性的存在。

今天的 CLI 不一样。**它是专门为非人类消费者设计的接口。**

这个区别首先体现在设计优先级上。30 年前的 CLI 以人类可读性为第一目标——man page 要详尽、错误提示要友好、交互式提示要引导用户一步步操作。今天的 CLI 在设计之初就将结构化输出作为一等公民（First-class Citizen）。飞书 CLI 和钉钉 CLI 的默认输出是人类可读的表格，但通过 `-f json` 即可切换为 Agent 友好的结构化 JSON。Google Workspace CLI 同样如此。结构化输出不是事后补丁，而是从第一天就并行设计的能力——因为这个 CLI 的消费者从一开始就包括 AI Agent。

发现机制也不同了。30 年前你靠记忆和 man page 学习一个 CLI。今天的 CLI 配合 Skill 文件工作[^9]——Skill 用自然语言描述工具能做什么、典型工作流是什么，Agent 读了 Skill 就知道该调什么命令。"学习使用一个 CLI"的认知负担从人类转移到了一个几百 token 的文档上。

生命周期也变了。30 年前的 CLI 命令一旦发布，要保持向后兼容几十年——`ls`、`grep`、`awk` 的参数 30 年没变。今天的 CLI 可以更快地迭代——Agent 不依赖肌肉记忆，Skill 文件更新了 Agent 就自动适应新的命令用法。Google Workspace CLI 明确标注自己是 pre-v1，会有 breaking changes。这在传统 CLI 的世界里是不可接受的事情。

如果只用一句话概括：**今天的 CLI 是软件暴露给 AI 的 API**——只是碰巧选了命令行的形式，因为 LLM 的训练数据里 shell 交互的比例远高于 JSON-RPC 协议，模型对这种形式天然更熟练。

## Skill：写给 AI 的"5 分钟上手教程"

理解了新 CLI 的本质，就能理解 Skill 为什么是它不可分割的配套。

CLI 单独使用时有一个问题：Agent 面对一个陌生的 CLI 工具，不知道有哪些子命令，不知道典型工作流是什么，只能靠 `--help` 一层层摸索。

Skill 文件补上了这个缺口。它是一个 Markdown 文件，用自然语言描述一个工具的核心能力、常用命令组合、参数约定和注意事项。Google Workspace CLI 内置了 100 多个 Skill 文件和 50 个 recipe[^3]，飞书 CLI 提供了 19 个 Skill[^4]，钉钉 CLI 也附带了完整的 skills 目录[^5]。

这跟人类学习工具的方式很像：你不会去读一个 CLI 的完整 man page，你会找一篇"5 分钟上手"教程，里面有几个常用命令和典型用法。Skill 就是写给 Agent 的"5 分钟上手"。

在 CLI + Skill 的组合里，Skill 承担了"能力发现"和"使用指导"的双重职责——不可或缺。这跟 MCP + Skill 的组合不同：MCP 的 tool schema 本身已经告诉 Agent"有哪些工具、每个工具接受什么参数"，Skill 在 MCP 场景里是增强项，用来叠加更高层的使用策略，但不是必需品。

不过 Skill 也有自己的脆弱性。它是非结构化的 Markdown 文档，跟 CLI 的版本之间没有自动化的一致性保证。CLI 改了一个参数名或者调整了子命令结构，Skill 文件可能没有同步更新。Agent 按照过时的 Skill 描述构造命令，就会调用失败。MCP 的 tool schema 虽然也需要维护，但它是代码生成的结构化定义，至少参数类型和必填项的约束是自动校验的。

自然语言描述还带来另一个风险：歧义。Skill 文件说"用 `--user` 参数指定用户"，但没说清楚是 user ID 还是 user name，Agent 可能猜错。MCP 的 JSON Schema 可以通过枚举值和类型约束减少这类歧义。

这是 CLI + Skill 组合目前最大的工程短板：**能力描述和能力实现之间缺少强绑定**。随着 CLI 工具的迭代速度加快，这个问题会越来越突出。一个可能的缓解方向是把 Skill 的生成纳入 CI/CD 流程，从 CLI 代码的注释或命令定义中自动生成 Markdown，而不是靠人工手写和手动同步。

## CLI 的适用边界：它够不到的地方

CLI 的优势有一个前提：Agent 运行在一个可以执行 shell 命令的环境里。这个前提在很多场景下不成立。理解 CLI 的边界，需要先区分两种 MCP 部署模式。

**Local MCP** 是在你的开发机上启动一个进程（比如 `npx dingtalk-mcp`），通过 stdio 跟 Agent 通信，由这个本地进程调目标服务的 API。CLI 做的事情几乎一模一样——在本地执行命令，调 API，返回结果。两者的执行环境、权限模型、认证方式高度重叠。区别在于 Local MCP 要加载 tool schema 到上下文，CLI 不需要。**CLI 真正在替代的是 Local MCP。**

不过这里有一个容易忽略的代价：环境漂移（Environment Drift）。Local MCP 通常是自包含的——一个 npm 包或 Docker 镜像，打包了运行所需的全部依赖。CLI 则高度依赖宿主机的 shell 环境：PATH 配置、前置依赖、环境变量、操作系统差异。在跨平台分发 Agent 能力时，让用户的机器跑通一个 CLI 的环境配置成本，有时反而比启动一个标准的 Local MCP server 更高。CLI 虽轻，但它把环境配置的包袱转移给了宿主机。

**Remote MCP** 是另一回事。它部署在云端——比如 claude.ai 里连接的 Notion MCP、Gmail MCP。用户在浏览器里点一下授权就能用，Agent 通过 HTTP 调远端服务，整个过程中 Agent 不接触用户的凭证，不需要 shell 环境。

CLI 替代不了 Remote MCP，原因有三：

1. **认证模型不同。** Remote MCP 走标准 OAuth，token 由 MCP 网关管理，Agent 永远接触不到用户凭证。CLI 需要把 API key 存在本地环境变量或文件里。在你自己的开发机上这没什么，但在多用户的云端 Agent 服务里，这是不可接受的安全模型。

2. **运行环境限制。** 浏览器里的 Agent、移动端的 Agent、API-only 的 Agent 都能通过 HTTP 调 Remote MCP，但不能执行 shell 命令。

3. **中心化管控。** 企业部署的 Remote MCP 可以在网关层做租户隔离、速率限制、审计日志。CLI 的所有执行发生在客户端，server 端没有拦截点。

同一个产品往往需要同时提供两种接口。以 Notion 为例：在 claude.ai 里走 Remote MCP（用户点一下授权就能用），在 Claude Code 里走 CLI + Skill（开发者本地配置 API key）。**不是产品决定用哪种方式，是 Agent 的运行环境决定的。**

## 三家 CLI 的技术路径：表面相似，底层分化

理解了 CLI 和 MCP 的关系之后，回头看 Google Workspace、飞书、钉钉三家的实现，就会发现它们对这个关系的处理方式完全不同。

| 维度 | Google Workspace `gws` | 飞书 `lark-cli` | 钉钉 `dws` |
|------|----------------------|----------------|------------|
| 语言 | Rust | Go | Go |
| 底层调用 | Google REST API（直接） | 飞书开放平台 REST API（直接） | DingTalk MCP 服务（JSON-RPC）[^10] |
| 命令生成 | 运行时读取 Discovery Service 动态生成 | 预定义命令集（200+） | 运行时从 MCP 注册表动态发现 |
| MCP 支持 | 可选：`gws mcp` 启动 Local MCP server | 独立项目：`lark-openapi-mcp`[^7]（Local MCP） | CLI 本身就是 MCP 的客户端 |
| Skill 数量 | 100+ Skill + 50 recipe | 19 个 Skill | 完整 skills 目录 |
| 新能力同步 | API 更新后自动可用（Discovery Service 驱动） | 需要 CLI 版本更新 | 后端 MCP 注册表更新后自动可用 |

Google 和飞书的模式是 **CLI 直接调 REST API**，CLI 和 MCP 是平行的两条路。优点是链路短、调试简单、不依赖额外的 MCP 基础设施。

钉钉的模式是 **CLI 作为 MCP 的前端外壳**[^10]。CLI 内部通过 discovery 管线从 MCP 注册表获取服务元数据，经 IR（中间表示）归一化后挂载为 Cobra 命令树，最终由 transport 层发起 MCP JSON-RPC 调用。CLI 把"理解 MCP schema"的认知负担从 LLM 的上下文里搬到了编译好的二进制文件里——协议照用，但 Agent 不需要"看到"协议细节。代价是多了一跳网络调用和 MCP 服务的运维负担。好处是所有 Agent 接口（CLI、MCP client、第三方集成）都经过同一层服务，能力注册和权限控制只做一次。

两种路径的选择反映了现有基础设施的现实映射——钉钉已经建设了 MCP 服务端，CLI 在上面加一层壳是阻力最小的路径。对于正在考虑做 CLI 的团队，结论很清晰：现有 REST API 足够规范，直接包装是最短路径；已有 MCP 基础设施或内部 API 风格杂乱，引入一层归一化的 MCP 网关反而能降低对外的适配复杂度。

## 从"加分项"到"基础设施"

技术层面的定位和边界讲清楚了，还有一个产品层面的问题：谁需要做 CLI？

一年前，答案取决于你的用户群里有没有开发者。面向开发者的产品（GitHub、Vercel、Cloudflare）需要 CLI，面向非技术用户的产品不需要。

Claude Code 和 OpenClaw[^8] 改变了这个逻辑。

Claude Code 本身是开发者工具，但 OpenClaw 不是。OpenClaw 是一个多渠道 AI 助手网关，一个进程就能把 AI 能力接入飞书、钉钉、企微等平台。它的终端用户是在聊天软件里跟 AI 对话的普通员工。OpenClaw 作为服务端组件，底层更可能直接调 REST API 或 SDK，而不是在服务器上 spawn CLI 进程。但关键在于：**搭建和维护这套系统的开发者，在开发过程中大量依赖 CLI。**

这形成了一个间接但关键的传导链：业务侧的非技术人员提出需求（"帮我做一个自动查日程并发提醒的机器人"），开发者用 Claude Code 搭建原型。Claude Code 在终端里通过飞书 CLI 获取日历数据、测试消息发送、验证权限配置。原型验证通过后，再把逻辑迁移到生产环境的 SDK 调用。CLI 在这个链路里扮演的角色不是生产运行时的调用接口，而是**开发者快速验证和迭代的工具**。

产品没有 CLI，开发者用 Claude Code 搭建原型时就缺少一个高效的交互入口。他们会转向非官方封装、用浏览器自动化模拟操作，或者推荐用户换一个有 CLI 的竞品。

## GUI 服务人类，CLI 服务 AI

回到开头的问题：为什么三家产品在同一个月做了同一个决定？

因为软件产品正在经历一次消费者结构的变化。以前产品只有一种消费者——人类。现在多了一种——AI Agent。人类通过 GUI 使用产品，Agent 通过 CLI（或 MCP 或规范的 API）使用产品。**同一个产品，两种形态，同时存在。**

飞书把整个产品——消息、文档、日历、邮件、表格、多维表格、任务、知识库——全部压缩成一套命令行接口，说的就是这件事：**我不只是一个给人用的 App，我也是一个给 AI 用的操作接口。**

这不是回归。30 年前的 CLI 之所以被 GUI 替代，是因为它对人类不友好。今天的 CLI 之所以回来，是因为它的消费者不再是人类。命令行的形式没变，但它在技术栈里的位置完全不同了——**不再是"落后的交互方式"，而是"面向 AI 消费者的原生接口"**。

如果你正在做一个 SaaS 产品，现在是时候认真思考这个问题了：你的产品有没有为 AI 消费者准备一套接口？对于存量系统，先做一层 CLI + Skill 包装是最快的路径——不需要改后端架构，只是在现有 API 上加一个命令行外壳和几个 Skill 文件。对于从零开始的新项目，可以直接考虑原生的 MCP 支持，或者从 API 设计阶段就把结构化输出、稳定的错误码、可机器发现的能力描述作为一等需求。

但如果什么都没有，你的产品在 Agent 生态里就是一个黑盒，只能通过截图识别和模拟点击来操作。那是最低效、最脆弱、也是最容易被竞品替代的集成方式。

[^1]: Notion MCP Server（v2.0 共 22 个 tools）：[github.com/makenotion/notion-mcp-server](https://github.com/makenotion/notion-mcp-server)
[^2]: GitHub MCP Server：[github.com/github/github-mcp-server](https://github.com/github/github-mcp-server)
[^3]: Google Workspace CLI（gws）：[github.com/googleworkspace/cli](https://github.com/googleworkspace/cli)
[^4]: 飞书 Lark CLI：[github.com/larksuite/cli](https://github.com/larksuite/cli)
[^5]: 钉钉 DingTalk Workspace CLI（dws）：[github.com/DingTalk-Real-AI/dingtalk-workspace-cli](https://github.com/DingTalk-Real-AI/dingtalk-workspace-cli)
[^6]: Model Context Protocol（MCP）：[modelcontextprotocol.io](https://modelcontextprotocol.io)
[^7]: 飞书 Lark OpenAPI MCP：[github.com/larksuite/lark-openapi-mcp](https://github.com/larksuite/lark-openapi-mcp)
[^8]: OpenClaw：[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
[^9]: Claude Code Skills 文档：[docs.anthropic.com/en/docs/claude-code/skills](https://docs.anthropic.com/en/docs/claude-code/skills)
[^10]: 钉钉 CLI 架构文档，描述了 discovery 管线、IR 中间层和 MCP JSON-RPC transport：[docs/architecture.md](https://github.com/DingTalk-Real-AI/dingtalk-workspace-cli/blob/main/docs/architecture.md)
