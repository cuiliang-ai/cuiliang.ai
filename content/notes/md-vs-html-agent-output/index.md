---
title: "跟风聊一下我的 Agent 输出选择：MD 还是 HTML"
date: 2026-05-12
draft: false
summary: "MD vs HTML 之争被 Thariq 那篇文章点燃，我也想了想。结论很朴素：默认 Markdown，特定场景切 HTML。判断标准只有两个：下游消费者是 Agent 还是人？会不会再改？文章顺带聊了我已经切到 HTML 的 4 个场景、一个双轨做法、以及 Agent 真正打开的新形态——让它顺手把消费内容的容器也做了。"
description: "MD vs HTML 之争被 Thariq 那篇文章点燃，我也想了想。结论很朴素：默认 Markdown，特定场景切 HTML。判断标准只有两个：下游消费者是 Agent 还是人？会不会再改？文章顺带聊了我已经切到 HTML 的 4 个场景、一个双轨做法、以及 Agent 真正打开的新形态——让它顺手把消费内容的容器也做了。"
tags:
  - AI Agent
  - Claude Code
  - Markdown
  - HTML
  - Workflow
categories:
  - AI Agent Engineering
ShowToc: true
---

最近这阵子，"Agent 该输出 HTML 还是 MD" 在网上吵得无比激烈。一时间 MD 像是被打入冷宫，HTML 成了 Agent 产物的新宠。

这阵风的源头是 Anthropic Claude Code 团队的 Thariq 写的[一篇文章](https://x.com/trq212/status/2052809885763747935)《Using Claude Code: The Unreasonable Effectiveness of HTML》，主张以后输出都尽量切到 HTML，理由从信息密度、可读性、可分享性一直讲到"造起来更有意思"。从 X 到公众号到处都在讨论。

我是 CC 和 Codex 的重度用户，每天都让它们生成各种东西——spec、code review 报告、代码分析文档、项目的 phase plan、给自媒体写的内容草稿。这个问题我也想过。结论比较朴素：**默认还是 Markdown，特定场景切 HTML**。

## 为什么默认还是 MD

反方反对 HTML 的几个主要论点——版本控制困难、源文件可读性丢失、注意力被冗余 token 稀释——我大多同意。这也是我默认还是 MD 的一部分原因。但更核心的原因是，我的产物**绝大部分不是终点**。

我让 coding agent 写一份 spec，写完它就要被人或者另一个 agent review，通过后让 coding agent 按照这份 spec 进行后续的开发工作。**开发过程中 spec 也会跟着实现持续修订**。这个 spec 是个**活的源文件**，不是一份终稿。

HTML 在这种场景下处处别扭：

- diff 一片红，review 的人根本看不出来改了什么
- 下一轮 Claude 要改一个小地方，定位锚点不稳定，容易牵动周围
- 我自己想搜个关键词，得先开浏览器看渲染版，源码里 grep 不出来

更别说项目里那些 `CLAUDE.md`、`AGENTS.md`、`SKILL.md`，本质上都是给 Agent 看的配置文件。这些东西只能是 Markdown，HTML 完全不适用。

所以"默认 MD"这件事是工作流约束的结果。我的工作流里，一份文档的下游消费者主要是 Agent 和我自己，我需要保证这份产物能够被工作主力 Agent 高效地使用。

## 但有几种场景我已经切换为 HTML

讲完默认值，说一下已经切到 HTML 的几种情况。共同特征是：**这份产物的下游消费者不是 Agent，是人；而且只消费一次，不再改**。

如果只能挑一个先试，我推荐从 **Slide deck** 开始——替代品（PowerPoint）你已经熟，对比立刻能感受到，而且讲完归档不留长期维护负担，几乎不可能踩坑。代码文档和 Code review 报告涉及工作流改造，门槛更高，建议有了第一次正反馈再上。

### 1. Slide deck

需要给团队或者跨组讲一个主题——架构介绍、阶段汇报、技术分享——以前的默认动作是开 PowerPoint，或者让 Claude 生成一份 pptx 文件。现在我直接让它生成一份 self-contained HTML deck：每张 slide 一个 section，键盘左右翻页，图表用 SVG 内联，要演示某个交互的地方直接嵌一个可点的小 demo。

比 pptx 顺手的几个点：改一个标点不用开 PowerPoint，直接让 Claude `str_replace`；嵌入的是活的图表，不是截图；分享只要一个文件，本地打开或者发链接都行；视觉风格可以跟项目文档统一。

讲完一次就归档，没有后续迭代——从交付那一刻起就是冻结的，HTML 的劣势全部消失，优势全部生效。当然如果对方要求必须 pptx 落档（比如要传内部系统），那还得走传统路线。

顺便推荐 Zara Zhang 的 [frontend-slides](https://github.com/zarazhangrui/frontend-slides)——一个 Claude Code 的 slide 生成 skill，可以直接用，也可以 fork 一份改成更适合自己的风格。

### 2. 代码文档

这是我最近最认可的一类切换。模块说明、调用关系、状态机、配置参数空间——这些东西**本身就是结构化和动态的**，硬塞进 Markdown 等于把三维信息压成一维文本。读者读完还要在脑子里重建一遍结构，这一步成本很高，只是平时没人量化它。

前段时间让 CC 分析自己的代码、生成一本电子书，把一个 query 的全流程做成了动态交互图，理解起来比静态架构图直观很多。副作用是：让 Agent 生成这类交互文档之前，你得先**把信息架构显式化**——哪些是顶层节点、哪些是展开的下一层、哪些通过 hover 关联。这个过程逼着你把系统理解想清楚，你自己的认知提升可能比读者收益更大。

### 3. Code Review 报告

我最近把自己的 code review skill 改成了生成 HTML 格式的报告。HTML 报告里能装下：问题分类、严重程度、新引入 vs 历史遗留、建议修复方案，加一段可以直接粘贴回 PR 的 review comments。同样的内容塞进 MD 要么过载要么过简，HTML 版本能做到详尽和好读兼顾。

判断标准很简单：**这份产物会落在哪个渲染 surface 上**。落在 GitHub / Obsidian / Notion 上就是 MD，作为独立文件单独打开就是 HTML。

### 4. PoC、Demo 与一次性小工具

这是我从 Thariq 那篇文章里**沉淀下来**的东西——不是 HTML 本身，而是这个用法：让 Claude 给你做一个一次性的可执行产物，专为眼下这个事情服务，用完就关。

可以是验证某个技术想法的 PoC，可以是给同事看效果的 Demo，也可以是为某次决策造一个专用界面。共同点是：寿命很短，做完就丢，不进 codebase。

最近用过的几个：

- 想确认 SSE streaming + 某个解析方式行不行，让 Claude 直接生成一个单文件 HTML，里面 mock 了 server 端的输出节奏，浏览器打开能看到完整效果——比开项目工程跑起来快得多
- 给自己一个"买 mini PC vs 翻新一台旧 ThinkPad 当 home server"的决策做了一个多维度对比矩阵，权重可调，分数实时刷新

PoC 和 Demo 这类用法以前也有，只是现在便宜很多。但为一次决策造一个专用界面——这是过去根本不存在的形态。造一个工具的成本一直高于它解决的问题，所以没人会做。Claude 把这个成本压到了几乎为零，这个区间才打开。

## 我最近在试的一个做法

代码文档这一类，我最近在试一个双轨做法：

- `docs/foo.md` 是 source of truth，进 git，团队 review，下一轮 Claude 读取的版本
- `docs/foo.html` 是 reading surface，从 MD + 一份 metadata JSON（图数据、参数）生成
- 生成那一步目前主要靠 Claude Code 手动跑——改完 MD 让它重新渲染 HTML，自动化（比如包成 slash command 或者 git hook）还在固化中

如果这套跑顺了，MD 那边的优势（可 grep、可 diff、可 review、可被 Agent 消费）一个不丢，HTML 那边的优势（结构表达、阅读体验、给非技术同事看）也都拿到了。两份在一起反而比任何一份单独存在更有用。

其实这就是 Markdown 当初被发明的初衷——John Gruber 设计它的时候本来就是"易读的源 + 编译到 HTML"。这个范式没错过，错的是中间二十年我们把 MD 直接当成了最终阅读形态。Agent 把"编译那一步"的成本压到零之后，这条老路反而走得通了。

## 最后

如果让我把这场讨论的核心收益提炼成一句话，不是"HTML 比 MD 好"，而是：

> Markdown for what lives. HTML for what's used once. Agents made the second category cheap enough to be worth doing.

真正打开的可能性是：你以前会让 Claude 写文档、写代码、写报告——这都是"内容"。现在你可以让它顺手把"消费这个内容的容器"也做了——一个临时的 ticket 排序器、一个参数 tuning 面板、一个 PR diff viewer。这些容器用完即弃，但解决问题的方式比纯文本顺畅很多。格式选 MD 还是 HTML 是术，理解"Agent 同时拓展了内容和承载内容的形态"才是道。

回到最初的问题——MD 还是 HTML？两个都用。但分清楚什么时候用哪个，比纠结哪个更好重要得多。
