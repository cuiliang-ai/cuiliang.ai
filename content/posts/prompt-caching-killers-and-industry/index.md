---
title: "Cache 杀手与行业实战：从踩坑到最佳实践"
date: 2026-01-24
draft: false
summary: "六大 Cache 杀手的诊断与对策，以及 Claude Code、Codex、Manus、Gemini CLI 的 Cache-Aware 架构全景扫描。"
description: "六大 Cache 杀手的诊断与对策，以及 Claude Code、Codex、Manus、Gemini CLI 的 Cache-Aware 架构全景扫描。"
tags: ["Prompt Caching", "KV Cache", "Agent", "Claude Code", "Manus", "OpenAI Codex"]
categories: ["AI Agent Engineering"]
series: ["Agent 工程师的 Prompt Caching"]
---

> 📌 **本文是「Agent 工程师的 Prompt Caching」系列的第 3/4 篇**。第一篇讲 KV Cache 底层原理，第二篇讲 Prompt Cache 核心约束，第三篇讲实战踩坑与行业全景，第四篇讲 Context Engineering 架构手艺。

## Cache 破坏的六大杀手与对策

前缀精确匹配的铁律意味着：**任何位置的任何改动 → 该位置之后的 Cache 全部失效。**

在 Agent 系统中，有六类操作是最常见的 Cache 杀手。好消息是，每一个都有成熟的应对方案。

> 以下归纳综合自 Manus 官方博客（纪一超）、Claude Code 团队（Thariq）的实战经验，以及 chaofa 的系统性分析。

---

### 杀手一：时间戳放在 System Prompt 开头

**症状**：每次请求的 System Prompt 开头都带当前时间，导致第一个 token 就不一样，整个 Cache 废掉。

这是 Manus 早期踩过的坑。纪一超在官方博客中直言：

> "A common mistake is including a timestamp—especially one precise to the second—at the beginning of the system prompt. Sure, it lets the model tell you the current time, but it also kills your cache hit rate."
> （一个常见错误是在 system prompt 开头放精确到秒的时间戳。虽然模型能告诉你现在几点，但代价是 Cache 命中率归零。）

很多开发者喜欢在 System Prompt 的第一行写：

```
You are an AI assistant. Current time: 2026-02-24 14:30:05
```

时间戳每秒都在变。结果呢？你精心构造的几万 token 的 System Prompt、工具定义、历史对话——全部 Cache Miss，每次从头算。

**对策：动态信息后置。**

把时间戳、环境状态等变化频繁的信息，从 System Prompt 的开头移到**对话历史末尾的 user message** 中。

```
// ❌ 错误：时间戳放在 System Prompt 开头
{role: "system", content: "Current time: 2026-02-24...\nYou are..."}

// ✅ 正确：System Prompt 保持稳定，时间戳放在最新的 user message
{role: "system", content: "You are..."}  // 永远不变
...
{role: "user", content: "<system-reminder>Current time: 2026-02-24...</system-reminder>\n用户的实际问题"}
```

Claude Code 就是这么做的——用 `<system-reminder>` 标签把动态信息嵌在 user message 中。OpenAI Codex 则是追加新的 developer/user 消息。

**核心原则：稳定内容放前面，变化内容放后面。**

---

### 杀手二：动态增删工具定义

**症状**：根据 Agent 当前阶段动态加载不同的工具集，每次工具列表变化，后续所有 Cache 失效。

Claude Code 团队的 Thariq 指出这是他们遇到的**最常见的 Cache 破坏方式**。

场景很典型：你的 Agent 有 30 个工具，规划阶段只需要 5 个，执行阶段需要 20 个，总结阶段需要 3 个。如果你按需加载不同的工具子集——恭喜，每次状态切换，Cache 全废。

Manus 博客也直接说明了原因：

> "In most LLMs, tool definitions live near the front of the context after serialization, typically before or after the system prompt. So any change will invalidate the KV-cache for all subsequent actions and observations."
> （在大多数 LLM 中，工具定义在序列化后位于 context 的前部。所以任何变动都会使后续所有 action 和 observation 的 KV Cache 失效。）

**对策：工具列表永远不变，通过其他机制限制可用范围。**

三家给出了三种不同但殊途同归的方案：

### 方案 A：Claude Code —— 状态转换工具化 + defer_loading

最精巧的设计。两个关键思路：

**思路一：把状态切换变成工具调用。** Plan Mode 不是通过切换工具列表实现的，而是通过 `EnterPlanMode` / `ExitPlanMode` 两个**工具**实现的。工具列表始终完整不变，模型通过调用这两个工具来"切换状态"。

**思路二：defer_loading 延迟加载。** 对于大量工具（比如通过 MCP 接入的），先注册一个轻量的 stub（只有名字和简介，没有完整 schema），再提供一个 `ToolSearch` 工具让模型按需检索完整定义。这样 tools 数组的大小和内容保持稳定。

### 方案 B：Manus —— Logits Masking + 命名约定

更底层的控制。所有工具始终在 Prompt 中，但通过**推理时的 logits masking** 来控制模型实际能调用哪些工具。

具体做法：
- 工具命名遵循约定：`browser_click`、`shell_exec`、`file_read`
- 不同模式下，对 token logits 做 masking，让模型"看得到但选不了"某些工具
- 支持三种粒度：Auto（模型自选）、Required（必须调用）、Specified（限定范围）

这个方案需要 self-hosting 推理引擎，因为你要控制推理时的 logits。对于调用商业 API 的开发者不适用，但思路值得了解。

### 方案 C：OpenAI Codex —— allowed_tools 参数

最简单直接的方案。API 层面提供 `allowed_tools` 参数：

```json
{
  "tools": [...],           // 完整工具列表，永远不变
  "allowed_tools": ["shell", "file_read", "file_write"]  // 当前可用子集
}
```

`tools` 数组不变（保 Cache），`allowed_tools` 限制当前可用范围。

**三种方案对比：**

| 方案 | 实现方式 | 优点 | 限制 |
|------|----------|------|------|
| Claude Code | 工具化状态转换 + stub | 灵活，模型自主决策 | 需 API 支持 defer_loading |
| Manus | logits masking | 精细控制，零额外 token | 需 self-hosting |
| OpenAI | allowed_tools 参数 | 最简单，API 原生支持 | 仅粗粒度控制 |

**本质是同一个思路：工具定义不变（保 Cache），通过其他机制限制可选范围。**

---

### 杀手三：MCP 工具注册顺序不确定

**症状**：MCP 服务器重连后，返回的工具列表顺序可能变化。工具集完全相同，但序列化后 token 顺序不同，Cache 照样失效。

这是一个极其隐蔽的问题。你的 Agent 接入了三个 MCP 服务器，分别提供 5 个工具。某次请求时服务器 A 先响应，下次可能服务器 B 先响应，工具定义的拼接顺序就不一样了。从语义上看完全等价，从 token 序列上看完全不同。

OpenAI Codex 在实战中就遇到过这个问题——MCP 工具的动态注册让工具列表变得不稳定。

**对策：在 Agent 接入层对工具列表做排序稳定化。**

```python
# ❌ 直接使用 MCP 返回的工具列表（顺序不确定）
tools = mcp_client.list_tools()

# ✅ 排序后再注入 Prompt
tools = sorted(mcp_client.list_tools(), key=lambda t: t["name"])
```

原则很简单：无论底层工具来源如何动态，最终注入 Prompt 的 tools 数组必须**顺序确定、内容稳定**。这是 Client 侧的责任，不要指望 MCP 服务器替你保证。

更进一步，如果某个 MCP 服务器会动态新增/移除工具（比如用户安装了新插件），你需要在接入层做版本化管理——只在会话开始时拉取工具列表，会话过程中锁定不变。

---

### 杀手四：修改或删除历史消息

**症状**：为了"美化"上下文，编辑或删除之前的 action/observation，导致修改位置之后的 Cache 全部失效。

有些开发者会做这些操作：
- 删除失败的工具调用结果（觉得"脏数据"会干扰模型）
- 修改之前的 assistant 回复（修正格式错误）
- 截断过长的 observation（把 10KB 的网页内容缩减为摘要）

这些操作看起来是在"优化"context，实际上每一个都在破坏 Cache。因为历史消息在 Prompt 的中部，一旦修改，后面所有内容的 Cache 都失效了。

Manus 博客专门用了一节 "Keep the Wrong Stuff In" 来强调这个问题：

> "Erasing failure removes evidence. And without evidence, the model can't adapt."
> （擦除失败就是移除证据。没有证据，模型就无法自我调整。）

**对策：严格 Append-only，错误内容保留。**

这带来一个双重好处：
1. **保护 Cache**：历史消息不变，前缀稳定
2. **模型学习**：模型看到自己之前犯的错，会在后续步骤中主动避免——这才是真正的 agentic 行为

如果确实需要"清理"过长的 observation，应该在**未来的新消息**中做摘要引用，而不是回去修改历史消息：

```
// ❌ 错误：修改第 5 轮的 observation，把 10KB 网页缩减为摘要
messages[5].content = "摘要：..."  // 破坏 Cache

// ✅ 正确：保留原始 observation，在第 10 轮的 user message 中追加摘要
messages.append({role: "user", content: "之前第 5 轮获取的网页核心信息是：..."})
```

**核心原则：永远追加，永远不修改已有内容。**

---

### 杀手五：会话中切换模型

**症状**：在同一个对话中切换模型（比如从 Opus 切到 Haiku），Cache 全部失效。

原因很简单：Cache 是 **model-specific** 的。不同模型的权重不同，同样的 token 序列计算出的 KV 矩阵完全不同，无法复用。

这里有一个反直觉的推论：

> **在 100K token 的长对话中，切换到更便宜的模型可能反而更贵。**

算一笔账：Opus 的 100K cached input token 只需 $0.30/MTok × 100K = $0.03。换成 Haiku 后，100K token 全部 Cache Miss，按 Haiku 的全价算（虽然单价低，但量大），加上重新 Prefill 的延迟成本——综合下来可能更贵。

**对策：单次会话内不切换模型，用子代理隔离。**

三家的做法：

- **Claude Code**：主 Agent 用 Opus，需要轻量探索时 fork 出 Haiku 子代理（Sub-Agent Handoff），子代理有自己独立的 context
- **OpenAI Codex**：同一对话保持同一模型，不切换
- **Manus**：任务级路由——Claude 做代码、Gemini 做多模态、OpenAI 做数学——不同任务不同模型，但**单次对话内不变**

---

### 杀手六：非确定性序列化

**症状**：同样的工具调用结果，两次请求序列化出来的 JSON token 序列不一样，Cache 失效。

这是最隐蔽的杀手。Manus 踩过这个坑，纪一超在博客中明确警告：

> "Ensure your serialization is deterministic. Many programming languages and libraries don't guarantee stable key ordering when serializing JSON objects, which can silently break the cache."
> （确保你的序列化是确定性的。很多编程语言和库不保证 JSON 对象的 key 排序稳定，这会悄无声息地破坏 Cache。）

问题出在哪？Python 的 `json.dumps()` 默认不保证 key 的排序。同一个 dict，两次序列化可能输出不同的 key 顺序：

```python
# 第一次
{"name": "张三", "age": 30}

# 第二次（可能）
{"age": 30, "name": "张三"}
```

从语义上完全一样，但从 token 序列上完全不同。Cache 废了。

**对策：统一序列化函数，保证确定性输出。**

```python
# ✅ 始终使用 sort_keys=True
json.dumps(data, sort_keys=True, ensure_ascii=False)
```

不只是 JSON，任何涉及到 Prompt 构建的序列化操作，都必须保证确定性：

- JSON key 排序
- 工具参数排序
- 列表元素排序（如果语义上无序的话）
- 浮点数格式化（避免精度差异）

**核心原则：序列化必须是确定性的。相同语义 → 相同 token 序列。**

---

### 六大杀手速查表

| # | 杀手 | 一手来源 | 对策 | 一句话 |
|---|------|----------|------|--------|
| 1 | 时间戳放开头 | Manus 博客（纪一超） | 动态信息后置到 user message | 稳定内容放前面 |
| 2 | 动态增删工具 | Thariq + Manus 博客 | 三种 Cache-Safe 工具管理方案 | 定义不变，控制可选 |
| 3 | MCP 工具顺序不确定 | Codex 实战经验 | Client 侧排序稳定化 | 注入前先排序 |
| 4 | 修改/删除历史消息 | Manus 博客（纪一超） | 严格 Append-only | 永远追加不修改 |
| 5 | 会话中切换模型 | Cache model-specific 基本事实 | 子代理隔离 | 单会话单模型 |
| 6 | 非确定性序列化 | Manus 博客（纪一超） | 统一序列化函数 | 相同语义→相同 token |

---


拆解完六大杀手，接下来我们做一次全行业扫描：从 Claude Code 到 Gemini CLI，从 LangGraph 到 CrewAI，各家 Agent 产品和框架在 Prompt Cache 上到底做了什么？

## 行业全景：谁在认真对待 Prompt Cache？

前面几节讲了原理和通用策略。这一节我们做一次全行业扫描：从 Claude Code 到 Gemini CLI，从 LangGraph 到 CrewAI，各家 Agent 产品和框架在 Prompt Cache 上到底做了什么？

扫描完之后，你会看到一个令人惊讶的分野：**做 Agent 产品的团队把 Cache 当命，做 Agent 框架的团队几乎没碰 Cache。** 这个鸿沟本身就说明了 Prompt Cache 在当前行业中的位置——它还没有成为共识，但已经是头部玩家的核心竞争力。

---

### 第一梯队：Cache-Aware 原生架构

这四家的共同特征是：**Prompt Cache 不是事后优化，而是从第一天就影响了整个系统架构的设计。**

### Claude Code（Anthropic）

Cache 意识最强的 Agent 产品，没有之一。

LMCache 团队对 Claude Code 做了详细的 trace 分析（2025.12），发现整体 **prefix 复用率高达 92%**，子代理的 ReAct 循环中更高达 97%。这不是刻意优化的结果，而是架构本身就为 Cache 而生。

**核心设计：**

**四层 context 架构，按稳定性递减排列。** tools（最稳定）→ system prompt → CLAUDE.md（项目级） → 对话历史（持续增长）。越稳定的内容越靠前，越动态的内容越靠后。

**自动 cache_control 断点管理。** Claude Code 自动在每个 user/assistant message 的最后一个 content block 放置 cache 断点。Thinking blocks 明确排除在外。开发者无需手动配置，也无法直接配置——这是一个经过深思熟虑的产品决策。

**Cache-Safe Compaction。** 当对话接近 context window 上限需要压缩时，压缩请求复用父会话的完整 prefix（system prompt + tools + 对话前缀），只在末尾追加压缩指令。从 API 的视角看，压缩请求和正常请求几乎一样——所以压缩本身也能命中 Cache。

**工具管理的两个创新。** 一是 `defer_loading`——大量 MCP 工具先注册轻量 stub（只有名字和简介），提供 `ToolSearch` 工具让模型按需检索完整 schema。二是"状态转换工具化"——Plan Mode 不是通过切换工具列表实现的，而是通过 `EnterPlanMode` / `ExitPlanMode` 两个工具实现的。工具列表始终完整不变。

**Cache 命中率是生产级指标。** Thariq（Claude Code 团队）在公开演讲中说过："工具变更是最常见的 cache 破坏方式"，cache 命中率下降被视为 **production incident**（生产事故），需要立即排查和修复。

### OpenAI Codex / GPT-5.2-Codex

API 层面对 Cache-Safe 设计支持最完善。

**自动 Prompt Caching。** ≥1024 token 的前缀自动缓存，按 128 token 块匹配。无需开发者配置，无额外费用。OpenAI 的路由系统会将相同前缀的请求路由到已缓存的服务器。

**allowed_tools：最优雅的工具管理方案。** 这是 OpenAI 在 Prompt Cache 方面最重要的 API 设计。tools 数组永远完整不变（保 Cache），通过 `tool_choice.allowed_tools` 限制当前可用子集：

```json
{
  "tools": [...全部工具定义，永远不变...],
  "tool_choice": {
    "type": "allowed_tools",
    "tools": [
      {"type": "function", "name": "get_weather"},
      {"type": "function", "name": "search_docs"}
    ]
  }
}
```

OpenAI 官方 Prompt Caching 201 Cookbook 明确这样描述：

> Leverage allowed_tools tool_choice option that lets users restrict the tools the model can call for a request without changing the tools array and busting the cache.
> （利用 allowed_tools 选项限制模型可调用的工具，同时不改变 tools 数组、不破坏 cache。）

**context_management compaction。** GPT-5.2 新增原生压缩支持，设置 `compact_threshold` 自动触发：

```python
response = client.responses.create(
    model="gpt-5.2-codex",
    input=conversation,
    context_management=[{"type": "compaction", "compact_threshold": 100000}]
)
```

**Cookbook 中的其他最佳实践：** metadata（如 timestamp、request ID）不要放在 prompt 里（放到 API 的 metadata 字段），动态内容放末尾，工具定义和 schema 的排序不要变。

### Manus

Cache 命中率被定义为"单一最重要的指标"。

纪一超（Manus CEO）在官方博客中系统性地阐述了 Cache-Aware 设计。核心策略在前面章节已经详细讨论过，这里提炼关键差异：

**Logits masking 方案。** 与 Claude Code 的 defer_loading 和 Codex 的 allowed_tools 不同，Manus 采用了更底层的控制——所有工具始终在 Prompt 中，通过推理时的 logits masking 控制模型实际能调用哪些。这需要 self-hosting 推理引擎，对调用商业 API 的开发者不适用。

**确定性序列化的明确警告。** 纪一超在博客中专门强调了 JSON key 排序不稳定会"悄无声息地破坏 cache"——这是很多开发者踩过但不知道原因的坑。

**MCP 工具通过 CLI 执行。** Manus 不把 MCP 工具绑定到 tools 数组，而是通过 Bash 工具在沙盒中执行 MCP 的 CLI 接口。这样 tools 数组保持精简（< 20 个原子工具），能力范围却无限——工具列表稳定，保 Cache。

**任务级模型路由。** Claude 做代码、Gemini 做多模态、OpenAI 做数学——不同任务不同模型，但单次对话内不切换。从 Cache 角度看，每个模型都在自己的对话中保持前缀稳定。

### Gemini CLI

利用 Gemini API 的隐式 + 显式双模缓存体系。

**隐式缓存（Implicit Caching）。** Gemini 2.5+ 模型默认开启。API 自动缓存对话的前缀部分，后续请求只需发送新增的 turn。开发者无需任何配置即可享受 **90% 的 token 折扣**。

**显式缓存（Explicit CachedContent）。** 开发者可以创建命名的 cache 对象，设置 TTL（默认 60 分钟），跨请求通过 cache name 引用。适合大型静态文档（如代码库、规范手册）的反复查询。最低 4096 token。

```python
cache = client.caches.create(
    model="gemini-2.5-flash",
    config=types.CreateCachedContentConfig(
        display_name="project_docs",
        system_instruction=instruction,
        ttl=3600  # 1 小时
    )
)
```

**每次请求重建完整对话历史。** Gemini CLI 的 GeminiChat 类每次都发送完整的 conversation history。这看起来低效，但配合隐式缓存，API 端自动识别相同前缀并复用——实际效果和 Claude Code 的 append-only 模式类似。

**GEMINI.md 层级发现。** 类似 CLAUDE.md，但支持层级化加载——项目根目录、子目录都可以有自己的 GEMINI.md。每次启动时从头扫描文件系统构建 context（防止 stale data），会话中可通过 `/memory refresh` 命令刷新。

**一个有趣的差异：TTL 模型。** Gemini 的显式缓存有明确的 TTL 和存储费用——你为缓存的 token 按时间付费。这和 Anthropic/OpenAI 的"5 分钟免费自动过期"模型不同，适合需要长时间保持缓存的场景（比如全天候的客服系统）。

---

### 第二梯队：应用层缓存——框架的 Cache 盲区

扫描完四家 Agent 产品，我们再看看主流 Agent 框架的情况。结果令人意外：**几乎没有一个主流框架在 KV Cache / Prompt Cache 层面做了设计。**

### LangGraph（LangChain）

**节点级计算缓存，不是 Prompt Cache。**

LangGraph 在 2025 年 5 月推出了 Node-level CachePolicy，支持基于节点输入的结果缓存——如果相同输入的节点已经执行过，直接返回缓存结果。支持 TTL 和多种后端（Memory、SQLite）。

但这是**计算结果缓存**，不是 **LLM Prompt Cache**。LangGraph 不控制发送给 LLM 的 prompt 结构，不管理前缀稳定性，不处理工具列表排序。Cache 效果完全依赖底层 LLM provider 的自动缓存能力。

社区也反馈了相关问题：LangChain 的 `set_llm_cache` 在 LangGraph 中不总是按预期工作，GitHub Discussion #1230 中有多位开发者报告了这个问题。

### CrewAI

**工具结果缓存 + 角色稳定性间接有利。**

CrewAI 的所有工具内置 caching 支持，通过 `cache_function` 属性提供细粒度控制。这可以避免重复调用外部 API（比如同一个搜索查询不会执行两次）。

CrewAI 的角色化设计（每个 agent 有固定的 role/goal/backstory）间接有利于 Prompt Cache——因为 agent 的 system prompt 天然稳定。但框架本身**不管理 prompt 的 prefix 结构**，不保证工具定义排序，不处理动态信息位置。

### AutoGen / Microsoft Agent Framework

**请求级去重缓存。**

AutoGen 支持基于 disk/Redis 的 API 请求缓存——相同的 LLM 请求直接返回之前的结果。多个 agent 可以共享同一个 cache store。

Microsoft Agent Framework（AutoGen + Semantic Kernel 的统一继任者，2025.10 发布 Preview）主要关注 multi-agent orchestration、MCP/A2A 协议集成、Azure AI Foundry 部署。**未发现任何专门的 Prompt Cache 文档或策略**——依赖底层 Azure OpenAI / Foundry 的自动 caching。

---

### 产品 vs 框架：Cache 意识的鸿沟

把上面的调查结果整理成一张表：

| 系统 | 类型 | Cache 层面 | Prefix 稳定设计 | 工具管理策略 | 压缩策略 |
|------|------|-----------|----------------|-------------|---------|
| **Claude Code** | 产品 | KV Cache (Prompt Cache) | ✅ 四层架构 | defer_loading + 工具化状态 | Cache-Safe Compaction |
| **OpenAI Codex** | 产品 | KV Cache (Auto Caching) | ✅ 自动前缀匹配 | allowed_tools | /responses/compact |
| **Manus** | 产品 | KV Cache (Provider) | ✅ 确定性序列化 | Logits masking + CLI 卸载 | Raw→Compact→Summarize |
| **Gemini CLI** | 产品 | KV Cache (Implicit + Explicit) | ✅ 完整历史重建 | Bash + GEMINI.md | 隐式自动管理 |
| **LangGraph** | 框架 | 节点级计算缓存 | ❌ | 不涉及 | 不涉及 |
| **CrewAI** | 框架 | 工具结果缓存 | ⚠️ 角色间接稳定 | 不涉及 | 不涉及 |
| **AutoGen** | 框架 | 请求级去重 | ❌ | 不涉及 | 不涉及 |
| **MS Agent Framework** | 框架 | 无专门策略 | ❌ | 不涉及 | 不涉及 |

这个表格揭示了一个清晰的分野：

**Agent 产品团队**（Claude Code、Codex、Manus、Gemini CLI）在 KV Cache / Prompt Cache 层面做了深度架构设计——前缀稳定、工具管理、Cache-Safe 压缩、确定性序列化。对他们来说，Cache 命中率直接决定了产品的成本和延迟，是生死攸关的指标。

**Agent 框架**（LangGraph、CrewAI、AutoGen、MS Agent Framework）主要做应用层缓存——计算结果缓存、请求去重、工具输出缓存。它们把 Prompt Cache 的责任完全留给了底层 LLM provider 和开发者自己。

**为什么会有这个鸿沟？**

一个可能的解释：Agent 产品团队自己承担 API 成本和延迟后果——Claude Code 每天处理海量 token，Cache 命中率每下降 1% 都意味着真金白银的损失和用户体验的劣化。而 Agent 框架的开发者感知到的是"模型 API 变贵了"或"响应变慢了"，但很难把问题归因到框架的 prompt 结构设计上。

**这对开发者意味着什么？** 如果你在用 LangGraph、CrewAI 或 AutoGen 构建 Agent，**你需要自己承担 Cache-Safe 设计的责任**。框架不会帮你做这件事。具体来说：

1. **自己管理 prompt 的前缀稳定性**——确保 system prompt 和工具定义不在请求间变化
2. **自己做工具列表排序**——`sorted(tools, key=lambda t: t["name"])` 
3. **自己确保确定性序列化**——`json.dumps(data, sort_keys=True)`
4. **自己设计压缩策略**——不要依赖框架的默认行为

---

### 学术验证：PwC 的系统性评估

2026 年 1 月，普华永道（PwC）的研究团队发表了论文 *"Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks"*——据我所知，这是第一篇系统性评估 Agent 场景 Prompt Cache 策略的学术论文。

**实验设计：**
- 跨三家 LLM provider（OpenAI、Anthropic、Google）
- 对比三种缓存策略：全量 Cache、只 Cache System Prompt、排除动态 Tool Results
- 在 DeepResearch Bench（多轮 agentic 搜索基准）上评估
- 消融实验覆盖 500-50000 token prompt 和 3-50 tool calls

**核心发现：**

**1. Cache 能降低 API 成本 41-80%，TTFT 改善 13-31%。** 这个数据和各家 provider 的官方折扣比例一致，验证了在真实 agentic workload 中 Cache 确实能带来显著收益。

**2. System Prompt Only Caching 在 cost + latency 两个维度最稳定。** 这个结论非常实用——如果你不想做复杂的 Cache 架构设计，至少确保你的 System Prompt 被稳定缓存。

**3. Naive full-context caching 反而可能增加延迟。** 这是最反直觉的发现。原因是：动态 tool 结果会触发 cache write（写入成本）但在后续请求中不会被复用（因为 tool 结果每次都不同）。你在为永远不会被命中的 cache 付出写入代价。

**4. 最大化成本节省的策略 ≠ 最大化延迟改善的策略。** 不同的优化目标可能需要不同的 Cache 策略。论文建议开发者根据自己的优先级（成本 vs 延迟）选择策略。

这篇论文的价值在于：它用受控实验证实了 Manus、Claude Code、Codex 团队在实战中摸索出来的经验——**策略性的 Cache 边界控制优于天真的全量缓存**。

---

### 共性模式：四家产品的 Cache 设计共识

虽然四家产品的实现细节各不相同（Claude Code 用 defer_loading、Codex 用 allowed_tools、Manus 用 logits masking、Gemini CLI 用隐式缓存），但它们在架构层面达成了惊人的共识：

**共识一：稳定前缀，动态后置。**

所有产品都把稳定内容（system prompt、工具定义）放在 context 最前面，把动态内容（时间戳、环境状态、用户消息）放在最后面。这是前缀匹配约束的直接推论。

**共识二：工具定义永远不变。**

没有一家产品通过增删工具列表来"优化" context。它们用各自的方式（API 参数、logits masking、defer_loading、CLI 卸载）限制模型的可选范围，但 tools 数组本身保持稳定。

**共识三：Append-only 历史。**

所有产品都不修改、不删除历史消息。错误的 tool 调用结果被保留（帮助模型学习），过时的内容通过追加摘要引用来处理。

**共识四：文件系统作为外部记忆。**

Claude Code 用 glob/grep，Manus 用沙盒文件系统，Gemini CLI 用 GEMINI.md 层级。它们都不用向量数据库做运行时检索——文件系统的零索引延迟和精确匹配更适合 Agent 场景。

**共识五：压缩必须 Cache-Safe。**

Claude Code 的 Cache-Safe Compaction、Codex 的 /responses/compact、Manus 的 Full→Compact→Summarize 三层策略，都确保压缩操作本身不破坏 Cache 前缀。

---

> 1. **行业存在清晰的 Cache 意识鸿沟**：Agent 产品（Claude Code、Codex、Manus、Gemini CLI）把 Cache 当成架构核心；Agent 框架（LangGraph、CrewAI、AutoGen）把 Cache 留给开发者自己处理。
>
> 2. **如果你在用框架构建 Agent，Cache-Safe 设计是你的责任**：框架不会帮你做前缀稳定、工具排序、确定性序列化。这些"小事"可能占你 API 成本的 80%。
>
> 3. **四家产品的五大共识**是经过实战验证的最佳实践：稳定前缀、工具不变、append-only、文件系统记忆、Cache-Safe 压缩。不管你用什么框架，都可以参照实现。
>
> 4. **PwC 论文的关键发现**：naive full-context caching 反而可能增加延迟。至少确保 System Prompt 被稳定缓存，是投入产出比最高的优化。
>
> 5. **allowed_tools（OpenAI）是目前最优雅的 API 设计**：一个参数同时解决了工具管理和 Cache 稳定两个问题。如果你在用 OpenAI API 构建 Agent，这是第一个应该采用的特性。

---

下一篇我们将拉高视角，聊 context 管理的完整策略——Context Rot 是什么、三层压缩怎么做、文件系统如何成为 Agent 的第二大脑，以及从 Prompt Engineering 到 Context Engineering 的范式转变。

---

**「Agent 工程师的 Prompt Caching」系列导航**

1. [KV Cache 原理：LLM 推理的底层机制](/posts/prompt-caching-kv-cache-fundamentals/)
2. [Prompt Cache：Agent 成本控制的核心约束](/posts/prompt-caching-core-constraints/)
3. **Cache 杀手与行业实战：从踩坑到最佳实践**（本文）
4. [Context Engineering：Agent 架构师的核心手艺](/posts/prompt-caching-context-engineering/)
