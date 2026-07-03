---
title: "Agent 工程师的 Prompt Caching"
description: "Prompt Caching 不只是省钱的开关，它是 Agent 成本结构和上下文工程的底层约束。这个系列从 KV Cache 的推理原理讲起，到 Agent 成本控制的核心约束、常见的 cache 杀手，再到缓存如何反过来塑造整个上下文工程。"
params:
  status: complete
weight: 20
---

对 Agent 工程师来说，Prompt Caching 是绕不开的一课：它决定了你的成本曲线，也决定了 context 该怎么组织才不会把缓存打穿。这个系列从底层机制讲到工程实战，最后把它和 Context Engineering 串成一条线。

**推荐阅读顺序：**

1. KV Cache 原理：LLM 推理的底层机制 —— 缓存为什么存在
2. Prompt Cache：Agent 成本控制的核心约束 —— 它如何决定成本结构
3. Cache 杀手与行业实战：从踩坑到最佳实践 —— 哪些写法会打穿缓存
4. Context Engineering：Agent 架构师的核心手艺 —— 缓存视角下的上下文工程
