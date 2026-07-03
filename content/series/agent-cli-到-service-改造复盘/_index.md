---
title: "Agent CLI 到 Service 改造复盘"
description: "把一个跑在命令行里的 Agent 改造成服务，听起来只是套层 API，实际撞到的是并发、Channel、资源限制这些真问题。这是一次真实改造的复盘：撞到的三堵墙，和我最后没做的那些取舍。"
params:
  status: complete
weight: 40
---

CLI Agent 改成 Service，最难的不是写接口，而是那些在单机命令行下从不出现、一上服务就全冒出来的问题。这个两篇复盘记录了真实踩过的坑和做过的权衡——包括那些明明想做、最后决定不做的部分。

**推荐阅读顺序：**

1. [Agent CLI→Service 1/2] 我撞到的三堵墙 —— 改造中最难的三个问题
2. [Agent CLI→Service 2/2] Channel、资源限制，和我没做的那些事 —— 架构取舍与克制
