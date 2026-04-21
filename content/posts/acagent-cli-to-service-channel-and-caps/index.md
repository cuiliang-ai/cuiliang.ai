---
title: "[Agent CLI→Service 2/2] Channel、资源限制,和我没做的那些事"
date: 2026-04-21T11:00:00+08:00
draft: false
summary: "Service Mode 改造下篇:Channel 推广到 1:N 的 AttachManager、5 层 Resource Cap 的职责切分、错误信封 / observability / CLI≡serve,以及三类我没做的事。"
description: "ACAgent 从 CLI 到 Service 改造的工程层细节:AttachManager 慢 attacher 驱逐、Cap 哨兵值与 -1 语义、CLI 内部 uvicorn,以及 trade-off 自陈。"
tags: ["AI Agent", "Architecture", "Service Mode", "Resource Cap", "Observability", "ACAgent"]
categories: ["AI Agent Engineering"]
series: ["Agent CLI 到 Service 改造复盘"]
ShowToc: true
TocOpen: true
---

> 📌 **本文是「Agent CLI 到 Service 改造复盘」系列的第 2/2 篇**。[上篇](/posts/acagent-cli-to-service-three-walls/)讲三次认知转变:会话 ≠ 进程、事件流 ≠ 输出、持久化 ≠ 存档。本篇展开剩下的工程层细节和 trade-off 自陈。

> 比上篇更具体、更密集。如果你也在做类似改造,这篇里的坑和数字可能直接能抄。

## 开场

[上篇](/posts/acagent-cli-to-service-three-walls/)讲完三堵墙之后,Service Mode 的骨架算是立起来了:Runner 是会话的载体、Bus 是事件的真相、JSONL 是事件的持久化。但骨架上还要挂几样东西才能真跑起来:

- Channel 要从"只服务一个终端"扩展到"服务 N 个 attacher"
- 要有真正意义上的资源限制(CLI 不需要,Service 必须有)
- 错误要能被客户端机读
- 服务要能被观察、能被健康检查、能被另一个进程发现
- 用户视角上得消除"CLI 模式 vs Service 模式"的二分

这些单独拎出来都不是大题目,但缺一个就有一个场景跑不动。这篇按实装顺序讲。

最后还有一节是 trade-off 自陈,我决定不做的、想做没时间做的、不确定要不要做的。这一节在上篇故意省掉了,只有把实装细节都讲完,读者才有判断力看我的取舍是否合理。

## 4. Channel:从 1:1 推广到 1:N

> **Channel 在 CLI 里是 stdin/stdout 的抽象,1:1 直连。到 Service 里它必须是"N 个 attacher 订阅同一个 session"的 fan-out 抽象 —— CLI 只是 N=1 的退化情形。**

### 我一开始怎么做

CLI 里 `Channel` 的角色很朴素,就是一个"把 Loop 的输出变成人眼能看的东西"的适配器,`CLIChannel.stream_text()` 直接 `print()` 就完了。

到 Service 我最初的想法也简单:给 Runner 加一个 `channel` 字段,Dashboard 连进来的时候 "替换" 掉这个 channel。

### 撞到的墙

两个很快就暴露的问题:

**问题 1**:Dashboard 断开之后,CLI 终端再也收不到输出。原来我真的把 channel 换掉了,CLI 那个 `print` channel 被顶掉没恢复。

**问题 2**:两个 Dashboard 同时 attach 时,只有后连上来的那个收到事件。原来 channel 是"替换"不是"追加"。

第二个问题我试过改成"channel 链表",让 stream_text 依次调所有 channel 的方法。几天后又撞到第三个:

**问题 3**:Dashboard 那个浏览器 tab 我切到后台,Chrome 把它的 timer / 网络回调 throttle 了,WS 那一端 ack 变得忽快忽慢。我的 channel 链里逐个 await,只要其中一个没及时回,整个 agent 的 turn 就卡在那等,直到 socket 超时。

一个慢 attacher 能拖死整个进程。

### 我怎么想

盯着"替换 channel → 链表 channel → 慢 attacher 拖死"这条演进轨迹看,发现我一直在给错的抽象打补丁。

Channel 这个概念承担了两个职责:

1. **事件产生**(Loop 内部用 `channel.stream_text()` 发事件)
2. **事件消费**(终端/WS 把事件渲染给人看)

在 CLI 里这俩合一了(Loop 直接写到终端),但 Service 里必须拆开。而上篇的 EventBus 其实已经把职责 1 剥离出来了。所以 **Channel 应该退化成"事件消费端的适配器"**,它的实例可以有 N 个,不再是 Loop 的私有字段。

引入新概念:**Attacher**。一个 Attacher 就是一个消费端,有自己的类型(terminal / ws)、模式(editor / observer)、自己的事件队列。Channel 是 Attacher 的一种实现。

### 改成什么

```python
@dataclass
class Attacher:
    id: str
    type: Literal["terminal", "ws"]
    mode: Literal["editor", "observer"]
    queue: asyncio.Queue
    dropped_count: int

class AttachManager:
    def attach(self, session_id: str, attacher: Attacher) -> None: ...
    def detach(self, session_id: str, attacher_id: str) -> None: ...
    def broadcast(self, session_id: str, event: dict) -> int: ...
    def attachers(self, session_id: str) -> list[Attacher]: ...
```

CLI 终端不再特殊,它也是一个 Attacher,`type="terminal"`。一套代码路径同时走通 CLI 和 WS。

### 几个要注意的细节

**a) broadcast 用 `put_nowait` + 驱逐**

慢 attacher 拖死进程那个坑,解法和 EventBus 完全一样(这不是巧合,fan-out 问题本质都是这样):

```python
def broadcast(self, session_id, event) -> int:
    delivered = 0
    to_evict = []
    for att in self.attachers(session_id):
        try:
            att.queue.put_nowait(event)
            delivered += 1
        except asyncio.QueueFull:
            att.dropped_count += 1
            if att.dropped_count >= self.slow_drop_threshold:
                to_evict.append(att.id)
    for att_id in to_evict:
        self.detach(session_id, att_id)
    return delivered
```

`slow_drop_threshold` 我设的 64,连续 drop 64 个事件就踢,大约对应"这个 attacher 彻底跟不上了"。⚠️ **这个数字是拍脑袋的,目前没做过真实负载测试,生产部署前应该根据实际事件速率重算。**

**b) detach 时推 close sentinel**

和 EventBus 同款机制:被踢的 attacher 必须收到 `None` 哨兵,否则它的消费循环永远 hang 在 `queue.get()` 上。这个我第二次踩才记住。

**c) 输入广播的语义选择**

多个 attacher 同时输入怎么办?这是个**产品决策**不是技术决策:

| 策略 | 优点 | 缺点 |
|---|---|---|
| Last-writer-wins | 简单,符合直觉 | 可能 race 丢字 |
| Leader election | 公平,唯一输入源 | 协议复杂 |
| Queue-and-merge | 无歧义 | 破坏 turn 边界,行为诡异 |

我选了 last-writer-wins,预期使用场景是"我在笔记本上用 CLI,然后手机 Dashboard 同时连上来",两个都是我自己。多人协作场景下"上一个发的覆盖之前的"符合直觉。Observer 模式的 attacher 直接拒绝输入,返回 `observer_readonly` 错误。

多人同时用同一个 session 互相抢 turn 的场景我没设计,真需要再改。

### 回头看

这堵"墙"其实是上篇墙 2(事件流)的副产品。**如果 EventBus 先想清楚了,AttachManager 的抽象就是水到渠成的**:一个 attacher = 一个 Bus 的订阅者 + 一个特定的输出适配器。

现在回看最满意的点是"CLI 终端也是一个 attacher",这让 CLI 模式和 Service 模式**不需要有两套代码**。上篇结尾那个反直觉结论(CLI 是 Service 的退化情形)在这里得到了最干净的体现。

## 5. Resource Cap:CLI 不需要,Service 必须有

> **Service 模式下用户可能离线、可能不在乎花了多少钱。必须有强制上限 —— 而且必须分层,每一层对应一个特定的失控场景。**

### 为什么 CLI 不需要

CLI 里你自己控制:钱花光了就停,时间长了 Ctrl-C。你是**唯一的使用者,也是唯一的付费者**,在场监督。

Service 模式下这两个"唯一"都没了:session 可能在你睡觉时跑、账单可能不是你付(内部工具给同事用)、进程可能服务多个 session 共享预算。强制上限不再是锦上添花,是**必须设的护栏**。

### Cap 的层级矩阵

做到一半我才反应过来单一 cap 不够,至少要 5 层:

| 层级 | 单位 | 作用 | 默认值 |
|---|---|---|---|
| per-call | LLM API 调用超时(秒) | 防 provider 卡死 | 60s |
| per-session | input + output token | 单会话失控保护 | None(默认关) |
| per-session | wall-clock(秒) | 长跑会话超时 | 3600s |
| per-instance | 并发活跃 session 数 | 进程容量保护 | 5 |
| per-instance / day | rolling 24h token | 成本保护(按账单周期) | None(默认关) |

每一层对应一个特定的失败模式。缺哪一层就有一个场景控制不住:

- 没 per-call → 一个卡死的 API 调用让整个 session hang
- 没 per-session token → 一个跑飞的 agent 可以烧几百刀
- 没 per-session wall-clock → agent 陷入无限工具调用没人拉停
- 没 per-instance 并发 → 同时 100 个 session 把 provider 打到 rate limit
- 没 daily cap → 账单月末爆炸

### 实装里的几个坑

**a) Cap 检查必须在 `Runner.step()` 入口,不能在 Loop 内部**

这个坑我踩过。一开始我把 token cap 检查放在 `AgenticLoop.run_turn()` 的 stream 循环里,超了就 raise。

结果用户看到的是**"半个回答 + 错误"**。stream 已经吐了一部分给客户端才 raise,前端显示一段截断的文字,然后红底错误。体验灾难。

改成在 `Runner.step()` 入口先检查:

```python
async def step(self, user_input: str):
    self._check_session_caps()       # 入口拦截
    if self._step_lock.locked():
        raise RunnerBusyError(...)
    await self._step_lock.acquire()
    return self._step_iter(user_input)
```

用户看到的是"请求被拒绝(429)",没有"半个回答"。**Cap 是 Runner 层的事,不是 Loop 层的事**。这也是 M6 阶段的一次重构,上篇讲 Runner ≠ Loop 的职责边界,这里是个具体应用。

**b) Daily cap 必须跨重启持久化**

Daily cap 如果只在内存里,进程一重启就清零,"每天 100 万 token" 的语义破产。

解法:状态文件 + atomic write + debounce:

```
~/.acagent/state/daily_token_cap.json
{"used": 234567, "window_started_at": 1729...}
```

三个细节:

- `add()` 之后标记 dirty,启动 1s 后的 debounced save(不是每次都同步写盘,IO 过重)
- 写到 `.tmp` 再 `os.replace`,atomic 保证半截文件不会污染 state
- 启动时加载,如果 `window_started_at` 过了 24h 自动重置

Debounce 是关键,没有 debounce 的话每次 token 累加都触发 fsync,agent 的热路径变得巨慢。

**c) 错误码细分:同样 429,客户端处理策略完全不同**

| 错误码 | 含义 | 客户端建议 |
|---|---|---|
| `session_busy` | 同 session 第二个请求撞锁 | 稍后重试(会立刻好) |
| `concurrency_cap` | 进程并发上限 | 稍后重试(等别的 session 结束) |
| `session_token_cap` | 单 session token 用尽 | 不要重试,开新 session |
| `session_wallclock_cap` | 单 session 时间用尽 | 不要重试,开新 session |
| `daily_cap_exhausted` | 全实例 daily cap 用尽 | 等到明天 |

全都是 HTTP 429,但客户端收到不同 `code` 该做的事完全不同。如果你用 HTTP code 表达原因(比如全部 429 然后让客户端从 message 里 grep 字符串),就是把契约塞到字符串里,早晚出事。

这就是下面错误信封的动机。

**d) Prometheus 指标的哨兵值**

`acagent_daily_tokens_remaining` 这个 gauge,我选择用 **-1 表示"未设 cap"**,不是 0。

```
acagent_daily_tokens_remaining 765433   # 有 cap:剩余量
acagent_daily_tokens_remaining -1       # 无 cap:哨兵值
```

如果用 0 表示"无限",Dashboard 看到 `remaining=0` 没法区分是"用完了"还是"没设 cap"。用 -1 作哨兵,PromQL 里 `< 0 or > threshold` 可以可靠分支。**metric 设计里的哨兵值往往决定了下游能不能写出干净的查询,这种细节不提前想,后面是 Grafana 里一堆 hack。**

### 回头看

Resource Cap 这部分改得不算优雅(5 层是分批加上去的,不是一次设计好的),但现在这 5 层我没想删掉任何一个。每一层都对应一个真实发生过的失控场景。

如果说这部分有什么可迁移的经验:**不要试图用一个"聪明"的 cap 覆盖所有失控场景。** 失控的形态不是一种,cap 的层级也不应该是一种。5 层不算多,每层的职责分得干净就行。

## 6. 配套基础设施(一笔带过)

剩下几件事单独都不够开一节,但没它们 Service Mode 也跑不起来。

### 6.1 错误信封

CLI 里出错用户看 traceback,Service 里出错客户端必须机读:

```json
{
  "error": {
    "code": "session_token_cap",
    "message": "Session sess-abc token cap exceeded: 105000 >= 100000",
    "details": {"session_id": "sess-abc", "used_tokens": 105000, "cap_tokens": 100000}
  }
}
```

三个原则:

- **错误码可枚举**,客户端 `switch (code)`,不 grep message
- **details 给结构化上下文**,UI 直接渲染"已用 X / 上限 Y"
- **HTTP code 表达类别,error.code 表达原因**:429 说"暂时不行",code 说"现在再试 vs 明天再试"

### 6.2 Observability 三件套(default OFF)

| 工具 | 作用 | 何时打开 |
|---|---|---|
| OpenTelemetry trace | 跨进程关联(HTTP → tool → sub-LLM) | 调性能问题 |
| Prometheus metrics | 容量 / 成本量化 | 要 SLO / 报警 |
| structlog 结构化日志 | 可检索复盘 | 事故复盘 |

**三件套都 default OFF**。纯 CLI 用户不该被逼着装 OTel 依赖。配置里 `observability.metrics_enabled = true` 才打开。这是工具型项目的礼貌:能力可用,但不强迫。

### 6.3 多实例服务发现

最朴素但够用的方案:

```
~/.acagent/instances/
├── 12345.json
├── 12346.json
└── 12347.json
```

每个 instance 启动写 `{pid}.json`,退出删除,运行时定期更新 heartbeat。Dashboard 扫这个目录就能列出所有本机 instance。

**零依赖、零运维**。缺点是跨机器只能靠共享 fs 或 Tailscale overlay,PID 复用时 stale 文件清理也脆弱。这些缺点接受,不值得为此引入 mDNS / Consul。

### 6.4 Ops 端点

| 端点 | 作用 |
|---|---|
| `GET /healthz` | k8s liveness |
| `GET /readyz` | k8s readiness(registry / scheduler / watchdog 都 up) |
| `GET /api/doctor` | 自检(provider 连通性、配置、optional extras) |
| `POST /api/shutdown` | 优雅 drain |
| `GET /metrics` | Prometheus pull |

加上 hot reload:watch 配置文件,非破坏性改动(log_level、cap、webhook)热更新,破坏性改动(端口、provider)仍需重启。

## 7. CLI ≡ serve:消除模式二分

前面所有事(Runner / Bus / JSONL / AttachManager / Cap / Ops 端点)都在回答"Service 模式怎么实现",但还剩一个产品层面的问题没答:**用户什么时候从 CLI 模式"切换"到 Service 模式?**

最直觉的答案是 `acagent serve` 是单独的子命令,跟 `acagent`(CLI)并列。但越想越别扭:这俩用的是同一套 Runner、同一套 Bus、同一套 JSONL,只是 attacher 的来源不同(终端 vs WS)。强行做成两个子命令,等于给用户造了一个本来不存在的认知负担。

最后落地的方案:**默认 CLI 启动就开一个内部 uvicorn**。

```bash
$ acagent
> 你好  # 用户看到的还是普通 REPL
```

但背后已经在 `127.0.0.1:随机端口` 跑了 FastAPI,端口写进 `~/.acagent/instances/{pid}.json`。Dashboard 立刻能 attach。`--no-serve` 显式关掉。

带来的变化:

- 用户视角:还是熟悉的 `acagent` REPL,什么都没变
- Dashboard 视角:任何时候列 instances 目录就有可 attach 的会话,不需要用户先"切到服务模式"
- 代码视角:**没有"CLI 模式 vs Service 模式"的代码分叉**,CLI 终端就是 N=1 的 attacher

这个设计把服务化从"功能开关"降级为"实现细节"。配合上篇结尾"CLI 是 Service 的退化情形"那个反直觉结论,这里给出工程对应物:**没有什么"改成服务模式",只有"是否允许别人 attach"**。

代价是默认起进程多占 1 个端口(~10MB 内存),以及一旦本机有多 instance 就要靠 InstanceRegistry 区分。这两个代价我都能接受。

如果让我从这两篇文章里挑一个最值得抄的设计,就是这个。

## 8. 我没做的那些事

Service Mode 很多"本该做"的事我没做。不是忘了,是主动选择不做,或者想做但排不上。这一节把这些坦白摊出来,帮读者判断这套设计在**你的场景**里够不够用。

分三类,每类姿态不一样。

### 8.1 我决定不做的(短期内不会做)

**真正的多租户 cap**

当前的 cap 是 per-session / per-instance / daily,没有 per-user。多用户场景下 Alice 和 Bob 共用一个 instance,daily cap 先到谁谁挤掉另一个。

为什么不做:ACAgent 的预期使用场景是**单用户**(我自己,或小团队每人自建 instance)。per-user cap 需要做用户认证、配额分配、账单归因,复杂度爆炸。做了反而让简单场景变复杂。

什么时候会重新考虑:用户数上到 5+,或者开始给不熟的人用。

**RBAC**

当前认证是 bearer token,全有全无。没有"只读观察" / "可以调工具但不能写文件"这种细粒度。

为什么不做:同上,单用户场景下你是自己的 root。

**真正的安全沙箱**

工具执行目前跑在 agent 进程里。恶意工具能碰到本机所有东西。

为什么不做:工具集合是我自己写的,不是第三方。如果有一天接入 MCP 或允许用户自定义工具,就必须上 sandbox 了。现在不做。

### 8.2 我想做但没时间做的

**Session live migration**

进程重启时所有 in-flight turn 丢失。JSONL 能恢复历史,但正在跑的那一轮(工具调用跑到一半)恢复不了。

为什么没做:要做 "turn-level checkpoint + 外部工具状态序列化",至少再写两周。没到这个紧急程度。

**跨进程 Runner**

`AgentRunner` 是 Protocol,`SubProcessRunner` 的接口预留好了但没实装。工具崩溃目前会把 agent 进程一起带走。

为什么没做:把工具跑在子进程要解决 IPC 协议、状态同步、超时回收一大堆事。当前工具集合稳定,没撞到这个问题。

**真实的负载测试**

几个拍脑袋的数字(queue `maxsize=256`、`slow_drop_threshold=64`、`max_concurrent=5`)都没做过压测。

为什么没做:没真正的生产部署需求。但这会是第一个上生产前必须补的功课。

### 8.3 我不确定要不要做的

**多实例聚合 cap**

各 instance 各自算 daily cap,多实例并行跑时总消耗可能超出"我以为的总预算"。

为什么犹豫:做的话要引入中心化状态(Redis / 共享 fs 的原子计数器),把"零运维"这个优点破坏了。目前的妥协是:单用户场景下多实例是低频行为,每个 instance 各自设的 cap 取 min 够用。不严格但没出过事。

到底该不该做?不确定。要看跨机器多实例的真实使用频率。

**多 attacher 的 HITL permission 路由**

当前 `ask_permission` 问"第一个能响应的 attacher"。多人同时 attach 时谁先看到谁先答,逻辑散落。

为什么犹豫:多人协作场景里 permission 路由是个小型产品决策。应该问谁?投票?leader?还是"session owner"专属?我没想清楚什么是**对的**,所以没做。

---

这三类的语气应该有区分:

- **决定不做**:有判断、有边界,不是懒惰
- **想做没时间**:坦承局限,给出条件
- **不确定要不要做**:承认自己没想清楚,这是最诚实的一类

同行读复盘文最反感两种:"我们把一切都做了"(吹)、"还有很多可以做的"(敷衍)。我尽量不说这两种话。

## 整个系列的收尾

两篇文章讲了 8 件事:上篇的 3 堵墙 + 这篇的 Channel / Cap / 配套 / 没做的。把它们抽象成一条线的话:

**Service 化的本质不是"加 HTTP wrapper",是把 CLI 里被进程隐式承担的所有抽象都显式化。**

每一个隐式抽象都对应一次改造:

| CLI 里的隐式 | Service 里的显式 |
|---|---|
| 进程 = 会话 | `AgentRunner` + `SessionRegistry` |
| print 即输出 | `RunnerEventBus` + 订阅者 |
| 退出时快照 | JSONL append-only + checkpoint |
| 单一终端 | `AttachManager` + N 个 Attacher |
| 用户控制成本 | 5 层 Resource Cap |
| traceback | 错误信封 |
| 不需要被看见 | `/healthz` / `/metrics` / `/doctor` |
| 单机单实例 | InstanceRegistry + CLI≡serve |

每一行都是一堵墙的产物。

如果上篇的反直觉结论是 **"CLI 是 Service 的退化情形"**,这篇给的补充是:**退化是自然的,倒推则要撞墙。** 如果你正打算从头设计一个允许 attach 的 agent,上面这张表的 8 行可以当 checklist,不必一墙一墙地撞再补。每一次撞墙后的认知转变本身,就是这类项目最有价值的部分,但不必每个人都重撞一遍。

---

## 附录

**ACAgent** 是一个 Python 写的个人 AI Agent 轮子,最开始的想法是学习 Claude Code 架构设计,目前逐渐演变为 agent 想法的一个 playground。

- [上篇:我撞到的三堵墙](/posts/acagent-cli-to-service-three-walls/)
