# 接入 Simulator

路径均以 Roboenv 仓库为根。设备侧接口见
[接入 Devices](adding-devices.md)，实际换装步骤见
[组合与切换](simulator-device-recipes.md)。

## 1. 职责边界

| 层 | 负责 | 不负责 |
|---|---|---|
| Simulator adapter | 创建场景/机器人，观测、控制、预算、benchmark 原始判定 | Agent 安装、LLM |
| WorldSession | 持有唯一后端实例、生命周期、共享执行锁 | 设备动作定义 |
| Binding | 把设备接口接到这个世界的实体/控制器 | 创建第二份世界或重置 episode |
| DeviceService | 发现、分发、统一结果/错误 | 了解具体物理引擎 |
| ScoreAuthority + verifier | 环境评分与证据检查 | 采信 agent 的成功声明 |

现有 EnvAdapter 包含 benchmark 接口，不是纯物理引擎抽象。
WorldSession 包装它；评分仍由环境权威调用，不属于 device。

## 2. 固定依赖

在 upstreams.lock.json 添加公开 URL、精确 commit、后端归属、依赖关系，
必要时添加带 SHA256 的受管理补丁。源码放 third_party/ 的 Git submodule。

    roboenv sources sync --backends BACKEND
    roboenv sources status --backends BACKEND

为重依赖准备独立解释器与资产。源码 pin、Python 包集合、GPU/渲染配置、
资产身份和最终镜像 digest 是不同层，不能用其中一个替代全部可复现性。
Roboenv 不自动接受资产再分发条款或下载所有模型。

## 3. 实现 adapter

接口：[core/adapter.py](../runtime/src/roboenv_runtime/core/adapter.py)。
最小可运行例子：[FakeShell](../runtime/src/roboenv_runtime/adapters/fakeshell.py)。

至少实现以下真实语义：

- name、embodiment：稳定身份；与实际世界匹配。
- reset(seed)、close()：确定性初始化及本实例资源释放。
- observe()、situation()、facts()、goal()：符合任务观测权限的状态与事实。
- budget()：benchmark 的真实有限预算，不沿用无界占位。
- score()：原有 benchmark 判定，返回 ScoreReport，不放宽成功阈值。
- capture()/frames()：有相机才提供；缺失能力要明确说明。
- tools()：可选兼容 primitive 表。直接实体 binding 可以完全不依赖它。

AdapterBase 的默认观测会查询约定 primitive；不提供这些 primitive 时，
必须自行实现 observe()。延迟导入 simulator 重依赖，不能在 SDK 导入时启动引擎。
使用 register_adapter 注册，在 BUILTIN_ADAPTERS 添加模块映射。

## 4. 接上 devices

提供已有实体和控制/观测接口，让 binding factory 解析 entity。
默认设备方案放 catalog/robots/ROBOT/bindings.yaml。

当前生产 catalog 使用五个后端专属 binding：
libero.device/1、robocasa.device/1、robotwin.device/1、
robodojo.device/1、behavior.device/1。
它们在 runtime/bindings 中直接调用 worker 注入的、已计费的控制器函数，
不再执行 catalog 里的 Python driver，也不连接回自己的 socket。

这是一层后端控制器适配，不是所有模拟器都已实现统一低级关节控制。
fakeshell.puck/1 是不使用 primitive 表的
直接实体示例。中立 DeviceService 不依赖以上任何具体绑定。

## 5. 接上任务发现与镜像

新增后端须同时完成：

1. tools/build_simulator_image.py 中解释器、源码和资产切片映射。
2. src/roboenv/tasks/catalog.py 中轻量 discoverer 和默认 robot。
3. 从上游真实 registry/config 提取 id、初始化参数、任务语言及来源哈希。
4. 每个生成定义能被 Task.load 验证，并通过真实 Harbor loader。
5. 缺失场景、配置或资源的任务明确排除，并记录原因。

不得通过导入一个会启动 GPU 的模块来列 task。不要推测任务数或编造场景。
批量 task 不入库；生成方法见 [Harbor tasks](harbor-tasks.md)。

## 6. 生命周期

    Harbor 启动 sim
      → adapter 初始化 episode
      → WorldSession 持有后端和锁
      → binding 校验并解析现有实体
      → DeviceService 注册完成
      → 健康检查通过
      → agent 开始

设备与兼容 primitive 共享世界、预算和锁。只读请求不能推进物理世界。
Owner reset/restore 先关闭旧 binding，再改变世界并重新绑定；
device.close() 不得关闭整个世界。

## 7. 验收与交付

    bash tools/check.sh

还必须增加后端专项测试：缺失资产失败、错误 embodiment/entity 被拒绝、
控制和观测单位正确、预算计费不重复、失败不伪装为成功、reset/close 不泄漏。
运行真实 Docker Harbor task，并用保存程序完成 fresh-process replay。

提交时分别说明源码可获取、格式通过、容器启动、任务完成和 replay 认证覆盖。
CPU fixture 通过不能替代 GPU 证据；当前覆盖见 [质量审计](quality-audit.md)。
