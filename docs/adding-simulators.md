# 接入新的 Simulator

新接入按 [DESIGN.md](../DESIGN.md) 的设备服务设计实现。RoboCasa 是参考实现；
旧后端的 primitive API、ground-truth 观测和旧 driver 生成方式不是新接入规范。
配置编写见 [接入 Devices](adding-devices.md)，改动范围见
[组合与切换](simulator-device-recipes.md)。

## 1. 确定任务、硬件与观测边界

先查上游任务 registry、机器人模型、控制器配置、相机定义和成功判定。
明确每个目标 embodiment 的设备、动作空间、单位、控制频率、frame 与预算。
相同机器人和传感器配置应跨任务复用；场景布局、物体位置、seed 不是设备配置。
不同硬件或相机 arrangement 使用不同 robot ID，不靠 launch 重写同一配置。

新后端默认 `sensors-only`：agent 获得任务指令、设备事实和传感器读数，
自行感知、估计和规划。不得把物体坐标、fixture 几何、layout/style、scene seed
放进 task context、world profile 或新增的 scene device。机器人本体状态和
FK/IK 属于设备读数；读取任意 simulator state 不属于公开接口。
特殊 benchmark 若明确要求特权观测，必须单独定义、标记和验证观测政策。

| 组件 | 职责 |
|---|---|
| Adapter | 初始化/关闭世界，原生控制支持，原 benchmark 预算与判定 |
| WorldSession / worker | 唯一世界、执行锁、生命周期、预算和证据 |
| Native binding | 连接已有实体，实现基本设备操作，不创建第二个世界 |
| DeviceService / supervisor | 校验、分发、身份、去重、HTTP/owner 边界 |
| Complex / 其他 harness | perception、planning、反馈控制程序、模型调用 |
| Harbor | trial 创建、资源调度、agent/verifier 启动与清理 |

## 2. 锁定源码与准备独立运行环境

在 `upstreams.lock.json` 登记公开来源、精确 revision、后端归属和依赖，
源码以选定的 `third_party/` submodule 管理。修改上游前先执行：

```sh
roboenv sources check-updates --backends BACKEND
roboenv sources sync --backends BACKEND
roboenv sources status --backends BACKEND
```

按照项目规则审核 upstream 更新、维护带校验和的补丁；不在 submodule 内提交，
不依赖未记录的 checkout 修改。不要递归下载所有后端。

接入 `runtime/environments.json` 的安装 recipe、对应 lock/资产获取步骤和
`src/roboenv/provisioning/` 的准备/诊断流程，使下面的操作可用：

```sh
roboenv prepare BACKEND
roboenv doctor BACKEND
```

开发/Harbor 环境与 simulator 解释器隔离。源码 revision、Python lock、资产身份、
渲染驱动要求和镜像 digest 都要明确，不能用一个版本号替代全部。不要把所有
模拟器依赖安装进 SDK 环境。列任务、读配置、导入 SDK 均不得启动 GPU 或下载模型。

## 3. 实现最小 adapter 与 native binding

参考 [RoboCasa adapter](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py)、
[backend](../runtime/src/roboenv_runtime/adapters/robocasa/backend.py) 和
[native devices](../runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py)。
基类在 [core/adapter.py](../runtime/src/roboenv_runtime/core/adapter.py)。

Adapter 提供稳定的 `name`/`embodiment`、`reset(seed)`、`close()`、任务指令
`goal()`、真实有限 `budget()`、原始 `score()`，以及所需的 owner 观测/证据支持。
Override 基类依赖旧 primitive 的默认方法；新接入不需要新增 `tools()` 宏操作表。
评分可以读取特权状态，但它与 agent 传感器读取必须分开。

Binding 直接调用当前世界的实体和原生 servo/controller。每次 write 的输入
先验证，再经统一 execution 入口产生效果并计费。一个命令只推进一次声明的
原生控制区间；未被控制的执行器使用明确的 neutral/hold/expiry 规则。
read、status、发现、录制和 IK/FK 不推进时间。设备关闭不关闭世界。

在 adapter registry/`BUILTIN_ADAPTERS` 注册后端，并在
[runtime binding registry](../runtime/src/roboenv_runtime/bindings/__init__.py)
显式注册 binding ID。不要让配置指定任意 Python import，不要从 binding
通过 socket 回调正持锁的本 worker，也不要新增第二套 driver/runtime。

## 4. 预生成可直接复制的机器人配置

```text
catalog/robots/ROBOT/
  workspace.yaml
  devices/NAME/profile.yaml
  devices/NAME/driver.py      # world 除外
catalog/bindings/ROBOT.yaml
catalog/tasks/TASK/task.yaml
```

robot 根目录只有 `workspace.yaml` 与 `devices/`，供直接复制到 Complex。
Binding plan 是 RoboEnv 私有组装信息，不属于 Complex 配置。README、来源 receipt
放在 `docs/`；生成器放在 `tools/`；测试用的额外 profile 放在 fixtures。

用上游模型/控制器/相机定义预生成稳定参数，必要时只编译一个 CPU 机器人模型。
保存来源哈希并测试重复生成一致。不要初始化随机任务再把其场景或世界位姿写进
静态 profile。参考 [RoboCasa 生成器](../tools/generate_robocasa_config.py) 和
[配置研究](robocasa-config.md)。完整 schema、driver、frame 规则见
[接入 Devices](adding-devices.md)。

`roboenv launch` 只读取 catalog。运行时可以校验实际硬件参数、返回 live
capability，但不能静默更新 catalog 或改变相同配置所声明的控制语义。

## 5. 接入任务发现、launch 和 Harbor

在 `src/roboenv/tasks/catalog.py` 添加轻量 discoverer 与默认 robot 映射，并
检查所有后端枚举、准备流程和镜像构建映射是否覆盖新后端。只从锁定上游的
registry/config/AST 获取任务 ID、参数、语言和来源；无法解析的任务记录排除原因。
任务使用 `action_abstraction: devices`，复用 robot ID，并保留原始 horizon 和判定。

参考 [小型 RoboCasa task](../catalog/tasks/robocasa-pick-place-microwave/task.yaml)。
在小样例通过后才批量发现/导出，生成集合不入库。

```sh
roboenv launch TASK_ID --port 8123
```

本地与 Harbor 都走同一个 task resolver、世界和 device registry。每个任务一个
endpoint，所有设备共享它；并行任务用不同端口，或隔离网络中的同一内部端口。
Ready 必须意味着场景、binding 和设备都已就绪。连接失败不得悄悄新建世界。

为 `tools/build_simulator_image.py` 和部署映射补齐解释器、源码、资产切片和
真实镜像 digest。Harbor 接入同时覆盖启动、健康检查、agent endpoint、verifier
和失败清理。格式验收使用实际安装的 Harbor loader，详见
[Harbor tasks](harbor-tasks.md)。不要把 agent/model 安装塞进 simulator adapter。

## 6. 审核所有 agent 可达路径

不能仅隐藏 Python 方法。检查 worker 的 public description 投影和 HTTP allowlist；
目前 RoboCasa 的脱敏逻辑包含后端分支，新 adapter 不会自动继承全部 sensor-only
行为。新接入必须实现同等边界并测试：

- task description 只含允许的指令/上下文，不含私有 manifest、seed 或场景状态。
- device list/profile/read 不返回对象列表、布局、fixture 参数、成功信号。
- owner observe/score/reset 等不能经 HTTP 调用。
- `operations.get` 不得返回 owner 结果，也不能复用缓存绕过限制。
- health、错误详情、日志和 artifact 路径不能成为特权状态通道。
- agent 不可读取 simulator 资产、私有任务文件、owner socket、journal 或快照。

只读挂载仍然可以泄露内容。Harbor 的 agent 与 verifier 权限需分开，owner
证据目录不可读；同 Unix 用户运行本地 TUI 只能做功能测试，不能声称文件隔离。

## 7. 验收与证据

```sh
bash tools/check.sh
```

新增正向与负向测试：离线加载/再生成、真实接口与单位、错误参数先于效果失败、
只读不 step、stop 语义、IK/FK 不污染 live state、命令去重、部分初始化清理、
端口并行和 episode 隔离、catalog 在 launch 前后不变、特权观测拒绝。
Complex schema 校验之外，还需用复制后的 driver 验证实际调用和图像文件生命周期。

分别报告配置/CPU checks、真实 GPU boot、device checks、任务成功、fresh-process
replay。TUI 录制可用不等于任务完成，录像也不等于逐帧 simulator replay。
实际 GPU 或模型运行需明确范围，不能从文档/CPU 验收隐式启动。保留旧证据，
不要用新版本标签覆盖旧结果。
