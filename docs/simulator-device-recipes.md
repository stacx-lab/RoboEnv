# 新 Simulator 与配置变更：操作路径

详细步骤见 [Simulator 接入](adding-simulators.md) 和 [设备配置](adding-devices.md)。
先确定改动属于任务、硬件、逻辑接口还是部署，不要用 launch 重写 robot 配置。

| 需求 | 应修改 | 不应修改 |
|---|---|---|
| 相同 rig 换 task/seed/布局 | 私有 task 参数与指令 | 静态 profile/driver |
| 同时运行多个任务 | 端口、endpoint、独立 harness 进程/网络 | 每份 robot 配置的连接文件 |
| 修改逻辑设备名称 | profile 名称、目录、binding key、mount 引用 | 原生硬件实体 |
| 增加/更换相机 | backend 启用的 sensor、binding config、静态 intrinsics/mount | 物体坐标或 layout snapshot |
| 更换硬件/控制器 | 新 rig 配置、native binding、资产、参数与验证 | 用旧 rig 名称掩盖单位或运动学变化 |
| 全新 simulator | 源码/安装 recipe、adapter、binding、任务发现、镜像与验证 | 新的公共 env 方法或 task 宏操作 |

## 相同配置复用于多个任务

标准 RoboCasa 任务都引用 `robot: robocasa-panda-omron`。其目录只有
`workspace.yaml` 和八个设备目录，包含 body、torso、arm、gripper、左右/腕部
RGB 相机及 world。没有 scene device，也不向 task context 放 ground truth。

两个终端分别运行：

```sh
roboenv launch robocasa-pick-place-microwave --port 8123
```

```sh
roboenv launch robocasa-arrange-vegetables --port 8124
```

给两个独立 Complex workspace 复制相同的 robot 目录，在各自进程中设置
`ROBOENV_ENDPOINT`。服务变更不修改已复制配置。具体 TUI 操作见
[本地录制计划](complex-local.md)。

## 重命名或选择设备

RoboCasa 的公开 `body` 仍绑定原生 entity `base`；逻辑名不是引擎对象名。
调整名字时同步更新目录、profile.name、binding key 和所有 mount.parent 引用。
多个相机都可以使用 entity `camera`，但各自 config.camera 指向不同原生 sensor。
验证它们不是错误地返回同一个画面。

通用 task 格式允许 `device_bindings` 完整替换默认映射。它不是逐项 overlay，
也不会创建新的硬件。显式 subset/custom mapping 必须匹配实际提供给 Complex
的配置目录；不要复制全套配置却让其中设备不存在。相同 rig 的普通任务应直接
复用默认 `catalog/bindings/ROBOT.yaml`，避免维护重复映射。

## 修改硬件、mount 或动作空间

先确认 simulator 是否支持目标硬件，再写新的可复用配置。不要仅修改 robot ID
就宣称完成换装。相同方法名不意味着相同单位、控制频率或动作维数。

静态 camera extrinsics 必须相对其真实安装 frame。RoboCasa 左右相机随 lift，
腕部相机随 arm flange；body 世界位姿和 torso 高度是动态状态。不能通过合并
设备把动态变换写成常量。Complex frame graph 不支持的设备结构需要先设计
相应 frame 接口，再宣布兼容。

预生成稳定事实，保存来源到 `docs/`，确认同一 rig 跨姿态和 task 不需重写。
渲染尺寸、相机型号或 controller 变化后更新配置和验证证据。Agent 用摄像头
估计场景，不从 profile、task context、错误日志或 owner 状态读取物体真值。

## 验证与发布范围

配置变化运行 schema/再生成/复制 driver 检查；binding 变化增加真实 CPU 世界的
控制、预算、只读、异常、并行与观测隔离测试，再运行 `bash tools/check.sh`。
修改运行时代码后重建 runtime wheel/镜像；只重导出 task 不会更新镜像里的实现。

另行在明确范围内验证 GPU boot、设备、TUI 录制、任务成功和 fresh-process replay。
不同机器人不能复用旧任务成功证据，录制 MP4 也不是 replay 证书。
不覆盖历史 task、镜像 digest 或证据；生成/执行见 [Harbor tasks](harbor-tasks.md)。
