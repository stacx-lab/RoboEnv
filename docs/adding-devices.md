# 编写设备配置与接口

本指南对应新的设备服务设计。先阅读 [DESIGN.md](../DESIGN.md)，世界生命周期见
[Simulator 接入](adding-simulators.md)。参考配置是
[RoboCasa workspace](../catalog/robots/robocasa-panda-omron/workspace.yaml)。

## 1. 先决定设备边界

配置描述真实硬件/传感器及其基本操作，不描述任务解法或场景状态。逻辑名称由
workspace 定义，`type` 使用 Complex 的设备类型，不用名称暗示另一套接口。

| RoboCasa 名称 | Complex type | 边界 |
|---|---|---|
| `body` | `body` | 移动底盘，三个平面控制分量 |
| `torso` | `manipulator` | 独立升降关节和 live flange |
| `arm` | `manipulator` | Panda 手臂，关节状态、Cartesian delta 命令、IK/FK |
| `gripper` | `end_effector` | 夹爪本体状态与开闭控制 |
| `camera_left`, `camera_right` | `rgb` | 两个第三人称视角，随升降机构移动 |
| `camera_wrist` | `rgb` | 腕部视角，随手臂 flange 移动 |
| `world` | `world` | 坐标根，仅 profile |

不要添加 scene device 暴露 simulator 的物体坐标。也不要改放 task context；
agent 必须通过传感器感知环境。仅因为某字段可以从 MuJoCo/Isaac 读取，不代表
它是允许的传感器读数。

设备合并需要同时考虑控制与 frame。Complex 当前 body 只有一个 frame，
manipulator 支持 base→flange 的 live 变换。RoboCasa 因此保留单关节 torso；
把 lift 合入 body 还需要扩展 Complex 的 moving-frame 支持，不能只改目录名。

## 2. 文件布局与静态事实

```text
catalog/robots/ROBOT/
  workspace.yaml
  devices/
    NAME/
      profile.yaml
      driver.py
    world/
      profile.yaml
catalog/bindings/ROBOT.yaml
```

robot 根目录只放 `workspace.yaml` 与 `devices/`，不放 `bindings.yaml`、
`profiles/`、README、provenance、task/session 文件。背景说明、来源 receipt 和
生成脚本分别放 `docs/`、`docs/` 和 `tools/`。不需要的实验设备不要混进可发现目录。

`workspace.yaml` 放 assembly identity、平台类型和操作说明。不要把特定任务的
布局、模型路径或 seed 写进去。设备列表来自 `devices/`，不维护另一份人工名单。

| Profile 字段 | 内容 |
|---|---|
| `name`, `type` | 与目录和 Complex 类型一致 |
| `intrinsics` | 关节、限制、动作分量/单位、控制时序，或传感器内参 |
| `extrinsics.mount` | parent、必要的 parent_frame、刚性安装变换 |
| `access` | 服务访问方式、控制模式、driver 入口/解释器/操作列表 |
| `meta` | schema 要求的创建/更新元数据，不是启动时记录 |

完整实例见 [arm](../catalog/robots/robocasa-panda-omron/devices/arm/profile.yaml)、
[body](../catalog/robots/robocasa-panda-omron/devices/body/profile.yaml)、
[left camera](../catalog/robots/robocasa-panda-omron/devices/camera_left/profile.yaml)
和 [world](../catalog/robots/robocasa-panda-omron/devices/world/profile.yaml)。
Camera pinhole 参数位于 `intrinsics.value`；不要靠字段形似绕过实际 schema 验证。

从锁定的 robot/controller/camera 源码提取事实，必要时 CPU 编译单独机器人模型。
为生成器提供独立输出目录，测试重复生成一致，并把源文件哈希保存到机器人目录外。
参考 [生成器](../tools/generate_robocasa_config.py) 与 [来源研究](robocasa-config.md)。
不从随机任务 reset 的结果复制对象位置、底盘世界位姿或 countertop 高度。

同一 rig 跨任务使用相同文件。`launch` 不生成/更新 catalog。静态 profile 是
客户端理解设备的契约；运行时 capability 描述不能成为静默漂移单位/维度的理由。
变更相机分辨率、控制器或硬件后需要相应新配置与验证，不只是换 task 名称。

## 3. Frames、单位与控制语义

Mount 表达 child frame 在 parent frame 中的位姿，单位米、rotation 为 quaternion
xyzw。相机使用 OpenCV optical frame：+X 向右、+Y 向下、+Z 向前。MuJoCo 相机
朝向约定不同，需要真实转换，不能直接照抄 quaternion。

RoboCasa 的 frame 链为 `world → body → torso → arm → gripper`；左右相机接
`torso:flange`，腕部相机接 `arm:flange`。静态 mount 不包含 lift 的当前高度，
该高度通过 torso 的 `flange_pose` 参与 frame 计算。机器人世界位姿按声明的本体
传感器读数处理；不能给未知/动态 mount 写一个虚假的 identity。

Profile 明确动作顺序、units、bounds、control_mode 和控制区间。RoboCasa arm 的
六个 normalized 分量不是六个关节角，也不是绝对 Cartesian pose。每次 write
是一个原生控制区间；其它执行器的 neutral/hold 行为与 velocity expiry 必须明确。
不能把一次 composite action 拆成多次推进世界的设备 write。

## 4. 编写可直接使用的薄 driver

每个需要 driver 的设备在 catalog 中就包含 `driver.py`。Complex 复制目录后
可以直接调用；不要求 agent 编写 simulator driver，也不依赖 launch 临时补文件。
参考 [薄客户端模板](../src/roboenv/assets/device-client.py)。它只依赖 SDK，
模拟器实现与重依赖留在 server 的独立解释器。

```text
python3 driver.py read '{}'
python3 driver.py write '{"target":[...]}'
```

stdout 返回一个 JSON 对象；诊断不要混进 stdout。成功结果包含 `ok: true` 和
设备数据，失败包含 `ok: false` 与错误。Camera 返回 `file`、宽高等元数据；
文件必须在一次性 driver 退出后仍可被 Complex 录制器读取。

用 `ROBOENV_ENDPOINT` 指向当前服务。RoboCasa profile 的 loopback URL 是默认值，
运行时环境变量可覆盖；换端口不重写 profile。Catalog 不存 connection/session 文件。
SDK 长连接对象固定发现的 episode；新的 one-shot driver 连接当前 endpoint，
需要强制 episode 固定时由 runner 传显式 session/generation。

| Type | Driver 必须提供 |
|---|---|
| 普通 sensor | `read`, `load`, `unload`, `status` |
| mobile body / end_effector | 上述操作加 `write`, `stop` |
| manipulator | 上述操作加 `ik`, `fk` |
| world | 无 driver |

其它类型以 Complex 的 `src/device/interface.ts` 和 schema 为准。不要给相机
添加无意义的 write/IK。Required action 必须真实可用，不能用成功的 stub 代替。
Read/IK/FK 不修改世界；stop 明确同步区间、hold 行为和实际限制。

## 5. 实现 server-side binding

[数据接口](../src/roboenv/devices/model.py) 提供 DeviceDefinition、Action、
BoundDevice、ExecutionContext。Factory 为 `factory(spec, root, context)`：

- `spec` 是已解析的逻辑名称、binding ID、entity、profile、config、overrides。
- `root` 是只读 robot definition 根，profile 路径必须在其内部。
- `context.native` 指向现有世界；backend/embodiment 必须核对。
- `context.devices` 用于把原生实体映射到当前逻辑名称，尤其是 mount parent。

参考 [RoboCasa binding](../runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py)。
在 `default_registry()` 注册固定 binding ID。旧 public_tools/primitive bridges
仅用于尚未迁移的后端，新 binding 应直接连接原生实体。

每个 handler 的修改经 `execution.perform(..., mutating=True)`，在效果前完成
schema、有限值、范围、维度和前置条件校验。输入/输出 schema 与 driver 声明一致。
不能实现的操作显式失败，不返回假的停止、IK 或 benchmark success。Scratch
kinematics 不能改 live qpos/qvel/ctrl/time。Binding close 只释放绑定资源。

## 6. 私有绑定配置

以下是 `catalog/bindings/ROBOT.yaml` 的片段，完整文件还需其余设备：

```yaml
schema_version: roboenv/bindings/1
backend: robocasa
embodiment: panda_omron
devices:
  body:
    binding: robocasa.device/1
    entity: base
    profile: devices/body/profile.yaml
  camera_left:
    binding: robocasa.device/1
    entity: camera
    profile: devices/camera_left/profile.yaml
    config:
      camera: robot0_agentview_left
```

逻辑名称 body 与内部 entity base 不必相同。相机 config 选择真实传感器，并须由
backend 启用；profile 不会凭空创建摄像头。Factory 拒绝未知 entity/config。
`overrides` 只调整允许的设备参数，不增加硬件能力或观测权限。

新 catalog 使用外置 plan；旧机器人和打包后的 server bundle 仍可带本地
`bindings.yaml`，loader 优先读本地文件。Complex 副本不需要这个文件。
Task 的 `device_bindings` 是完整替换，而非默认列表上的 patch；自定义列表必须
同时提供匹配的 Complex 设备目录，不能留下未提供或不允许访问的设备。

## 7. 验证配置能用且不会泄露状态

1. 用 Complex 的实际 Device/RobotManifest schema 和 required-action 定义验证文件；
   只有 YAML 能解析并不够。测试 robot 根目录只有预期内容。
2. 从隔离源模型重新生成到临时目录，比较文件并记录来源；移动机器人不改变刚性 mount。
3. 把配置复制到新的 `.complex/robots/ROBOT/`，设置 endpoint，测试复制后的 driver。
4. 验证每个相机确实路由到不同传感器、图像在 driver 退出后存在、read 不 step。
5. 测试错误输入先于效果失败、stop/IK/FK、共享预算、命令去重和 episode 隔离。
6. 测试 task/context/profile/health/status 不泄露 scene truth，owner 请求不能经
   HTTP 或缓存结果查询绕过限制。
7. 运行 `bash tools/check.sh`；另行报告真实 GPU boot、TUI recording、任务成功和 replay。

复制、TUI 和录像导出命令见 [本地 Complex 测试](complex-local.md)。录制 outcome
是人工注释，不是 RoboEnv verifier 的 benchmark 结论。
