# 接入 Devices

本文对应 roboenv/devices/1 与 roboenv/bindings/1。
路径以 Roboenv 仓库为根；世界初始化见
[Simulator 接入](adding-simulators.md)。

## 1. Definition、binding 和 world

- Definition：设备类型、内参、外参、公开控制模式；不持有 simulator。
- Binding：把公开接口接到具体后端实体，负责单位/坐标转换和能力限制。
- World：实体实际存在的世界，拥有物理状态、生命周期和执行协调。

manipulator 类型不自动保证 IK/FK、力控或急停。参数未知时写未知，
不能用单位矩阵假装已标定，也不能让不同后端悄悄改变同一接口的单位。

## 2. 编写设备事实

文件是 catalog/robots/ROBOT/devices/NAME/profile.yaml。
内参属于设备事实，安装姿态与 entity/config 属于组装。
当前迁移任务通常允许 benchmark ground truth，不能标为纯视觉。

profile 中 access.driver.actions 是离线提示；运行时 devices.describe
返回的实际动作表才是权威。新的 profile 目录不需要 driver.py：
exporter 会为 agent 生成统一薄客户端。

## 3. 实现 factory

接口在 [devices/model.py](../src/roboenv/devices/model.py)：
DeviceDefinition、Action、BoundDevice、ExecutionContext。
factory 签名为 factory(spec, root, context) -> BoundDevice。

- spec：逻辑名称、binding ID、entity、profile、config 和 overrides。
- root：设备定义目录，只读。
- context.backend/embodiment：实际后端身份。
- context.native：环境注入的后端实例。
- context.public_tools：已过滤、已计费的控制器表，仅兼容控制器 binding 需要。
- context.public_dispatch：受限世界服务入口；直接实体 binding 不必使用。

完整例子：[bind_puck](../runtime/src/roboenv_runtime/bindings/fakeshell.py)。
它检查真实后端与 puck 实体，直接读写原生状态，不需要旧 driver 或 RPC。
后端控制器例子：[simulators.py](../runtime/src/roboenv_runtime/bindings/simulators.py)；
静态槽位映射与动作实现位于 runtime/bindings/backends。

每个 handler 接收 args 和 execution。物理修改必须经过
execution.perform(op, args, callable, mutating=True)，以记录证据并执行只读限制。
不要向正持锁的自身 worker 再发 socket 请求。

Action.input_schema 由 server extra 中的标准 JSON Schema 验证器在执行前校验，
远程引用被禁止。部分迁移接口仍只有宽泛的 object schema；handler 还必须验证
尚未完整建模的范围、单位和前置条件。不能实现的能力用
Action(reason=...)；不能返回假的停止、IK 或成功结果。
on_close 仅释放此 binding 的资源，不重置/关闭 world。

Binding 是可信环境代码，不是 Python 沙箱；绕过 execution 写后端对象能破坏
契约，必须通过代码审查和测试阻止。

最后在 runtime/src/roboenv_runtime/bindings/__init__.py 的
default_registry 显式注册。配置只能选注册 ID，不能指定任意 Python import。

## 4. 定义组装方案

    schema_version: roboenv/bindings/1
    backend: fakeshell
    embodiment: general
    devices:
      controller:
        binding: fakeshell.puck/1
        entity: puck
        profile: devices/controller/profile.yaml

profile 必须位于 robot 定义目录内部。entity 的含义由 factory 定义，
不是可随意访问 simulator 属性的通用路径。
config 由 factory 验证；现有后端控制器 binding 拒绝未知 config。
overrides 只允许 intrinsics、extrinsics 和 control_mode，不会凭空增加硬件能力。

Task 的 device_bindings 会替换整个默认设备映射，而非逐项合并：

    device_bindings:
      puck:
        binding: fakeshell.puck/1
        entity: puck
        profile: devices/controller/profile.yaml

自定义映射以 task.backend 为后端，不继承默认 plan 的可选 embodiment 限制。
factory 必须验证实际 embodiment 和 entity 的兼容性。
Owner reset/restore 之后重新创建 binding，不能缓存旧世界句柄继续使用。

## 5. 运行时与 agent

    profile + binding plan + live world
                      ↓
                  BoundDevice
                      ↓
                 DeviceService ← SDK / CLI / 薄 driver ← agent

生产 binding 实现在 runtime wheel。task 的 sim 目录只持有定义与组装配置；
agent 只得到所选设备的事实和薄客户端。设备名称变化不要求 agent 核心理解
是哪一种 simulator。

在已启动的 CPU fixture 内：

    roboenv devices --name controller
    roboenv device controller write '{"target":[0.3]}' --request-id move-001
    roboenv device controller read '{}'
    roboenv operation move-001

这里 target 是一维 puck 位置，不是通用机械臂姿态。
动作成功不等于 benchmark 成功；verifier 仍检查环境判分与预算。
结果不确定时查询原 request ID，不要用新 ID 盲目重发运动。

## 6. 验收清单

运行 tools/check.sh，并为新增 binding 测试：

- 错误后端/实体/路径拒绝；部分初始化失败释放已取得资源。
- 单位和姿态约定明确；只读动作不写世界；缺失 controller 标 unsupported。
- 动作失败传播，不返回成功；日志与预算使用同一个 world。
- reset/restore 重新绑定；close 不重置 simulator。
- 真实 socket、实际 Docker task 和独立 verifier 完成一次端到端验证。

现有 controller binding 不等于自动 URDF 组装、ROS 接口、高频流或真实急停。
新增这些能力应注册真实实现并另做硬件/安全验收。
