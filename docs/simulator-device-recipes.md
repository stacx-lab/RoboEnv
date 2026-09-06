# 新 Simulator、换 Devices：实际操作

详细 API 见 [Simulator](adding-simulators.md) 与 [Devices](adding-devices.md)。
先分清要改的是世界、接口还是机器人。

| 需求 | 修改位置 | 必须验证 |
|---|---|---|
| 同 simulator 换 task | backend_kwargs 和任务语言 | registry 存在、初始化参数、原 benchmark 判定 |
| 给已有实体换逻辑名字 | device_bindings 的键与 profile | entity 不变、客户端重新发现 |
| 减少/增加已存在的设备接口 | 完整 device_bindings 映射 | factory 支持对应槽位、设备不会凭空创建 |
| 换实际机器人 | simulator 初始化参数、robot catalog、binding | 机器人模型、控制器、动作维数和 benchmark 可比性 |
| 接全新 simulator | source lock、adapter、binding、discoverer、镜像映射 | 从依赖获取到真实 Harbor replay 全链路 |

## 1. 只更换设备视图

CPU fixture 的 world 里已有 puck。把逻辑设备 controller 改名为 puck：

    device_bindings:
      puck:
        binding: fakeshell.puck/1
        entity: puck
        profile: devices/controller/profile.yaml

保存到输入 task.yaml 后，重新生成到一个新目录。agent 的
roboenv devices 将显示 puck；原来的 controller 不再暴露。
这没有创建第二个 puck，也没有改变物理世界。

生产后端也遵循这个原则。例如 LIBERO 的 entity 槽位有 arm、gripper、
camera、world。可以选择暴露子集，但新的设备类型或相机选择参数必须先有
factory 实现，不能在 YAML 中填写任意 entity/config 就假定有效。

## 2. RoboTwin 更换真实机器人

现有 adapter 接受 embodiment；Piper 和 ALOHA-AgileX 的任务配置分别是：

    backend: robotwin
    backend_kwargs:
      task_name: handover_mic
      embodiment: piper
    robot: robotwin-dual-piper

    backend: robotwin
    backend_kwargs:
      task_name: handover_mic
      embodiment: aloha_agilex
    robot: robotwin-aloha-agilex

这只是 task.yaml 的相关字段，仍需 id、backend_task、instruction 等完整定义。
上游发现器会分别生成 robotwin.* 和 robotwin-aloha.*，无需手工展开全部 task。
机器人参数和 robot catalog 必须同时匹配；仅换 robot 字段会被 embodiment
校验拒绝，或无法得到正确控制器。

新机器人需要相应模型资产与 adapter 支持；跨机器人动作预算、任务可达性和
成功标准可能不同，必须分别记录回放结果，不能沿用旧机器人的证书。
RoboDojo 的 YAM catalog/绑定存在，不代表资产和所有任务已完成回归。

## 3. 给新 simulator 暴露现有设备类型

1. 先让 adapter 独立完成 reset/observe/control/score/close。
2. 在 factory 检查真实后端与实体，转换成已有公开设备语义。
3. 复用 profile 中确实相同的事实，重新声明安装外参和可用动作。
4. 用测试证明相同输入语义、不同后端执行均正确，不能只比 JSON 字段名。
5. 加入 discoverer 和镜像映射，生成 task，再通过真实 Harbor/回放验收。

若后端只能做 symbolic primitives，就如实声明 symbolic-primitives。
不能把它包装成 joint-space motion 后与连续控制结果混合报告。

## 4. 生成与回归

    roboenv tasks generate --source MY_CATALOG \
      --backends robotwin --deployments deployments/local/map.json \
      --wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
      --out generated/device-variant --check-images

重建 runtime image 才会带上修改后的 factory；仅重新生成 task 不会更新
镜像中的 Python 实现。修改 profile/plan 只需重新生成 task。
不要覆盖已用来做实验的旧 task、镜像 digest 或证据。
