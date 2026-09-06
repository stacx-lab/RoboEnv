# Harbor task 生成、来源与验收

所有命令以 Roboenv 仓库根目录为工作目录。统一入口是
tools/generate_tasks.py，也可用等价的 roboenv tasks generate。
只提交生成器和小型样例，不提交批量展开的 tasks；generated/ 被 Git 忽略。

## 1. 格式契约

生成器使用 Harbor task schema 1.4，并逐个调用 Harbor 0.22.0 的真实
Task.is_valid_dir 和 Task 加载器。不是仅检查几个同名文件。
目录、可选 solution 和 verifier 奖励文件规则遵循
[Harbor 官方 task 文档](https://www.harborframework.com/docs/tasks)。

    task/
      instruction.md
      task.toml
      environment/
        Dockerfile
        docker-compose.yaml
        definition/        agent 可读设备事实与薄客户端
        sim/               环境持有的任务参数与 binding plan
        sdk/               本次冻结的 SDK wheel
      tests/
        test.sh
        verify.py
        task.json
      solution/            默认不存在；显式启用参考解才生成
        solve.sh
        solution.py
        source.json
      bundle.json          Roboenv 附加的内容哈希，不替代 Harbor 配置

标准入口是 tests/test.sh；verifier 输出 Harbor 要求的
/logs/verifier/reward.json。容器内的 /app、/env、/tests 等是标准运行路径，
不是开发机器的绝对路径。

Compose 中 main 是 agent 所在容器，sim 是环境容器的逻辑服务名，不是
物理机器名称。main 等待设备服务健康检查通过才启动。当前 verifier 在
main 中由 Harbor 执行，权威状态和 journal 在独立环境中；不要把它描述为
完全隔离、独立认证的 verifier 容器。

当前支持 Linux Docker Compose。符合 Harbor 格式不代表每个云 sandbox
provider 都支持多容器、GPU 和只读资产挂载。

## 2. 准备与生成

先执行 tools/bootstrap.sh 和 tools/check.sh，生成 SDK/runtime wheel。
按需执行 sources sync，再用 tools/build_simulator_image.py 构建后端镜像。
镜像构建读取本仓库锁定源码；准备好的解释器和资产通过
--dependencies 显式提供，不由 task 在 agent 启动后安装。

本地部署映射示例，保存为 deployments/local/map.json：

    {
      "libero": "../../.cache/images/libero/deployment.json",
      "robocasa": "../../.cache/images/robocasa/deployment.json"
    }

映射中的相对路径以映射文件所在目录为基准。deployment 中需要真实的
不可变 image digest；有外部资产时必须是可用的只读 depot。
本地部署文件不进入 Git。

    .venv/bin/python tools/generate_tasks.py \
      --backends libero robocasa \
      --deployments deployments/local/map.json \
      --wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
      --out generated/benchmark --seeds 0 --check-images

默认从锁定上游发现任务，不需要旧 task 目录。可用 --ids 精确选取任务；
也可传 --source catalog/tasks 使用小型样例，或指定自己维护的
规范化 task.yaml 目录。--seeds 生成不同 seed 的独立 Harbor task。

    .venv/bin/roboenv tasks discover --backends libero \
      --out generated/libero-source
    .venv/bin/roboenv tasks list --source generated/libero-source/tasks

发现器只读取上游 Python AST、YAML、JSON、BDDL，不导入模拟器引擎。
源文件、SHA256、上游 revision、受管理补丁及排除原因均进入 inventory。
缺失 BDDL、未展开的组合 suite、缺少明确场景的任务不会伪造配置补齐。

产物包含 tasks/、相对路径 job.json、taskset.json，以及存在时的上游
source-inventory.json。名称经过标准化并附加短哈希，避免大小写和标点
变化造成碰撞。格式、目标文件冲突和输入检查完成后才开始写任务。
有手工修改或旧生成文件会遗留时拒绝覆盖，要求使用新输出目录。

## 3. solution 是谁写的

任务来自官方 benchmark，不意味着参考程序也是官方的。
本项目历史导入与样例里的 solution.py 属于项目参考实现，不应称作官方解。

默认不读取、不复制 solution。需要 Oracle 回归时，显式传
--include-references，并在输入 task.yaml 旁提供 solution.py。
没有参考解的任务仍是合法 Harbor task，但不能用 Oracle 做解题回归。

可在旁边放 solution-source.json，记录 origin、official、upstream_url、
revision。没有来源说明时默认标记 project-reference 和 official=false。
声称官方来源必须给上游链接和 revision，维护者还需核对原文件和许可证。
生成器记录程序 SHA256，并将 replay_verified 设为 false：
旧解的存在、旧证书或作者自述，都不能认证这个新 task。

## 4. 运行与资产可移植性

    .venv/bin/roboenv tasks run generated/benchmark \
      --deployments deployments/local/map.json \
      --agent codex --model MODEL

runner 只把本机 depot 路径放入子进程环境变量，不写进公开 task。
它随后调用标准 harbor run；--agent 可换为当前 Harbor 已安装的 agent。
Complex 自定义安装器由独立 integration 提供，不打包进 Roboenv。

也可在 taskset 目录直接调用 harbor run -c job.json -a AGENT -m MODEL。
需要先设置 taskset.json.required_env 列出的 ROBOENV_DEPOT_* 变量；
GPU 选择使用 ROBOENV_GPU。路径、凭证和模型端点只能进入本地运行配置。

## 5. 什么才算通过

- 格式通过：Harbor 加载了 task，不代表镜像存在或任务能完成。
- --check-images：额外确认镜像在本机，仍未启动模拟器。
- 启动通过：真实容器、资产和设备 registry 就绪。
- 任务通过：环境 verifier 检查 score、completed、预算和 journal。
- 回放认证：另起新进程执行保存程序，由模拟器再次确认。

生成结果始终标记 execution_verified=false。不能把发现的任务数量，
或者批量生成成功的数量，写成实验成功数量。
