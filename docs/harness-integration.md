# Building a harness or benchmark on RoboEnv

RoboEnv is the simulator service, the SDK, the device contract and the journal
certification. A harness (an agent runner such as Kinex or rsi) and a benchmark (a
trial scheduler such as RLE-Bench) build on those four things through the SDK and
the CLI; nothing in RoboEnv knows which harness or benchmark is on the other side.
This document lists what RoboEnv provides for that, what it deliberately does not,
and, for the maintainers reviewing this proposal, where every Harbor-specific piece
that used to live here went.

This scope change is a public-interface change and would ship as **0.10.0**. The
version in this branch is not bumped; the release decision belongs to the
maintainers.

## What RoboEnv provides

| Need | RoboEnv entry point |
|---|---|
| Start a world and wait for its devices | `roboenv.launch(task, ...)` / `roboenv launch TASK [--port P] [--host 0.0.0.0] [--advertise-host HOST] [--allow-stop] [--detach --json] [--python-path DIR] [--runtime-root ROOT --catalog C --interpreter PY] [--run-dir DIR]`; the launch record (`run.json`, schema `roboenv/local-run/1`) carries endpoint, socket, session and generation |
| Attach an agent | `roboenv.connect(endpoint)` (`Environment`: `task`, `devices()`, `device(name)`), `roboenv.client.Client` (wire client, standard library only), the CLI device commands (`roboenv devices|device|describe`), `ROBOENV_ENDPOINT` / `ROBOENV_SOCKET` / `ROBOENV_SESSION` / `ROBOENV_GENERATION` |
| Give the agent portable device files | `roboenv.devices.export.export_devices(profiles, directory, connection)`; launch writes them beside the run record, and a supervisor started with `ROBOENV_PUBLISH_DEVICES=DIR` (and `ROBOENV_ENDPOINT` naming the public URL for `connection.json`) publishes them itself at readiness |
| Give a coding agent a language reference | `roboenv.devices.documentation.export_documentation(robot, devices, directory, connection)` and `robot_description` |
| Find and resolve tasks | `roboenv.tasks.catalog.load_catalog` / `discover`, `roboenv tasks list|discover`, `roboenv.tasks.resolve.resolve_task` (task, robot directory and frozen binding plan), `roboenv.tasks.model.Task` |
| Run a reference program as an agent | `roboenv replay PROGRAM --endpoint URL`: public operations only, no verdict |
| Certify a run | `roboenv.verify.certify(...)` and `roboenv verify --run RUN --out DIR`: identity check over the owner socket, one submit, `observe`, score validation and journal validation, fail-closed `submit.json` / `reward.json` / retained `actions.jsonl`; `validate_journal`, `validate_score`, `attach_budget`, `strict_json`, `public_error` and `refuse` are public |
| Stop a world | `roboenv.environment.stop`, `roboenv stop`, `RunningEnvironment.close()`; `simulation.stop` over HTTP only with `--allow-stop` |
| Prepare a simulator | `roboenv prepare BACKEND`, `roboenv.provisioning.setup.install` / `prepare`, `roboenv.provisioning.sources.sync`, `roboenv.provisioning.robocasa_assets.prepare_assets`, `roboenv.provisioning.robodojo_assets.prepare_assets`, the executable-depot verification under `runtime/packaging/` |
| Build an immutable simulator image | `tools/build_simulator_image.py` writes `deployment.json` (schema `roboenv/deployment/1`: image digest, runtime wheel hash, source revisions, depot manifests); its consumer is the harness |
| Validate a prepared simulator without a model | `tools/validate_simulators.py`: launch, replay, verify, stop per task and seed, allowlisted receipt ([simulator validation](gpu-validation.md)) |

`roboenv.verify` is the reference validator. It imports nothing else from the
package, so a verifier container can copy that one file; a harness running in the
same environment imports it. Re-implementing journal validation elsewhere is how
rules drift: RLE's `native_tasks/verify.py:validate_episode` currently lacks the
single-submit rule, the post-submit mutation rule, the `effects: none` exemption,
strict JSON parsing and the protocol/`journal_version` checks that
`roboenv.verify.validate_journal` applies. rsi's local runner (rsi commit 093afbc)
already calls `roboenv.verify.certify` when the module is importable and falls back
to the packaged `assets/verify.py` only for RoboEnv 0.9.0 and earlier, which lack it.

## What RoboEnv does not provide

Trial scheduling, retries, timeouts and concurrency; containers, compose files,
images for the agent side and network isolation; agent execution, model
credentials, workspaces and controller development; task bundles, tasksets and
their loaders; GPU pools and resource allocation; reports across trials. A
benchmark owns those and talks to RoboEnv only through the table above. RoboEnv
owns the environment state and the verdict; agents own reasoning and programs.

Local launch shares one Unix user with the harness; that is a functional boundary,
not filesystem isolation. Enforced observation isolation needs a separate agent
identity or container without simulator assets, private task definitions, owner
sockets or evidence access. The harness provides that boundary.

## Typical composition

```python
import roboenv
from roboenv.verify import certify

with roboenv.launch("libero/bowl-onto-plate", host="0.0.0.0", port=8123) as run:
    # Hand run.endpoint and run.record["devices"] to the agent process.
    ...
    report = certify(
        socket_path=run.record["socket"], session=run.record["session"],
        generation=run.record["generation"], task_id=run.record["task"], seed=0,
        budget_mode="practice",
        journal=f"{run.record['directory']}/evidence/journal/actions.jsonl",
        out="verdict",
    )
```

`roboenv verify --run RUN_JSON --out verdict/` does the same from the launch record.
A world serves one episode: certify once, then stop it.

## Moved out of RoboEnv in this proposal

The downstream benchmark, RLE-Bench-inhouse (`feat/task-integration`), already
implements the Harbor layer; the table names its files. "Port" marks pieces RLE
does not have yet.

| Removed from RoboEnv | Role | Home in RLE-Bench-inhouse |
|---|---|---|
| `src/roboenv/harbor/export.py` (`export_task`, `verifier_script`, `registry_name`) | Harbor 1.4 task bundle: private `environment/sim` (task, bindings, robot) versus public `environment/definition`, compose, `task.toml`, `tests/verify.py` | `native_tasks/build.py:generate`; public facts exported live per episode by `coordinator.py` via `export_documentation`; the verifier is `native_tasks/verify.py` (see prerequisites) |
| `src/roboenv/harbor/taskset.py` (`export_taskset`, `run_taskset`, `validate_harbor_task`, `deployment_map`, `deployment_environment`) | Batch generation, `job.json`, pinned deployment identity, `ROBOENV_DEPOT_*` required environment, `harbor run` | `tasksets.discover` / `tasksets.publish`, `runner.job_config` / `runner.run`, `build.validate_task` (add the `schema_version == "1.4"` assertion) |
| `src/roboenv/harbor/deployment.py` (`load_deployment`, `deployment_contract`, `native_service`) | Deployment manifest validation, read-only depot mounts, compose service defaults, `runtime_options` | `roboenv.py:deployment` (rejects mounts, non-image interpreter modes and `PYTHONPATH`/`ROBOENV_*` overrides), `prepare.simulator_ready`; `shm_size` is still unvalidated there |
| `src/roboenv/harbor/workflow.py` and `roboenv export` | One-task export with wheel build and loader check | `rlebench generate` |
| `src/roboenv/harbor/kinex.py` (`KinexAgent`, `inspect_bundle`) | Harbor agent adapter: bundle identity re-hash, `no-model` and `develop` modes | `agents/install.py:stage_kinex` (image-baked, `installation.json` identity), `kinex.py`, `kinex_run.mjs`; port `no-model` from RoboEnv PR #45 |
| `roboenv tasks generate|validate|run` | CLI over the above | `rlebench prepare-sim|prepare|generate|run|summarize|clean` |
| `tools/generate_tasks.py` | Wrapper for `roboenv tasks generate` | `rlebench generate` |
| `tools/smoke_device_harbor.py` | CPU Harbor/Docker/Kinex trial on the synthetic world | none; `test_native_tasks::test_generation_uses_real_harbor_loader_and_separates_private_inputs` covers generation, a container smoke is RLE's to add |
| `tools/package_kinex_agent.py` | Kinex checkout plus bridge as a hashed tarball | `agents/install.py:stage_kinex` (`vendor.py:copy_inputs` excludes `node_modules`, `.venv`, `dist`, `build`, `.git`) |
| `tools/report_kinex_harbor.py` | Trial receipts: bridge execution versus native success | `report.summarize` (usage, learning curve, attribution) |
| `integrations/kinex/run.ts`, `controller.ts` | Agent-side composition of Kinex tools, sessions and recording; `no-model` and `develop` modes; controller completion | `kinex_run.mjs`; port `no-model` and frozen-controller execution from PR #45 |
| `integrations/kinex/provider-evidence.ts` | Asserts the real model, effort and delivered camera images at the provider boundary (zstd Codex bodies, forced SSE in PR #51) | `kinex_run.mjs:auditFetch` records but never asserts; port `requestEvidence` / `requestText` |
| `integrations/kinex/artifacts.py`, `freeze.py`, `prepare.py` | Freeze/restore/unchanged checks of reusable controller content; workspace preparation from `ROBOENV_ROBOT` | `vendor.py:copy_inputs`, `installation.json`; frozen-controller validation: port from PR #45 |
| `integrations/kinex/recording.py`, `verify_media.py` | Decode recordings, check timestamps, rates and black frames | `recording.recording_status` (file set and sha256 only); a decode/rate check is optional |
| `integrations/kinex/probe/` (`integration_probe`) | Bounded no-model sensor/move/stop controller | port with PR #49 (timed segments, `world.wait`) |
| `src/roboenv/assets/main.Dockerfile` | Agent container: SDK wheel, `agent` user, ffmpeg | `agents/Dockerfile` (`userdel ubuntu`, `useradd --uid 1000 agent`, as in PR #46) |
| `src/roboenv/assets/serve-native.sh` | Simulator container entrypoint: depot verification, fresh session, supervisor arguments | `sim.command = bootstrap python -m rlebench2.native_tasks.coordinator` (no depots) |
| `src/roboenv/assets/healthcheck.py` | Compose health probe: `health` plus `devices.list` over the owner socket | `test -S .../control.sock`; an RPC-level `health` probe is optional |
| `ROBOENV_DEPOT_<BACKEND>_<n>` | Depot locations for compose volumes | not needed: RLE rejects mounts |
| `ROBOENV_SDK_PATH`, `ROBOENV_FRESH_SESSION`, `ROBOENV_HTTP_HOST`, `ROBOENV_HTTP_PORT` | `serve-native.sh` inputs | coordinator and compose configuration; `roboenv launch --host/--port/--python-path` covers the same for a local owner |
| `ROBOENV_ROBOT`, `ROBOENV_MANIFEST`, `ROBOENV_VERIFIER_OUT` | Agent-image and verifier-shell paths | RLE agent image and `native_tasks/verify.py` |
| `ROBOENV_GPU` | Compose device reservation | RLE GPU pools; the validation tool uses `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID` |
| `harbor` extra and its 80 pins in `locks/dev.txt` | Loader validation dependency | RLE's own environment |
| `docs/harbor-tasks.md` | Deployment, generation and execution guide | RLE `docs/architecture.md`, `docs/agents.md`, `docs/validation.md` |
| `tests/test_export.py`, `tests/test_taskset.py`, `tests/test_kinex_harbor.py`; the deployment, healthcheck, taskset-runner and verifier-shell cases of `test_security_contracts`, `test_interpreter_depot`, `test_device_binding_separation`, `test_task_contract`, `test_workflows`, `test_device_service` | Harbor-side regressions | `test_native_tasks`, `test_tasksets`, `test_clean`, `test_shared`, `test_kinex_auth`, `test_prepare`; missing there: an execution receipt for a zero-exit agent, and `runtime_options` validation |
| Harbor Oracle mode of `tools/validate_simulators.py`, `gpu-validation.yml` | GPU validation through Harbor bundles | RoboEnv keeps validation through launch/replay/verify; RLE has no reference opt-in or probe mode yet |

Unchanged and still available to a harness: `roboenv launch --host 0.0.0.0
--advertise-host`, the image mode of `tools/build_simulator_image.py` (RLE consumes
its `deployment.json` through the `roboenv/deployment/1` schema in
`rlebench2/roboenv.py`), the Kinex local acceptance gates (`tools/accept_*.py`,
`tests/libero_kinex.py`, `tests/native_rigs_kinex.mjs`, [local Kinex
use](kinex-local.md)) and the privacy audit.

Retained without a current consumer, pending a maintainer decision: the depot mode
of `tools/build_simulator_image.py` (`--interpreter-mode depot`,
`runtime/packaging/Dockerfile.depot`, `relocate.py`, `verify_dependencies.py`,
`tests/test_interpreter_depot.py`) and `ROBOENV_PUBLISH_DEVICES` in the supervisor.
RLE rejects mounts and non-image interpreter modes and builds its own images; rsi
and Kinex use neither.

## Behaviour changes in this branch

- The SDK client and the portable shared driver defaulted the session identity to
  `harbor`; both default to `local` now, and the catalog copies of `_roboenv.py`
  are regenerated. A deployment that relied on the implicit default must set
  `ROBOENV_SESSION`; launch already pins a unique session per run.
- `docs/robocasa-config.provenance.json` records the hash of the previous
  `_roboenv.py`; it is a historical receipt of the last RoboCasa catalog generation
  and is refreshed by the next `tools/generate_robocasa_config.py --provenance` run
  with the prepared interpreter, not by this change.
- `roboenv.verify.READ_ONLY_OPS` lists `simulation.status`. The supervisor proxies
  it to the worker, whose status is a pure read, and counts it among its own
  read-only operations; a status poll after the verifier's submit, which
  `roboenv replay` makes on completion, no longer voids the certificate.
- `submit.json` names a certification failure by exception type and a path-free
  detail (`roboenv.verify.public_error`); the full message, which may carry the run
  directory, goes to `roboenv verify`'s stderr or to `certify(log=...)`.
  `submit.json`, `reward.json` and the retained `actions.jsonl` are replaced
  atomically, so a reader never sees a torn verdict.
- `tools/validate_simulators.py` writes receipts with schema
  `roboenv/simulator-validation/1` instead of `roboenv/gpu-validation/1`. Its
  `--deployments` and `--sdk-wheel` arguments are gone; `--interpreter`,
  `--runtime-root`, `--python-path` and `--timeout` pass through to `roboenv launch`.
- `tools/clean_install.py --without-harbor` is gone with the `harbor` extra, and
  `roboenv export` with the Harbor workflow.

## What RLE and rsi must close before RoboEnv drops the layer

1. Land RoboEnv `feat/simulator-integration` (12 commits ahead of `main`): RLE's
   coordinator calls `launch(record_episode=True)` and reads
   `evidence/native-initialization.json` and `native-episode.json`, which exist only
   on that branch. This is a prerequisite regardless of Harbor.
2. Re-stage Kinex 0.9.0: RLE's `third_party/kinex` is 0.8.2 (4db60e9, six commits
   behind 0.9.0 at 18bcbdc). The 0.9 port work sits in RoboEnv PRs #45, #51 and
   #49 and should be re-targeted to RLE.
3. Enforce provider evidence in `kinex_run.mjs` (model and effort assertion, at
   least one delivered image, zstd Codex bodies) from PRs #51 and #45.
4. Harden `native_tasks/verify.py` by importing or copying `roboenv.verify`: single
   terminal submit, post-submit mutation, response identity, `journal_version`,
   strict JSON, `shutdown` forbidden.
5. Port the no-model and frozen-controller mode and the PR #49 probe if model-free
   device conformance runs are wanted in RLE.
6. Add a reference-solution opt-in and a probe mode with an allowlisted,
   privacy-checked receipt (RLE's `prepared.json` and `summary.json` may contain
   host paths today).
7. Minor: `schema_version` assertion in `build.validate_task`, an RPC-level
   simulator healthcheck, `shm_size` validation, and a decision on `cap_drop` for
   the `main` service (the root `chown` in `kinex.py` needs `CAP_CHOWN`).
8. rsi: its local runner already certifies through `roboenv.verify.certify`
   (commit 093afbc, with the packaged `assets/verify.py` as the fallback for
   RoboEnv 0.9.0 and earlier), but its default `harbor` backend
   (`executor/src/run.ts:29`) imports `roboenv.harbor.taskset` and
   `roboenv.harbor.deployment` (`executor/python/rsi_harbor/run.py:42-43`) and
   consumes `roboenv/taskset/1`. It must move to RLE's generator or pin RoboEnv
   to 0.9.0 or earlier before 0.10.0 ships.
