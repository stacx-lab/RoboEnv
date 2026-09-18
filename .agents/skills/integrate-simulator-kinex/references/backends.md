# Backend migration checkpoints

Read only the selected backend section after inspecting the shared owners below.
These are implementation checkpoints, not backend acceptance declarations. The
device migration began from RoboEnv `9f3f18a` and Kinex `6eef667`; files below also
include subsequent working-tree changes. Verify current code, dirty inputs and
pinned upstreams on every integration. Paths are relative to their named
repository. Do not restore retired paths or infer current status from these notes.

## Shared owners and completion boundary

| Concern | RoboEnv source to trace |
| --- | --- |
| Source/recipe/assets | `src/roboenv/provisioning/`, `runtime/environments.json`, `runtime/locks/`, `third_party/` |
| Native lifecycle | `runtime/src/roboenv_runtime/adapters/BACKEND/`, `runtime/world.py` and `runtime/worker.py` under that runtime package |
| Binding selection | `runtime/src/roboenv_runtime/bindings/{__init__,native_devices,simulators}.py`, `src/roboenv/devices/bindings/registry.py`, `src/roboenv/devices/bindings/defaults/` |
| Public projection/transport | `src/roboenv/devices/`, `runtime/src/roboenv_runtime/runtime/http_service.py` |
| Catalog and discovery | `catalog/robots/`, `catalog/tasks/`, `src/roboenv/tasks/catalog.py`, `tasks/resolve.py` under the SDK package |
| Portable driver export | `src/roboenv/devices/export.py`, `src/roboenv/assets/`, `tools/prepare_kinex_workspace.py` |
| Native acceptance examples | `tools/check_native_{robotwin,robodojo,behavior,transport}.py`, `tools/replay_native_devices.py`; LIBERO/RoboCasa have their own record formats and checks |

The pre-migration baseline used primitive-oriented handlers and incomplete static
catalogs. The typed route now lives in each backend's `devices.py` and `rig.py`,
with shared `adapters/device_control.py` and `adapters/native_record.py`. Verify
which route the resolved task selects. A registered binding, generated rig or old
successful reference program does not prove typed Kinex integration.
Inventory actual supported rigs and public operations before editing. Update
task discovery's `action_abstraction`, default robot and controller mapping along
with runtime registration; test generated tasks through the normal resolver and
workspace preparation, not only handwritten examples.
Trace clock-override validation in `src/roboenv/tasks/resolve.py` and CLI launch as
well as backend support; implementing a clock does not automatically enable its
public `--sync` or `--realtime` selection.

Legacy bindings may contain in-process helpers named `rpc`; inspect their actual
implementation before diagnosing a second socket stack. Preserve useful native
control code and independent policy entrypoints when compatible with the requested
scope. Remove primitive invocation and scene truth from the default device path.

## RoboTwin

- Inspect `adapters/robotwin/{adapter,devices,rig,backend,embodiments}.py`
  under `runtime/src/roboenv_runtime/`, its selected upstream robot configs, and
  `catalog/robots/robotwin/robotwin-dual-piper` / `robotwin-aloha-agilex` if still present.
  Confirm the intended upstream edition; RoboTwin and RoboTwin2.0-Plus are not
  interchangeable source or benchmark identities.
- The default typed route must use the pinned RoboTwin source directly.
  Trace where the native SAPIEN world actually lives and how calls are counted.
  A pose planner may
  execute many steps; wrapping it as `write` does not provide the single-interval
  absolute-joint contract. Attach typed bindings to the shared episode authority.
- Audit native reset itself: the baseline tries seed offsets and nested setup
  retries, then calls expert `play_once()` before rebuilding the scene because
  some success predicates require attributes initialized by that prepass. Check
  whether the selected upstream still needs it. Preserve required predicate
  initialization or prove an equivalent replacement; do not silently delete it.
  If retained, isolate the expert prepass as owner initialization, record its
  time/work and errors, and rebuild the evaluated initial state afterward. Its
  actions/state are never candidate inputs or evidence of candidate success.
  Record requested seed, `actual_seed` and every internal retry; predeclare whether
  seed substitution is allowed. Replay uses the effective state identity. Two
  requested seeds that fall back to the same state are not distinct cases.
- Extract both arms' actual joint order, limits, flange/TCP transforms, gripper
  mapping and cameras from the selected embodiment. Prevent alias/composite
  commands from controlling the same joints twice. Verify atomic left/right/hand
  updates, retained targets and independent stop scopes under each supported clock.
- Replace public object-list/pose/score observations with declared sensors and
  robot measurements. Add the portable driver and camera resources missing from
  the old profiles; FK/IK stays local to Kinex.
- Audit `robotwin-devices.in`, `robotwin-devices-resolved.txt`, source pins and
  actual import origins against the selected benchmark source. The
  historical API-call-budget reference is not a typed-control benchmark result.
  Record physics frequency, action decimation and horizon; retain the original
  task predicate and label any changed control/budget condition.
- Cover every rig advertised by the completed integration. Include a task needing
  actual dual-arm coordination and a distinct manipulation mechanism in the small
  task matrix, within authorization. For alternative layouts/randomization, verify
  that the chosen native initialization settings change effective state.

## RoboDojo

- Inspect `adapters/robodojo/{adapter,devices,rig,backend}.py`, retained legacy
  callers, and the selected upstream environment, robot, camera and task
  configuration. The typed catalogs include ARX-X5 and YAM; their default route
  uses source-backed `EvalEnv` with an injected client.
- Match catalog selection to actual native creation parameters. In the baseline,
  YAM requires `env_cfg_type=yam`; changing only the robot ID is insufficient.
  Resolve single/dual-arm devices and aliases against the selected native inventory.
- Published scene layouts currently use the `arx_x5` collection. Generated YAM
  practice aliases explicitly set `layout_config=arx_x5` while keeping the YAM
  robot configuration. Record that adaptation in initialization/replay and do not
  call it an official YAM benchmark. Missing layouts must never trigger a silent
  fallback; inspect the selected collection and preserve its original contents.
- Preserve useful motor/held-control machinery while moving `pick`, `place`, pose
  planners and object queries out of the public device surface. Neither a websocket
  policy client nor a second agent service is required for Kinex device control.
  Check controller and camera initialization without contacting a policy server.
- Distinguish `step_lim` action frames, observation collection intervals and raw
  physics steps. Define which operations advance each counter and preserve native
  truncation and task-specific success requirements. Cover rendering/read calls
  that might accidentally step the world; do not infer harmlessness from a name.
- Inspect seed/layout mapping and the existing soft-reset path. A restored layout
  and warmed controllers may differ from a fresh process. Log initialization and
  settling separately; validate acceptance/replay from fresh native creation,
  including tasks whose success depends on a final robot posture.
- Check deferred native creation: `EvalEnv.sim` may be absent until first reset.
  Compare configured and live physics timestep after creation. Isolate native
  resume/video output IDs between workers; a constant default run ID can cause
  concurrent jobs to overwrite or remove one another's files.
- Validate the pixels produced by the native tiled capture manager. Authored
  camera aperture intrinsics can differ from the effective render projection;
  compare the render product's camera parameters and a rendered marker. Source
  camera rotations use intrinsic XYZ, while URDF rpy uses a different convention.
  CPU contact reports may be disabled by Isaac Lab: an owner fixture can enable
  the observation channel, but must still prove physical contact and release.
- For YAM, verify the managed USD conversion version and actual finger collision
  hierarchy. Current conversion decomposes the two concave finger meshes while
  preserving source visuals, joints and older USD/receipt archives. A fixed
  contact fixture can pass while a single-hull finger ejects a free object;
  require unsupported dynamic retention and release as well as loaded contacts.
  Quiesce the asset depot before upgrading and refresh affected native/Kinex
  evidence afterward. Conversion subprocess success requires both its exit code
  and a saved success receipt: Kit shutdown can hide a Python exception.
- Verify the selected Isaac Sim/Isaac Lab, cuRobo and asset dependencies together
  in a new interpreter. Source-lock syntax success is not a clean-install proof.
  Configure cold-start/warmup deadlines from observed boot behavior and include
  real cameras in native probes. Retain fluid/deformable checks when such tasks
  are in scope; a rigid pickup task does not cover them.

## BEHAVIOR

- Inspect `adapters/behavior/{adapter,devices,instances,rig,backend}.py`, the
  `behavior-r1pro` catalog and external binding, and upstream robot/controller/task
  definitions. The historical symbolic controller exposes privileged object/goal
  helpers; it is not the default typed route.
- For the default integration in this skill, implement continuous motor/device
  control with permitted sensors. Symbolic primitive success cannot satisfy this
  target. Preserve symbolic evaluation only as an explicitly separate mode when
  retained; never silently select it to make difficult tasks pass.
- Define the selected robot's movable base, arms, torso/head and grippers from its
  actual inventory. Use current Kinex types and units, live base/attachment frames,
  coordinated actuator updates and measured stop behavior. An old `type: body`
  profile with `invoke` is not a substitute for the current device schema.
- Pin activity, `scene_model` and `activity_instance_id`, plus assets and task
  definition identity. In the baseline, changing `seed` does not select another
  presampled instance. A distinct-state case must change an effective native
  initialization input and verify the resulting state; do not claim a hold-out
  from different seed labels alone.
- Inspect the released instance format. The current 2026 training assets use a
  base scene template plus per-instance task-relevant states and robot poses.
  Follow the upstream evaluator's restore/settle/snapshot order before starting
  the episode clock. Changing the full-scene filename alone does not load these
  instances. Record the selected state-file hash and effective native state.
- Keep BDDL evaluation owner-side with its original predicate semantics. Never
  replace task success with distance, symbolic-controller return values or a
  partial goal fraction. Robot sensors are public; object states and privileged
  goal diagnostics are not perception inputs. Choose tasks covering the declared
  manipulation/navigation capabilities and retain task-state details in replay.
- Inspect `behavior-devices.in`, `behavior-devices-resolved.txt`, provisioning
  registration and asset acquisition receipts. Revalidate the final recipe in a
  fresh prefix; a local freeze or an earlier successful installation is not proof
  for changed dependencies. Keep optional dataset/recording imports lazy so device
  boot does not require policy-training stacks. Acquire only authorized assets
  and explicitly authorized license acceptances through a reproducible setup path.
- Use the selected upstream's installation instructions. The official
  [BEHAVIOR installation guide](https://behavior.stanford.edu/getting_started/installation.html)
  describes license flags and substantial first-start initialization; do not copy
  LIBERO warm-start timeouts. Snapshot/replay must cover relevant object/system and
  controller state, including particles or transitions when selected tasks use them.
