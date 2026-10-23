# Booster T2

The Isaac Lab backend exposes the Booster T2 with Revo2 hands through
`base`, `head`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, `head_camera`
and `world`. The native owner runs a pinned ONNX whole-body controller at 50 Hz
and PhysX at 200 Hz. Agent reasoning, training, benchmarks and Kinex sessions
remain outside RoboEnv. The original locomanip project is not a runtime or
image-build dependency.

Prepare the Isaac runtime and download the pinned robot, controller and scene assets:

```sh
roboenv prepare isaaclab
python tools/prepare_booster_t2.py --download
roboenv launch isaaclab/t2-reach-table --record
```

The depot contains the T2 URDF/meshes, fixed controller, and selected SIMPLE
house, GraspNet objects and articulated assets. `src/roboenv/assets/booster-t2-assets.json` pins every file
and the upstream scene revision. Downloads require read access to the pinned
upstream repository; set `GITHUB_TOKEN` (or `GH_TOKEN`) when authentication is
required. The token is used only for GitHub API requests. To reuse an offline depot, pass
`--source /path/to/portable-t2-depot` instead of `--download`. Preparation preserves differing existing files
and fails on checksum mismatch. It honors `ROBOENV_ASSET_CACHE`; use the same
setting at launch. `--robot-only` prepares flat-ground touch tasks. Generated
USD conversions live in process-local temporary storage, leaving the depot intact.

`realtime` continuously advances physics on the shared owner scheduler;
`world.wait` waits for that clock. `stepped` advances only while waits are pending.
Both modes use the same physics, buffers and devices. Realtime has no control-frame
cap; the external harness owns its wall-clock deadline. Stepped uses `max_steps`.
Reset includes 100 standing-control frames outside the episode budget. Seed zero
retains the source layout; other seeds add small bounded start-position variations.
Falling ends an attempt. No seed substitution occurs.

Arm writes specify joint configurations whose FK defines palm targets. The fixed
controller chooses the measured arm configuration: joint arrival is not the same
as palm arrival. Public geometry supports local head/arm FK, arm IK, and coupled
hand grasp-center FK without simulator access. Hand stop preserves preload;
opening requires an explicit command. No world pose or object truth is exported.

RoboEnv includes two standalone examples: `isaaclab/t2-reach-table` for
flat-ground locomotion and touch, and `isaaclab/t2-can-in-bowl` for kitchen objects
and grasp/release. The complete benchmark definitions live in RLE-Bench-inhouse
`tasks/task11/catalog/`. All use the single `t2-household` factory with an explicit
`backend_kwargs.task_spec`. RoboEnv owns scene construction, specification validation
and native predicates; RLE owns benchmark definitions, instructions and selection.

Task specifications use version 2. Supported goals include touch, lift, placement,
containment, direct handover, sliding, rotation, stacking, level or loaded carrying,
articulated-joint operation and chair pushing. Ordered touch goals require a new
contact after the preceding stage. Placement tests every object-bound corner
against the visible rectangular zone, with release, low-speed and support checks.
Pick-and-place requires a prior 8 cm lift near a hand sustained for 0.15 s.
Carry goals monitor tilt and retained contents throughout the lift; sliding and
rotation require continuous support. Each stage has its specified dwell time.
These are geometric predicates, not force-contact classifiers. Task definitions
and short physics checks do not establish agent performance or solvability.

Owner evidence retains reset state, source/asset/controller identities, task
definition, commands at control-frame boundaries and final physical/controller
state. Replay never restores final state:

```sh
python tools/replay_booster_t2.py owner-record.json --task-file task.yaml --out replay.json
```

Retain the original resolved task YAML alongside the record. Multi-object replay
requires it to preserve PhysX actor creation order: sorted evidence JSON retains
the task values but loses mapping order. Replay checks the task definition and
instance against the record, records the YAML checksum, and keeps the same state
and verdict tolerance. RLE task generation retains this input as
`environment/sim/episode_01.yaml`.

Boot, device conformance, media, replay and native task success are separate
acceptance claims. RLE-Bench-inhouse `task11` owns selection, deadlines, execution
mode, model runs and result publication.
