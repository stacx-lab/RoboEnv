# Booster T2

The existing Isaac Lab backend exposes the Booster T2 with Revo2 hands through
`base`, `head`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, `head_camera`
and `world`. The native owner runs a pinned ONNX whole-body controller at 50 Hz
and PhysX at 200 Hz. Agent reasoning, training, benchmarks and Kinex sessions
remain outside RoboEnv. The original locomanip project is not a runtime or
image-build dependency.

Prepare the existing Isaac runtime and a separately retained, verified asset depot:

```sh
roboenv prepare isaaclab
python tools/prepare_booster_t2.py --source /path/to/portable-t2-depot
roboenv launch isaaclab/t2-reach-table --record
```

The depot contains the T2 URDF/meshes, fixed controller, and selected SIMPLE
house/GraspNet assets. `src/roboenv/assets/booster-t2-assets.json` pins every file
and the upstream scene revision. Preparation preserves differing existing files
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

RoboEnv includes two standalone debug examples: `isaaclab/t2-reach-table` for
flat-ground locomotion and touch, and `isaaclab/t2-can-in-bowl` for kitchen objects
and grasp/release. The complete benchmark definitions live in RLE-Bench-inhouse
`tasks/task11/catalog/`. All use the single `t2-household` factory with an explicit
`backend_kwargs.task_spec`; new layouts and goal sequences need no registry entry
or provider catalog copy. RoboEnv owns scene construction, specification validation
and native predicates; RLE owns benchmark definitions, instructions and selection.

Placement requires a prior 8 cm lift near a hand sustained for 0.15 s,
release clearance, low speed, support geometry and a 0.5 s dwell. Container goals
use conservative object bounds inside an upright, nearly stationary bowl.
This is a geometric grasp/containment predicate, not a force-contact classifier.
New task definitions do not establish task solvability or model performance.

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
