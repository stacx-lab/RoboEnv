# Booster T2

The Isaac Lab backend runs Booster T2 with Revo2 hands and a pinned whole-body
controller. It exposes `base`, `head`, `left_arm`, `right_arm`, `left_hand`,
`right_hand`, `head_camera` and `world`. Control runs at 50 Hz and physics at 200 Hz.

## Setup

Follow the [Isaac Lab requirements](isaaclab.md), then prepare the runtime and
download the pinned robot, controller and scene assets:

```sh
roboenv prepare isaaclab
python tools/prepare_booster_t2.py --download
roboenv launch isaaclab/t2-can-in-bowl --record
```

Downloads require access to the pinned upstream repository. Set `GITHUB_TOKEN`
or `GH_TOKEN` if authentication is needed; the token is sent only to GitHub.
To reuse an offline depot, replace `--download` with
`--source /path/to/portable-t2-depot`. Preparation verifies checksums and refuses
to replace differing files. Set `ROBOENV_ASSET_CACHE` consistently for preparation
and launch when using a custom cache location. Generated USD files use temporary
storage and leave the depot unchanged.

Two standalone examples use the benchmark house: `isaaclab/t2-can-in-bowl`
for grasping and placement, and `isaaclab/t2-ordered-reach` for walking to the
table and touching the blue and red blocks in order with the specified hands.
Launch either task with `roboenv launch TASK_ID --record`. Both keep the session
open after success for inspection. The complete twenty-task benchmark is defined in
[RLE-Bench Task 11](https://github.com/stacx-lab/RLE-Bench-inhouse/tree/main/tasks/task11).

## Control and timing

Use the [device API](device-documentation.md) to read sensors and issue commands.
Arm joint commands define palm targets through forward kinematics; the fixed
controller chooses the measured joint configuration. Palm arrival and joint
arrival can therefore differ. Public geometry supports local head/arm FK, arm IK
and hand grasp-center FK. Object poses and hidden world state are not exposed.
Stopping a hand preserves its preload; opening requires an explicit command.

`realtime` advances physics during reasoning and tool calls, with no control-frame
cap; the caller sets the wall-clock deadline. `world.wait` waits for that clock.
In `stepped` mode, physics advances only during waits and `max_steps` limits control
frames. Both modes exclude 100 standing initialization frames from the episode
budget. Seed zero uses the source layout; other seeds slightly vary the robot's
start position. Falling ends the attempt.

## Household scenes and task semantics

Version 5 uses `scene: hssd_household`: the pinned SIMPLE/HSSD 102344022 residence
with its original architecture, furniture, materials and lighting. GraspNet
supplies calibrated scans for the tomato soup can, bowl, apple and cracker box.
Task fixtures include an open bookcase, consoles, cart rails, a doorway and
functional mechanisms. Selected furniture and decorations are removed or
replaced to provide access and clearance. These are adaptations of the source
house, not its untouched layout.

Mechanisms use authored collision geometry and passive joints. Scanned props
use calibrated collision approximations; the bowl has an open compound shell.
Imported visuals add no extra rigid bodies or colliders. Not every object is a
scan, and the render mesh is not an exact collision mesh.

Tasks select the `t2-household` factory through `backend_kwargs.task_spec`.
RoboEnv validates the specification and evaluates physical contacts, support,
containment, release and settling. Instructions can also require ordered touches,
tools, level carrying or both hands; a final pose alone does not satisfy those
requirements. RLE-Bench owns task instructions, selection and evaluation budgets.

## Recording and replay

`--record` captures the overview, head and hand spectator views. These do not add
agent observations. See [spectator recording](spectator.md) for output handling.
Owner evidence records initialization, source and asset identities, the task
definition, commands and final state.

Replay requires the original frozen source and assets. Retain the resolved task
YAML with each recording, then run this in the Isaac runtime:

```sh
python tools/replay_booster_t2.py owner-record.json --task-file task.yaml --out replay.json
```

The YAML preserves object creation order and is required for multi-object replay.
RLE-Bench retains it as `environment/sim/episode_01.yaml` (and the corresponding
file for each later episode). Replay starts fresh physics, rejects identity
mismatches and compares state and native verdict.

Only version 5 task definitions with `scene: hssd_household` are supported.
Native task outcome, replay, recording and agent completion are separate results.
See [validation](gpu-validation.md) for the distinction between CPU checks and simulator evidence.
