---
name: integrate-simulator-kinex
description: Implement a new simulator in RoboEnv and connect it to Kinex from source preparation through native device tests, live task solving, and independent frozen-controller acceptance. Use for complete simulator onboarding across these repositories.
---

# Integrate a simulator with RoboEnv and Kinex

Deliver a working integration and executable acceptance protocol in one continuous
workflow. Discover the current code before choosing interfaces. RoboCasa is the
reference integration; simulator-specific facts must come from the new simulator.
Never stop at generated profiles, fake-provider checks, or a service that merely boots.

## Establish scope once

Find both repositories, read their nearest AGENTS.md, inspect status/diffs and
upstreams, and preserve existing work. Use the user's requested branches and
publication scope. Do not infer authorization to commit, push or run paid models
from this skill alone.

Resolve minor choices from the code. If essential information is missing, ask one
combined question at the beginning: simulator/upstream and task family, native
assets or credentials, exact model/effort and bounded execution budget, and any
required benchmark mode. Continue independent inspection while waiting. Once
these are settled, implement, test, diagnose ordinary failures, and iterate without
requiring a new prompt for each phase. Stop only for an exhausted agreed budget,
missing external access, or a choice changing scientific meaning or protected data.

Record exact source revisions, effective configuration, seeds, initial-state IDs,
commands, interpreter/dependency identities and artifact locations in machine-readable
receipts outside source control. Update only necessary existing documentation; the
implementation, tests, protocol and receipts are the primary deliverables.

## Trace the active contract

Read RoboEnv README.md, DESIGN.md, docs/simulator-integration.md and the current
RoboCasa adapter/backend/bindings/generator. Read Kinex's actual schemas in
src/calibration/, required actions in src/device/interface.ts, dispatch and
Python-runner code, python/kinex/ helpers, and builtin/device/templates/.

Trace the complete path before editing:

1. Selected upstream revision and assets → isolated interpreter and preparation.
2. Native world → adapter → shared WorldSession/worker → device bindings/service.
3. Static rig catalog → portable typed drivers → Kinex workspace and tools.
4. Owner submission → native verdict and recording → new-process replay.

Prefer the existing abstractions. Keep one native world per episode, with one
authority for reset, serialization, physical time, budgets, termination and score.
Bindings reference that world; they never start another simulator or loop through
their own worker socket. Reasoning and controller programs belong in Kinex.

Treat Kinex core edits as exceptions supported by a reproducible failure and a
focused regression test. Keep simulator-specific controllers, registration and
task logic out of its core. Do not relax schemas to accommodate an incomplete rig.

## Keep the integration small

Before adding files, map each requirement to an existing owner: source preparation,
world lifecycle, device validation/transport, catalog export, task discovery, or
acceptance. Reuse those implementations. Add a simulator-specific branch or helper
only for behavior that the existing contract cannot express. Do not retain a second
primitive/driver stack merely to resemble an older integration.

Share device definitions and portable drivers across tasks. Represent rigid mounting
differences with existing binding overrides; create another robot catalog only when
hardware, control or sensor interfaces materially differ. Keep a small set of
representative tasks in Git, generate the full task registry from upstream metadata,
and put episode seeds, budgets and temporary variants in run configuration.

For cleanup, distinguish runtime callers, independent CLI/policy entrypoints, scanned
skills, and historical source/validation records. No import references proves only
absence from that call graph. Before deletion, establish the replacement or retired
scope and identify affected entrypoints. Preserve source/patch identities needed by
retained receipts. Test changed receipt readers against the original format as well
as newly generated records; passing tests after removing coverage is not equivalence.

Remove duplicate checks only when the shared boundary still rejects the same invalid
inputs before effects. Compare resolved profiles and behavior before/after factoring.
After deleting packaged files, build from fresh staging and check wheel contents.

## Build the native integration

- Pin the selected upstream and checksum-verified patches using RoboEnv's source
  management. Check its current default branch with `roboenv sources check-updates`.
  Never commit inside submodules or initialize unrelated simulators.
- Add the isolated dependency recipe, asset checks, noninteractive setup and
  diagnostics. Prove `roboenv prepare BACKEND`, `doctor`, source status and task
  discovery work. SDK import and catalog discovery must not initialize a GPU.
- Implement a thin adapter for lifecycle, instruction, budget, owner observation
  and the original task predicate. Put native controllers/rendering in the backend.
  Register the lazy adapter and fixed binding ID in the existing registries.
- Define joint order, units, frames, command bounds, measured feedback, stop scope,
  camera acquisition clocks and controller semantics before exposing devices.
  Validate malformed/nonfinite/out-of-range input before any effect.
- For manipulators, accept absolute joint targets and retain other actuators' goals.
  Acceptance means the target was accepted; determine arrival from later feedback.
  Preserve measured soft-limit overshoot; only private solver seeds may be projected.
  FK/IK belongs in Kinex's local driver helpers, without simulator RPC or live edits.
- State clock semantics explicitly: synchronous writes advance one native interval;
  realtime goals advance through the shared scheduler. Reads, lifecycle, stop and
  local geometry must not request physical steps. Test coordinated writes against
  the actual batching contract; concurrent calls alone do not guarantee atomicity.
- Preserve native success and declared budgets. If typed control or an unbounded
  practice mode differs from a published benchmark, mark that mode explicitly and
  never label its results as equivalent benchmark scores.
- Expose instructions, robot facts and declared sensors publicly. Keep object truth,
  seeds, state restoration, owner sockets and verdicts outside agent workspaces.
  Use the existing Harbor isolation/export path when that is the evaluation target.

## Generate a portable rig and connect Kinex

Extract stable robot/controller/camera facts from pinned sources or a task-free
compiled model. Never encode sampled object locations, seed-specific poses or
session identity in a rig. Generate shared device facts and the necessary mounting
overrides, not one catalog per task or scene name. Verify deterministic regeneration,
resolved-profile equality and mount invariance while changing robot joint pose.

Follow the current RoboCasa catalog: workspace.yaml and devices under the robot
directory, binding plans outside it. Reuse src/roboenv/devices/export.py and the
existing typed client assets with their adjacent transport support files. Copy the
whole required driver bundle, including local geometry assets. Do not copy session
connection files into the static catalog.

Parse every generated device with Kinex's current Device schema and required-action
checks. Validate its calibration plan. Match native inventory against exported
joint names, bounds, controller, camera dimensions, intrinsics and mounts at startup.
Use explicit parent frames; attach moving cameras to a live flange, with OpenCV
optical axes and declared depth units when applicable.

Prepare a fresh `.kinex/` workspace using the current preparation helper. Use
`kinex python` for the agent's separate environment. Pin the launched episode's
session/generation beside copied drivers, activate devices through Kinex lifecycle,
then exercise discovery, sensor reads, local FK/IK, a bounded servo and stop through
the normal Kinex tool runner. Simulator packages must not be needed by the drivers.

## Implement and run the acceptance protocol

Write the pass conditions and test matrix before evaluating solutions. Reuse
RoboEnv's native probes and Kinex's examples/simulator-acceptance machinery where
appropriate; add the new backend's protocol and owner replay implementation.
Do not copy a different simulator's numerical limits or task definitions.

**Gate A — static and isolated installation.** Run required source checks and
packaging checks. Validate all advertised tasks without starting the simulator,
every rig's current Kinex schema/actions/calibration, deterministic generation,
dependency consistency and pinned asset identity. Exercise positive and negative
configuration cases; record skips as uncovered checks.

**Gate B — real native device conformance.** Cover every rig/controller family and
at least two fixed initial states with a fresh process per case. Check small measured
motion, convergence versus acceptance, stop and gripper contact/release, FK against
native geometry at varied poses, IK residuals, dynamic mounts, render dimensions and
optical projection. Check both supported clocks, finite-budget terminal freeze,
read-only invariance, invalid commands with zero effects, and reset rebinding.

**Gate C — transport and Kinex.** Run simultaneous independent episodes. Test request
deduplication, rejected stale identities after endpoint reuse, denied owner operations
and private cached results, and camera artifact access. Run real Kinex device/tool
execution and background recording. Decode the actual videos; check nonblack frames,
readable files, timestamps, observed cadence, recorder shutdown and immutable profile
facts. Replay the owner command record in a fresh simulator process and compare state,
budget and native verdict under a declared tolerance. Failure must produce a nonzero
exit code and preserved diagnostic artifacts.

**Gate D — task completion by a frozen controller.** Use the requested task matrix;
otherwise choose a small set covering distinct mechanisms. Keep extra stress tasks
only when their coverage is needed. Configure the exact model/effort and finite
development rounds/time within authorization; distinguish exactly N rounds from
up to N rounds and do not silently stop an exact-round study after early success. Use the
real Kinex agent to observe, implement and debug a reusable controller tool. Freeze
all reusable code plus its Python manifest and dependency lock. Each validation uses
a new Kinex process/workspace and a new simulator world; no model, repair, conversation,
live state or hidden owner information carries into validation. Run two identical
initial-state trials for repeatability and one held-out initial state. Default to
3/3 successful trials per task; agree any different threshold before running.

For each validation require all of: normal controller exit, recorded real device
interaction, owner/native success, unchanged controller and device-configuration
hashes, valid recordings and
successful independent replay. An agent's success statement, zero process exit code,
or matching replay of a failed episode cannot pass this gate. Record failure categories
and continue only within the predeclared development budget. Do not lower thresholds
after observing results. Distinguish local API conformance from adversarial filesystem
isolation; local host runs do not certify a Harbor container deployment.

The current LIBERO protocol demonstrates the command structure; create the equivalent
backend-specific file and probes for the new simulator:

```sh
# Infrastructure gates, no model calls:
ROBOENV_REPO/.venv/bin/python ROBOENV_REPO/tools/accept_BACKEND.py \
  --python SIMULATOR_PYTHON --kinex KINEX_REPO --out FRESH_INFRASTRUCTURE_OUTPUT

# Full bounded development and independent frozen-program validation:
SIMULATOR_PYTHON KINEX_REPO/examples/simulator-acceptance/campaign.py \
  --roboenv ROBOENV_REPO --python SIMULATOR_PYTHON \
  --protocol KINEX_REPO/examples/simulator-acceptance/BACKEND.json \
  --model AUTHORIZED_MODEL --effort AUTHORIZED_EFFORT --out FRESH_TASK_OUTPUT
```

Replace these arguments with discovered paths and the authorized scope. The generic
campaign expects `prepare_kinex_workspace.py`, `finalize_episode.py`, a backend replay
script selected by the protocol, and normal `solve_task` Kinex tool discovery. If the
new simulator needs different initialization keys, set them in the protocol rather
than adding simulator branches to the campaign. Extend a helper only when its actual
contract cannot represent the new backend.

Validate the acceptance machinery itself with a controller that claims success and
exits normally without solving the task: the owner verdict must reject it. Also test
artifact mutation and missing evidence. Run the real gates after those checks, not
only the negative fixtures.

## Finish based on evidence

Inspect final diffs against the starting upstream revisions, run the repositories'
required checks, verify preserved reference work and publish only within authorization.
Report changed integration surfaces, exact commands and receipts, each gate's status,
task/seed counts, and remaining failures. Do not call the integration accepted while
an agreed gate is failed or unrun. If an external prerequisite blocks the final gate,
deliver the implementation and precise restart command with that gate explicitly
pending; never fabricate successful evidence or require the user to rediscover context.
