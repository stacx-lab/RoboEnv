---
name: integrate-simulator-kinex
description: Implement or migrate a RoboEnv simulator to Kinex typed devices, from source and isolated installation through native code, portable catalogs, and complete acceptance. Use for new or changed simulator integrations; use test-simulator-kinex to run jobs on an existing integration.
---

# Integrate a simulator with RoboEnv and Kinex

Carry the requested integration through implementation, environment preparation,
tests, and executable acceptance in one continuous workflow. RoboCasa is an
architecture reference; LIBERO supplies native probe and replay examples. Inspect
their actual coverage before using either as acceptance evidence. A catalog,
fake-provider test, successful boot, or historical receipt cannot certify a new
integration or a different source revision.

This skill guides integration work; it does not bundle a simulator or campaign
runner. Missing backend code, recipes, drivers, and acceptance helpers are work to
implement within the requested integration, not reasons to return only a plan.
For a review-only request, inspect and report without executing this workflow.

Use [test-simulator-kinex](../test-simulator-kinex/SKILL.md) for the executable
acceptance stages of this integration. This is a division of workflow ownership,
not a separate agent or a required user handoff. Continue into that skill in the
same authorized task. A request only to run an existing simulator job starts there
and does not require this implementation workflow.

For multiple simulators, write a plan identifying shared transport/driver/runner
changes, backend-specific source/recipe/rig changes, dependencies, execution order,
and affected regression gates. Stabilize shared contracts before collecting final
evidence. Keep each native world and result directory independent; a shared change
invalidates every affected backend receipt, while a backend-only change need not.

## Establish scope and executable prerequisites

Find both repositories, read their nearest AGENTS.md, inspect status/diffs, HEAD,
tracking and upstreams, and preserve existing work. When asked for the latest code,
verify remote tips without silently replacing a checkout. Follow the requested
branch/publication scope; this skill does not authorize commits, pushes or merges.

Resolve routine choices from the current code and existing session authorization.
If essential scope is missing, ask one combined question covering only unresolved
choices: upstream/task family, assets and required license/access authorization,
GPU allocation, exact model/effort, bounded execution cost/time, and benchmark or
deployment mode. Continue independent implementation while waiting. Do not invoke
paid models or accept licenses implicitly. Once scope is settled, execute and
diagnose ordinary failures without requesting another prompt for each phase.

Read the testing skill's [acceptance contract](../test-simulator-kinex/references/acceptance.md)
before choosing tasks or running solutions. Record its effective run specification
outside source control. Default
to local sensor-only, typed-device integration acceptance; official benchmark
equivalence and Harbor deployment are separate scopes. Explicit user choices win.
Integration acceptance establishes a working agent/simulator loop, not a minimum
task success rate. Require real model observation and device control, correctly
reported native outcomes, recording and replay; inspect failures to distinguish
integration defects from controller/perception failures. Do not keep optimizing a
difficult task solely to obtain success unless task-performance acceptance was
requested. Frozen 3/3 task success belongs to that separate scope.

Preflight the actual executables, imports and entrypoints, not just their names:

- Bootstrap RoboEnv's development CLI using `tools/bootstrap.sh`; inspect the
  selected recipe, source lock, asset preparation and diagnostics.
- Prepare Kinex with its declared Node version (`engines`) and npm lock (`npm ci`,
  `npm run build`). Trace current
  schemas, lifecycle, Python tools, recording, provider and model/effort routing.
- Locate a working campaign runner and all its imports/bridges/protocol files in
  the selected Kinex checkout. A README reference or archived experiment is not an
  installed dependency. If absent, implement the small owner-side harness described
  in the acceptance reference using current Kinex APIs. Record its source identity.
- Verify the requested model/effort reaches the provider request with a no-network
  request-construction test. Check credentials without printing them. Do not assume
  an old `--effort` CLI flag exists or silently use another model/default effort.

Do not spend the model development budget until the native and no-model Kinex
paths work. Track implementation, infrastructure and controller failures separately.

## Trace the current integration and migration boundary

Read RoboEnv README.md, DESIGN.md, docs/setup.md, docs/simulator-integration.md,
and the selected adapter/backend/bindings/generator. Inspect the RoboCasa and
LIBERO implementations where their contracts apply. In Kinex, read
packages/core/src/{calibration/,device/interface.ts,device/dispatch.ts,device/driver.ts},
packages/core/src/tool/python-runner.ts, packages/core/src/python/,
packages/core/assets/python/kinex/ and packages/core/assets/builtin/device/templates/.

Trace this path before choosing edits:

1. Pinned upstream and assets → isolated interpreter and installation recipe.
2. Native world → adapter → WorldSession/worker → bindings and device service.
3. Static rig → copied portable drivers → Kinex lifecycle and normal tool runner.
4. Generated task → owner submission → native verdict and command record → replay.

For RoboTwin, RoboDojo or BEHAVIOR, read the corresponding section of
[backends.md](references/backends.md). Classify existing surfaces as reusable,
requiring migration, or independently retained. These backends may already boot
through older primitive APIs; that does not make their catalogs Kinex-compatible.
Migrate registry selection, task discovery, launch/export and tests together so
generated tasks use the same typed path as the small checked-in examples.

Keep one native world per episode and one authority for reset, physical time,
serialization, budgets, termination, score and evidence. Bindings reference that
world; they do not create another simulator, agent loop or worker-socket round trip.
Reasoning, perception and task-solving programs belong in Kinex tools/skills.

Reuse existing owners for preparation, lifecycle, transport, export and acceptance.
Keep simulator-specific registration/controllers/task logic out of Kinex core.
Generic Kinex core changes require a reproduced compatibility failure and a focused
regression test; adding an acceptance example is distinct from changing its core.
Do not relax schemas to accommodate incomplete profiles.

Before removing old paths, identify runtime callers, generated-task consumers,
independent policy/CLI entrypoints, scanned content and historical receipt readers.
Keep retained modes explicit and test their boundaries. No import matches alone
does not establish that a file is unused. Preserve source/patch identities and old
evidence; do not delete coverage to make a migration pass.

## Prepare a reproducible native environment

Use `roboenv sources check-updates --backends BACKEND` and follow the repository's
source policy when selecting the upstream base. Pin changes and checksum-verified
patches; never commit inside submodules or initialize unrelated simulators. Use
documentation matching the selected source revision and review changes from it.

Implement missing isolated recipes, import probes, asset acquisition/verification,
noninteractive setup and diagnostics through src/roboenv/provisioning/. Separate
development, simulator and agent Python environments. Resolve dependency conflicts
instead of hiding them with `--no-deps`; a prepared interpreter is not an install lock.

For Gate A, install the final recipe into a new prefix using
`roboenv prepare BACKEND --prefix NEW_PREFIX`. Reused environments can accelerate
development, but `prepare` reuse and `doctor` alone do not prove fresh installation.
Check dependency consistency, exact source/import origins from outside the repo,
and verified asset identities. Reuse existing assets only after verification;
do not overwrite datasets, interpreters or previous evidence.

Inspect the upstream boot sequence and renderer requirements. Make cold-start,
device activation, execution, replay and shutdown timeouts separately configurable.
Capture progress/logs and clean up only processes/resources owned by this run.
SDK imports and task discovery must not initialize a GPU. Asset or credential
blockers leave dependent gates pending while independent implementation continues.

## Implement the native device contract

- Keep the adapter thin: lifecycle, instruction, budget, private observation and
  original task predicate. Native controllers and rendering belong in the backend;
  register its lazy adapter and fixed binding IDs in the existing registries.
- Define joint order, units, reference frames, bounds, measured feedback and stop
  scope before exposing devices. Reject malformed, nonfinite, out-of-range and
  overlapping actuator commands before effects.
- Manipulators accept absolute joint targets and preserve other actuators' goals.
  Acceptance means a target was accepted; arrival requires measured feedback.
  Preserve measured soft-limit overshoot; project only private solver seeds.
  FK/IK uses Kinex's local geometry helpers without simulator RPC or state edits.
- Declare supported clocks. Synchronous writes advance one native control interval;
  coordinated writes must compose according to an explicit atomic-step contract.
  Realtime goals advance through the shared scheduler. Reads, stop and geometry
  calls do not request extra physics steps. Reset may perform declared initialization
  settling, recorded separately from episode control; device load never resets.
- Honor native thread affinity for creation, stepping, sensing and destruction.
  A simulator that works in a direct adapter test can deadlock when a background
  clock calls its physics API. Keep native execution on the required owner thread
  and verify the actual worker path, including idle connections and clock progress.
- Preserve native success and budget accounting, including initialization, control
  decimation, truncation and terminal freeze. Declare practice modes when control
  or horizons differ from the benchmark; do not relabel their results as official.
- Expose instructions, robot facts and permitted sensors. Keep object truth, seeds,
  restoration, owner sockets, journals and verdicts outside public device responses
  and agent workspaces. Remove legacy invoke/scene/score shortcuts from the default
  Kinex path; shared transport alone does not sanitize a backend's observations.

## Generate complete rigs and connect Kinex

A rig is a stable embodiment, controller, sensor arrangement and mounting geometry.
Reuse it across tasks with that arrangement. Different stable mount configurations
may have separate complete catalogs, as in `catalog/robots/libero/libero-panda-*`.
Preserve those scene rigs; do not collapse them into shared runtime mount overrides.
Share generators and driver implementations without making a copied rig incomplete.
Bindings belong outside the robot directory; use safe relative catalog paths and
the existing resolver. Keep only representative task examples in Git.

Extract robot/controller/camera facts from pinned sources or a task-free compiled
model. Never encode sampled object placements, seed-specific poses or session
identity. Verify deterministic regeneration, resolved-profile equality and rigid
mount invariance while joints move. Match live inventory to exported joint names,
bounds, controllers, camera dimensions, intrinsics and mounts at startup. Use live
flange parents for moving sensors, OpenCV optical axes and explicit depth units.

Reuse src/roboenv/devices/export.py and typed client assets. Copy each required
driver bundle including transport support and local geometry resources. Audit the
actual workspace helper: copying only top-level `*.py` files loses URDF/JSON or
nested resources. Extend copying with an explicit resource set and portability
test when needed; never copy task assets, owner evidence or simulator packages.

Parse every profile with the selected Kinex Device schema, required-action checks
and calibration plan. Prepare a fresh `.kinex/` workspace through the current
helper, call the core runtime's `python.prepare()`, and verify the actual
driver/tool interpreter under `.kinex/python/.venv`. Pin the
launched session/generation beside drivers, outside the static catalog. Activate
devices through Kinex lifecycle and exercise discovery, sensor reads, local FK/IK,
servo, world waits, measured feedback and stop through normal tool execution.
Servo computes its trajectory duration and accepts no `duration_s` override;
submission and elapsed duration do not prove arrival.

Test persistent drivers and their support files in the agent environment. Cached
profiles/connections must not survive an episode identity change unnoticed; reload
or restart through lifecycle. No driver may start its own simulator. World exposes
read/wait through its driver; fixed-base profiles remain driverless static mounts.

## Complete acceptance through the testing skill

Follow [test-simulator-kinex](../test-simulator-kinex/SKILL.md) in full-integration
mode, carrying the established source identities, authorization, task/rig matrix,
clocks, limits, numerical tolerances and private output location. Do not ask the
user to repeat the original instruction or authorize the same resources again.

Implementation owns any missing native conformance, replay, asset preparation or
runner capabilities. Add them to the existing RoboEnv or Kinex owners; the testing
skill then executes them. A helper for another backend is not interchangeable.

Required evidence is layered:

| Gate | Completion evidence |
| --- | --- |
| A | Static/catalog checks and installation of the final recipe in a new prefix |
| B | Measured native devices, geometry, cameras, contact, clocks and lifecycle |
| C | Real transport and no-model Kinex execution, recording and independent replay |
| D | Authorized visual model session, real control, truthful outcome, recording/replay and failure diagnosis |
| Optional task performance | Frozen fresh-process trials meeting the requested native success threshold |

A standalone smoke job cannot substitute for these gates. Keep harness execution,
controller completion, native task outcome, media integrity, device/source identity
and replay as separate checks. An honestly reported controller failure can support
integration acceptance when the integration path works and its failure is diagnosed.
Missing observations, broken drivers or unverifiable replay cannot. Preserve failed
attempts; require unchanged frozen inputs and held-out history for task-performance
trials when that scope is selected.

Fix integration failures within the authorized scope, rerun affected gates, then
resume acceptance. Do not edit a frozen candidate during final validation. Harbor
deployment requires the additional exported-task, loader, agent/verifier and image
evidence defined by the testing skill; local acceptance cannot establish isolation.

## Finish or resume from evidence

Run required repository checks for the changed surfaces; integration code changes
normally require RoboEnv `tools/check.sh` and Kinex tests/typecheck when affected.
Revalidate gates affected by later source, recipe, rig or helper changes. Preserve
older receipts; never combine incompatible revisions into one accepted result.

Report changed files, exact source/environment identities, executed commands,
per-gate status, task/rig/initial-state counts and artifact locations. Acceptance
requires all agreed integration gates to pass. Budget exhaustion, external prerequisites and
unsolved tasks must remain explicit in their own result fields, with completed code
and checks delivered. Save a precise resume command/configuration and compatible
artifact references outside source control; do not require the user to reconstruct
context or authorize the same work again. Never promise task success within an
arbitrary finite model budget.
