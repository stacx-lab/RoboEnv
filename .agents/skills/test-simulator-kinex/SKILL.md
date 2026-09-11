---
name: test-simulator-kinex
description: Run real Kinex jobs on a RoboEnv simulator, including no-model probes, controller development, frozen task validation, and full integration acceptance. Use for simulator integration tests, regression jobs, or reviewing recorded acceptance; new backend implementation belongs to integrate-simulator-kinex.
---

# Test a simulator through real Kinex jobs

Execute the requested job using the current RoboEnv owner lifecycle and Kinex
tool runner. Deliver source-matched evidence for that scope. A boot, schema check,
controller return, or matching replay alone does not establish task success.
For an evidence review, inspect artifacts without launching jobs.

This skill is independently usable and is also the acceptance phase of
[integrate-simulator-kinex](../integrate-simulator-kinex/SKILL.md). It does not
require a second agent or another user prompt. Carry existing authorizations,
failed attempts and development/held-out case history across phases.

## Select the job scope

| Request | Workflow and completion claim |
| --- | --- |
| Smoke test or check an existing connection | Bounded no-model probe: real sensor, measured motion, stop, recording, owner finalization, command replay |
| Develop a controller or solve a task | Authorized live Kinex sessions after native/infrastructure checks; preserve development history |
| Run a frozen controller | Fresh no-model validation with immutable artifact and specified cases; report every trial |
| Complete or certify an integration | All gates A–D below, covering selected rigs, clocks and tasks |
| Measure task-solving performance | Integration evidence plus immutable trials and an explicit task success threshold |
| Run a Harbor job | Explicit deployment scope and real loader/agent/verifier boundary, plus applicable checks |

Use the user's mode, matrix and limits. For an unspecified test job on an existing
integration, begin with a no-model probe; do not silently add paid development or
a three-trial campaign. Full integration requests inherit gates A–D; native task
success is reported but is not their pass threshold. Task-performance acceptance
is separate and must not become an implicit unlimited controller-optimization task.
A regression without live-development lineage is a controller regression, not a
new certified development campaign.

Read [acceptance.md](references/acceptance.md) before full acceptance, model
development or final case selection. It owns the detailed specification, freeze
and exposure policy, replay contract, failure tests and pass conditions.

## Resolve the actual executable path

Inspect both repositories' guidance, status, revisions and dirty inputs. Read the
current task resolver, robot/bindings, interpreter registration, Kinex schemas,
Python workspace helper and runner. Record effective identities: HEAD alone does
not identify an uncommitted implementation.

Check these entrypoints instead of assuming archived commands still work:

- RoboEnv: `roboenv doctor BACKEND`, task discovery and `roboenv launch`,
  `tools/prepare_kinex_workspace.py`, `tools/finalize_episode.py`, and the selected
  backend's native conformance/replay tools.
- Kinex: `examples/simulator-acceptance/{campaign,infrastructure,aggregate}.py`
  when present, all required imports and `--help`, the pinned Bun version and
  `kinex python`. Read the current example README and parsers.
- LIBERO/RoboCasa may use separate native/protocol runners. Select their actual
  supported route. The generic `replay_native_devices.py` currently accepts
  RoboTwin, RoboDojo and BEHAVIOR, not every backend's record format.

Use a task ID or YAML through the shared resolver. Generated aliases may select
different robot and backend settings; a robot label alone does not configure the
native embodiment. Record effective seed/layout/activity instance and clock.
BEHAVIOR seed changes alone do not select another presampled activity instance.

Create an owner-side run specification before launch: interpreters, task/cases,
exact model/effort when used, recording devices/rates, stage deadlines, GPU,
fresh output paths and prior artifact/evidence references. Pass only fields the
selected parser consumes. For the current campaign runner, `role` belongs to the
final aggregation spec, not the per-job spec.

Inspect GPU/disk capacity and accepted licenses/credentials. Existing session
authorization persists; ask only about missing access or material scope. Never
accept licenses or invoke paid models implicitly. Keep secrets, owner state,
receipt paths and final-case identities out of agent content. Large environments
and caches can use verified external storage; preserve assets and prior evidence.
Check the actual native owner/capture root as well as the Kinex workspace: moving
campaign outputs alone may leave native images on the quota-limited filesystem.
Do not relocate an active world's files or change its resolved storage paths.

Missing native/runner capability is implementation work. In an authorized
integration, repair it through the integration skill and resume here. In a job-only
request, diagnose a required migration rather than silently changing the native
contract or substituting a fake path.

## Execute the applicable gates

**A — Static and installation.** Validate selected tasks/catalogs and copied driver
resources against current schemas. For new-integration acceptance, install the
final recipe with `roboenv prepare BACKEND --prefix NEW_PREFIX` and verify
dependencies, import origins and assets. A standalone job may reuse a verified
interpreter; reuse is not fresh-install proof.

**B — Native conformance.** Full acceptance covers every advertised rig and
supported clock with at least two effective initial states in fresh processes.
Measure motion/arrival, scoped stop, contact/release, local FK/IK, optical
projection and moving mounts. Check invalid inputs before effects, atomic steps,
read invariance, budget exhaustion and reset/lifecycle ownership. Owner fixtures
can test geometry/contact but cannot prove task success. Missing required native
checks remain pending or failed even when optional CPU tests skip them.

**C — Transport and no-model Kinex.** For full acceptance, test independent
concurrent worlds, deduplication/uncertain replies, stale generations after endpoint
reuse, denied owner/cached-result access, sensor artifact ownership and cleanup.
Use actual cameras and effective states: the native transport helper supports
`--camera`, `--seeds FIRST SECOND`, and `--instances FIRST SECOND` for BEHAVIOR.
Run a bounded sensor/move/stop probe through normal Kinex activation and Python
tool execution, recording, owner finalization and fresh-process command replay.
Check measured movement as well as accepted writes. Native task success may be
false for a passing infrastructure-only probe.
Exercise the real worker scheduler for each supported clock. Direct calls to a
native adapter do not establish threaded worker compatibility. Record concurrent
GPU load when measuring recording cadence, and diagnose a rate failure without
silently lowering the declared rate or accepting only an encoded video FPS.

**D — Live agent integration.** Begin paid development after required
infrastructure gates pass for current native/Kinex identities. Verify exact
provider/model/effort and successful requests containing real camera images.
Require representative sessions to use those observations, execute real device
control, preserve device definitions, finalize the original native outcome, record
media and pass independent replay. Inspect session messages, tool results, device
events and actual video frames together. A failed grasp, perception mismatch or
honest `completed:false` does not invalidate a functioning integration. A driver
failure, missing sensor, broken lifecycle or mismatched replay requires diagnosis
and affected regression checks; never attribute an unexplained failure to model
reasoning. Record residual intermittent issues without claiming they were fixed.
Iterate to resolve integration defects within user limits; stop when this evidence
is sufficient unless further task development was requested. No overall limit
still requires finite child-stage deadlines. Controllers use permitted devices and
perception, never native object truth, verdicts or reference solutions.

The current `infrastructure.py` covers the static/native/transport/probe evidence,
and `campaign.py develop` records live-agent evidence. Audit their files together
with installation evidence and failure classifications for integration acceptance.
The existing `aggregate.py` is a task-performance aggregator requiring native
success; it does not implement a task-success-optional integration report. Do not
add ignored spec fields or relabel its failed output as passed.

**Optional task performance — Frozen validation.** When requested, establish
trial counts and success criteria separately from integration readiness.

Freeze selected reusable tool/skills/libraries and Python lock before final cases.
Check the discovered tool ID against its manifest and directory. The current
example freezer expects `.kinex/tool/<tool>/tool.yaml`; align its directory with
the manifest name during development, before freezing and verifying lineage.
Every final case starts a new native world and Kinex process/workspace, restoring
the exact artifact without model calls, repairs or prior conversation/state.
Runtime observations stay outside frozen code. Default task-performance acceptance is 3/3
successes per task: two equivalent initial states and one distinct state excluded
from development. Honor explicit alternative counts without calling them this
default protocol.

Run the real aggregation path. A final trial requires normal completion (for the
example runner, JSON `completed: true` and no error), successful writes, original
native task success, unchanged artifacts/devices, valid media and independent
command replay. A matching replay of failure cannot substitute. Preserve final
failures; further authorized development must mark exposed cases as tuning data
and reserve a new held-out case.

**Harbor, when requested.** Read `docs/harbor-tasks.md`, validate exports with the
actual loader, and run the requested job with image, mounts, agent/verifier and
cleanup provenance. Local checks do not establish deployment isolation. Local
job authorization does not imply publication or deployment authority.

## Diagnose, deliver and resume

Separate installation, boot, infrastructure, controller, native task and replay
failures. Retain failed logs and partial evidence. Finalize and close only owned
worlds/drivers/recorders with bounded cleanup. Fix ordinary in-scope failures in
new attempts instead of overwriting results.

A denied loop-confirmation request is a failed execution. Retain it and diagnose
the repeated operation. Stateful scans should have explicit viewpoints, visited
state, finite motion limits and a stopping condition. Do not disable safeguards
or keep retrying the same denied sequence; continue with a bounded workflow within
the existing authorization, or resolve a genuinely missing permission.

Inspect each child result file as well as its exit code. Some Isaac shutdown
paths terminate Python directly with code zero before pending returns or errors
reach the caller. Save native verdicts before shutdown; a missing, malformed or
non-passing replay/media receipt fails its gate. Match replay to the native
record checksum, rather than accepting an unrelated passing result.

After fixes, recheck source/artifact identities. Reuse evidence only when its
relevant source/dependency/asset scope still matches. Shared runtime/Kinex changes
can invalidate multiple backends. Do not relabel historical results.

Report mode, task/rig/case coverage, native outcomes, normal completion, media,
replay and artifact paths separately. Save executed commands and a resume spec,
preserving authorization and exposure history. Recheck prerequisites on resume;
an old blocker cannot override later user authorization. No skipped required gate
can yield complete integration acceptance.
