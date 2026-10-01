# Executable acceptance contract

Read for full integration acceptance, controller development, or frozen final validation.
For a standalone probe, use only its applicable infrastructure and evidence rules.
The requested job mode determines scope; this document does not authorize extra
model calls, trials, deployment, or implementation work.
Use current repository APIs; this reference defines required behavior, not an
already-implemented command or a new simulator/agent framework.

## Integration readiness and task performance

Integration acceptance requires gates A–D from the testing skill: installation,
native conformance, transport/no-model execution, and representative live visual
agent sessions with real writes, owner finalization, recording and independent
replay. Native task success remains an observed outcome. It is not a prerequisite
for accepting the integration unless the user also requests task-performance
acceptance. Do not continue task optimization merely to turn an integration pass
into a task pass.

For live sessions, verify current source/device identities, actual provider/model/
effort and successful image inputs, meaningful device execution, native outcome
reporting, cleanup and matched media/replay. Inspect messages, event traces and
sampled video frames to classify a failure. Controller perception, planning and
grasp failures may demonstrate a working integration; transport, sensor, actuation,
clock, owner or replay defects require repair or an explicit unresolved limitation.
Unknown causes are not model-reasoning diagnoses. A controller exception must stay
visible even when the surrounding harness finalizes and replays correctly.

Record `integration_passed`, `task_success` and `controller_completed` separately in
the owner review. Preserve the original receipts. Audit external harness session evidence alongside installation, native replay and failure review.
A failed task result must remain a failure in the task-performance report.
The freeze, held-out selection and 3/3 rules below apply when task performance is
requested. Fresh-process command replay remains required for integration itself.

## Effective run specification

Store a machine-readable run configuration and receipts outside source control.
Reuse an existing protocol format where possible. Extend its parser/tests for
missing semantics; adding ignored JSON keys does not implement a requirement.
Keep portable backend/task protocols with their code and host paths, credentials
references and results in private run configuration. Never put secrets in receipts.

Resolve and validate these fields before the operations they govern:

| Field group | Required meaning |
| --- | --- |
| Sources | Selected RoboEnv/Kinex roots and HEADs; dirty input hashes; upstream revisions, patch and dependency-lock hashes; runner/helper identities |
| Target | Backend/edition, advertised rigs/tasks, representative acceptance tasks with coverage reasons, control/observation/budget mode, local or isolated-container scope |
| Installation | Development, simulator and agent interpreters; new installation prefix; asset identities; existing access/license authorization and unresolved blockers |
| Execution | GPU allocation and concurrency; cold boot, activation, development, validation, media decoding, replay and cleanup timeouts; episode horizon and clock; maximum infrastructure retries |
| Model | Exact provider/model/effort and existing authorization; user-selected development limits or explicit no-overall-limit scope; finite child-stage deadlines; cost/token limit when specified; stop-on-success versus exact-round policy |
| Cases | Native task identifiers; robot/controller configuration; development/tuning cases; final repeatability pair and distinct held-out case; seed plus effective scene/layout/instance/state identities |
| Checks | Expected devices/camera rates; geometry, motion, rendering and replay tolerances; native success rule; trial counts and per-task success threshold |
| Evidence | Fresh output directories; controller/driver/configuration manifests; commands and per-gate receipts; failed-attempt history; resumable state |

Use the user's requested matrix; otherwise choose a small set covering the declared
rigs and distinct mechanisms. State the selection before evaluation. For optional
task-performance acceptance, default finals are 3/3 trials per selected task, with two identical initial states
and a third effective state excluded from development. If the backend cannot offer
that variation, resolve the scope before running; do not relabel duplicates.
Do not equate discovery coverage with solved-task coverage.

All actual invocations must consume these bounds. Include development, compaction
or other model calls in any agreed total model limit. Expiry cancels outstanding
work and closes owned workers/drivers/recorders with a bounded grace period.
Retries are logged attempts within the same budget, not invisible replacements.
Once real validation interaction starts, a controller failure counts as a trial;
only predeclared infrastructure retry rules can justify a replacement attempt.

Preflight the enforceability of any strict token/cost ceiling separately from
time/round limits. The audited Kinex provider request has no remaining-budget or
maximum-output field, and usage arrives after execution. A timer plus post-run
usage is not a hard cost cap. If the user requires one, implement and test request
admission/reservation and supported output limits across normal calls, compaction
and retries, including interrupted calls with uncertain usage. If the provider
cannot enforce the requested ceiling, resolve that execution scope before paid
calls while continuing independent work. Do not invent a monetary ceiling when
the user has authorized finite time/round limits instead.

## Inspect the external harness

RoboEnv contains native probes, replay tools and owner lifecycle APIs. Agent
sessions, campaign scheduling, artifact freezing and model configuration belong
to an external harness. Inspect that harness's actual commands and receipts before
claiming it supports the requested scope; do not add an agent runner to RoboEnv.
Missing external capabilities must remain explicit. Reuse:

- RoboEnv's task resolver, `roboenv.launch`, owner finalization and
  `roboenv.verification` score/journal checks, separately from task scheduling.
- Kinex's core runtime (`createKinexRuntime` in `packages/core`: devices, tools,
  recording, Python) and the application (`createApplication`, `kinex run`) at their
  current paths.
  Use normal tool discovery and permission/device boundaries, not direct calls to
  the candidate's Python function or a second implementation of the agent loop.
- The repository's actual Python preparation and content-root behavior. Currently,
  the Kinex runtime prepares `.kinex/python/{pyproject.toml,uv.lock}` and
  `.kinex/python/.venv` on first use (`python.prepare()`); there is no `kinex python` command.
  Require the lock before freezing; fresh validation recreates the environment
  with locked dependencies. Do not copy its venv or inherit simulator `PYTHONPATH`.

In Kinex 0.9.0, `kinex run` takes `--model provider/model` and `--thinking LEVEL`
(`--effort` is an alias). Choose a supported current route or make a focused generic
plumbing fix with a request-construction regression.
Do not assume that passing a flag records the effective provider configuration.
Test the supported model/effort path without a network request before live use.

The runner must provide configuration preflight, no-model infrastructure execution,
bounded live development, frozen-artifact final validation, and a precise resume
path. These may be existing commands rather than one new CLI. Run their `--help`
and no-model smoke path, then record exact runnable command arrays/cwd/environment
overrides in the receipt. Never deliver `accept_BACKEND.py` placeholders as commands
that have been tested. Include all needed interpreters, imports and bridge assets.

## Native and no-model path before development

Implement a short bounded controller through Kinex that reads a real sensor,
commands measured native motion and stops. Run it from a newly prepared workspace
through activation, recording, owner submission and independent replay. A task
failure is expected for this probe; Gate C assesses infrastructure without claiming
live-agent or task-performance evidence. A fake provider can test plumbing but cannot replace this probe.

For gripping rigs, native contact conformance includes loaded contacts, retention
of an unsupported dynamic fixture, and release after opening. A pinned fixture
alone can mask collision approximations that eject movable objects. Keep these
owner fixtures separate from task controllers and original task predicates.

Verify complete copied resources, per-process episode pinning and actual tool/driver
interpreter identity. Isolate global/parent workspace content during validation or
include every used dependency in the frozen manifest. Fresh workspaces nested in
an old project can otherwise inherit tools/configuration through find-up discovery.
Builtin Kinex tools remain usable through the recorded Kinex revision.

Declare the public/owner boundary. The runner may create episodes and collect private
verdicts; the candidate may use only declared device/perception inputs. Deny legacy
scene/invoke shortcuts and owner-result access through public transport. For local
runs record the absence of adversarial filesystem isolation; for container evaluation validate
the agent identity, mounts, owner socket, verifier and image provenance in deployment.

Configure renderer warmup and all stage timeouts; do not impose LIBERO's historical
180-second launch limit on an Isaac backend. In receipts distinguish cold boot from
steady-state stepping. Test concurrent independent native worlds only within the
authorized resource allocation; insufficient capacity leaves that check pending.
Use a finite, supported media-check timeout in the external harness for complete
recording decoding; do not add ignored configuration keys or weaken validation.
In realtime read-only tests, distinguish ordinary scheduler ticks from additional
steps requested by a read. Compare serialized observations and step accounting;
elapsed wall time alone cannot establish a read-induced physics effect.

## Development and final freeze

Use real Kinex model sessions to observe, implement and debug a reusable controller
tool, conventionally `solve_task`, using only allowed inputs. Development cases and
tuning-validation feedback can guide later rounds within budget. Honor exactly N
rounds when requested; otherwise stop early only if the declared policy permits it.
Verify camera observations reach the model as valid image inputs through normal
Kinex tool results, not merely as files in the recorder. Inspect attachment/source
identity without retaining credentials; a text-only model request does not prove
the requested visual control path.

Select the final artifact using development/tuning results only. Freeze tool/skill/
library code, local resources, Python manifest/lock, Kinex builtin dependency identity
and effective tool arguments. Verify devices and drivers were not changed by the
developing agent. Hash static profiles/driver bundles before and after each trial;
record new endpoint/session connection files separately and fence them per episode.
Do not carry observations, recordings, live state, owner metadata or conversations
into a new validation workspace. Learned reusable controller code may carry over.

Check tool naming through normal discovery before freezing. Kinex resolves the
manifest's `name`; inspect the external harness's artifact manifest and selected
tool directory together. Keep their identifiers consistent (for example, `solve_task`). Normalize a draft during development and
revalidate it there; do not rename frozen content or repair its manifest after
final validation starts.

Final acceptance creates a new Kinex process/workspace and native world for every
case. Recreate Python from the frozen lock and call the controller through normal
Kinex tool execution, without model calls or repairs. Verify that model devices,
network credentials or inherited tools cannot silently supply agent reasoning in
this mode; any separately authorized perception model must be pinned and declared.

Do not send final hold-out results back into a development round and continue to
call the same cases held-out. On failure, report it. If further development is
authorized, use the exposed cases as tuning data and reserve new untouched final
cases under an amended protocol. If none are available, state that limitation.
For an exact-round study, record per-round tuning results and run final hold-out
only after the predeclared final-artifact selection; do not pick the best hold-out.

Verify repeatability cases load the same effective initialization and the third
case loads a distinct one. Seeds alone are insufficient: record native scene,
layout, activity instance and initialization fingerprints as applicable. Keep these
owner-side. Compare state identities under the predeclared native tolerance when
byte equality is inappropriate; do not weaken it after observing failures.

## Owner record and independent replay

Inspect `finalize_episode.py`: it submits through the owner channel and saves the
adapter's post-submit observation. A generic finalizer does not create missing
backend replay evidence. Implement the selected adapter's private record and replay
reader together, including:

- Source, assets, task definition, robot/controller and initialization identities.
- Native/control frequencies, initialization settling, budgets and terminal reason.
- Ordered commands at native/control frames, composed actuator groups, retained
  targets, stop events and any controller-mode changes needed to reproduce physics.
- Relevant final robot/object/controller/system state and native predicate result.

Replay in a separate simulator process from verified initialization, using recorded
commands and no candidate controller/model/owner final-state restoration. Compare
state, budget and native verdict. A second execution of the controller or playback
of video is a different check and cannot replace command replay.

Define state coverage and tolerances before evaluation. MuJoCo flattened-state
equality is not a universal SAPIEN/Isaac test. Include task-relevant dynamic systems
when applicable; a joint-only comparison cannot establish object/task replay.
If deterministic replay cannot be established, leave the required gate failed or
pending and explain the evidence needed; do not quietly reduce its coverage.

## Pass conditions and failure tests

Each task-performance final trial must have normal, non-timeout tool completion, real successful
device writes corroborated by owner evidence, native task success, unchanged frozen
artifacts/configuration, valid recordings and passed independent replay. Decode
actual requested camera streams; check timestamps, observed acquisition cadence,
nonblack/readable frames, state samples and recorder shutdown. Do not derive sensor
cadence solely from encoded MP4 FPS. Device acceptance is not arrival evidence.

Check the backend's native terminal behavior. Some episodes end before a
controller reaches its planned final waypoint; the current RoboDojo write refusal
also covers its native end flag. A controller may use readable terminal sensor
observations to verify its task objective and finish without further physics.
A rejected write alone proves neither success nor horizon exhaustion. Preserve
native termination semantics and keep the evaluator verdict owner-only.

Before paid development, test that the applicable acceptance machinery rejects:

- A task-success claim based only on a controller return for an unsolved task.
- A task-success claim based on matching command replay of a failed episode.
- Missing/skipped required checks, missing media/replay, or a failed child process.
- Mutated frozen code, dependency lock, driver/profile or episode identity.
- Substituted initialization, reordered commands or incompatible replay sources.

Run both a passing fixture and these negative checks through the actual aggregation
path; then execute the real gates. Required native checks must fail closed when
their simulator/interpreter/camera prerequisites are absent, even if the ordinary
CPU suite skips optional GPU tests. Missing gates cannot yield overall success.

## Receipt, regression and resumption

Record configuration, commands, effective identities, completed/skipped checks,
per-trial conditions and failures in machine-readable receipts. Identify the frozen
artifact chosen for acceptance and preserve every development/retry attempt. A
budget limit or infrastructure error is not a native task failure; report both
individually while keeping incomplete acceptance unsuccessful.

Separate infrastructure passes from task solving and local API conformance from
container isolation. Use historical RoboCasa/LIBERO evidence only for its recorded
scope and source identity; architecture reference status does not imply full gates.

After implementation changes, rerun affected checks against the final source set.
Changes to physics/control require new native/replay evidence; catalog/transport/
runner changes require the corresponding gates. Do not mix incompatible sources
or relabel an earlier receipt. Finish with all agreed gates passed or clearly
reported failed/pending, including exact restart commands, remaining budget,
compatible artifacts and unresolved external inputs. Resumption preserves prior
evidence and cannot reset consumed budget or reuse an exposed final hold-out.

On resumption, check the campaign parent, Kinex runner and native supervisor
separately. An interrupted parent can leave the other two running. Verify their
workspace/session identities before taking over; honor the original child deadline
and use the existing owner finalization and shutdown APIs. Keep recovered records,
media and replay beside the interrupted attempt with explicit recovery provenance.
Do not invent the missing parent exit status or call partial recovery a passing
campaign. Preserve reusable code for a new development round when necessary.
