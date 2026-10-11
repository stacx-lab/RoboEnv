# Reproducible simulator validation

Use the backend-specific native checks below with prepared interpreters and assets.
Select GPU, task, rig, clock and output scope explicitly. These commands can execute
real motion; neither documentation updates nor CPU tests authorize GPU/model runs.
There is no bundled task scheduler or unified GPU CI runner.

## Evidence boundaries

Report boot, device behavior, controller completion, native task success, recording,
and fresh-process command replay separately. A successful probe need not solve the
task. Replaying a failed episode does not make it successful; a controller return
or video playback does not establish native success or command replay.

`tools/finalize_episode.py RUN_RECORD --out NEW_DIRECTORY` freezes an owned episode
and saves private verdict/observation evidence. `roboenv.verification` provides
`attach_budget`, `validate_score`, and `validate_journal` for owner-side evidence
checks. Journal checks reject changed episode identity, incomplete operations,
forbidden resets, repeated submission and post-submit mutation. They retain the
single-generation trial contract and do not create a replay certificate.

Preserve original receipts, failed attempts and source/artifact identities. Do not
relabel historical evidence after changing code. Raw logs, journals, observations,
credentials and local paths stay private. Share only reviewed, sanitized summaries.

## Terminal evidence

Normal episode completion and explicit stop also persist the sealed owner
observation in `native-episode.json` for native device backends, or
`final-observation.json` for other adapters. `terminal.json` binds these files by
SHA-256 and records saving failures separately from the native verdict. This
evidence is independent of spectator recording and stays in the private run
directory; it is not part of agent observations.

RoboDojo additionally writes `native-diagnostics.json`: registered goal/score
condition trees, actual predicate evaluations and stage progression, and
read-only terminal checks with native measurements, comparison operators and
units. Detailed terminal checks currently cover the predicates used by Build
tower and Fasten screws. History-dependent conditions retain their actual
evaluation frame and are not rerun; unsupported predicates or condition groups
are explicitly reported. Terminal geometry, historical completion, and the
authoritative verdict are separate fields. An abruptly killed worker cannot
guarantee a final snapshot. Existing deployments must be prepared with the new
runtime and managed native patch before this evidence is available.

## Native device checks

Native tools use prepared simulator interpreters and assets. Select the tool for
the backend and use its `--help` for task, rig, clock and output arguments:

| Tool | Scope |
|---|---|
| `tools/accept_libero.py` | LIBERO device, transport, recording and fresh-process replay gates without model calls |
| `tools/accept_simple.py` | SIMPLE native/transport/Kinex checks and replay without model calls |
| `tools/check_go2_d1_transport.py`, `tools/replay_go2_d1.py` | Go2–D1 transport checks and command replay |
| `tools/check_native_robotwin.py`, `tools/check_native_robodojo.py`, `tools/check_native_behavior.py` | Backend-specific native device conformance |
| `tools/check_behavior_cameras.py --out NEW_DIRECTORY` | BEHAVIOR first-read camera delivery after fixture movement, across three agent and three spectator cameras; unchanged owner state |
| `tools/check_native_clock.py` | Worker clock progress, motion, terminal freeze and independent replay |
| `tools/check_native_transport.py` | Concurrent episodes, deduplication, lost replies, public/owner separation and endpoint reuse |
| `tools/replay_native_devices.py` | Fresh-process replay of matching RoboTwin, RoboDojo or BEHAVIOR owner records |

Transport checks select an RGB/RGB-D camera from the live inventory; `--camera`
overrides it. `--seeds FIRST SECOND` selects native layouts for RoboTwin and
RoboDojo. BEHAVIOR also needs distinct `--instances FIRST SECOND`; seeds alone
do not select different presampled activity instances. The check rejects
effectively identical loaded states.

Use `CUDA_VISIBLE_DEVICES` for every backend and native/replay tool. RoboEnv maps
logical CUDA device zero to the matching renderer device and supplies internal
settings; no separate `OMNIGIBSON_GPU_ID` or EGL selector is required. Verify the
actual renderer/physics device in native logs. A source or import check does not
establish GPU readiness.

Record the effective task, rig, clock, initialization, source/artifact identities
and commands with each result. Preserve failed attempts. CPU/schema checks,
native motion, recording, task success and command replay are separate evidence.
For optional Kinex device checks and recording, use [local Kinex use](kinex-local.md).
For standalone images and depot validation, see [setup](setup.md#standalone-simulator-images).
External CI operators must enforce trusted-runner access, resource limits and
artifact privacy; backend acceptance is not established by configuring a runner.

Kinex smoke and rig checks use the built `packages/core` runtime with Node >=22.19.
Select a checkout with `--kinex` (or `KINEX_ROOT`) and a Node executable with
`--node` (or `NODE`) in the LIBERO and SIMPLE acceptance runners. Rebuild Kinex core
after updating its source before running these checks. Python driver tests use
that checkout's `packages/core/assets/python` library.

## Additional LIBERO control smoke

With LIBERO prepared and Kinex core built (Node >=22.19), run:

```sh
.venv/bin/python tests/libero_control_wait.py --kinex /path/to/kinex \
  --out /tmp/libero-control-wait
```

This starts and disposes fresh services on free ports, exercises Kinex `servo`
and `world.wait`, and records motion outcomes without model calls or a task-success
claim. Existing services and workspaces are preserved.

## Outstanding native validation

Current humanoid backends still need native GPU pacing, camera-load, balance,
and fresh-process realtime replay validation. Retired Isaac Lab G1 implementation
notes are historical, not evidence for current SIMPLE G1 behavior. CPU checks do
not certify GPU replay or task completion.
