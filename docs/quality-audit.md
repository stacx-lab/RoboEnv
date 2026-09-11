# RoboEnv quality and support status

This is a private, history-free source preview, not an approved public package or
image release. The core license is intentionally unset by the owner. A passing
development/source-preview gate must not be reported as formal release approval.

## Implemented engineering gates

- Independent SDK and runtime distributions, environment-owned device bindings,
  versioned wire contracts, and no agent/training dependency.
- Strict request/config parsing, declared action-schema validation, immutable
  terminal snapshots, retry/dedup evidence, and negative security tests.
- Selective public source pins, exact managed-patch verification, source-only
  bridge recipes, and normal-resolver installation checks.
- One Harbor task generator, real Harbor 0.22.0/schema 1.4 validation, no implicit
  solution, provenance checks, portable deployments, and safe output preflight.
- Independent wheel installation, reproducible contract-dependency staging,
  package RECORD/source checks, SBOM and vulnerability-audit tooling.
- CPU/Python CI, source-catalog CI, protected manual GPU validation, contribution
  and security guidance, privacy checks, and clean source-snapshot preparation.

The private snapshot includes source, small examples, instructions and pinned
Gitlinks. It excludes old Git history, generated benchmark expansions, experiment
records, credentials, machine-specific deployment files, assets and checkpoints.

## Evidence and backend boundaries

| Backend | Current integration | Validation boundary |
|---|---|---|
| CPU fixture | Device, worker and Harbor contract tests | Synthetic world; no native physics claim |
| LIBERO | Official source, scene-specific Panda devices | Native replay and task success need matching receipts |
| RoboCasa | Panda-Omron devices and continuous control | Installation, native execution and replay are separate gates |
| RoboTwin | Source-backed dual-arm typed devices | Each embodiment and control protocol needs its own evidence |
| RoboDojo | Source-backed ARX-X5 and YAM devices | Layout, assets and native execution must match the selected task |
| BEHAVIOR | R1Pro typed devices and clock support | Symbolic receipts do not validate continuous device control |
| Isaac Lab | Experimental Go2 posture devices | Not an officially supported backend |

Current recipes and installation boundaries are documented in [setup](setup.md)
and [the installation reference](installation-reference.md). Discovery counts
and execution coverage must be measured against the selected current sources.

## Integrity and scoring

Read-only interpreter depots are independently copied executable dependencies,
not ordinary asset data and not original source-tree mounts. Image-owned system
Python verifies the complete file, mode and symlink manifest before a depot
interpreter runs. Cross-depot links must terminate in the verified set; health
checks never execute the depot interpreter before validation.

Reference programs are project-authored unless a reviewed upstream source proves
otherwise. Oracle exit, simulator completion, a valid verifier verdict, and a
fresh-process replay certificate are distinct evidence levels. The validator
records them separately and does not call a reward-zero reference a success.
Every receipt applies only to its exact program, SDK/runtime and image identities.

## Remaining release constraints

- License approval and upstream asset/binary redistribution review are not done.
- A shared agent-container verifier does not defend against a hostile container
  root. This is a controlled evaluation environment, not a multi-tenant hardware
  safety system.
- Some action schemas remain broad; generic IK/FK, emergency stop, ROS and
  high-frequency telemetry are not implied by the device type.
- Full backend/embodiment/seed regression and all-task runtime coverage are not
  complete. Historical BEHAVIOR symbolic receipts do not validate its current continuous device protocol.
- Strict lint covers the new SDK/tools; migrated runtime still has type/style
  debt, with fatal-error checks enforced across the complete runtime.
- A wheel/core SBOM or audit does not cover a complete simulator image. Image
  scanning, signatures, public publishing and configured GPU-runner protection
  remain explicit operator/release responsibilities.

## Reproduce the checks

    bash tools/bootstrap.sh
    bash tools/check.sh
    .venv/bin/python tools/clean_install.py --wheels build/wheels
    .venv/bin/roboenv sources status
    ROBOENV_TEST_UPSTREAMS=1 .venv/bin/python -m pytest \
      tests/test_taskset.py::test_every_discoverable_upstream_task_passes_harbor_loader -q -s

See [GPU validation](gpu-validation.md) for explicit model-free execution and
[release engineering](releasing.md) for artifact-bound audits and promotion.
