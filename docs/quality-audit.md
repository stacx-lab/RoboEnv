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

| Backend | Completed evidence | Still required |
|---|---|---|
| CPU fixture | Real Docker/Harbor device trial, reward 1 | Recheck after material contract changes |
| LIBERO / LIBERO-PRO | Pinned-source Docker reference trial; new consistent environment installs and fresh-process device/reference checks | Full candidate-image matrix and additional tasks/seeds |
| RoboCasa | Docker with read-only assets, reference and verifier success after transient-XML fix | Fully fresh simulator dependency installation and wider regression |
| RoboTwin | New consistent environment, real device bridge imports, fresh-process reference and verifier success; terminal budget fix tested | Candidate Docker/embodiment matrix and additional tasks/seeds |
| RoboDojo | Source/binding tests, verified interpreter-depot image, Isaac startup | Complete device/task regression after optional policy-client decoupling; YAM validation |
| BEHAVIOR | Source discovery, symbolic binding tests, verified interpreter-depot image | Real task/device regression and a satisfiable clean-install recipe |

LIBERO and RoboTwin now have reviewed default resolved recipes; original freezes
remain unchanged. Their environment receipts document exactly which dependencies
changed, which sources were installed or source-only, and the single-seed evidence.
See [setup](setup.md) and the reports under docs/validation.

BEHAVIOR's observed package set has independently demonstrated conflicting
mandatory metadata constraints. Its candidate is explicitly blocked, not a clean
lock. A prepared interpreter snapshot is useful but does not prove a consistent
cold installation. RoboDojo/BEHAVIOR remain experimental for this source preview.

The source-catalog format test covers 1,055 definitions: LIBERO 484, RoboCasa 367,
RoboTwin 100, RoboDojo 54, BEHAVIOR 50. It records 534 excluded discovery entries,
including missing definitions, unexpanded composite suites and missing scenes.
Those counts are not execution or success rates.

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
  complete. BEHAVIOR is symbolic-primitives, not continuous motor control.
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
