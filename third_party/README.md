# Pinned upstream dependencies

These are real registered submodules, not a proposed directory convention.
The authoritative revisions, public URLs, dependency graph and patch SHA256s are
in upstreams.lock.json. Source initialization is selective, not recursive.

    roboenv sources sync --backends libero robocasa
    roboenv sources status --backends libero robocasa
    roboenv sources check-updates --backends robotwin

The sync command initializes missing registered submodules, checks origin URLs,
checks out exact reachable commits and applies only checksum-verified managed
patches. It refuses to switch a dirty checkout to another revision. Status checks
pins, origins, patch presence and the exact managed working tree; it is not a
claim of GPU or asset readiness. Extra unrecorded source changes fail the check,
without modifying the contributor's real Git index.

| Backend | Source closure |
|---|---|
| LIBERO / LIBERO-PRO | cap-x; nested LIBERO-PRO and its robosuite fork |
| RoboCasa | robocasa; top-level robosuite |
| RoboTwin | cap-x; top-level RoboTwin |
| RoboDojo | cap-x; RoboDojo; nested IsaacLab and curobo |
| BEHAVIOR | BEHAVIOR-1K |

Do not substitute the two robosuite checkouts or recursively fetch unrelated
policy repositories. RoboTwin is explicitly versioned here instead of relying
on a hidden nested clone.

Our SDK, adapters, bindings, tests and scoring remain outside submodules. Changes
to an upstream are recorded in patches/ and in the lock; do not make unrecorded
development-tree modifications part of a published environment.

## Maintaining upstreams

Never create commits inside submodules. `check-updates` queries live upstream
default-branch tips for the selected source closure; it does not initialize,
checkout, stage, or modify sources. It exits nonzero when pins are behind or a
remote query fails. Ordinary `sync` continues to use exact pins for reproducibility.

When updating, first verify that `sources status` reports an exact managed tree.
Review the upstream diff and test the existing patches against the latest commit
in an isolated checkout or temporary index. If they need changes, preserve local
work, regenerate the patch against the new base, and update its SHA256 in the lock.
Advance the submodule gitlink and lock revision together, check license notices,
and rerun source checks and tests before committing in RoboEnv. Do not force a
checkout over local edits. Keep historical receipts tied to their original base;
CPU and patch checks do not certify simulator replay on the new revision.

RoboCasa tracks upstream `main` in `.gitmodules`, while the gitlink and source
lock pin an exact upstream commit for reproducibility. The refresh from
`a07e365` to `4f8a298` changes only three upstream test scripts (robosuite API,
task names, and fixture-style coverage); its MIT license bytes are unchanged.
RoboCasa carries two patches. `robocasa-readonly-assets.patch` redirects
temporary object XML through RoboEnv's writable staging helper.
`robocasa-deterministic-counter-regions.patch` de-duplicates candidate
counter-top regions in their original order instead of through a `set`: a set
of XML elements iterates in memory-address order, so at a fixed seed the
region picked for an object (and so the kitchen) changed from one build or
process to the next. Upstream `main` still has the `set`.
Device profiles, RGB-D transport, kinematics and control bindings live in
RoboEnv. No downstream commits are created in the RoboCasa submodule.
Fresh CPU/source/packaging checks for this base are recorded in
[the validation receipt](../docs/validation/robocasa-device-cpu-4f8a298.json).
Historical simulator receipts remain unchanged; kitchen GPU boot, task success
and fresh-process replay are not certified by this CPU check.

The 2026-09-06 source review advances RoboDojo to `ee67a14` (optional
articulation drive reset and evaluation/download tooling) and LIBERO-PRO to
`47aaa80` (additional task variations, torch loading defaults and CLI entry
points). Their managed patches and license bytes are unchanged. Isolated patch,
syntax and notice checks are recorded in
[the source review](../docs/validation/upstream-source-review-20260906.json);
these new bases have no new simulator boot, device, success or replay certificate.
Historical receipts retain their original revisions.

LIBERO's robosuite fork remains at `a498b08`: latest upstream `cb173eb` removes
`SingleArmEnv` and `load_controller_config`, which LIBERO and its runtime still
require. Updating this source needs a compatibility migration, so
`check-updates --backends libero` currently reports this one stale pin. All other
source pins matched their upstream default-branch tips on 2026-09-06.

A patched submodule normally appears dirty in `git status`, including untracked
files added by a patch. This is expected when `sources status` reports
`managed_tree_matches: true`: the patch in RoboEnv already records those changes.
Do not hide this state with Git ignore flags or commit the patched tree upstream.

Assets and checkpoints are separate licensed downloads, not submodule contents
guaranteed by this command. LFS smudging is disabled during source sync. The image
builder accepts prepared assets as read-only depots and records the final image
digest. Existing external checkouts and experiments are not moved or rewritten.

Each upstream retains its own license. The source lock is not permission to
redistribute every asset, simulator binary or checkpoint. Review those licenses
before publishing images; the core project's license decision is still a release
gate, documented in the quality audit.

See [installation and asset receipts](../docs/setup.md) for strict per-simulator
preflight, current clean-install blockers, and portable SHA256 depot manifests.
