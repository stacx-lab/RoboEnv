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
| LIBERO | official LIBERO |
| RoboCasa | robocasa; top-level robosuite |
| RoboTwin | top-level RoboTwin |
| RoboDojo | RoboDojo; nested IsaacLab and curobo forks |
| BEHAVIOR | BEHAVIOR-1K |
| Isaac Lab | standalone IsaacLab |
| SIMPLE | SIMPLE; nested decoupled_wbc, gear_sonic and unitree_sdk2_python |

Initialize only the selected source closure. RoboDojo's IsaacLab fork is separate
from standalone Isaac Lab; their dependency recipes are not interchangeable.

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
and rerun source checks and tests before committing in RoboEnv. Materialize intended
local source changes in managed patches before refreshing.
`prepare` treats selected source trees as disposable: it discards tracked edits
and untracked source files, restores the lock and reapplies current patches on
every call, including interpreter reuse. It preserves ignored assets, environments,
caches, recordings and nested checkouts. Ordinary `sources sync` remains
conservative about dirty checkouts. Keep historical receipts tied to their original base;
CPU and patch checks do not certify simulator replay on the new revision.

RoboCasa tracks upstream `main` in `.gitmodules`, while the gitlink and source
lock pin an exact upstream commit for reproducibility. The current base is
`456174f`; the update from `4f8a298` changes only the upstream README, and its
MIT license bytes are unchanged.
RoboCasa carries two patches. `robocasa-readonly-assets.patch` redirects
temporary object XML through RoboEnv's writable staging helper.
`robocasa-deterministic-counter-regions.patch` de-duplicates candidate
counter-top regions in their original order instead of through a `set`: a set
of XML elements iterates in memory-address order, so at a fixed seed the
region picked for an object (and so the kitchen) changed from one build or
process to the next. Upstream `456174f` still has the `set`.
Device profiles, RGB-D transport, kinematics and control bindings live in
RoboEnv. No downstream commits are created in the RoboCasa submodule.
The [historical CPU validation receipt](../docs/validation/robocasa-device-cpu-4f8a298.json)
remains evidence for `4f8a298`, not this base. The scene-determinism regression
checks repeated CPU kitchen construction through the current adapter; device and
Kinex compatibility use the current conformance tests. These checks do not certify
kitchen GPU boot, task success or fresh-process command replay.

The 2026-09-29 refresh advances RoboTwin to `ea8b211`, RoboDojo to `726e9aa`,
BEHAVIOR to `bd049de`, standalone Isaac Lab to `998bf88`, and SIMPLE's nested
gear_sonic to `d9bfffb`. The other eight pins already match their default branches.
RoboTwin and RoboDojo retain their runtime/optional-client patches. BEHAVIOR's
optional-workflow patch follows the new environment exports and makes Warp a
base dependency because the new normal runtime imports it. RoboEnv's BEHAVIOR
adapter selects scene zero, its robot and its BDDL object scope through the new
batched environment API. Isaac Lab now defines
dependencies in `pyproject.toml`: its old coverage patch is obsolete, while the
CloudXR opt-in patch moves to the root teleoperation extra and package metadata.
The source notices record the new BEHAVIOR root license (same MIT text) and
Isaac Lab's copyright-year update; the other inventoried notice bytes are unchanged.
Historical validation receipts retain their original source revisions. See
[Isaac Lab](../docs/isaaclab.md) for its current installation compatibility limits.

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
