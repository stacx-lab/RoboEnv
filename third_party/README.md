# Pinned upstream dependencies

These are real registered submodules, not a proposed directory convention.
The authoritative revisions, public URLs, dependency graph and patch SHA256s are
in upstreams.lock.json. Source initialization is selective, not recursive.

    roboenv sources sync --backends libero robocasa
    roboenv sources status --backends libero robocasa

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
