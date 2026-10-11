# Working on RoboEnv

RoboEnv provides agent-independent robotics environments: simulator workers,
device APIs, task discovery, budgets, and benchmark verification.
Read README.md first; docs/README.md indexes the design contracts.
Use docs/setup.md for installation and
docs/device-documentation.md for coding-agent integration.

## Layout

- src/roboenv/: public client/lifecycle and shared validation infrastructure.
  Domain packages: devices/, tasks/, provisioning/, cli/, control/.
- runtime/: separate runtime package, simulator adapters, scoring, workers, image recipes.
- catalog/: robot/device facts and small task examples, not a full generated benchmark.
- third_party/: pinned upstream submodules and checksum-verified patches.
- tools/, tests/, locks/: development, validation, packaging, and dependency inputs.
- docs/, .github/: integration guides, release rules, and CI.
- .cache/, build/, generated/: local artifacts; do not commit them.

RoboEnv owns environment state and verdicts. Agents own reasoning and programs;
agent-specific integration and training belong outside this repository.
One world owns lifecycle and budgets; device bindings connect to that world.
Do not reintroduce a second driver stack or an agent loop. Policy inference and
rollout are agent-side work: there is no VLA subsystem, no policy client or
server, and no `vla` operation. Devices are the only control surface.

## Development environment

Use Linux, uv, and Python 3.12 for development tooling.
The SDK supports Python 3.10+; server-side schema validation uses its server extra.
Each simulator has an isolated interpreter and separately acquired assets.
GPU runs require compatible NVIDIA drivers; MuJoCo uses EGL and Isaac needs Vulkan.
Never install all simulator dependencies into the development environment.

    bash tools/bootstrap.sh
    source .venv/bin/activate
    bash tools/check.sh
    python tools/clean_install.py --wheels build/wheels \
      --out .cache/clean-install-receipt.json

## Sources and tasks

Initialize only selected upstreams; do not recursively fetch every submodule.

    roboenv sources sync --backends libero
    roboenv sources status --backends libero
    roboenv prepare libero
    roboenv tasks discover --backends libero --out generated/libero-source

Bootstrap the CLI with `bash tools/bootstrap.sh`, activate `.venv`, then use
`roboenv prepare SIMULATOR` and `roboenv launch TASK_ID`.
Preparation uses a managed default environment or an explicit --prefix;
`roboenv doctor SIMULATOR` only audits. Registered interpreters may be reused; existing environments
are never overwritten. Clean-install verification is for developers, not users.
Task discovery reads pinned upstream definitions or lists an explicit source catalog.
It does not install assets or start an agent. Large interpreters may use separately
copied, read-only, content-verified depots; never mount the original project tree.
Keep only small task examples in Git. Validate discovered task schemas and binding
resolution. Agent runners, task scheduling and controller campaigns belong outside
RoboEnv; Kinex-compatible device catalogs and local checks remain supported.

## Changes and verification

- Preserve benchmark success rules, horizons, units, and observation restrictions.
- Validate inputs before effects; unsupported actions must fail explicitly.
- Keep read-only operations, lifecycle ownership, retries, and final snapshots consistent.
- Treat third-party source working trees as disposable. Materialize every intended
  source change in a managed patch and record its checksum in upstreams.lock.json;
  never rely on direct checkout edits or create downstream submodule commits.
  `prepare` discards local source edits, restores the locked upstream commits, and
  reapplies managed patches even when reusing an existing interpreter. Preserve
  downloaded assets, environments, caches, recordings, and nested checkouts.
  Before an upstream refresh, audit existing edits against the recorded patches
  and materialize any intended unrecorded changes before discarding them.
- Never create commits inside submodules. Before maintaining a selected upstream,
  run `roboenv sources check-updates --backends BACKEND`. Use the latest upstream
  default-branch commit as the base, review changes, and rebase managed patches.
  Commit only the upstream gitlink, lock revision, patch files/checksums, and
  corresponding notices in RoboEnv. Preserve historical validation receipts;
  a new upstream base requires new validation, not relabeled old evidence.
- Add focused positive and negative tests, then run tools/check.sh. Keep default
  CPU, privacy and wheel gates; avoid redundant full-process parameter matrices.
  Gripper stop preserves the applied target/preload; explicit open releases it.
- CPU checks and task-format checks are not GPU replay certificates.
- Report boot, device checks, task success, and fresh-process replay separately.
- Real GPU/model runs require explicit scope; do not invoke paid models implicitly.

## Privacy and publication

Do not commit credentials, hostnames, user-specific paths, caches, assets,
generated task collections, or historical experiment output. Use portable
configuration and keep private diagnostics local. Preserve existing evidence.

The current target is the private stacx-lab/RoboEnv repository, initialized from
the reviewed source snapshot without old history. The core license is intentionally
unset. Do not invent a license or publish packages/images automatically.
Follow docs/releasing.md; source-preview approval is not formal release approval.
