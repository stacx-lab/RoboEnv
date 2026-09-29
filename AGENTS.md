# Working on RoboEnv

RoboEnv provides agent-independent robotics environments: simulator workers,
device APIs, task catalogs, budgets, and run certification.
Read README.md first; use docs/setup.md for installation and
docs/harness-integration.md for building a harness or benchmark on it.

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
Discovery reads pinned upstream task metadata without importing simulator engines
and does not install assets or start an agent. Task bundles, trial scheduling and
agent execution belong to the benchmark harness (see docs/harness-integration.md).
The depot interpreter mode of `tools/build_simulator_image.py` (`runtime/packaging/`)
has no current consumer and is retained pending a maintainer decision; where it is
used, a depot is a separately copied, read-only, content-verified tree, never a
mount of the original project tree.

Every discovered task must resolve against the checked-in robot catalog. Keep only
the discovery code and small examples in Git. Project reference programs are not
official solutions and need explicit opt-in and provenance.

## Changes and verification

- Preserve benchmark success rules, horizons, units, and observation restrictions.
- Validate inputs before effects; unsupported actions must fail explicitly.
- Keep read-only operations, lifecycle ownership, retries, and final snapshots consistent.
- Record upstream changes as locked patches; never rely on unrecorded checkout edits.
- Never create commits inside submodules. Before maintaining a selected upstream,
  run `roboenv sources check-updates --backends BACKEND`. Use the latest upstream
  default-branch commit as the base, review changes, and rebase managed patches.
  Commit only the upstream gitlink, lock revision, patch files/checksums, and
  corresponding notices in RoboEnv. Preserve historical validation receipts;
  a new upstream base requires new validation, not relabeled old evidence.
- Add positive and negative tests, then run tools/check.sh.
- CPU checks and catalog checks are not GPU replay certificates.
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
