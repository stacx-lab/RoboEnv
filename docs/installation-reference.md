# Installation reference and validation evidence

Roboenv has three separately verified inputs: its own Python distributions,
simulator source and dependencies, and licensed asset depots. A passing check at
one boundary is not evidence that the next boundary works.

## CPU development and an independent wheel installation

From the Roboenv repository root, install uv using its
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
bash tools/bootstrap.sh
bash tools/check.sh
.venv/bin/python tools/clean_install.py --wheels build/wheels \
  --out .cache/clean-install-receipt.json
```

Bootstrap uses Python 3.12 and the development lock, disables user site-packages,
and finishes with a dependency consistency check. An optional first argument
selects a different virtual environment. Existing non-environment directories are
rejected; an existing development environment is reused, never deleted.

The clean-install check does something different from editable-source tests: it
creates a new temporary environment, installs both wheels with normal dependency
resolution and Harbor 0.22.0, checks dependencies, changes outside the source
tree, and probes the SDK, worker, supervisor, and Harbor commands. It verifies
that imports actually originate in the new environment and that the real Harbor
task model accepts schema 1.4. No parent workspace, agent implementation, GPU,
simulator installation or credential is needed. Dependencies still require an
accessible package index; `--offline` requires a populated uv cache.

The receipt records wheel hashes and completed checks, not host locations.
Failed installation diagnostics remain in the ignored, owner-readable file
`.cache/clean-install/last-run.private.log`. Do not publish raw installer logs.

## Select and inspect one simulator

```bash
.venv/bin/roboenv sources sync --backends robocasa
.venv/bin/roboenv sources status --backends robocasa
.venv/bin/python -m roboenv.provisioning.setup robocasa
```

Source readiness verifies public origin URLs, exact commits, patch hashes, and
the exact tracked tree after those patches. The check uses a temporary Git index:
it does not overwrite a contributor's working tree or staging area. Unrecorded
source edits fail readiness. Ignored caches and asset depots are checked separately.

`runtime/environments.json` names each interpreter version, resolved dependency
lock, editable packages, **source-only** checkouts, Git source pins, package
indexes, and upstream asset instructions. The default setup command is read-only.
It rejects duplicate, ranged, malformed, or unresolved requirements and reports
known conflicts explicitly. Recipe paths must stay inside the selected checkout.

An explicit installation can download many GB and creates a **new** environment:

```bash
.venv/bin/python -m roboenv.provisioning.setup robotwin \
  --install --prefix .venv-sim/robotwin --wheels build/wheels
```

It never modifies an existing environment or silently bypasses the dependency
resolver. One normal resolver invocation installs the lock, selected SDK/runtime,
and required editable packages together. It then runs dependency consistency,
installed Git identity and import-origin checks. Source-only checkouts are not
installed as Python distributions. Import probes disable GPU visibility and do
not create an episode. Failure preserves the partial new environment and a unique
owner-readable log under `.cache/setup/`; it never produces a successful certificate.

Package inputs are explicit and independent of the shell's active environment:

```bash
# Default: SDK at this project root and runtime at this project's runtime/.
.venv/bin/python -m roboenv.provisioning.setup libero \
  --install --prefix .venv-sim/libero

# Both selected wheels; neither package falls back to a neighboring workspace.
.venv/bin/python -m roboenv.provisioning.setup libero \
  --install --prefix .venv-sim/libero-wheels --wheels build/wheels

# Mixed inputs, when intentionally validating a different package build.
.venv/bin/python -m roboenv.provisioning.setup robotwin \
  --install --prefix .venv-sim/robotwin-mixed \
  --sdk build/wheels/roboenv-0.2.0-py3-none-any.whl --runtime runtime
```

`--sdk` and `--runtime` each accept a package source directory or a wheel.
They cannot be combined with `--wheels`. `--project` selects a different Roboenv
checkout explicitly; omitted package arguments then resolve inside that checkout,
not its parent. Running the installed simulator outside that checkout requires
`ROBOENV_ROOT` to identify the selected, pinned source root. Generated native
images set their own container-local source root.

## Resolved RoboCasa recipe

RoboCasa uses `runtime/locks/robocasa-resolved.txt` on Python 3.11 and the explicit
PyPI index. The original `robocasa.txt` remains a historical freeze: its
`gym-notices==0.2.0` pin is unavailable. The resolved recipe changes that pin to
`0.1.0`, preserves all other historical versions, and pins five previously
implicit SDK/server dependencies. Resolution and an install dry-run cover the
complete recipe, including both simulator source packages and the SDK/runtime.
A fresh isolated installation of all 143 packages also passed dependency
consistency and SDK/runtime import-origin checks. These checks did not boot a
RoboCasa world or validate assets or GPU replay.

## Promoted LIBERO and RoboTwin recipes

The defaults now use `runtime/locks/libero-resolved.txt` and
`runtime/locks/robotwin-resolved.txt`. Original `libero.txt` and `robotwin.txt`
freezes and the candidate receipts remain untouched as historical evidence.
The promoted locks match the tested candidate requirements exactly.

Both install **one** OpenCV distribution, `opencv-python==4.11.0.86`, because
the selected robosuite/SAPIEN packages require that distribution. NumPy stays at
1.26.4; installing desktop and headless OpenCV together is not a substitute for
dependency consistency. RoboTwin retains Torch 2.4.1, SciPy 1.10.1, and JAX 0.4.29.
LIBERO retains its tested MuJoCo, Torch, NumPy, and JAX core pins; its full resolved
dependency changes are recorded in the [LIBERO validation receipt](validation/libero-candidate.md).

The full cap-x distribution includes agent/training requirements that are not
the Roboenv simulator dependency boundary. Instead, these recipes load the needed
bridge modules from the source-locked checkout. LIBERO-PRO and its robosuite fork
still install normally as editable packages; RoboTwin's benchmark source is
source-only. No fake `capx` distribution is installed, and no dependency resolver
is bypassed. Current cap-x initialization eagerly imports Franka IK types, so
the locks still include PyRoKi/JAXLS even on RoboTwin's native-planner path.

The two `*-source-pins.txt` files only bind PyRoKi's floating JAXLS Git URL to the
same exact Git source already declared in the resolved lock. The setup audit
rejects numeric version overrides, a different repository/revision, extras,
markers, or packages absent from the lock. Installation additionally checks the
installed commit and the actual consuming packages' source/version requirements.
It is not permission to overwrite incompatible dependency constraints.

Current validation boundaries:

| Backend | Setup preflight | Remaining installation evidence |
|---|---|---|
| LIBERO / LIBERO-PRO | Resolved recipe; clean installation checked | One fresh host seed-0 reference and device checks passed; new clean-environment container verification remains separate |
| RoboCasa | Corrected recipe; fresh installation and package checks passed | Simulator boot, assets, and GPU replay remain unverified |
| RoboTwin | Resolved recipe; clean installation checked | One fresh host seed-0 reference and unchanged independent verifier passed; new clean-environment container verification remains separate |
| RoboDojo | Source and freeze syntax pass | Execute and verify a fresh full simulator installation, assets, and GPU episode |
| BEHAVIOR | Blocked | No clean-install dependency lock published yet; upstream license acceptance remains explicit |

Read-only audit results keep `clean_install_verified` and `gpu_replay_verified`
false for the current invocation. `recorded_recipe_validation` separately reports
the checked-in historical evidence. A successful new installation sets only its
clean-install result; it does not inherit another machine's GPU result.

The host proofs cover LIBERO `libero.90_8`, seed 0, at 2779/4000 simulation steps,
and RoboTwin `robotwin.handover_mic`, seed 0, at 64/60000 API calls. The reference
programs are project-authored, not claims of official benchmark solutions.
See the [RoboTwin candidate receipt](../runtime/locks/robotwin.candidate.json)
for its exact package/source changes and the independently checked trial.
Neither proof covers all tasks, seeds, embodiments, optional services, training,
or a newly built Docker image. The locks pin package versions and Git revisions;
they are not yet hash-locked binary wheel sets.

## Assets: explicit provenance and content verification

`roboenv prepare robocasa` automatically downloads the pinned upstream kitchen
asset bundles and keeps local content receipts for reuse and repair. These local
receipts do not establish redistribution approval or GPU readiness. The low-level
installer still installs packages only.

For other assets, follow the instructions in the selected **pinned upstream**,
review its license, and acquire them independently. Roboenv setup does not accept dataset
terms, download checkpoints or imply permission to redistribute an image.

Record a data-only depot after acquisition, using the actual source/version and
license URL rather than copying the placeholders below:

```bash
.venv/bin/python -m roboenv.provisioning.assets create ASSET_DEPOT \
  --asset-id example-assets \
  --source-url https://example.org/assets \
  --source-revision RELEASE_VERSION \
  --license-url https://example.org/license \
  --out deployments/local/example-assets.json
.venv/bin/python -m roboenv.provisioning.assets verify ASSET_DEPOT \
  deployments/local/example-assets.json
```

Manifests contain relative filenames, byte sizes, SHA256s and declared provenance,
never the depot's host location. Absolute or escaping symlinks, special files,
source-checkout roots and tokenized provenance URLs are rejected. Internal links
are recorded without following them outside the depot. Keep depots quiescent while
hashing; full verification deliberately reads all bytes and may take time for
large assets. Verification detects changed, missing, or extra files.

The manifest does **not** authenticate an upstream release or approve its license.
Reviewing redistribution rights is still a separate release gate. Store the
receipt outside the depot; a new output is required, so an old receipt cannot be
silently overwritten. Asset verification is an explicit operator check today,
not an automatically enforced part of every deployment manifest.

## Device-contract dependencies in simulator images

The image builder accepts --interpreter ENV/bin/python independently of
--dependencies, so a freshly installed environment need not use a historical
directory layout. --interpreter-mode image embeds it; --interpreter-mode depot
creates an independently copied, relocated, read-only executable depot and a
smaller image. The latter verifies complete content/modes/symlink targets using
trusted image Python before executing the interpreter. It is not a mount of the
original prepared environment, and it is not an ordinary non-executable asset.

Staging and Docker storage are checked separately. Failed build copies are
reproducible scratch data, but never remove original interpreters, assets or
experiment evidence to make space. Installed Kit extension links must be copied
into the frozen dependency group; unknown external links are rejected.

The image builder stages `runtime/locks/device-service.txt` into a fresh directory
using `tools/contract_dependencies.py`. It selects wheels for the supplied
simulator interpreter without installing into that interpreter. This keeps JSON
Schema validation available even when an old simulator freeze did not contain it.

The complete six-package closure is pinned; its Python 3.10-compatible `rpds-py`
pin comes from the existing RoboTwin freeze. Actual isolated probes on Python
3.10, 3.11, and 3.12 verify Draft 2020-12 schema resources, accepted/rejected
instances and native-extension imports. Targets are Python-ABI-specific and must
not be copied between incompatible interpreter versions. This is a contract
dependency check, not simulator replay evidence.
