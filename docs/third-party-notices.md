# Third-party source notices

The [source lock](../upstreams.lock.json) records URLs and exact revisions.
[source-notices.json](source-notices.json) records the SHA-256 of each listed
license/notice at that revision. The preflight checks coverage, revisions and
available bytes; `--verify-upstreams` also requires the actual checkouts and patches.
This inventory covers source notices, not asset or binary redistribution approval.

The managed Isaac Lab CloudXR patch makes streaming dependencies opt-in through
``isaaclab_teleop[cloudxr]``. Default RoboEnv installs retain the teleoperation
retargeters and UI without CloudXR's incompatible WebSocket requirement.

The managed RoboDojo predicate diagnostics patch adds an optional owner observer
and measurement collection to native reward functions. It preserves native
comparisons, short-circuit evaluation, and history updates. The runtime uses
these hooks for private terminal evidence, without exposing it to agents.

The managed SIMPLE runtime patch routes both renderers through the verified data
depot, aligns the shared G1 USD waist origins with Sonic's physics model, mirrors
full-precision MuJoCo state into Isaac without advancing a second physics world
during capture, and disables process-exiting fast shutdown so the
owner retains exception and replay exit-status handling. It also accepts the
owner's explicit Vulkan GPU selection, with CUDA device zero after visibility
remapping and single-GPU rendering. Patch checksums are recorded in the source lock.

The BEHAVIOR source pin includes the upstream presampled-pose selection option
and CI runner update through `a8247a8`. Its inspected code licenses are unchanged.
The managed GPU patch separates the Vulkan rendering index from CUDA physics
index zero; existing optional-workflow patches remain applied. Historical
validation receipts describe their original source revisions.

| Locked source | Notices inspected within that checkout |
|---|---|
| libero | LICENSE (MIT text) |
| robocasa | LICENSE (MIT text) |
| robosuite | LICENSE (MIT text) |
| robodojo | LICENSE (MIT text) |
| isaaclab | LICENSE (BSD-3-Clause), LICENSE-mimic (Apache-2.0) |
| isaaclab_native | LICENSE (BSD-3-Clause) |
| curobo | LICENSE (Apache-2.0), LICENSE_ASSETS (component-specific notices) |
| robotwin | LICENSE (MIT text) |
| behavior | OmniGibson/LICENSE and bddl3/LICENSE (MIT text) |

The BEHAVIOR entries cover those code components, not the entire benchmark's
assets or data. The cuRobo asset notice refers to additional component licenses.
The IsaacLab code notice is not permission to redistribute Isaac Sim binaries.
Keep applicable upstream notices with redistributed material; do not replace them
with a RoboEnv core license. Assets, datasets, checkpoints and binaries require
separate review; published images need their own component inventory and notices.
Core licensing and contribution rights, including managed patch authorship,
remain subject to the [release gates](releasing.md).

Before changing a source pin, inspect the new notice files and deliberately update
the inventory. A changed hash must never be accepted merely to silence the check.

The managed RoboTwin GPU patch selects the same device for its initial SAPIEN
renderer and scene render system. The legacy renderer wrapper ignores device
arguments; the patch uses the device-aware constructor. The upstream source pin
is unchanged.

## ABC

ABC (`amazon-far/abc`) is pinned in `upstreams.lock.json` under `third_party/abc`.
Its source includes an Apache-2.0 license. Simulator assets retain their upstream
notices and are acquired from its checksum-pinned asset manifest; assets and model
weights are not included in RoboEnv wheels. No policy weights are downloaded.
The managed `abc-unified-arm-mounts.patch` raises both arm bases from 0.75 m to
0.76 m in six task scenes so all supported tasks use one robot configuration.
The managed `abc-blender-live.patch` adds calibrated live camera specifications,
read-only state export, and bounded renderer lifecycle handling. Optional rendering
downloads Blender 4.5.3 under its bundled GPL notices and the CC0 Photo Studio 01
HDRI from Poly Haven. These downloads remain separate from RoboEnv packages.

## RoboBallet

The eight-arm reaching task adapts transition rules from Google LLC's 2025
RoboBallet release, distributed under Apache-2.0. Its original geometry and
collision modules are loaded from the separately acquired archive at
<https://doi.org/10.5281/zenodo.16546403>, pinned by SHA-256
`ffb8447e6d0f309026d9baa8538d0c77931655323cbb532e0bf8c11a617ad537`.
Preparation retains the archive's `LICENSE` and `README.md`; original code and
mesh assets are not embedded in RoboEnv wheels. The checked-in scene records
its source file hashes and the shared target/workpiece translation. Panda/tool
geometry retains the archive's component notices. No policy weights or reference
solutions are included.
