# Third-party source notices

The [source lock](../upstreams.lock.json) records URLs and exact revisions.
[source-notices.json](source-notices.json) records the SHA-256 of each listed
license/notice at that revision. The preflight checks coverage, revisions and
available bytes; `--verify-upstreams` also requires the actual checkouts and patches.
This inventory covers source notices, not asset or binary redistribution approval.

The managed Isaac Lab CloudXR patch makes streaming dependencies opt-in through
``isaaclab_teleop[cloudxr]``. Default RoboEnv installs retain the teleoperation
retargeters and UI without CloudXR's incompatible WebSocket requirement.

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
