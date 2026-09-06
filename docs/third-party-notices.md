# Third-party source notices

The source lock is authoritative for URLs and exact revisions.
[source-notices.json](source-notices.json) records the SHA-256 of each listed
license/notice at that revision. The preflight checks coverage, revisions and
available bytes; --verify-upstreams also requires the actual checkouts and patches.

This is a factual inventory, not a legal opinion or redistribution approval.
It neither grants a license to Roboenv nor transfers upstream copyrights.

| Locked source | Notices inspected within that checkout |
|---|---|
| capx | LICENSE (MIT text) |
| libero_pro | LICENSE (MIT text) |
| libero_robosuite | LICENSE (MIT text) |
| robocasa | LICENSE (MIT text) |
| robosuite | LICENSE (MIT text) |
| robodojo | LICENSE (MIT text) |
| isaaclab | LICENSE (BSD-3-Clause), LICENSE-mimic (Apache-2.0) |
| curobo | LICENSE (Apache-2.0), LICENSE_ASSETS (component-specific notices) |
| robotwin | LICENSE (MIT text) |
| behavior | OmniGibson/LICENSE and bddl3/LICENSE (MIT text) |

The BEHAVIOR entries cover those code components, not the entire benchmark's
assets or data. The cuRobo asset notice refers to additional component licenses.
The IsaacLab code notice is not permission to redistribute Isaac Sim binaries.
Keep applicable upstream notices with redistributed material; do not replace them
with the future Roboenv core license.

## Still requiring a decision or review

- The copyright holders must approve a core Roboenv license. Neither package
  currently has an approved LICENSE; strict packaging preflight blocks release.
- Audit migrated first-party files and managed patch authorship; a public Git URL
  alone is not a contribution-rights review.
- Review each asset, texture, robot model, dataset, checkpoint and binary separately.
  Download access, a local asset installation, or an upstream code license is not
  recorded here as redistribution clearance.
- Image publication requires its own component inventory and notices, including
  system packages and simulator dependencies. Package wheels do not bundle the
  third-party checkouts or large assets.

Before changing a source pin, inspect the new notice files and deliberately update
the inventory. A changed hash must never be accepted merely to silence the check.
