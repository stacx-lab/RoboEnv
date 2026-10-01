# Release engineering

The core license is not yet declared, so strict package release remains blocked.
The repository provides release checks and candidate artifacts; publication
requires a separate maintainer decision. No workflow automatically publishes
packages or simulator images.

## Local checks and the license gate

    bash tools/bootstrap.sh
    bash tools/check.sh
    .venv/bin/python tools/clean_install.py --wheels build/wheels \
      --out .cache/clean-install-receipt.json
    .venv/bin/roboenv sources sync --backends BACKEND
    .venv/bin/python tools/release_preflight.py \
      --verify-upstreams --wheels build/wheels \
      --output build/release/preflight.json

Initialize the selected backends needed for the candidate; strict upstream
verification requires every locked source to be available and match its patches.
The clean-install check creates a temporary environment, installs both wheels
with normal dependency resolution, and checks dependencies, import origins,
CLI entry points outside the checkout, without agent frameworks installed. It needs
an accessible package index or, with `--offline`, a populated uv cache. Receipts
record wheel hashes; raw installation diagnostics stay private.

The default mode is strict: any error or release blocker returns nonzero. It checks
package/module versions, SDK/runtime dependency alignment, long descriptions,
source registrations and exact managed patch state, notice inventory, wheel metadata,
every wheel RECORD checksum, payload/source agreement, known privacy patterns and
obvious credential signatures.
Malformed inputs fail without printing private filesystem locations.

For ordinary development, `--mode development` allows unresolved release-only
prerequisites while still failing implementation errors. Inspect `packaging_ready`;
development or `source-preview` success is not package-release approval.

The core license is deliberately unresolved. Authorized maintainers must select it,
place the approved full text in LICENSE and runtime/LICENSE, and declare the same
valid SPDX expression plus license-files = ["LICENSE"] in each project's metadata.
Custom licenses require an explicit review and corresponding gate change. Do not
create a placeholder license merely to make CI green. Rebuild both wheels after
the decision and verify that their included license text matches the approved text.
See [the notice inventory](third-party-notices.md) for separate upstream reviews.

## Required evidence before publication

| Gate | Required evidence |
|---|---|
| Source and packaging | Strict preflight, exact commit, clean source snapshot, both wheel hashes |
| Clean installation | Installation from built wheels in a new environment, dependency check, CLI and imports outside the checkout |
| Reproducibility | Independent builds using the same locked inputs and SOURCE_DATE_EPOCH; compare outputs, explain any difference |
| CPU and contracts | Passing tests on the declared Python support matrix; negative security tests and protocol-compatibility coverage |
| Simulator claims | Representative boot, device, verdict and fresh-process replay on every claimed backend/embodiment; multiple seeds |
| Security and supply chain | Reviewed scanner results, artifact SBOM, image-specific SBOM/scans, retained notices and provenance |
| Public operation | Approved core license, private disclosure channel, maintainer ownership, reviewed support matrix and upgrade notes |

The packaging script cannot prove the other rows. Missing GPU evidence means the
backend remains experimental or unsupported in that release; do not infer support
from a successful source discovery or schema-validation run. Do not relabel old runs
as evidence for changed binaries. Keep successes and failures in the release notes.

## CI policy

CPU and Python compatibility jobs run on unprivileged hosted runners without
submodules or simulator assets. Source discovery has a separate scheduled/manual
job and uses selective public pins. The manual candidate job additionally creates
an SPDX SBOM, runs vulnerability/secret scans and enforces strict preflight.
It does not turn an unsuccessful preflight into a passing release.

Top-level actions use immutable commit pins, checkouts do not retain credentials,
and default token permissions are read-only. Scanner tools have explicit versions.
Dependency changes are reviewed manually; there is no active Dependabot configuration.
Review tool transitive downloads and action changes.
These choices follow [GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).
An action pin does not by itself make every transitive download reproducible.

Never run untrusted pull-request code on a long-lived GPU runner. GPU automation
needs a disposable runner, trusted/ref-reviewed input, read-only depots, exclusive
resource allocation and reviewed evidence retention. Configure that infrastructure
before enabling a GPU schedule. See [simulator validation](gpu-validation.md)
for evidence boundaries and backend-specific checks.

## Candidate artifacts and promotion

### Real core audit and SBOM

The release tools are fully version/hash pinned in locks/release-tools.txt.
Use separate tool and minimal runtime environments, both on Python 3.12:

    uv venv --python .venv/bin/python .cache/release-tools
    uv pip sync --python .cache/release-tools/bin/python \
      --require-hashes --index-url https://pypi.org/simple locks/release-tools.txt
    uv venv --python .venv/bin/python .cache/release-core
    uv pip install --python .cache/release-core/bin/python \
      --index-url https://pypi.org/simple build/wheels/*.whl
    uv pip check --python .cache/release-core/bin/python
    .venv/bin/python tools/audit_core_release.py \
      --python .cache/release-core/bin/python \
      --tool-python .cache/release-tools/bin/python \
      --wheels build/wheels --out build/release/core

Use new environment locations for a new candidate, or explicitly reinstall both
changed local wheels. The audit refuses unrelated dev packages and mismatched
installed candidate bytes. It queries only public third-party dependency names
and versions; unpublished project wheels are checked locally, not sent to PyPI.
Unavailable advisory data, skipped dependencies, malformed reports and failed SBOM
validation all return nonzero. Known vulnerabilities also block this audit.

The CycloneDX 1.6 SBOM inventories the actual core environment, including the
server extra. Candidate wheel file URLs are removed, replaced by real wheel
hashes, and the resulting document is schema-validated again. The report records
generator versions, UTC time, dependency versions and candidate/SBOM digests.
This checks known advisories, not undiscovered vulnerabilities. The default
development environment and simulator images are different audit scopes.
See the [pip-audit usage and security model](https://github.com/pypa/pip-audit).

Keep package digests, sanitized preflight/security summaries and the SPDX artifact
SBOM together. The wheel SBOM is not a complete installed-environment or image SBOM.
Raw scanner output can contain source snippets or sensitive paths; never upload it.
Scan the final image digest, not only its Dockerfile or a previous tag.

Before promotion, verify the candidate is from the reviewed source commit, add
release notes and a public support matrix, confirm the evidence refers to the exact
artifact hashes, and obtain independent approval. Configure a protected publishing
environment and short-lived identity-based credentials when adding a publisher.
Do not store registry tokens or signing keys in this repository.

Keep old immutable artifacts available for rollback. Publish a new version for a
fix; never replace a released wheel or move an already-certified image tag to
different bytes. Increment protocol versions for incompatible wire semantics and
document migration separately from implementation/package versions.
