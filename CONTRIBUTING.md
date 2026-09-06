# Contributing to Roboenv

Roboenv owns environment state, simulator adapters, device bindings, budgets and
verification. Agent loops, model credentials, skills, training and agent-specific
Harbor installation belong in their respective projects.

## Before proposing a change

Read [the architecture boundary](README.md), [device bindings](docs/adding-devices.md)
and [simulator integration](docs/adding-simulators.md). Open a design issue before
changing a public protocol, adding an engine dependency to the SDK, or introducing
a new execution authority. Small fixes can go directly to a pull request.

The core license is not yet approved. A contribution does not resolve that decision;
do not copy code or assets without recording their origin and applicable notices.
Never assume that a benchmark's code license also covers its downloaded assets.

## Local acceptance

Run from this repository's root:

    bash tools/bootstrap.sh
    bash tools/check.sh
    .venv/bin/python tools/release_preflight.py \
      --mode development --wheels build/wheels

The development preflight can pass while reporting explicit release blockers.
Its packaging_ready field remains false in that case. It is not release approval.
No simulator, GPU, model credential or parent workspace is needed for CPU checks.

Add a failing regression test before fixing behavior. For devices, cover unsupported
capabilities, invalid arguments, read-only calls, reset/rebind, exhausted budgets,
timeouts, duplicate requests and an exception after a state-changing call.
For exporters, test a clean directory and a locally modified existing export.
Never weaken a safety check simply to make a fixture pass.

## Simulator and task changes

Update a public source pin and its managed patch hashes together. Do not edit
vendored checkouts and leave the change unrecorded. Never initialize submodules
recursively; select only the backends required by the change.

Check in the task generator and small fixtures, not thousands of generated tasks.
Each discovered task must point to actual upstream metadata; exclusions must be
reported. Solutions are opt-in references with explicit provenance, not automatic
claims of an official expert solution or a successful replay.

A physical acceptance report must distinguish source discovery, Harbor loading,
simulator boot, device calls, verifier result and fresh-process replay. Record the
task/seed, controller condition, source commits, image and wheel digests, candidate
program hash and simulator verdict. Report failures as well as successes. A CPU
mock or old image cannot certify a changed simulator implementation.

## Review checklist

- Public contracts are versioned; breaking changes include a migration note.
- New optional dependencies remain optional at SDK import time.
- Tests cover the failure being fixed, not just a happy-path example.
- Documentation commands work from a standalone checkout.
- Source/task/license provenance is explicit and original evidence is preserved.
- No secrets, host identities, personal directories, machine-local deployment
  files, generated task sets, simulator assets or checkpoints are included.
- A second reviewer checks permission, verifier and release-workflow changes.

Keep pull requests focused and explain externally observable behavior. Do not
reformat an entire migrated backend in a functional change. Do not publish private
logs in an issue; follow [the security policy](SECURITY.md).
