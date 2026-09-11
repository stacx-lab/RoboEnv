# Kinex Harbor integration

These external bridge, recording, frozen-controller and evidence tools belong to
RoboEnv. They run against an explicitly packaged Kinex source tree without modifying
its checkout; TypeScript imports of Kinex internals require matching versions.

See [the execution guide](../../docs/simulator-kinex-testing.md) for packaging,
standard Harbor tasks/jobs, no-model probes, development, freezing and receipts.
Use `roboenv tasks run --agent kinex`; there is no local campaign scheduler.

TypeScript imports resolve inside the packaged Kinex tree at
`kinex/integrations/roboenv`. Run its tests with the packaged Bun and dependencies.
Python artifact tests run with `python -m unittest discover -s integrations/kinex
-p 'test_*.py'` from RoboEnv. Use `tools/report_kinex_harbor.py` to collect
Harbor trial receipts. A process exit alone cannot certify a trial.
