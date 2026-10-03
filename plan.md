# Current work

## Velocity feedback and native limits

- [x] Remove global base pose from public velocity feedback across all four backends.
- [x] Preserve local torso/height feedback, explicit frame estimates and native command semantics.
- [x] Publish RoboCasa's normalized input bounds and retain named-axis writes.
- [x] Validate the refactor with CPU/source/package checks and prepared RoboCasa CPU physics.

## Failed-write cleanup

- [x] Cancel the submitting execution when portable actuator feedback validation fails.
- [x] Preserve replacements and return structured, latched cleanup failures.
- [x] Refresh catalog drivers and correct stale feedback/FK documentation.
- [x] Complete regression, source and RoboEnv packaging gates with Kinex.

Validation: tools/check.sh passed with 1,087 tests and 40 skips, privacy audit
and wheel checks. Kinex typecheck, standalone/core/setup suites, build and local
tarball packaging passed.

Existing copied workspaces need refreshed profiles, actuator drivers and shared
transport resources. Restart existing RoboEnv services and reload Kinex devices
after upgrading.

## Outstanding native validation

- [ ] Native GPU pacing, camera-load, balance and fresh-process realtime replay
  validation for current humanoid backends. Retired Isaac Lab G1 implementation
  notes are historical, not evidence for current SIMPLE G1 behavior.

CPU checks do not certify GPU replay or task completion; report those separately.

## Model-defined joint frames

- [x] Remove computed-frame placeholders from backend builders, hand geometry and catalogs.
- [x] Use Kinex model-defined frame discovery in exported joint/hand drivers.
- [x] Update documentation and regression coverage.
- [x] Complete CPU, catalog consistency and cross-repository packaging checks.

Validation: tools/check.sh passed with 1,088 tests and 40 skips, privacy audit and
wheel checks. Kinex typecheck, 70 standalone/140 core/25 setup tests, build and local
tarball packaging passed. GPU replay was not run.

Refresh existing workspace profiles and runtime/driver resources together, then
reload devices. Preserve connection and calibration settings.
