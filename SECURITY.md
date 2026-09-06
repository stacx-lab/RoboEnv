# Security policy

Roboenv is pre-release software for controlled evaluation. No stable release,
long-term support window, response-time guarantee, or multi-tenant security
certification is currently offered.

## Reporting

Use the repository's private vulnerability-reporting interface if maintainers have
enabled it. Otherwise open an issue asking only for a private contact channel:
do not include exploit details, credentials, private endpoints or sensitive logs
in that public request. Maintainers must enable and test a private channel before
the first public release. No email address or service is assumed here.

Include affected versions, a minimal reproduction, expected versus observed
behavior, and the relevant trust boundary. Redact identifiers and provide a
synthetic fixture where possible. Never test against other people's services or
robots without their permission.

## Trust boundaries

An agent is not an environment owner. Agent-controlled code and returned claims
must not be accepted as the benchmark's verdict. Public device operations,
owner lifecycle controls, simulator state, assets and verifier authority are
separate concerns even when processes share a machine.

Current Harbor deployments use a separate simulator container and read-only
asset mounts. The verifier script runs in Harbor's shared agent/verifier
container; this is not a fully isolated, adversarially secure verifier boundary.
Container isolation also does not make device bindings safe for a physical robot.
Do not expose an unauthenticated device socket to a network or unrelated tenant.
Do not mount a Docker socket, model credentials, the source checkout or writable
asset depots into an agent's container.

Use disposable jobs, least-privilege mounts, bounded operations and explicit image
digests. Treat plugins, upstream code, task programs and checkpoints as supply-chain
inputs requiring review. An unsupported emergency-stop method is not a safety
system; Roboenv must not substitute for a robot's independent safety controls.

## Release controls

The local preflight scans owned text and wheel contents for known private-path
patterns and several credential signatures. It is a regression check, not a
complete secret detector. The candidate workflow additionally runs an external
vulnerability/secret scanner and emits only a sanitized summary.

Before release, review the supported deployment threat model, negative tests,
scanner results, source licenses and artifact provenance. Scan each actual
simulator image separately; an SDK wheel scan says nothing about an Isaac,
CUDA, system-library or checkpoint dependency inside a container.

Keep raw security reports private until reviewed. Rotate a leaked credential;
removing it from a later commit does not invalidate it. Preserve the original
evidence privately and publish only a redacted reproducer and remediation note.
