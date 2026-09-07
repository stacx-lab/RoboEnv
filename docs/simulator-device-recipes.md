# Choosing the integration change

Use the [integration guide](simulator-integration.md) for implementation steps and
[RoboCasa](robocasa.md) for the reference catalog. Choose the scope before editing:

| Change | Files and behavior to update |
|---|---|
| Same rig, new task/seed/layout | Private task configuration and instruction; reuse the catalog |
| Concurrent tasks | Separate endpoints and harness workspaces or isolated networks |
| Rename a device | Directory, profile name, binding key, and mount references |
| Add or replace a camera | Enabled native sensor, binding selection, intrinsics, and rigid mount |
| Change hardware or controller | Matching rig catalog, native binding, assets, and validation |
| Add a simulator | Source/installation recipe, adapter, bindings, task discovery, deployment, and tests |

For example, launch the same RoboCasa rig in two independent worlds:

```sh
roboenv launch robocasa-pick-place-microwave --port 8123
roboenv launch robocasa-arrange-vegetables --port 8124
```

Run these in separate terminals. Copy the same robot directory into two harness
workspaces and set each process's `ROBOENV_ENDPOINT`. Changing the endpoint does
not require editing profiles. See [Complex local testing](complex-local.md).

A task's `device_bindings` replaces the full default map. Publish a matching device
selection; profiles do not create native hardware. When renaming, remember that
logical names and native entities differ: RoboCasa's `body` binds entity `base`.

Keep moving frames dynamic. Cameras on the lift attach to `torso:flange`, and the
wrist camera to `arm:flange`. Changing device boundaries may require frame-interface
work, not just renamed directories. New sensor or controller settings need updated
static facts and tests; launch never regenerates the catalog.

Rebuild runtime wheels/images after runtime changes; re-exporting tasks alone does
not update their service implementation. Preserve previous deployment identities
and evidence. Report CPU checks, GPU boot, device checks, task success, and replay
separately; a recording is not a replay certificate.
