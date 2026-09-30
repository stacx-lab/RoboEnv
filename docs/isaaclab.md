# Isaac Lab

Isaac Lab source management, isolated installation and diagnostics remain available:

```sh
roboenv sources sync --backends isaaclab
roboenv prepare isaaclab
roboenv doctor isaaclab
```

The experimental [Go2–D1 environment](go2-d1.md) provides a bundled task and
device adapter. The former G1 practice task,
profiles, bindings, runtime and task-specific asset tools have been removed.
SIMPLE's G1 integration is separate and remains supported.

Standalone Isaac Lab remains pinned to `99f1423`, with its existing resolved
dependency lock, coverage compatibility patch and opt-in CloudXR patch. The
proposed update to `998bf88` was reverted because its dependencies conflict with
Isaac Sim 6.1 package metadata. `sources check-updates --backends isaaclab` will
therefore report a newer upstream revision; this pin is intentionally retained.

Existing environments, cached assets and user workspaces are preserved.
See [setup](setup.md) for installation requirements.
