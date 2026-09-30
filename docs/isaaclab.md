# Isaac Lab

Isaac Lab source management, isolated installation and diagnostics remain available:

```sh
roboenv sources sync --backends isaaclab
roboenv prepare isaaclab
roboenv doctor isaaclab
```

There is no bundled Isaac Lab task or device adapter. The G1 practice task,
profiles, bindings, runtime and task-specific asset tools have been removed.
SIMPLE's G1 integration is separate and remains supported.

The current source pin follows upstream's default `develop` branch (`998bf88`).
Its dependencies now live in the root `pyproject.toml`, including new workspace
packages and CUDA 13 PyTorch packages. CloudXR remains optional: the root
`cloudxr` extra and `isaaclab_teleop[cloudxr]` explicitly enable it. The old
`coverage==7.4.4` installation patch was removed because that requirement no
longer exists.

Fresh installation is currently blocked. Upstream requires PyTorch 2.12 and
torchvision 0.27, while its Isaac Sim 6.1 dependency requires torchvision 0.26.
Upstream's UV overrides can resolve an environment, but that environment violates
published dependency metadata and cannot pass RoboEnv's strict dependency check.
Other overrides replace Isaac Sim's USD/Newton requirements. RoboEnv therefore
reports this blocker before installation instead of reusing the old recipe as
proof for the new source. `runtime/locks/isaaclab-resolved.txt` remains the
historical lock for `99f1423`; it is no longer selected by the current recipe.
An updated dependency-consistent recipe and new native validation are required.

Existing interpreters and cached assets are preserved. Reusing an interpreter
refreshes its source checkout but does not update dependencies or certify that
old environment against this new source.
See [setup](setup.md) for installation requirements.
