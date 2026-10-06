# Agent-neutral device documentation

`roboenv.devices.documentation` exports a language reference for coding agents
that connect through the RoboEnv SDK or CLI. It does not depend on an agent framework.
The caller owns task goals, instructions, and evaluation policy.

```python
from roboenv.devices.documentation import export_documentation

export_documentation(
    robot,                       # public robot workspace manifest
    env.devices(),               # descriptors from the running environment
    output_directory,
    {"endpoint": endpoint, "session": session, "generation": 0},
)
```

Use a fresh output directory per environment session. The output is `harness.md`,
`devices/<name>.yaml`, `devices.json`, `connection.json`, and `workspace.yaml`.
No drivers or agent auto-discovery instruction files are written. Device names
must be unique logical names. Public exports omit binding selectors, access/driver
configuration, and implementation metadata. Supply only a public robot manifest.

The Markdown explains connections, supported action arguments and result schemas,
zero-based joint vector order and units, gripper direction commands, named base
velocity fields, feedback frames, mount transforms, and RGB/depth interpretation.
The YAML preserves exact public geometry and schemas for code to consume. Use
live descriptors so action support, dimensions, limits, and control settings match
the running environment; static driver action lists are not SDK capabilities.

`robot_description(robot, devices)` supplies a short model/simulator/inventory
summary for embedding in an orchestrator's instruction. `render_documentation`
returns the full Markdown without writing files. Neither function starts or
changes a simulator, and neither defines benchmark goals or rewards.

Kinex-style file drivers remain a separate projection through
`roboenv.devices.export.export_devices`; orchestration adapters choose when to
create those resources. That projection places a single `_roboenv.py` in the
shared `devices/` root, alongside the individual device directories.

Camera projection uses integer pixel centers: the top-left pixel is `(0, 0)`.
For a centered image the principal point is `((width - 1) / 2, (height - 1) / 2)`.
Registered depth is optical-axis Z in the profile's declared units, sampled at
those same pixel centers. RoboCasa uses single-sample offscreen rendering for
paired RGB-D; RoboTwin retains native supersampled RGB and captures pixel-center
depth with a matching raster camera. These corrections change rendered sensor
data, so older image recordings retain their original calibration.

Camera `timestamp` uses the simulator clock, matching joint feedback.
RGBD returns one timestamp for the pair; transport preserves it. Separate
reads in realtime mode can still describe different simulation instants. Matching
clock units alone does not make an image and a later joint read an atomic snapshot.

Re-export devices when upgrading a runtime or changing camera calibration so
the copied profiles reflect the current intrinsics and extrinsics.

## Camera read output

SDK camera reads accept `camera.read(output="file")` (the default),
`output="array"`, or `output="base64"`. File mode returns `file` and optional
`depth_file`. Array mode returns owned NumPy `color` (uint8 RGB, H×W×3) and
optional `depth` (uint16, H×W); install `roboenv[camera]` for decoding.
Base64 mode returns PNG/PGM16 strings under `color`/`depth`. Every mode includes
one `timestamp`; profile dimensions and depth units are unchanged.

The server protocol and artifact production are unchanged. Clients convert the
received images; array/base64 HTTP reads avoid local temporary files. File mode
supports artifacts larger than the existing in-memory response limit. Exported
Kinex drivers accept the same option; arrays cross their JSON boundary as base64
and are decoded in the calling Kinex Python runtime.

For tasks with changing instructions, call `env.refresh_task()` between stages.
It updates `env.task` through the existing public description endpoint.
`roboenv describe` and copied Kinex workspaces' `python task.py` read the same
current instruction without exposing native task state.
