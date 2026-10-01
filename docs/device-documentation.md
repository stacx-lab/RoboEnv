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
