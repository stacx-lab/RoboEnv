# ChemLab instruments

ChemLab uses the existing Isaac Lab adapter, worker, world clock, device server,
workspace exporter and spectator recorder. There is no robot in this mode.
An agent selects instrument commands, reads agent-visible measurements and uses
`world.wait` to allow the instrument process models to evolve. RoboEnv owns task
initialization, evaluator-only state, scoring, episode limits and evidence.
Benchmark scheduling and Kinex policy evaluation belong outside RoboEnv.

## Prepare

The complete `MBZUAI/ChemLab-Assets` dataset release is pinned to
`a6c006f921c9bee62f5585dd7f769163a7f51573`. Its catalog contains 203 instruments.
Acquisition includes source, contracts, URDF/GLB/Blender models, accessory meshes,
textures, data tables, examples and notices. Nothing is fetched during launch.

```sh
roboenv prepare isaaclab --prefix /path/to/new/isaaclab-env
uv pip install --python /path/to/new/isaaclab-env/bin/python \
  -c runtime/locks/isaaclab-resolved.txt -r runtime/locks/chemlab.txt
export ROBOENV_ASSET_CACHE=/path/to/asset-cache
python tools/prepare_chemlab.py
python tools/prepare_chemlab.py --verify
```

The asset tool requires Git and Git LFS. It stages a new checkout on the target
filesystem, checks the exact Git revision, tracked-file integrity and LFS object
hashes, then records the instrument resource hashes. Existing depots are verified
and never reset or overwritten. Preserve a partial or modified depot and choose a
new `--directory` to prepare another copy. Supply the same cache setting at launch;
an explicit `backend_kwargs.asset_directory` selects a depot instead.

ChemLab adds `yourdfpy` for URDF visual geometry and `CoolProp` for upstream
thermodynamic models. The supplemental dependency lock constrains these additions
against the existing Isaac Lab lock; it does not replace Isaac Lab's source pin.
See [third-party notices](third-party-notices.md#chemlab) for separate code, asset
and embedded-data terms.

## Run and connect

```sh
roboenv launch isaaclab/chemlab-hood-light-sash \
  --interpreter /path/to/new/isaaclab-env/bin/python --port 8123 --record
python tools/prepare_kinex_workspace.py isaaclab/chemlab-hood-light-sash \
  --endpoint http://127.0.0.1:8123 --out /path/to/new/agent-workspace
```

Use the **live endpoint** when preparing the Kinex workspace: each instrument has
a different command set, resolved from its contract after initialization. Static
setup contains lifecycle and reading actions only. The live export uses existing
portable drivers and pins the episode identity. It contains no simulator assets,
owner configuration, hidden state or solutions. Load `world` and `instrument`
through the normal Kinex device lifecycle. A controller Python tool can call:

```python
from kinex import RPCError, device, finish

device("instrument", "light_set", {"on": True})
device("instrument", "sash_set", {"height_m": 0.457})
try:
    device("world", "wait", {"duration_s": 1.0})
except RPCError as error:
    if error.code != "episode_finished":
        raise
finish(device("instrument", "read"))
```

Public action names replace native dots with underscores (`sash.set` becomes
`sash_set`) to satisfy Kinex's driver schema. The owner record retains native IDs.
The exported `devices.md` documents the live input schemas and unsupported actions.

Runtime `succeeded` means that the instrument accepted/completed that command;
it does not mean that the task succeeded. Public `read` returns only contract
slots marked `agent`. Owner/debug/simulation capabilities are excluded. Native
preconditions, units and unsupported capabilities remain authoritative. Asynchronous
hosts also expose `operation_poll` and `operation_cancel`. Panel captures return
PNG bytes as base64 with dimensions and checksum, never private host filenames.

There is one world clock with a 0.1-second interval. Commands change process
targets; they do not independently advance time. Reads do not request ticks.
`world.wait` advances all instruments together in stepped mode. Realtime mode uses
the existing owner-thread scheduler and has no step cap, following RoboEnv policy.
These are instrument process simulations, not contact-driven robot manipulation.

## Practice tasks

| Task ID under `isaaclab/` | Instrument | Goal |
| --- | --- | --- |
| `chemlab-hood-light-sash` | Esco fume hood | Light on, sash at 0.457 m |
| `chemlab-water-activity-temperature` | AQUALAB water activity meter | Indicated temperature near 303.15 K |
| `chemlab-hotplate-stirring` | IKA hotplate/stirrer | Indicated speed near 600 rpm |
| `chemlab-balance-power` | METTLER TOLEDO balance | Powered on with stable reading |

Each condition must hold for one simulation second. Tasks use seed 0, a 6000-frame
practice budget and stop-on-success. An unchanged initial state does not solve
these tasks. They are integration exercises, not an upstream ChemLab benchmark.

To add an instrument or combine instruments, reuse `task_name: chemlab-instruments`
and name each instance in `task_spec.instruments`, selecting its catalog `asset`.
Owner-only `initialize` commands configure samples and physical conditions. Set
declarative `goals` using `instrument`, `state`, optional nested `path`, and either
`equals` or numeric `min`/`max`. `dwell_s` applies to their conjunction. Define a
matching device/binding per instance using `isaaclab.chemlab/1`; device names and
entities must match. Instances have separate upstream hosts and share one clock.
Instrument mounts are read from the existing device profiles in world coordinates;
the spectator camera fits the resulting collection automatically.
No new adapter, transport or driver implementation is required.

## Geometry and validation boundaries

The scene imports URDF visual geometry, mesh colors and UV textures. Process state
drives joint bindings with compatible scalar units and explicitly audited transforms.
Unbound controls and unsupported state-to-geometry mappings are listed in the private
`scene_mappings` record. Prose-only Matterix bindings, contact/button dynamics,
material/chemical transfers and dynamic screen textures are not automatically
inferred from the contracts. The upstream panel API remains available independently
of the 3D display. This mode does not run a second physics clock.

Every host has a pinned factory entry in `src/roboenv/assets/chemlab-runtimes.json`.
Updating the release requires auditing this table and rerunning native checks;
contracts do not select arbitrary import paths. `render: false` can be used in
owner `backend_kwargs` for process/transport checks without Isaac rendering. Such
checks are not GPU or camera evidence.

```sh
python tools/check_chemlab.py --geometry --out /path/to/new/audit.json
ROBOENV_CHEMLAB_TEST_ASSETS=/path/to/verified/depot \
  python -m pytest tests/test_chemlab.py -q
python tools/finalize_episode.py /path/to/run.json --out /path/to/new/owner
python tools/replay_chemlab.py /path/to/new/owner/owner-record.json \
  --out /path/to/new/replay.json
python tests/chemlab_probe.py --assets /path/to/verified/depot --render \
  --kinex /path/to/kinex --out /path/to/new/native-probe
```

Run the audit/replay tools in the prepared runtime interpreter. They record exact
coverage; an import/read/clock/geometry audit does not certify every capability or
chemical model. Replay checks implementation and asset identities, reconstructs
the initial state, reissues recorded commands at their original frames, and compares
terminal state, budget and native outcome. Keep owner evidence outside the agent
workspace. Model evaluation and benchmark publication are separate work.
