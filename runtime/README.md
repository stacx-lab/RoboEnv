# Environment runtime

The separate roboenv-runtime distribution owns simulator adapters, controller
bindings, world lifecycle, policy execution, budgets, scoring, and the journaled
worker/supervisor. It is installed in simulator images, not agent containers.

Device definitions and service dispatch are in the lightweight SDK. Runtime
bindings connect those interfaces to the same world and counted controller
functions used by the benchmark. The former parallel rig/device driver package
has been archived; there is one active device assembly path.

data/ holds remaining environment facts; locks/ records prepared simulator
dependency sets; packaging/ contains image relocation and container recipes.
Heavy assets and interpreters are separate build inputs. Source code comes from
the pinned upstreams in this repository, not from the prepared dependency tree.

Native tasks require an immutable image/depot deployment. Whole development-tree
mounts are no longer supported by the task exporter.
