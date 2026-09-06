# LIBERO resolved-environment acceptance

[The machine-readable receipt](libero-candidate.json) records a real clean
interpreter installation and local GPU replay. It does not certify Docker/Harbor,
every task, every seed, or bitwise-equivalent trajectories.

## What changed

The historical runtime/locks/libero.txt remains byte-for-byte unchanged.
The candidate has separate inputs, a resolved lock and one source-identity override:

- runtime/locks/libero.candidate.in: selected simulator/control dependencies.
- runtime/locks/libero.candidate.txt: their normally resolved transitive closure.
- runtime/locks/libero.candidate-source-pins.txt: fixes PyRoKi's floating jaxls
  dependency to the historical commit in the same repository; it does not override
  numerical dependency version ranges.

NumPy 1.26.4, MuJoCo 3.5.0, JAX/jaxlib 0.4.29, Torch 2.8.0, SciPy 1.17.0 and
PyOpenGL 3.1.6 remain at their historical versions. OpenCV 4.11.0.86 accepts
NumPy 1.26.4 on Python 3.12; 4.12.0.88 instead requires NumPy 2 on that interpreter.
The actual package is opencv-python, because the pinned robosuite declares that
dependency. No headless OpenCV package is co-installed: both distributions own cv2.
[OpenCV package metadata and installation guidance](https://pypi.org/project/opencv-python/4.11.0.86/)

The cap-x checkout is loaded as a locked source-only bridge by the existing
runtime loader. Its full distribution is not installed or represented as
dependency-consistent: it unconditionally declares training/model-server extras
outside this workload. LIBERO-PRO and its robosuite fork are installed normally.

Selected runtime imports require Gym/Gymnasium, BDDL, future, Open3D, Viser,
robot descriptions and local PyRoKi/JAX. Cap-x's eager registry imports explain
why some visualization dependencies remain even for privileged control.
Unused model servers, training libraries and pyrender are not direct candidate
inputs; in particular, PyOpenGL was not downgraded merely to satisfy the unrelated
historical pyrender pin.

Two real failures were resolved before acceptance:

1. PyRoKi's unpinned jaxls Git URL conflicted with the explicit historical SHA.
   The source-only override selects the same repository at commit 6fe7cf9.
   That revision requires JAX >=0.4.0; a newer floating revision requires >=0.6.
2. The first actual simulator boot found a missing future import. Restoring the
   historical future==0.18.2 dependency fixed the boot; resolver success alone
   had not caught the undeclared simulator import.

The full historical freeze also named unavailable gym-notices==0.2.0. The scoped
resolver selected available gym-notices==0.1.0 under Gym's declared requirement.

## Reproduce the accepted installation

Use a new location; do not replace an existing interpreter. The selected source
checkouts must already pass roboenv sources status --backends libero, and the
SDK/runtime wheels must already have been built.

    uv venv --python 3.12.12 .cache/libero-candidate
    UV_CACHE_DIR=.cache/uv uv pip install \
      --python .cache/libero-candidate/bin/python --link-mode hardlink \
      --index-url https://pypi.org/simple \
      --override runtime/locks/libero.candidate-source-pins.txt \
      -r runtime/locks/libero.candidate.txt \
      build/wheels/roboenv-0.2.0-py3-none-any.whl \
      build/wheels/roboenv_runtime-0.2.0-py3-none-any.whl \
      -e third_party/cap-x/capx/third_party/LIBERO-PRO \
      -e third_party/cap-x/capx/third_party/libero_dependencies/robosuite
    uv pip check --python .cache/libero-candidate/bin/python

The actual install used normal dependency resolution and isolated builds, without
--no-deps. Dependency checking reported 176 compatible installed distributions.
The interpreter occupied about 9.38 GB and the newly fetched robot/config cache
about 0.50 GB; installer-cache files were hardlinked, not separately copied into
the interpreter. Existing environments and benchmark evidence were not modified.

## GPU evidence and its limits

Choose an available physical GPU explicitly. This command launches two independent
worker processes and never calls a paid model:

    .cache/libero-candidate/bin/python -I tools/replay_libero_candidate.py \
      --gpu 0 --out .cache/libero-candidate-validation

The helper checks candidate dependency versions/VCS identities and module origins.
It uses new LIBERO configuration, robot-description, Matplotlib and XDG cache
locations. SDK/runtime imports come from the candidate environment; LIBERO and
robosuite come from the selected locked checkouts. The robot asset revision and
URDF hash are included in the receipt. Post-run exact source checks passed.

| Phase | Measured result |
|---|---|
| Device worker | Arm, camera, gripper and world reads passed; gripper motion charged 40 steps, from 10 to 50 |
| Fresh reference worker | Original saved libero.90_8 program, seed 0, historical capx initialization order |
| Simulator verdict | reward=1.0, completed=true, 2,779 / 4,000 steps |
| Finalization | submit and post-submit observation both reported 2,779 steps |

The device-probe and reference phases took approximately 33 and 60 seconds.
The saved program's historical note says 2,706 steps, so this is a success
regression check, not a claim of identical trajectories or identical efficiency.
No benchmark horizon, success predicate or observation condition was weakened.
The program remains a project-authored reference, not an official expert solution.
Docker read-only asset mounts and Harbor verification require their own subsequent
acceptance run; this local worker receipt must not be substituted for one.
