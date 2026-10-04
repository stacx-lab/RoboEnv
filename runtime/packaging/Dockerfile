# syntax=docker/dockerfile:1
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 libvulkan1 libx11-6 libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libasound2t64 libnss3 libdbus-1-3 ca-certificates git ffmpeg \
    libglu1-mesa g++ \
    && rm -rf /var/lib/apt/lists/*
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY sim-env/ /opt/sim-env/
COPY python-base/ /opt/python/
COPY runtime-data/ /opt/roboenv/runtime/
COPY relocate.py /opt/relocate.py
RUN --mount=type=secret,id=relocation,target=/run/secrets/relocation.json python3 /opt/relocate.py
COPY wheels/ /opt/wheels/
COPY contract-deps/ /opt/roboenv-contract-deps/
RUN python3 -m venv /opt/installer && \
    /opt/installer/bin/pip --python /opt/sim-env/bin/python install --no-index --no-deps /opt/wheels/*.whl
ENV ROBOENV_ROOT=/opt/roboenv/runtime \
    PYTHONPATH=/opt/roboenv-contract-deps \
    NVIDIA_DRIVER_CAPABILITIES=all MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    PATH=/opt/sim-env/bin:$PATH
COPY entrypoint.sh /opt/roboenv/entrypoint.sh
ENTRYPOINT ["/bin/sh", "/opt/roboenv/entrypoint.sh"]
WORKDIR /opt/roboenv/runtime
CMD ["/opt/sim-env/bin/python", "-m", "roboenv_runtime.runtime.supervisor", "--help"]
