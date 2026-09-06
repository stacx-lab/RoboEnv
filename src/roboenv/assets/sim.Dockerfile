# ETES sim shell: system libraries only; the simulator environment, the runtime
# harness and the assets are bind-mounted read-only from the host.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglx0 libopengl0 libosmesa6 libglvnd0 libvulkan1 \
    libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libgomp1 libxrandr2 libxinerama1 libxcursor1 libxi6 libglu1-mesa libxt6 \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# the runtime injects the NVIDIA driver libraries but neither the Vulkan ICD manifest
# nor the glvnd EGL vendor manifest; the Vulkan ICD's init NEEDS the EGL one
# (without it, vk_icdNegotiateLoaderICDInterfaceVersion fails and Isaac/SAPIEN see no GPU)
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '%s' \
    '{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.289"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && printf '%s' \
    '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

WORKDIR /opt
