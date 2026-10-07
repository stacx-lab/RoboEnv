"""One CUDA-visible GPU, matched to each renderer by physical identity.

Driver discovery lives in a bounded child. Importing this module does not load
CUDA, Vulkan, EGL, or a simulator, and launch never changes its caller's mask.
"""

import ctypes as c
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def graphics_apis(backend, options=None):
    options = options or {}
    if backend in {"behavior", "robodojo", "isaaclab", "robotwin"}:
        return ("vulkan",)
    if backend == "simple" and options.get("sim_mode", "mujoco_isaac") == "mujoco_isaac":
        return ("vulkan", "egl")
    if backend in {"libero", "robocasa", "abc", "simple"}:
        if options.get("enable_render", True):
            return ("egl",)
    return ()


def _cuda_device():
    """Query logical zero under the inherited mask without creating a context."""
    cuda = c.CDLL("libcuda.so.1")

    def check(code):
        if code:
            raise RuntimeError(
                f"CUDA discovery failed: CUresult {code}; "
                "check CUDA_VISIBLE_DEVICES and the NVIDIA driver"
            )

    check(cuda.cuInit(0))
    count = c.c_int()
    check(cuda.cuDeviceGetCount(c.byref(count)))
    if count.value < 1:
        raise ValueError("No GPU is visible to CUDA; check CUDA_VISIBLE_DEVICES")
    device = c.c_int()
    check(cuda.cuDeviceGet(c.byref(device), 0))
    identity = (c.c_ubyte * 16)()
    check(cuda.cuDeviceGetUuid_v2(c.byref(identity), device))
    physical = (c.c_ubyte * 16)()
    check(cuda.cuDeviceGetUuid(c.byref(physical), device))
    if bytes(identity) != bytes(physical):
        raise ValueError("MIG instances are unsupported: a full graphics-capable GPU is required")
    bus = c.create_string_buffer(32)
    check(cuda.cuDeviceGetPCIBusId(bus, len(bus), device))
    return {"uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity))),
            "pci_bus_id": bus.value.decode().lower(), "cuda_index": 0}


def _egl_devices():
    """Enumerate EGL devices without making a rendering context."""
    egl = c.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [c.c_char_p]
    egl.eglGetProcAddress.restype = c.c_void_p

    def function(name, result, *args):
        address = egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"EGL GPU discovery requires {name}")
        return c.CFUNCTYPE(result, *args)(address)

    query = function("eglQueryDevicesEXT", c.c_uint, c.c_int,
                     c.POINTER(c.c_void_p), c.POINTER(c.c_int))
    string = function("eglQueryDeviceStringEXT", c.c_char_p, c.c_void_p, c.c_int)
    count = c.c_int()
    if not query(0, None, c.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    devices = (c.c_void_p * count.value)()
    if not query(count.value, devices, c.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    result = []
    for index, device in enumerate(devices[:count.value]):
        extensions = (string(device, 0x3055) or b"").decode().split()
        # A Mesa device may describe the same PCI slot without CUDA interop.
        if "EGL_NV_device_cuda" not in extensions:
            continue
        entry = {"index": index}
        if "EGL_EXT_device_persistent_id" in extensions:
            binary = function("eglQueryDeviceBinaryEXT", c.c_uint, c.c_void_p,
                              c.c_int, c.c_int, c.c_void_p, c.POINTER(c.c_int))
            identity, size = (c.c_ubyte * 16)(), c.c_int()
            if binary(device, 0x335C, 16, identity, c.byref(size)) and size.value == 16:
                entry["uuid"] = "GPU-" + str(uuid.UUID(bytes=bytes(identity)))
        for extension, attribute in (("EGL_EXT_device_drm_render_node", 0x3377),
                                     ("EGL_EXT_device_drm", 0x3233)):
            if extension in extensions:
                node = string(device, attribute)
                if node:
                    path = Path("/sys/class/drm") / Path(node.decode()).name / "device"
                    if path.exists():
                        entry["pci_bus_id"] = path.resolve().name.lower()
                        break
        result.append(entry)
    return result


def match_graphics(device, candidates, api):
    matches = [item for item in candidates if
               item.get("uuid", "").lower() == device["uuid"].lower()]
    # Prefer UUID: multiple EGL vendors may expose the same PCI device.
    if not matches:
        matches = [item for item in candidates if
                   item.get("pci_bus_id", "").lower() == device["pci_bus_id"].lower()]
    if len(matches) != 1:
        raise ValueError(f"CUDA GPU {device['uuid']} has no unique {api} device")
    return matches[0]["index"]


def _vulkan_devices():
    """Query Vulkan 1.1 device UUIDs without creating a logical/rendering device."""
    class Application(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                    ("name", c.c_char_p), ("version", c.c_uint32),
                    ("engine", c.c_char_p), ("engineVersion", c.c_uint32),
                    ("apiVersion", c.c_uint32)]

    class InstanceInfo(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                    ("flags", c.c_uint32), ("application", c.POINTER(Application)),
                    ("layerCount", c.c_uint32), ("layers", c.c_void_p),
                    ("extensionCount", c.c_uint32), ("extensions", c.c_void_p)]

    class DeviceID(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                    ("deviceUUID", c.c_ubyte * 16), ("driverUUID", c.c_ubyte * 16),
                    ("deviceLUID", c.c_ubyte * 8), ("deviceNodeMask", c.c_uint32),
                    ("deviceLUIDValid", c.c_uint32)]

    class Properties(c.Structure):
        # VkPhysicalDeviceProperties is an output-only fixed-size Vulkan 1.0
        # structure. Reserve an aligned buffer larger than it; only ID is read.
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                    ("properties", c.c_uint64 * 128)]

    vk = c.CDLL("libvulkan.so.1")
    vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p,
                                    c.POINTER(c.c_void_p)]
    vk.vkCreateInstance.restype = c.c_int32
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkDestroyInstance.restype = None
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32),
                                             c.POINTER(c.c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = c.c_int32
    vk.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(Properties)]
    vk.vkGetPhysicalDeviceProperties2.restype = None

    def check(code):
        if code != 0:
            raise RuntimeError(f"Vulkan GPU discovery failed: VkResult {code}")

    app = Application(sType=0, name=b"roboenv-gpu-probe", apiVersion=(1 << 22) | (1 << 12))
    info = InstanceInfo(sType=1, application=c.pointer(app))
    instance = c.c_void_p()
    check(vk.vkCreateInstance(c.byref(info), None, c.byref(instance)))
    try:
        count = c.c_uint32()
        check(vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None))
        devices = (c.c_void_p * count.value)()
        check(vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices))
        result = []
        for index, device in enumerate(devices[:count.value]):
            identity = DeviceID(sType=1000071004)
            props = Properties(sType=1000059001,
                               pNext=c.cast(c.pointer(identity), c.c_void_p))
            vk.vkGetPhysicalDeviceProperties2(device, c.byref(props))
            result.append({"index": index,
                           "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity.deviceUUID)))})
        return result
    finally:
        vk.vkDestroyInstance(instance, None)


def _probe(apis):
    device = _cuda_device()
    for api in apis:
        candidates = _vulkan_devices() if api == "vulkan" else _egl_devices()
        device[api + "_index"] = match_graphics(device, candidates, api)
    return device


def resolve_gpu(apis):
    """Resolve the first CUDA-visible GPU; container masks are already applied."""
    if not apis:
        return None
    if set(apis) - {"egl", "vulkan"}:
        raise ValueError("Unknown graphics API")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "roboenv.provisioning.gpu", *apis],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"GPU discovery failed before launch: {detail}") from exc


def gpu_environment(selection):
    return {"CUDA_VISIBLE_DEVICES": selection["uuid"],
            "ROBOENV_GPU": json.dumps(selection, sort_keys=True),
            **({"MUJOCO_EGL_DEVICE_ID": str(selection["egl_index"])}
               if "egl_index" in selection else {}),
            **({"SIMPLE_ISAAC_ACTIVE_GPU": str(selection["vulkan_index"])}
               if "vulkan_index" in selection else {}),
            "ROBOTWIN_RENDER_DEVICE": "cuda:0"}


def configure_gpu(backend, options=None):
    """Apply worker-local settings, including when a native/replay tool starts directly."""
    apis = graphics_apis(backend, options)
    if not apis:
        return None
    inherited = os.environ.get("ROBOENV_GPU")
    if inherited:
        selection = json.loads(inherited)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != selection["uuid"]:
            raise ValueError("GPU configuration changed after launch")
        if any(api + "_index" not in selection for api in apis):
            raise ValueError("GPU configuration lacks required renderer identity")
    else:
        selection = resolve_gpu(apis)
    os.environ.update(gpu_environment(selection))
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("RoboEnv GPU selection: " + json.dumps(selection, sort_keys=True), flush=True)
    return selection


if __name__ == "__main__":
    print(json.dumps(_probe(tuple(sys.argv[1:]))))
