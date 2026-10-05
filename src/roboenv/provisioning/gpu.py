"""Bounded, process-local GPU selection for SIMPLE's CUDA/Vulkan renderer.

Importing this module does not load a GPU library. Vulkan discovery runs in a
short-lived child, before any simulator imports or world creation.
"""

import csv
import json
import os
import subprocess
import sys
import uuid


def validate_gpu(gpu):
    if gpu is not None and (type(gpu) is not int or gpu < 0):
        raise ValueError("gpu must be a nonnegative host GPU index")


def select_gpu(devices, vulkan, gpu, environment):
    """Match physical identity, independently of CUDA/Vulkan enumeration order."""
    validate_gpu(gpu)
    permitted = list(devices)
    # Container visibility is a restriction as well, when supplied by its runtime.
    for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        mask = environment.get(key)
        if mask is None or (key == "NVIDIA_VISIBLE_DEVICES" and mask == "all"):
            continue
        selected = []
        for token in mask.split(","):
            token = token.strip()
            matches = [d for d in devices if str(d["index"]) == token
                       or (token.startswith("GPU-") and d["uuid"].startswith(token))]
            if len(matches) != 1:
                raise ValueError(f"{key} must select available full GPUs by index or UUID")
            if matches[0] in selected:
                raise ValueError(f"{key} contains a duplicate GPU")
            selected.append(matches[0])
        permitted = [d for d in selected if d in permitted]
    if not permitted:
        raise ValueError("No GPU is permitted by inherited GPU visibility")
    if gpu is None:
        device = permitted[0]
    else:
        device = next((d for d in permitted if d["index"] == gpu), None)
        if device is None:
            raise ValueError(f"GPU {gpu} is unavailable or excluded by inherited GPU visibility")
    matches = [d for d in vulkan if d["uuid"].lower() == device["uuid"].lower()]
    if len(matches) != 1:
        raise ValueError(f"Selected GPU {device['index']} has no unique Vulkan device")
    return {**device, "vulkan_index": matches[0]["index"], "cuda_index": 0}


def _vulkan_devices():
    """Query Vulkan 1.1 device UUIDs without creating a logical/rendering device."""
    import ctypes as c

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


def _probe():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    devices = [{"index": int(index), "uuid": identity.strip(), "pci_bus_id": pci.strip()}
               for index, identity, pci in csv.reader(result.stdout.splitlines())]
    return {"devices": devices, "vulkan": _vulkan_devices()}


def resolve_gpu(gpu=None):
    validate_gpu(gpu)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "roboenv.provisioning.gpu"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        inventory = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"SIMPLE GPU discovery failed before launch: {detail}") from exc
    return select_gpu(inventory["devices"], inventory["vulkan"], gpu, os.environ)


def gpu_environment(selection):
    return {
        "CUDA_VISIBLE_DEVICES": selection["uuid"],
        "SIMPLE_ISAAC_ACTIVE_GPU": str(selection["vulkan_index"]),
        "ROBOENV_SIMPLE_GPU": json.dumps(selection, sort_keys=True),
    }


def configure_simple_gpu(gpu=None):
    """Called before SIMPLE imports, also by standalone native/replay tools."""
    validate_gpu(gpu)
    inherited = os.environ.get("ROBOENV_SIMPLE_GPU")
    if inherited and gpu is None:
        selection = json.loads(inherited)
        if any(os.environ.get(key) != value
               for key, value in gpu_environment(selection).items()):
            raise ValueError("SIMPLE GPU configuration changed after launch")
    else:
        selection = resolve_gpu(gpu)
        os.environ.update(gpu_environment(selection))
    return selection


if __name__ == "__main__":
    print(json.dumps(_probe()))
