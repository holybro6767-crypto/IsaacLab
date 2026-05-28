# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal Vulkan loader probe.

Loads ``vulkan-1.dll`` (Windows) or ``libvulkan.so.1`` (Linux/macOS) and
calls ``vkCreateInstance`` + ``vkEnumeratePhysicalDevices`` via ctypes. No
external dependencies; relies only on the OS-provided Vulkan loader and
whatever ICDs the platform has registered.

Mirrors what ``vulkaninfo --summary`` does at the loader/ICD level so we
can diagnose Vulkan availability on a CI runner without installing the
Vulkan SDK. Used by ``.github/workflows/windows-ci.yaml``'s Vulkan probe
step; equally useful as a standalone command:

    python tools/vulkan_probe.py
"""

from __future__ import annotations

import ctypes
import sys


def _load_loader() -> ctypes.CDLL | None:
    """Load the OS Vulkan loader, or return ``None`` if it isn't installed."""
    candidates = (
        ("vulkan-1.dll", ctypes.WinDLL) if sys.platform == "win32" else (None, None),
        ("libvulkan.so.1", ctypes.CDLL),
        ("libvulkan.so", ctypes.CDLL),
        ("libvulkan.1.dylib", ctypes.CDLL),
    )
    for name, ctor in candidates:
        if not name:
            continue
        try:
            return ctor(name)
        except OSError:
            continue
    return None


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def main() -> int:
    vk = _load_loader()
    if vk is None:
        print("vulkan loader NOT loadable on this platform")
        return 0
    print(f"vulkan loader loaded: {vk}")

    app = _VkApplicationInfo(0, None, b"probe", 0, b"probe", 0, (1 << 22))
    create_info = _VkInstanceCreateInfo(1, None, 0, ctypes.byref(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    result = vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
    status = "OK" if result == 0 else "ERROR"
    print(f"vkCreateInstance -> {result} ({status})")
    if result != 0:
        return 0

    count = ctypes.c_uint32(0)
    enum_result = vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
    print(f"vkEnumeratePhysicalDevices -> {enum_result}, physical-device count = {count.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
