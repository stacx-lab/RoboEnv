"""Compatibility imports for the original catalog-only device server.

New code composes DeviceService with registered bindings. This module remains so
existing Python callers do not break; it contains no device implementation.
"""

from .bindings.catalog import CatalogDeviceService
from .device_model import DeviceError
from .device_service import DeviceService

__all__ = ["CatalogDeviceService", "DeviceError", "DeviceService"]
