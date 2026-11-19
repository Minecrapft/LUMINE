"""PWM control for the camera's strip illumination."""

import atexit
import logging
import math
from threading import Lock
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class IlluminationUnavailable(RuntimeError):
    """Raised when GPIO-based illumination cannot be controlled."""


class IlluminationController:
    """Control an active-high MOSFET input on Raspberry Pi BCM GPIO 18."""

    def __init__(self, device_factory: Optional[Callable[[], Any]] = None):
        self._device_factory = device_factory
        self._device = None
        self._brightness = 0
        self._lock = Lock()

    def _create_device(self):
        if self._device_factory is not None:
            try:
                return self._device_factory()
            except Exception as exc:
                logger.exception("Could not initialize strip-light output")
                raise IlluminationUnavailable(
                    f"Could not initialize BCM GPIO 18: {exc}. Check that the "
                    "LUMINE user can read and write the Raspberry Pi GPIO chip "
                    "device (usually through the gpio group)."
                ) from exc

        try:
            from gpiozero import PWMOutputDevice
            from gpiozero.pins.lgpio import LGPIOFactory
        except ImportError as exc:
            raise IlluminationUnavailable(
                "GPIO support is unavailable. Install gpiozero and lgpio on the Raspberry Pi."
            ) from exc

        try:
            factory = LGPIOFactory()
            return PWMOutputDevice(
                pin=18,
                active_high=True,
                initial_value=0,
                frequency=1000,
                pin_factory=factory,
            )
        except Exception as exc:
            logger.exception("Could not initialize strip-light output")
            raise IlluminationUnavailable(
                f"Could not initialize BCM GPIO 18: {exc}. Check that the "
                "LUMINE user can read and write the Raspberry Pi GPIO chip "
                "device (usually through the gpio group)."
            ) from exc

    def _get_device(self):
        if self._device is None:
            self._device = self._create_device()
            atexit.register(self.close)
        return self._device

    def status(self) -> Dict[str, Any]:
        with self._lock:
            try:
                self._get_device()
            except IlluminationUnavailable as exc:
                logger.warning("Strip illumination is unavailable: %s", exc)
                return {"available": False, "brightness": 0, "error": str(exc)}
            return {"available": True, "brightness": self._brightness}

    def set_brightness(self, percentage: float) -> Dict[str, Any]:
        if (
            isinstance(percentage, bool)
            or not math.isfinite(percentage)
            or not 0 <= percentage <= 100
        ):
            raise ValueError("Brightness must be between 0 and 100")

        with self._lock:
            device = self._get_device()
            try:
                from gpiozero.exc import GPIOZeroError
            except ImportError:
                GPIOZeroError = RuntimeError
            try:
                device.value = percentage / 100
            except (GPIOZeroError, OSError, RuntimeError, ValueError) as exc:
                logger.exception("Could not set strip-light brightness")
                raise IlluminationUnavailable(
                    f"Could not set brightness on GPIO 18: {exc}"
                ) from exc
            self._brightness = round(percentage)
            return {"available": True, "brightness": self._brightness}

    def close(self):
        with self._lock:
            if self._device is None:
                return
            device = self._device
            self._device = None
            self._brightness = 0
            device.close()
