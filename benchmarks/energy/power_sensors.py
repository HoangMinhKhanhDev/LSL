"""Windows power-sensor adapters for real LSL energy evidence.

The adapters are intentionally strict: they either return measurements from a
hardware-backed sensor path or raise ``PowerSensorUnavailable``. No synthetic
or proxy energy is emitted from this module.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple


class PowerSensorUnavailable(RuntimeError):
    """Raised when no real hardware power sensor can be read."""


@dataclass
class PowerReading:
    timestamp: float
    watts: Optional[float]
    joules: Optional[float]
    source: str
    detail: str = ""


@dataclass
class EnergyMeasurement:
    sensor: str
    method: str
    device: str
    elapsed_seconds: float
    joules: float
    watts: float
    start: PowerReading
    end: PowerReading


class PowerSensor:
    name = "power_sensor"
    method = "unknown"

    def read(self) -> PowerReading:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def measure(self, workload: Callable[[], None]) -> EnergyMeasurement:
        start = self.read()
        t0 = time.perf_counter()
        workload()
        elapsed = time.perf_counter() - t0
        end = self.read()
        if start.joules is not None and end.joules is not None:
            joules = max(0.0, float(end.joules) - float(start.joules))
        elif start.watts is not None and end.watts is not None:
            joules = 0.5 * (float(start.watts) + float(end.watts)) * max(elapsed, 0.0)
        elif end.watts is not None:
            joules = float(end.watts) * max(elapsed, 0.0)
        else:
            raise PowerSensorUnavailable(f"{self.name} did not return watts or joules")
        watts = joules / max(elapsed, 1e-12)
        return EnergyMeasurement(
            sensor=self.name,
            method=self.method,
            device=platform.processor() or platform.machine(),
            elapsed_seconds=float(elapsed),
            joules=float(joules),
            watts=float(watts),
            start=start,
            end=end,
        )


def _positive(value: float, upper: float = 10000.0) -> bool:
    return value == value and 0.0 <= value <= upper


class IntelPowerGadgetSensor(PowerSensor):
    """ctypes bridge to Intel Power Gadget EnergyLib64.dll on Windows."""

    name = "intel_power_gadget"
    method = "intel_power_gadget_energylib"

    def __init__(self, dll_path: Optional[str] = None):
        if os.name != "nt":
            raise PowerSensorUnavailable("Intel Power Gadget EnergyLib is only supported on Windows here")
        path = dll_path or _find_energy_lib()
        if not path:
            raise PowerSensorUnavailable(
                "EnergyLib64.dll not found. Set INTEL_POWER_GADGET_DLL or install Intel Power Gadget."
            )
        self.dll_path = path
        self.dll = ctypes.WinDLL(path)
        self._bind()
        if not bool(self.dll.IntelEnergyLibInitialize()):
            raise PowerSensorUnavailable(f"IntelEnergyLibInitialize failed for {path}")
        self._last_energy_joules: Optional[float] = None
        self._last_power_watts: Optional[float] = None

    def _bind(self) -> None:
        self.dll.IntelEnergyLibInitialize.argtypes = []
        self.dll.IntelEnergyLibInitialize.restype = ctypes.c_bool
        self.dll.ReadSample.argtypes = []
        self.dll.ReadSample.restype = ctypes.c_bool
        if hasattr(self.dll, "GetNumNodes"):
            self.dll.GetNumNodes.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self.dll.GetNumNodes.restype = ctypes.c_bool
        if hasattr(self.dll, "GetNumMsrs"):
            self.dll.GetNumMsrs.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self.dll.GetNumMsrs.restype = ctypes.c_bool
        if hasattr(self.dll, "GetMsrFunc"):
            self.dll.GetMsrFunc.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            self.dll.GetMsrFunc.restype = ctypes.c_bool
        self.dll.GetPowerData.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.dll.GetPowerData.restype = ctypes.c_bool

    def _node_count(self) -> int:
        if not hasattr(self.dll, "GetNumNodes"):
            return 1
        count = ctypes.c_int(0)
        if not bool(self.dll.GetNumNodes(ctypes.byref(count))):
            return 1
        return max(1, int(count.value))

    def _power_msr_ids(self) -> List[int]:
        if not hasattr(self.dll, "GetNumMsrs") or not hasattr(self.dll, "GetMsrFunc"):
            return [1]
        count = ctypes.c_int(0)
        if not bool(self.dll.GetNumMsrs(ctypes.byref(count))):
            return [1]
        out: List[int] = []
        for msr in range(max(0, int(count.value))):
            func_id = ctypes.c_int(-1)
            if bool(self.dll.GetMsrFunc(msr, ctypes.byref(func_id))) and int(func_id.value) == 1:
                out.append(msr)
        return out or [1]

    def read(self) -> PowerReading:
        if not bool(self.dll.ReadSample()):
            raise PowerSensorUnavailable("Intel Power Gadget ReadSample failed")
        power_values: List[float] = []
        energy_values: List[float] = []
        for node in range(self._node_count()):
            for msr in self._power_msr_ids():
                values = (ctypes.c_double * 8)()
                n_values = ctypes.c_int(8)
                ok = bool(self.dll.GetPowerData(node, msr, values, ctypes.byref(n_values)))
                if not ok or n_values.value <= 0:
                    continue
                channel = [float(values[i]) for i in range(min(8, n_values.value))]
                if len(channel) >= 1 and 0.0 < channel[0] <= 1000.0:
                    power_values.append(channel[0])
                if len(channel) >= 2 and channel[1] >= 0.0:
                    energy_values.append(channel[1])
        watts = sum(power_values) if power_values else None
        joules = sum(energy_values) if energy_values else None
        if watts is None and joules is None:
            raise PowerSensorUnavailable("Intel Power Gadget returned no plausible power channels")
        if joules is not None:
            if self._last_energy_joules is not None and joules < self._last_energy_joules:
                joules = self._last_energy_joules
            self._last_energy_joules = joules
        if watts is not None:
            self._last_power_watts = watts
        return PowerReading(
            timestamp=time.time(),
            watts=watts,
            joules=joules,
            source=self.name,
            detail=os.path.basename(self.dll_path),
        )


class WindowsPowerMeterSensor(PowerSensor):
    """Reads the Windows ``Power Meter`` performance counter and integrates W over time."""

    name = "windows_power_meter"
    method = "windows_power_meter_integrated_watts"

    def __init__(self):
        if os.name != "nt":
            raise PowerSensorUnavailable("Windows Power Meter counters require Windows")
        self._read_counter()

    def _read_counter(self) -> float:
        script = (
            "$ErrorActionPreference='Stop'; "
            "$s=Get-Counter '\\Power Meter(*)\\Power'; "
            "$s.CounterSamples | Select-Object -ExpandProperty CookedValue | ConvertTo-Json"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise PowerSensorUnavailable(result.stderr.strip() or "Get-Counter Power Meter failed")
        text = result.stdout.strip()
        if not text:
            raise PowerSensorUnavailable("Power Meter counter returned no samples")
        values = json.loads(text)
        if not isinstance(values, list):
            values = [values]
        watts = [float(value) for value in values if _positive(float(value), upper=2000.0)]
        if not watts:
            raise PowerSensorUnavailable("Power Meter samples were empty or invalid")
        return float(sum(watts))

    def read(self) -> PowerReading:
        watts = self._read_counter()
        return PowerReading(timestamp=time.time(), watts=watts, joules=None, source=self.name)


def _candidate_energy_lib_paths() -> Iterable[str]:
    env = os.environ.get("INTEL_POWER_GADGET_DLL")
    if env:
        yield env
    found = ctypes.util.find_library("EnergyLib64")
    if found:
        yield found
    program_dirs = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    versions = ["Power Gadget 4.0", "Power Gadget 3.6", "Power Gadget 3.5", "Power Gadget"]
    for base in program_dirs:
        if not base:
            continue
        for version in versions:
            yield os.path.join(base, "Intel", version, "EnergyLib64.dll")
            yield os.path.join(base, "Intel", "Intel(R) Power Gadget", version, "EnergyLib64.dll")


def _find_energy_lib() -> Optional[str]:
    for path in _candidate_energy_lib_paths():
        if path and os.path.exists(path):
            return path
    return None


def available_sensor_names() -> List[str]:
    names: List[str] = []
    for name in ("intel_power_gadget", "windows_power_meter"):
        try:
            sensor = create_power_sensor(name)
            sensor.close()
            names.append(name)
        except PowerSensorUnavailable:
            continue
    return names


def create_power_sensor(name: str = "auto", dll_path: Optional[str] = None) -> PowerSensor:
    normalized = str(name or "auto").strip().lower()
    errors: List[Tuple[str, str]] = []
    if normalized in {"auto", "intel", "intel_power_gadget", "rapl"}:
        try:
            return IntelPowerGadgetSensor(dll_path=dll_path)
        except PowerSensorUnavailable as exc:
            errors.append(("intel_power_gadget", str(exc)))
            if normalized not in {"auto"}:
                raise
    if normalized in {"auto", "windows", "windows_power_meter", "power_meter"}:
        try:
            return WindowsPowerMeterSensor()
        except PowerSensorUnavailable as exc:
            errors.append(("windows_power_meter", str(exc)))
            if normalized not in {"auto"}:
                raise
    detail = "; ".join(f"{name}: {err}" for name, err in errors) or f"unknown sensor {name}"
    raise PowerSensorUnavailable(f"no usable real power sensor found ({detail})")
