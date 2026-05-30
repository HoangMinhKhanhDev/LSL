"""Focused infrastructure checks for the extreme strict gate."""
import json
import math
import os
import tempfile

import numpy as np

from benchmarks.strict.benchmark_goal_strict import validate_energy_evidence
from benchmarks.strict.target_registry import STRICT_TARGETS, registry_by_id
from benchmarks.energy.power_sensors import PowerReading, PowerSensor
from lsl import LivingSynapseLayer, NATIVE_AVAILABLE, require_native


def test_registry():
    registry = registry_by_id()
    required = {"G1.1", "G1.2", "G1.6", "G2.5", "G4.3", "G8.1", "Structural"}
    missing = sorted(required - set(registry))
    assert not missing, missing
    assert all(target.tier == "strict" for target in STRICT_TARGETS)


def test_capacity_math():
    bits = (math.lgamma(100001.0) - math.lgamma(41.0) - math.lgamma(99961.0)) / math.log(2.0)
    assert bits >= 500.0


def test_native_sparse_kernel():
    require_native()
    assert NATIVE_AVAILABLE
    layer = LivingSynapseLayer(32, 16, seed=1)
    active = np.asarray([3, 19], dtype=np.intp)
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    expected = (layer.W_slow[:, active] + layer.W_live[:, active]) @ values
    post = layer.forward_active(active, values)
    assert float(np.max(np.abs(expected - post))) <= 1e-6
    assert layer.last_forward_ops["mode"].startswith("native")
    layer.hebbian_update_active(1.0, lr=0.001, decay=0.001)
    assert layer.last_update_ops["mode"].startswith("native")


def test_energy_evidence_validation():
    payload = {
        "method": "meter",
        "device": "test",
        "timestamp": "2026-05-30T00:00:00Z",
        "tokens": 1000,
        "dense_watts": 100.0,
        "sparse_watts": 10.0,
        "dense_joule_per_token": 1.0,
        "sparse_joule_per_token": 0.01,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "energy.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        ok, detail, metrics = validate_energy_evidence(path, require_real=True)
    assert ok, detail
    assert metrics["saving"] >= 0.98


class FakeJouleSensor(PowerSensor):
    name = "fake_joule"
    method = "unit_test"

    def __init__(self):
        self.values = iter([1.0, 3.5])

    def read(self):
        return PowerReading(timestamp=0.0, watts=None, joules=next(self.values), source=self.name)


class FakeWattSensor(PowerSensor):
    name = "fake_watt"
    method = "unit_test"

    def __init__(self):
        self.values = iter([10.0, 14.0])

    def read(self):
        return PowerReading(timestamp=0.0, watts=next(self.values), joules=None, source=self.name)


def test_power_sensor_measurement_math():
    joule = FakeJouleSensor().measure(lambda: None)
    assert abs(joule.joules - 2.5) <= 1e-9
    watt = FakeWattSensor().measure(lambda: None)
    assert watt.joules >= 0.0
    assert watt.watts >= 0.0


def main():
    test_registry()
    test_capacity_math()
    test_native_sparse_kernel()
    test_energy_evidence_validation()
    test_power_sensor_measurement_math()
    print("Extreme strict infra tests: PASS")


if __name__ == "__main__":
    main()
