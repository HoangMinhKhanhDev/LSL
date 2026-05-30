# Hướng dẫn Testing

## Tổng quan

LSL sử dụng multi-tier testing strategy:
1. Unit tests cho individual components
2. Integration tests cho component interactions
3. Benchmark tests cho goal verification
4. Strict structural tests cho constraint enforcement

## Cài đặt Test Environment

### Cài đặt pytest

```bash
pip install pytest pytest-cov pytest-xdist
```

### Cấu hình pytest

Tạo `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

## Chạy Tests

### Chạy tất cả tests

```bash
pytest tests/
```

### Chạy specific test file

```bash
pytest tests/test_model.py
```

### Chạy specific test function

```bash
pytest tests/test_model.py::test_model_forward
```

### Chạy với coverage

```bash
pytest --cov=lsl --cov-report=html tests/
```

Coverage report sẽ được tạo trong `htmlcov/index.html`

### Chạy với parallel execution

```bash
pytest -n auto tests/
```

### Chạy với verbose output

```bash
pytest -v -s tests/
```

## Unit Tests

### Test Structure

```python
# tests/test_model.py
import pytest
import numpy as np
from lsl import LivingSynapseLM

class TestLivingSynapseLM:
    def test_initialization(self):
        """Test model initialization."""
        model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
        assert model.vocab_size == 100
        assert model.hidden_dim == 64
        assert model.step_count == 0
    
    def test_forward(self):
        """Test forward pass."""
        model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
        logits = model.forward(token_id=0)
        assert logits.shape == (100,)
        assert not np.any(np.isnan(logits))
    
    def test_observe(self):
        """Test observe method."""
        model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
        info = model.observe(0, 1)
        assert 'prediction_error' in info
        assert info['prediction_error'] >= 0
        assert model.step_count == 1
    
    def test_predict(self):
        """Test predict method."""
        model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
        probs = model.predict(0)
        assert probs.shape == (100,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np
from lsl import LivingSynapseLM, SDREncoder

@pytest.fixture
def simple_model():
    """Fixture for simple model."""
    return LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)

@pytest.fixture
def sdr_encoder():
    """Fixture for SDR encoder."""
    return SDREncoder(dim=1024, sparsity=0.2, seed=42)

@pytest.fixture
def sample_tokens():
    """Fixture for sample tokens."""
    return np.array([0, 1, 2, 3, 4])

# Usage in tests
def test_with_fixture(simple_model):
    logits = simple_model.forward(0)
    assert logits.shape == (100,)
```

### Parameterized Tests

```python
@pytest.mark.parametrize("vocab_size,hidden_dim", [
    (100, 64),
    (1000, 256),
    (10000, 512),
])
def test_model_sizes(vocab_size, hidden_dim):
    """Test model with different sizes."""
    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=42)
    assert model.vocab_size == vocab_size
    assert model.hidden_dim == hidden_dim
```

## Integration Tests

### Test Component Interactions

```python
# tests/test_integration.py
def test_model_with_sdr():
    """Test model with SDR encoding."""
    model = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_sdr=True,
        sdr_sparsity=0.2,
        seed=42
    )
    
    # Train
    for i in range(100):
        model.observe(i, (i+1) % 1000)
    
    # Test prediction
    probs = model.predict(0)
    assert probs.shape == (1000,)
    assert np.isclose(probs.sum(), 1.0)

def test_model_with_predictive_coding():
    """Test model with predictive coding."""
    model = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_predictive_coding=True,
        theta=0.02,
        seed=42
    )
    
    # Train
    for i in range(100):
        info = model.observe(i, (i+1) % 1000)
        assert 'prediction_error' in info
    
    # Check metrics
    metrics = model.metrics()
    assert 'e_emb_norm' in metrics
    assert 'e_ssm_norm' in metrics
    assert 'e_rec_norm' in metrics
```

### Test Memory Systems

```python
def test_long_context_memory():
    """Test long context memory."""
    model = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_long_context_memory=True,
        long_context_capacity=1000,
        seed=42
    )
    
    # Store transitions
    for i in range(100):
        model.observe(i, (i+1) % 1000)
    
    # Retrieve
    token, conf = model.long_context.predict_next(
        0,
        vocab_size=1000,
        return_confidence=True
    )
    assert token is not None
    assert 0 <= conf <= 1
```

## Benchmark Tests

### Strict Benchmark

```bash
python benchmark_goal_strict.py
```

Kiểm tra 18 goals:
- Phase 1: SDR (6 goals)
- Phase 2: Predictive Coding (6 goals)
- Phase 3: Cortical Column (6 goals)

Kết quả mong đợi:
```
Goals passed: 18/18
Overall: PASS
Structural PASS clean
```

### Phase Benchmarks

```bash
# Phase 1
python benchmark_sdr_phase1.py

# Phase 2
python benchmark_pc_phase2.py

# Phase 3
python benchmark_cortical_column_sequence.py

# Phase 4
python benchmark_semantic_sdr_scaling.py
python benchmark_sparse_physical_compute.py

# Phase 5
python benchmark_moonshot.py --profile quick
python benchmark_moonshot.py --profile full

# Phase 6
python benchmark_phase6.py --profile quick
python benchmark_phase6.py --profile full
```

### Custom Benchmark

```python
# tests/test_custom_benchmark.py
def test_custom_metric():
    """Test custom metric."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    # Train
    for i in range(1000):
        model.observe(i, (i+1) % 1000)
    
    # Compute metric
    metric = compute_custom_metric(model)
    
    # Assert threshold
    assert metric >= 0.8, f"Metric too low: {metric}"
```

## Structural Tests

### Strict Constraint Tests

```python
# tests/test_strict_constraints.py
def test_no_backpropagation():
    """Ensure no backpropagation in codebase."""
    import ast
    import os
    
    forbidden = ['backward()', 'autograd.grad', 'torch.autograd']
    
    for root, dirs, files in os.walk('lsl'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    for term in forbidden:
                        assert term not in content, f"Found {term} in {filepath}"

def test_no_optimizer_calls():
    """Ensure no optimizer calls."""
    import os
    
    forbidden = ['Adam', 'SGD', 'optimizer.step', 'optimizer.zero_grad']
    
    for root, dirs, files in os.walk('lsl'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    for term in forbidden:
                        assert term not in content, f"Found {term} in {filepath}"

def test_no_deep_learning_frameworks():
    """Ensure no deep learning framework imports."""
    import os
    
    forbidden = ['import torch', 'import tensorflow', 'import jax']
    
    for root, dirs, files in os.walk('lsl'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    for term in forbidden:
                        assert term not in content, f"Found {term} in {filepath}"
```

## Performance Tests

### Timing Tests

```python
import time

def test_forward_speed():
    """Test forward pass speed."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    start = time.time()
    for i in range(1000):
        model.forward(i)
    elapsed = time.time() - start
    
    # Should be fast (< 1 second for 1000 forwards)
    assert elapsed < 1.0, f"Too slow: {elapsed}s"

def test_sparse_vs_dense():
    """Test sparse computation is faster."""
    model_dense = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_sparse_computation=False,
        seed=42
    )
    
    model_sparse = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_sparse_computation=True,
        seed=42
    )
    
    # Time dense
    start = time.time()
    for i in range(100):
        model_dense.forward(i)
    dense_time = time.time() - start
    
    # Time sparse
    start = time.time()
    for i in range(100):
        model_sparse.forward(i)
    sparse_time = time.time() - start
    
    # Sparse should be faster
    assert sparse_time < dense_time, f"Sparse not faster: {sparse_time} vs {dense_time}"
```

### Memory Tests

```python
import tracemalloc

def test_memory_usage():
    """Test memory usage."""
    tracemalloc.start()
    
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    for i in range(1000):
        model.observe(i, (i+1) % 1000)
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    # Check memory is reasonable (< 100MB)
    total_memory = sum(stat.size for stat in top_stats)
    assert total_memory < 100 * 1024 * 1024, f"Memory too high: {total_memory / 1024 / 1024}MB"
    
    tracemalloc.stop()
```

## Regression Tests

### Test Known Issues

```python
def test_regression_issue_123():
    """Regression test for issue #123."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    # This used to fail
    for i in range(100):
        model.observe(i, (i+1) % 1000)
    
    # Should not raise error
    model.consolidate()
    model.replay(n=16)
```

## Test Categories

### Smoke Tests

Quick tests để verify basic functionality:

```python
def test_smoke_model():
    """Smoke test for model."""
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    logits = model.forward(0)
    assert logits is not None
```

### Unit Tests

Tests cho individual functions/methods:

```python
def test_sdr_encode():
    """Test SDR encoding."""
    encoder = SDREncoder(dim=1024, sparsity=0.2, seed=42)
    x = np.random.randn(1024)
    sdr = encoder.encode(x)
    assert sdr.shape == (1024,)
    assert np.isclose(sparsity_ratio(sdr), 0.2, atol=0.05)
```

### Integration Tests

Tests cho component interactions:

```python
def test_model_with_all_features():
    """Test model with all features enabled."""
    model = LivingSynapseLM(
        vocab_size=1000,
        hidden_dim=256,
        use_predictive_coding=True,
        use_sdr=True,
        use_sparse_computation=True,
        use_long_context_memory=True,
        seed=42
    )
    
    # Should work without errors
    for i in range(100):
        model.observe(i, (i+1) % 1000)
```

### End-to-End Tests

Tests cho complete workflows:

```python
def test_end_to_end_training():
    """Test complete training workflow."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    # Train
    for epoch in range(10):
        for i in range(100):
            model.observe(i, (i+1) % 1000)
        model.consolidate()
        model.replay(n=16)
    
    # Generate
    probs = model.predict(0)
    assert probs.shape == (1000,)
```

## Test Data Management

### Test Data Directory

```
tests/
├── data/
│   ├── sample_text.txt
│   └── sample_tokens.npy
├── conftest.py
├── test_model.py
└── test_sdr.py
```

### Fixtures for Test Data

```python
# tests/conftest.py
import numpy as np
import pytest

@pytest.fixture
def sample_text():
    """Load sample text."""
    with open('tests/data/sample_text.txt', 'r') as f:
        return f.read()

@pytest.fixture
def sample_tokens():
    """Load sample tokens."""
    return np.load('tests/data/sample_tokens.npy')
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=lsl tests/
      - name: Run strict benchmark
        run: python benchmark_goal_strict.py
```

## Test Best Practices

### 1. Use Descriptive Names

```python
# Good
def test_model_forward_returns_correct_shape()

# Bad
def test_forward()
```

### 2. Test One Thing

```python
# Good
def test_model_forward_shape()
def test_model_forward_no_nan()

# Bad
def test_model_forward()
```

### 3. Use Assertions

```python
# Good
assert result == expected
assert error < threshold

# Bad
if result != expected:
    print("Failed")
```

### 4. Isolate Tests

```python
# Good - each test independent
def test_model_initialization():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    assert model.vocab_size == 100

def test_model_forward():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    logits = model.forward(0)
    assert logits.shape == (100,)

# Bad - tests depend on each other
model = None
def test_init():
    global model
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)

def test_forward():
    global model
    logits = model.forward(0)
```

### 5. Use Fixtures

```python
# Good
@pytest.fixture
def model():
    return LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)

def test_forward(model):
    logits = model.forward(0)

# Bad
def test_forward():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    logits = model.forward(0)
```

## Debugging Tests

### Run with pdb

```bash
pytest --pdb tests/test_model.py::test_model_forward
```

### Print statements

```python
def test_with_debug():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    logits = model.forward(0)
    print(f"Logits shape: {logits.shape}")  # Use -s flag to see
    assert logits.shape == (100,)
```

### Stop on first failure

```bash
pytest -x tests/
```

## Test Coverage Goals

- Unit tests: >90% coverage
- Integration tests: >80% coverage
- Overall: >85% coverage

Kiểm tra coverage:
```bash
pytest --cov=lsl --cov-report=term-missing tests/
```

## Troubleshooting

### Tests fail randomly

- Sử dụng fixed seeds
- Kiểm tra race conditions
- Isolate flaky tests

### Tests too slow

- Sử pytest-xdist cho parallel execution
- Mock expensive operations
- Reduce test data size

### Import errors

- Kiểm tra PYTHONPATH
- Verify virtual environment
- Check dependencies

### Memory errors in tests

- Reduce batch sizes
- Use smaller models in tests
- Clean up after tests
