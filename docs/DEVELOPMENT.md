# Hướng dẫn Phát triển và Đóng góp

## Thiết lập môi trường development

### Cài đặt dev dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### Cấu hình IDE

**VS Code:**
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

**PyCharm:**
- Enable pytest runner
- Configure black formatter
- Enable flake8 inspection

## Quy trình phát triển

### 1. Fork và Clone

```bash
git clone <your-fork-url>
cd brain
git remote add upstream <original-repo-url>
```

### 2. Tạo branch mới

```bash
git checkout -b feature/your-feature-name
# hoặc
git checkout -b fix/your-bug-fix
```

### 3. Thực hiện thay đổi

- Viết code theo style guide
- Thêm tests cho new features
- Cập nhật documentation
- Đảm bảo tất cả tests pass

### 4. Commit changes

```bash
git add .
git commit -m "feat: add new feature"
# hoặc
git commit -m "fix: resolve bug in component"
```

**Commit message convention:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation update
- `test:` - Test addition/update
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance task

### 5. Push và tạo Pull Request

```bash
git push origin feature/your-feature-name
```

Sau đó tạo Pull Request trên GitHub với:
- Mô tả rõ ràng về thay đổi
- Link đến related issues
- Screenshots nếu applicable
- Checklist của requirements

## Code Style Guide

### Python Style

Sử dụng Black formatter:

```bash
black lsl/ benchmarks/ tests/
```

### Linting

Sử dụng flake8:

```bash
flake8 lsl/ benchmarks/ tests/
```

### Type Hints

Thêm type hints cho functions:

```python
from typing import Optional, Dict, List

def forward(self, token_id: int, target_id: Optional[int] = None) -> np.ndarray:
    pass
```

### Docstrings

Sử dụng Google style docstrings:

```python
def observe(self, token_id: int, target_id: int, reward: float = 0.0) -> Dict[str, float]:
    """Observe token pair and update model.

    Args:
        token_id: Input token ID.
        target_id: Target token ID.
        reward: Reward signal for neuromodulation.

    Returns:
        Dictionary with prediction_error, modulator, novelty, top1, p_target.
    """
    pass
```

## Testing

### Viết Tests

Tạo test file trong `tests/`:

```python
import pytest
import numpy as np
from lsl import LivingSynapseLM

def test_model_forward():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    logits = model.forward(token_id=0)
    assert logits.shape == (100,)

def test_model_observe():
    model = LivingSynapseLM(vocab_size=100, hidden_dim=64, seed=42)
    info = model.observe(0, 1)
    assert 'prediction_error' in info
    assert info['prediction_error'] >= 0
```

### Chạy Tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy specific test file
pytest tests/test_model.py

# Chạy với coverage
pytest --cov=lsl tests/

# Chạy với verbose output
pytest -v tests/
```

### Test Coverage

Mục tiêu coverage: >80%

```bash
pytest --cov=lsl --cov-report=html tests/
```

## Strict Constraints

### Forbidden Constructs

Dự án có strict constraints trong `GOAL.md`. Khi thêm code mới:

**❌ KHÔNG sử dụng:**
- Backpropagation: `loss.backward()`, `torch.autograd.grad()`
- Optimizers: `Adam`, `SGD`, `optimizer.step()`
- Deep learning frameworks: PyTorch, TensorFlow, JAX
- Attention matrices: Q/K/V attention, all-pairs interaction
- GPU-specific code: `.cuda()`, `.to(device)`
- Global backward pass

**✅ PHẢI sử dụng:**
- Local updates only
- NumPy operations
- CPU-only computation
- Online learning rules

### Verification

Chạy strict benchmark trước khi commit:

```bash
python benchmark_goal_strict.py
```

Đảm bảo kết quả:
```
Goals passed: 18/18
Overall: PASS
Structural PASS clean
```

## Architecture Guidelines

### Thêm Component Mới

1. Tạo file trong `lsl/`
2. Thêm imports vào `lsl/__init__.py`
3. Viết tests trong `tests/`
4. Cập nhật documentation
5. Thêm benchmark nếu cần

**Ví dụ:**

```python
# lsl/new_component.py
"""New component description."""
import numpy as np

class NewComponent:
    def __init__(self, param1: int, seed: int = 0):
        self.param1 = param1
        self.rng = np.random.default_rng(seed)
    
    def method(self, x: np.ndarray) -> np.ndarray:
        """Method description."""
        return x
```

```python
# lsl/__init__.py
from .new_component import NewComponent

__all__ = [
    # ... existing exports
    "NewComponent",
]
```

### Memory Components

Khi thêm memory component mới:
- Implement bounded lookup (không full scan)
- Sử dụng sparse computation
- Add consolidation mechanism
- Track metrics

### Predictive Coding Components

Khi thêm predictive coding component:
- Sử dụng local transition predictors
- Tính local prediction error
- Implement suppression mechanism
- Track error metrics

## Benchmark Guidelines

### Thêm Benchmark Mới

1. Tạo file trong `benchmarks/phaseX/`
2. Implement benchmark logic
3. Thêm vào runner
4. Document goals
5. Add to README

**Ví dụ structure:**

```python
# benchmarks/phaseX/test_new_feature.py
import numpy as np
from lsl import LivingSynapseLM

def run_benchmark():
    """Run benchmark for new feature."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    # ... benchmark logic
    return results

if __name__ == "__main__":
    results = run_benchmark()
    print(results)
```

### Benchmark Goals

Mỗi benchmark phải có:
- Rõ ràng success criteria
- Quantitative thresholds
- Baseline comparison
- Reproducibility (fixed seed)

## Documentation Guidelines

### Cập nhật Documentation

Khi thay đổi code:
1. Cập nhật `docs/API.md` nếu API thay đổi
2. Cập nhật `docs/ARCHITECTURE.md` nếu architecture thay đổi
3. Cập nhật `docs/USAGE.md` nếu usage thay đổi
4. Cập nhật `README.md` nếu user-facing changes
5. Thêm examples vào `examples/`

### Documentation Style

- Sử dụng Markdown
- Thêm code examples
- Sử dụng clear headings
- Provide parameter descriptions
- Include return value descriptions

## Performance Guidelines

### Profiling

Sử dụng Python profiler:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### Optimization Tips

- Sử dụng sparse computation khi có thể
- Tránh unnecessary copies
- Sử dụng in-place operations
- Batch operations
- Pre-allocate arrays

## Debugging Guidelines

### Logging

Thêm logging cho debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Token ID: {token_id}")
logger.info(f"Prediction error: {error:.4f}")
```

### Debug Mode

Thêm debug flag:

```python
def forward(self, x, debug=False):
    if debug:
        print(f"Input shape: {x.shape}")
    # ... code ...
```

### Assertions

Sử dụng assertions cho development:

```python
assert token_id >= 0 and token_id < self.vocab_size
assert error.shape == (self.hidden_dim,)
```

## Review Process

### Self-Review Checklist

Trước khi submit PR:
- [ ] Code follows style guide
- [ ] All tests pass
- [ ] New tests added
- [ ] Documentation updated
- [ ] Strict benchmark passes
- [ ] No forbidden constructs
- [ ] Performance acceptable
- [ ] Commit messages clear

### Code Review

Khi review code:
- Kiểm tra strict constraints
- Verify test coverage
- Check documentation
- Assess performance impact
- Verify reproducibility

## Release Process

### Version Bump

Sử dụng semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

### Changelog

Cập nhật `CHANGELOG.md`:

```markdown
## [1.0.0] - 2024-01-01

### Added
- New component X
- Feature Y

### Fixed
- Bug in component Z

### Changed
- Improved performance of A
```

### Release Checklist

- [ ] All tests pass
- [ ] Documentation complete
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Tag created
- [ ] Release notes published

## Contributing Examples

### Example 1: Thêm Utility Function

```python
# lsl/utils.py
def normalize_vector(x: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length.
    
    Args:
        x: Input vector.
        
    Returns:
        Normalized vector.
    """
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm
```

```python
# tests/test_utils.py
import numpy as np
from lsl.utils import normalize_vector

def test_normalize_vector():
    x = np.array([3.0, 4.0])
    normalized = normalize_vector(x)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
```

### Example 2: Thêm Benchmark

```python
# benchmarks/phaseX/test_new_metric.py
def run_benchmark():
    """Test new metric."""
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
    
    # Train
    for i in range(100):
        model.observe(i, (i+1) % 1000)
    
    # Test
    metric_value = model.compute_new_metric()
    
    # Assert threshold
    assert metric_value >= 0.8, f"Metric too low: {metric_value}"
    
    return {"metric": metric_value, "status": "PASS"}

if __name__ == "__main__":
    result = run_benchmark()
    print(result)
```

## Getting Help

### Resources

- `docs/ARCHITECTURE.md` - Architecture details
- `docs/API.md` - API reference
- `docs/USAGE.md` - Usage examples
- `GOAL.md` - Strict contract
- `README.md` - Quick start

### Questions

Mở issue trên GitHub với:
- Mô tả rõ ràng vấn đề
- Code example
- Expected vs actual behavior
- Environment details

### Discussions

Sử dụng GitHub Discussions cho:
- Feature requests
- Architecture questions
- General discussions

## Best Practices Summary

1. **Luôn** chạy strict benchmark trước khi commit
2. **Luôn** viết tests cho new features
3. **Luôn** cập nhật documentation
4. **Không bao giờ** sử dụng forbidden constructs
5. **Luôn** sử dụng fixed seed cho reproducibility
6. **Luôn** profile code cho performance
7. **Luôn** review code trước khi submit
8. **Luôn** follow commit message convention
