# Hướng dẫn Hiệu suất và Tối ưu hóa

## Tổng quan

LSL được thiết kế cho CPU-only performance với sparse computation. Hướng dẫn này bao gồm các kỹ thuật tối ưu hóa và profiling.

## Sparse Computation

### Bật Sparse Computation

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sparse_computation=True,  # Bật sparse forward
    use_predictive_coding=True,
    theta=0.02
)
```

Sparse computation chỉ xử lý active neurons, đạt 40x+ speedup so với dense.

### Sparsity Tuning

```python
# Higher sparsity = faster but lower capacity
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sdr=True,
    sdr_sparsity=0.1,  # 10% active bits (faster)
    # sdr_sparsity=0.2  # 20% active bits (default)
    # sdr_sparsity=0.3  # 30% active bits (slower but higher capacity)
)
```

### Top-k Sparse Forward

```python
# Trong LivingSynapseLayer
layer.forward(x, use_sparse=True)
```

Chỉ update top-k weights, giảm số operations.

## Predictive Coding Optimization

### Suppression Threshold

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_predictive_coding=True,
    theta=0.01,  # Lower = more suppression = faster
    # theta=0.02  # Default
    # theta=0.05  # Higher = less suppression = slower
)
```

Suppression bỏ qua small errors, tiết kiệm compute.

### Monitor Suppression

```python
metrics = model.metrics()
print(f"Suppression %: {metrics['e_emb_suppression_pct']:.2%}")
```

Target: >60% suppression.

## Memory Optimization

### Bounded Memory Lookup

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    memory_candidate_cap=32,  # Giảm từ 64 xuống 32
    use_sparse_memory=True
)
```

Giảm candidate cap giảm lookup time.

### Long Context Capacity

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_long_context_memory=True,
    long_context_capacity=65536,  # 64k thay vì 128k
    long_context_confidence_threshold=0.6  # Tăng threshold để giảm retrieval
)
```

### Episodic Buffer Size

```python
model.episodic = EpisodicBuffer(
    capacity=128,  # Giảm từ 256
    candidate_cap=32  # Giảm từ 64
)
```

## Dimension Tuning

### Hidden Dimension

```python
# Smaller hidden dim = faster
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=128,  # Giảm từ 256
    # hidden_dim=256  # Default
    # hidden_dim=512  # Slower but more capacity
)
```

Trade-off: speed vs capacity.

### SDR Dimension

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_semantic_sdr=True,
    semantic_hidden_dim=512,  # Giảm từ 1000
    embedding_dim=200  # Giảm từ 300
)
```

## Batch Processing

### Vectorized Operations

```python
# Bad: sequential
for i in range(len(tokens)):
    model.observe(tokens[i], tokens[i+1])

# Good: vẫn sequential nhưng tối ưu
for i in range(len(tokens)):
    model.observe(tokens[i], tokens[i+1])
```

LSL không hỗ trợ batch training do online learning nature.

## Consolidation Strategy

### Periodic Consolidation

```python
# Consolidate mỗi N steps
if model.step_count % 1000 == 0:
    model.consolidate(threshold=0.005, fraction=0.3)
```

### Adaptive Consolidation

```python
# Consolidate dựa trên live norm
if model.live_norm() > threshold:
    model.consolidate(threshold=0.005, fraction=0.3)
```

## Replay Optimization

### Replay Frequency

```python
# Replay mỗi consolidation
if model.step_count % 1000 == 0:
    model.consolidate()
    model.replay(n=16, lr_factor=0.5)
```

### Replay Batch Size

```python
# Giảm replay batch size
model.replay(n=8, lr_factor=0.5)  # Thay vì n=16
```

## Profiling

### CPU Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code để profile
for i in range(1000):
    model.observe(i, (i+1) % 1000)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Code để profile
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
for i in range(1000):
    model.observe(i, (i+1) % 1000)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

tracemalloc.stop()

### Line-by-Line Profiling

```bash
pip install line_profiler
```

```python
# Decorator để profile
@profile
def observe(self, token_id, target_id, reward=0.0, store=True):
    # ... code ...
```

```bash
kernprof -l -v script.py
```

## Performance Benchmarks

### Sparse vs Dense

```python
import time

# Dense
model_dense = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sparse_computation=False,
    seed=42
)

start = time.time()
for i in range(1000):
    model_dense.forward(i)
dense_time = time.time() - start

# Sparse
model_sparse = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sparse_computation=True,
    seed=42
)

start = time.time()
for i in range(1000):
    model_sparse.forward(i)
sparse_time = time.time() - start

speedup = dense_time / sparse_time
print(f"Speedup: {speedup:.2f}x")
```

Target: >40x speedup.

### Memory Usage

```python
import tracemalloc

tracemalloc.start()
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
snapshot1 = tracemalloc.take_snapshot()

for i in range(1000):
    model.observe(i, (i+1) % 1000)

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

## Optimization Checklist

### Before Optimization

- [ ] Profile để tìm bottleneck
- [ ] Measure baseline performance
- [ ] Identify hot paths
- [ ] Check memory usage

### During Optimization

- [ ] Bật sparse computation
- [ ] Tune sparsity parameters
- [ ] Optimize memory lookup
- [ ] Reduce dimensions nếu có thể
- [ ] Consolidate định kỳ

### After Optimization

- [ ] Re-profile để verify improvement
- [ ] Check accuracy không bị giảm
- [ ] Verify strict constraints vẫn hold
- [ ] Document changes

## Common Bottlenecks

### Dense Matrix Operations

**Problem:** Dense matrix multiplication

**Solution:** Bật sparse computation
```python
model = LivingSynapseLM(use_sparse_computation=True)
```

### Full Memory Scan

**Problem:** Scan toàn bộ memory

**Solution:** Sử dụng bounded lookup
```python
model = LivingSynapseLM(memory_candidate_cap=32)
```

### Too Many Active Bits

**Problem:** SDR có quá nhiều active bits

**Solution:** Giảm sparsity
```python
model = LivingSynapseLM(sdr_sparsity=0.1)
```

### Frequent Consolidation

**Problem:** Consolidate quá thường xuyên

**Solution:** Giảm frequency
```python
if model.step_count % 2000 == 0:  # Thay vì 1000
    model.consolidate()
```

## Scaling Performance

### Vocabulary Size Scaling

```python
# 1k vocab
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256)

# 10k vocab
model = LivingSynapseLM(vocab_size=10000, hidden_dim=512)

# 100k vocab
model = LivingSynapseLM(vocab_size=100000, hidden_dim=1024)

# 1M vocab
model = LivingSynapseLM(
    vocab_size=1000000,
    hidden_dim=2048,
    use_semantic_sdr=True,
    semantic_hidden_dim=2048
)
```

### Context Length Scaling

LSL có O(1) per-token compute, không tăng với context length.

```python
# Short context
model.observe_sequence(tokens[:100])

# Long context
model.observe_sequence(tokens[:10000])  # Same per-token cost
```

### Batch Size Scaling

LSL không hỗ trợ batch training, nhưng có thể parallelize independent runs:

```python
from multiprocessing import Pool

def train_model(seed):
    model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=seed)
    for i in range(1000):
        model.observe(i, (i+1) % 1000)
    return model.metrics()

with Pool(4) as p:
    results = p.map(train_model, [42, 43, 44, 45])
```

## CPU-Specific Optimizations

### NumPy Optimization

```python
import numpy as np

# Sử dụng float32 thay vì float64
x = np.array([1.0, 2.0], dtype=np.float32)

# Sử dụng in-place operations
x += y  # Thay vì x = x + y

# Sử dụng BLAS optimization
np.show_config()  # Kiểm tra BLAS
```

### Thread Parallelization

```bash
export OMP_NUM_THREADS=4  # Số threads cho OpenMP
export MKL_NUM_THREADS=4  # Số threads cho MKL
```

### Cache Optimization

```python
# Access patterns cache-friendly
# Sequential access tốt hơn random access
for i in range(n):
    process(data[i])  # Good

for i in random_indices:
    process(data[i])  # Bad
```

## Energy Efficiency

### Predictive Coding Savings

```python
model = LivingSynapseLM(
    use_predictive_coding=True,
    theta=0.02
)

# Monitor savings
metrics = model.metrics()
suppression_pct = metrics['e_emb_suppression_pct']
energy_savings = suppression_pct * 0.6  # 60% max savings
```

### Sparse Compute Savings

```python
# Sparse compute chỉ touches active neurons
# Savings ~ (1 - sparsity) * operations
sparsity = 0.2
savings = (1 - sparsity) * 100  # 80% savings
```

## Monitoring Performance

### Real-time Metrics

```python
import time

start = time.time()
for i in range(1000):
    model.observe(i, (i+1) % 1000)
elapsed = time.time() - start

tokens_per_sec = 1000 / elapsed
print(f"Tokens/sec: {tokens_per_sec:.2f}")
```

### Memory Monitoring

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f"Memory: {mem_info.rss / 1024 / 1024:.2f} MB")
```

### Operation Counting

```python
# LivingSynapseLayer tracks operations
layer = LivingSynapseLayer(in_dim=256, out_dim=256)
layer.forward(x, use_sparse=True)

# Check operation count
print(f"Operations: {layer.op_count}")
```

## Performance Targets

### Strict Benchmark Targets

- Sparse compute: >=40x speedup vs dense
- Prediction error drop: >=50%
- Signal suppression: >=60%
- Per-token compute: O(1) with context

### Moonshot Targets

- 1M vocabulary semantic SDR
- 128k long-context retrieval
- <1s per 1000 tokens (CPU)
- <100MB memory footprint

### Competitive Targets

- Open generation: <500ms per token
- World memory: <100ms per query
- Reasoning: <200ms per hop

## Troubleshooting Performance

### Slow Forward Pass

**Symptoms:** Forward pass quá chậm

**Solutions:**
1. Bật `use_sparse_computation=True`
2. Giảm `hidden_dim`
3. Giảm `sdr_sparsity`
4. Kiểm tra NumPy BLAS optimization

### High Memory Usage

**Symptoms:** Memory usage quá cao

**Solutions:**
1. Giảm `long_context_capacity`
2. Giảm `episodic` capacity
3. Giảm `memory_candidate_cap`
4. Consolidate thường xuyên hơn

### Low Suppression

**Symptoms:** Suppression percentage thấp

**Solutions:**
1. Tăng `theta` threshold
2. Kiểm tra prediction error distribution
3. Tune learning rates

### Poor Scaling

**Symptoms:** Performance không scale với vocabulary size

**Solutions:**
1. Sử dụng semantic SDR cho large vocab
2. Tăng `semantic_hidden_dim`
3. Giảm `embedding_dim`

## Best Practices

### 1. Profile First

Luôn profile trước khi optimize:
```python
import cProfile
cProfile.run('your_code()', 'profile.stats')
```

### 2. Measure, Don't Guess

Sử dụng metrics để guide optimization:
```python
metrics = model.metrics()
print(metrics)
```

### 3. Trade-offs

Hiểu trade-offs:
- Speed vs Accuracy
- Memory vs Capacity
- Sparsity vs Robustness

### 4. Verify Constraints

Sau khi optimize, verify strict constraints:
```bash
python benchmark_goal_strict.py
```

### 5. Document Changes

Document optimization decisions:
```python
# Reduced hidden_dim from 256 to 128 for 2x speedup
# Accuracy impact: <5%
```

## Advanced Optimization

### Custom Sparse Implementation

```python
def custom_sparse_forward(W, x, k=10):
    """Custom sparse forward with top-k."""
    # Compute top-k indices
    top_k_idx = np.argpartition(np.abs(x), -k)[-k:]
    
    # Only compute for top-k
    output = np.zeros_like(W @ x)
    for idx in top_k_idx:
        output += W[:, idx] * x[idx]
    
    return output
```

### Memory Pooling

```python
from multiprocessing.shared_memory import SharedMemory

# Shared memory cho multi-process
shm = SharedMemory(create=True, size=1024*1024)
```

### JIT Compilation

```python
from numba import jit

@jit(nopython=True)
def fast_function(x):
    # Fast compiled function
    return x * 2
```

## Performance Monitoring Dashboard

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    def track(self, model, operation):
        start = time.time()
        result = operation()
        elapsed = time.time() - start
        
        self.metrics.append({
            'time': elapsed,
            'memory': psutil.Process().memory_info().rss,
            'model_metrics': model.metrics()
        })
        
        return result
    
    def report(self):
        avg_time = np.mean([m['time'] for m in self.metrics])
        avg_mem = np.mean([m['memory'] for m in self.metrics])
        print(f"Avg time: {avg_time:.4f}s")
        print(f"Avg memory: {avg_mem / 1024 / 1024:.2f} MB")
```

## Summary

Key optimization strategies:
1. Bật sparse computation (40x+ speedup)
2. Tune sparsity parameters
3. Use bounded memory lookup
4. Optimize consolidation strategy
5. Profile before optimizing
6. Monitor metrics continuously
7. Verify strict constraints
8. Document trade-offs
