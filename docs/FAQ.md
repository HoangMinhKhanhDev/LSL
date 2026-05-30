# Câu hỏi Thường gặp (FAQ) và Khắc phục sự cố

## Câu hỏi Thường gặp (FAQ)

### Tổng quan

#### LSL là gì?

LSL (Living Synapse Language Model) là một mô hình ngôn ngữ bio-inspired sử dụng học cục bộ (local learning) online mà không cần backpropagation, optimizer state, GPU, hay các deep learning frameworks. Mô hình được thiết kế để hoạt động trên CPU với NumPy.

#### LSL khác gì so với Transformer/LLM thông thường?

- **Không backpropagation:** LSL sử dụng local learning rules thay vì gradient descent
- **Không GPU:** Chỉ sử dụng CPU với NumPy
- **Sparse computation:** Chỉ xử lý active neurons, đạt 40x+ speedup
- **Online learning:** Học từng token theo thời gian thực
- **Biological inspiration:** Dựa trên cortical column và predictive coding

#### LSL có thể thay thế GPT/LLM không?

Không. LSL là research prototype để chứng minh khả năng học local online, không phải production-ready LLM. Nó không cạnh tranh với frontier models về performance.

#### Tôi có thể dùng LSL cho production không?

Không khuyến nghị. LSL là research prototype với strict constraints. Nếu muốn dùng production, nên cân nhắc các frameworks khác.

### Cài đặt

#### LSL yêu cầu GPU không?

Không. LSL được thiết kế để chạy trên CPU-only.

#### LSL yêu cầu Python version nào?

Python 3.8 hoặc cao hơn.

#### LSL có tương thích với Windows không?

Có. LSL sử dụng NumPy thuần, tương thích với Windows, Linux, và macOS.

#### Tôi có thể cài đặt LSL với pip không?

Hiện tại LSL không có PyPI package. Clone repository và cài đặt từ source.

#### LSL có dependencies gì ngoài NumPy?

Không. NumPy là dependency duy nhất bắt buộc.

### Sử dụng

#### Làm sao để bắt đầu nhanh?

```bash
python benchmark_goal_strict.py
```

Nếu pass 18/18 goals, cài đặt thành công.

#### Làm sao để train mô hình?

```python
from lsl import LivingSynapseLM

model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
for i in range(len(tokens) - 1):
    model.observe(tokens[i], tokens[i+1])
```

#### Làm sao để generate text?

```python
from lsl import GenerationController

controller = GenerationController(model, tokenizer)
text = controller.generate("prompt", max_tokens=50)
```

#### LSL hỗ trợ batch training không?

Không. LSL sử dụng online learning, train từng token.

#### LSL có hỗ trợ multi-GPU không?

Không. LSL là CPU-only.

#### Làm sao để sử dụng SDR?

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sdr=True,
    sdr_sparsity=0.2
)
```

#### Làm sao để sử dụng predictive coding?

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_predictive_coding=True,
    theta=0.02
)
```

### Performance

#### Tại sao mô hình của tôi chạy chậm?

- Kiểm tra `use_sparse_computation=True`
- Giảm `hidden_dim`
- Giảm `sdr_sparsity`
- Xem `docs/PERFORMANCE.md` cho optimization tips

#### LSL có thể xử lý vocabulary lớn không?

Có, với semantic SDR:
```python
model = LivingSynapseLM(
    vocab_size=1000000,
    use_semantic_sdr=True,
    semantic_hidden_dim=2048
)
```

#### LSL có hỗ trợ long context không?

Có, với LongContextMemory:
```python
model = LivingSynapseLM(
    vocab_size=1000,
    use_long_context_memory=True,
    long_context_capacity=131072  # 128k
)
```

#### LSL có O(n) per-token compute không?

Có. Compute không tăng với context length.

### Benchmarks

#### Làm sao để chạy strict benchmark?

```bash
python benchmark_goal_strict.py
```

#### Làm sao để chạy Moonshot benchmark?

```bash
python benchmark_moonshot.py --profile quick
python benchmark_moonshot.py --profile full
```

#### Làm sao để chạy phase benchmark cụ thể?

```bash
python benchmark_sdr_phase1.py
python benchmark_pc_phase2.py
python benchmark_cortical_column_sequence.py
```

#### Benchmark fail thì sao?

- Kiểm tra NumPy version
- Kiểm tra seed reproducibility
- Xem error message
- Kiểm tra strict constraints

### Development

#### Làm sao để đóng góp code?

Xem `docs/DEVELOPMENT.md` cho hướng dẫn chi tiết.

#### LSL có code style guide không?

Có. Sử dụng Black formatter và flake8 linter.

#### Làm sao để viết tests?

Xem `docs/TESTING.md` cho hướng dẫn.

#### LSL có CI/CD không?

Không có official CI/CD, nhưng có thể setup GitHub Actions (xem DEVELOPMENT.md).

### Strict Constraints

#### Tại sao không có backpropagation?

Đây là research contract để chứng minh local learning có thể hoạt động.

#### Tôi có thể thêm PyTorch không?

Không. Vi phạm strict contract trong `GOAL.md`.

#### Tôi có thể thêm attention mechanism không?

Không. Vi phạm strict contract (no attention matrix).

#### Làm sao để verify không vi phạm constraints?

```bash
python benchmark_goal_strict.py
```

Kiểm tra "Structural PASS clean".

## Khắc phục sự cố (Troubleshooting)

### Installation Issues

#### ImportError: No module named 'lsl'

**Cause:** Python path không include thư mục project

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows
```

Hoặc chạy từ thư mục gốc của project.

#### NumPy version error

**Cause:** NumPy version quá cũ

**Solution:**
```bash
pip install --upgrade numpy
```

#### Permission error khi tải dữ liệu

**Cause:** Không có quyền ghi vào thư mục data

**Solution:**
```bash
chmod +w benchmarks/data/  # Linux/macOS
# Chạy với admin quyền trên Windows
```

### Runtime Issues

#### MemoryError: Unable to allocate array

**Cause:** Memory không đủ

**Solution:**
- Giảm `hidden_dim`
- Giảm `long_context_capacity`
- Giảm `episodic` capacity
- Sử dụng profile `quick` thay vì `full`

#### ValueError: token_id out of range

**Cause:** Token ID lớn hơn vocab_size

**Solution:**
```python
assert token_id < model.vocab_size
```

Kiểm tra tokenizer và vocab_size.

#### NaN trong logits

**Cause:** Numerical instability

**Solution:**
- Kiểm tra learning rate
- Kiểm tra max_norm clipping
- Reset model state

#### Prediction error quá cao

**Cause:** Model chưa train đủ hoặc learning rate không phù hợp

**Solution:**
- Tăng số training steps
- Giảm learning rate
- Kiểm tra tokenization
- Thử với semantic SDR

### Performance Issues

#### Forward pass quá chậm

**Cause:** Không bật sparse computation

**Solution:**
```python
model = LivingSynapseLM(use_sparse_computation=True)
```

#### Memory usage quá cao

**Cause:** Memory buffers quá lớn

**Solution:**
```python
model = LivingSynapseLM(
    long_context_capacity=65536,  # Giảm
    episodic_capacity=128  # Giảm
)
```

#### Suppression percentage thấp

**Cause:** Theta threshold quá thấp

**Solution:**
```python
model = LivingSynapseLM(
    use_predictive_coding=True,
    theta=0.05  # Tăng từ 0.02
)
```

#### Sparse compute không nhanh hơn dense

**Cause:** Sparsity quá cao

**Solution:**
```python
model = LivingSynapseLM(
    use_sdr=True,
    sdr_sparsity=0.1  # Giảm từ 0.2
)
```

### Benchmark Issues

#### Strict benchmark fail

**Cause:** Vi phạm strict constraints hoặc goals không đạt

**Solution:**
- Kiểm tra error message
- Xem goal nào fail
- Kiểm tra structural scan
- Verify không có forbidden constructs

#### Moonshot benchmark fail

**Cause:** Goals không đạt hoặc data thiếu

**Solution:**
- Tải dữ liệu benchmark (TinyStories, WikiText-2)
- Kiểm tra profile (quick vs full)
- Xem specific goal fail

#### Phase benchmark fail

**Cause:** Component không hoạt động đúng

**Solution:**
- Kiểm tra component implementation
- Xem test output
- Debug với verbose mode

#### Benchmark quá chậm

**Cause:** Profile `full` hoặc data quá lớn

**Solution:**
```bash
python benchmark_moonshot.py --profile quick
```

### Model Issues

#### Model không học

**Cause:** Learning rate quá thấp hoặc consolidation quá thường xuyên

**Solution:**
- Tăng learning rate
- Giảm consolidation frequency
- Kiểm tra neuromodulator values

#### Model quên old patterns

**Cause:** Consolidation quá aggressive

**Solution:**
```python
model.consolidate(threshold=0.001, fraction=0.1)  # Giảm
```

#### Generation lặp lại

**Cause:** Repetition fatigue không hoạt động

**Solution:**
- Sử dụng GenerationController với repetition tracking
- Tăng temperature
- Sử dụng top-p sampling

#### Relation probability bằng 0

**Cause:** Không đủ training data cho relation

**Solution:**
- Tăng training data với relation pairs
- Tăng `relation_strength`
- Kiểm tra `assoc_window`

### Data Issues

#### Tokenizer không encode được text

**Cause:** Text chứa unknown tokens

**Solution:**
```python
tokenizer.train([your_text])  # Train tokenizer
# Hoặc sử dụng pretrained tokenizer
```

#### Semantic embeddings không load được

**Cause:** Vocabulary không match builtin embeddings

**Solution:**
- Sử dụng `use_pretrained=False`
- Hoặc train custom embeddings
- Kiểm tra vocabulary format

#### Benchmark data không tải được

**Cause:** Network issue hoặc URL thay đổi

**Solution:**
- Kiểm tra internet connection
- Tải manual từ source
- Sử dụng cached data nếu có

### Environment Issues

#### Python version không tương thích

**Cause:** Python version quá cũ

**Solution:**
```bash
python --version  # Cần >= 3.8
# Upgrade Python nếu cần
```

#### Virtual environment không hoạt động

**Cause:** VENV không activate đúng

**Solution:**
```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### Git clone fail

**Cause:** Permission hoặc network issue

**Solution:**
```bash
git clone --depth 1 <repo-url>  # Shallow clone
# Hoặc download ZIP
```

### Debugging Tips

#### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Print model metrics

```python
metrics = model.metrics()
print(metrics)
```

#### Trace error

```python
import traceback
try:
    model.observe(token_id, target_id)
except Exception as e:
    traceback.print_exc()
```

#### Profile code

```python
import cProfile
cProfile.run('your_code()', 'profile.stats')
```

### Getting Help

#### Kiểm tra documentation

- `docs/ARCHITECTURE.md` - Architecture details
- `docs/API.md` - API reference
- `docs/USAGE.md` - Usage examples
- `docs/PERFORMANCE.md` - Optimization tips
- `docs/TESTING.md` - Testing guide
- `docs/DEVELOPMENT.md` - Development guide

#### Mở issue trên GitHub

Khi mở issue, bao gồm:
- Python version
- NumPy version
- OS
- Error message
- Reproduction code
- Expected vs actual behavior

#### Kiểm tra existing issues

Tìm issue tương tự trước khi mở mới.

#### Discussion

Sử dụng GitHub Discussions cho:
- Feature requests
- Architecture questions
- General discussions

## Common Mistakes

### 1. Quên seed

**Bad:**
```python
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256)
```

**Good:**
```python
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
```

### 2. Không reset state giữa sequences

**Bad:**
```python
# Train sequence 1
for token in seq1:
    model.observe(token, next_token)
# Train sequence 2 (state vẫn từ seq1)
for token in seq2:
    model.observe(token, next_token)
```

**Good:**
```python
# Train sequence 1
for token in seq1:
    model.observe(token, next_token)
model.reset_state()
# Train sequence 2
for token in seq2:
    model.observe(token, next_token)
```

### 3. Sử dụng wrong token IDs

**Bad:**
```python
model.observe(1000, 1001)  # vocab_size=1000
```

**Good:**
```python
assert token_id < model.vocab_size
model.observe(token_id, target_id)
```

### 4. Không consolidate

**Bad:**
```python
for i in range(10000):
    model.observe(i, (i+1) % 1000)
# Không consolidate
```

**Good:**
```python
for i in range(10000):
    model.observe(i, (i+1) % 1000)
    if i % 1000 == 0:
        model.consolidate()
```

### 5. Bật quá nhiều features cùng lúc

**Bad:**
```python
model = LivingSynapseLM(
    vocab_size=1000000,
    hidden_dim=2048,
    use_predictive_coding=True,
    use_sdr=True,
    use_semantic_sdr=True,
    use_sparse_computation=True,
    use_long_context_memory=True,
    use_role_binding=True,
    use_hierarchical_routing=True
)
```

**Good:**
```python
# Bắt đầu với basic
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256)
# Thêm features dần dần
```

## Quick Reference

### Common Commands

```bash
# Verify installation
python benchmark_goal_strict.py

# Run all tests
python run_all.py

# Run specific benchmark
python benchmark_sdr_phase1.py

# Profile performance
python -m cProfile -o profile.stats your_script.py

# Check coverage
pytest --cov=lsl tests/
```

### Common Patterns

```python
# Basic training
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
for i in range(len(tokens) - 1):
    model.observe(tokens[i], tokens[i+1])

# With consolidation
for i in range(len(tokens) - 1):
    model.observe(tokens[i], tokens[i+1])
    if i % 1000 == 0:
        model.consolidate()
        model.replay(n=16)

# Generation
probs = model.predict(token_id)
next_token = int(probs.argmax())

# With sampling
import numpy as np
next_token = np.random.choice(len(probs), p=probs)
```

### Key Parameters

- `vocab_size`: Kích thước vocabulary
- `hidden_dim`: Hidden dimension (tăng = chậm hơn nhưng capacity cao hơn)
- `use_sparse_computation`: Bật sparse compute (40x+ speedup)
- `use_predictive_coding`: Bật predictive coding (60%+ energy savings)
- `use_sdr`: Bật SDR encoding (combinatorial capacity)
- `theta`: Suppression threshold (cao hơn = nhiều suppression hơn)
- `sdr_sparsity`: SDR sparsity (thấp hơn = nhanh hơn)
- `long_context_capacity`: Long context memory (cao hơn = nhiều memory hơn)

## Contact

Nếu vẫn gặp vấn đề sau khi đọc FAQ:
1. Kiểm tra documentation trong `docs/`
2. Tìm trong GitHub issues
3. Mở issue mới với đầy đủ thông tin
4. Sử dụng GitHub Discussions cho questions
