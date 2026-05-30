# Hướng dẫn Cài đặt và Thiết lập

## Yêu cầu hệ thống

### Phần cứng tối thiểu
- CPU: x86_64 hoặc ARM64
- RAM: 4GB (khuyến nghị 8GB+ cho các benchmark lớn)
- Disk: 1GB free space

### Phần mềm
- Python: 3.8 hoặc cao hơn
- Hệ điều hành: Windows, Linux, macOS
- Không yêu cầu GPU

## Cài đặt

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd brain
```

### Bước 2: Tạo virtual environment (khuyến nghị)

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Requirements hiện tại:
```
numpy>=1.24.0
```

### Bước 4: Xác nhận cài đặt

```bash
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
python -c "import lsl; print('LSL imported successfully')"
```

## Tải dữ liệu benchmark (tùy chọn)

### TinyStories (cho Phase 5)

```bash
python download_tinystories_full.py
```

Dữ liệu sẽ được tải về `benchmarks/data/tinystories/`

### WikiText-2 (cho Phase 5)

```bash
python download_wikitext2.py
```

Dữ liệu sẽ được tải về `benchmarks/data/wikitext-2-raw-v1/`

## Cấu trúc thư mục sau cài đặt

```
brain/
├── lsl/                    # Core package
│   ├── __init__.py
│   ├── model.py
│   ├── sdr.py
│   ├── semantic_sdr.py
│   ├── synapse.py
│   ├── memory.py
│   ├── generation.py
│   ├── reasoning.py
│   ├── hierarchy.py
│   ├── cortical_column.py
│   ├── associative_memory.py
│   ├── world_memory.py
│   └── data/
│       └── mini_semantic_embeddings.json
├── benchmarks/             # Benchmark implementations
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── strict/
│   └── data/
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── API.md
│   ├── DEVELOPMENT.md
│   ├── TESTING.md
│   └── PERFORMANCE.md
├── tests/                  # Unit tests
├── examples/               # Demo scripts
├── GOAL.md                 # Strict contract
├── README.md               # Quick start
├── requirements.txt        # Dependencies
└── run_all.py             # Umbrella runner
```

## Xác minh cài đặt

### Chạy unit tests cơ bản

```bash
python test_lsl.py
```

### Chạy strict benchmark

```bash
python benchmark_goal_strict.py
```

Kết quả mong đợi:
```
Goals passed: 18/18
Overall: PASS
Structural PASS clean
```

### Chạy tất cả tests

```bash
python run_all.py
```

## Cấu hình tùy chọn

### Environment variables

Không có environment variables bắt buộc. Mọi cấu hình được thực hiện qua parameters trong code.

### Seed control

Để reproductibility, sử dụng seed parameters:

```python
from lsl import LivingSynapseLM

model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    seed=42  # Fixed seed for reproducibility
)
```

## Xử lý sự cố

### ImportError: No module named 'lsl'

**Giải pháp:**
- Đảm bảo bạn đang ở thư mục gốc của project
- Kiểm tra Python path: `python -c "import sys; print(sys.path)"`
- Thêm thư mục hiện tại vào PYTHONPATH:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
  set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows
  ```

### NumPy version error

**Giải pháp:**
```bash
pip install --upgrade numpy
```

### Memory error khi chạy benchmark lớn

**Giải pháp:**
- Sử dụng profile `quick` thay vì `full`:
  ```bash
  python benchmark_moonshot.py --profile quick
  ```
- Giảm batch size trong benchmark script
- Tăng RAM hệ thống

### Permission error khi tải dữ liệu

**Giải pháp:**
- Đảm bảo quyền ghi vào thư mục `benchmarks/data/`
- Chạy với quyền admin nếu cần (Windows)
- Sử dụng `sudo` trên Linux (không khuyến nghị)

## Cài đặt cho development

### Cài đặt thêm dev tools (tùy chọn)

```bash
pip install pytest pytest-cov black flake8 mypy
```

### Cấu hình git hooks (tùy chọn)

```bash
# Pre-commit hook example
cat .git/hooks/pre-commit << 'EOF'
#!/bin/bash
python -m pytest tests/
python -m flake8 lsl/
EOF
chmod +x .git/hooks/pre-commit
```

## Cập nhật

### Cập nhật code

```bash
git pull origin main
```

### Cập nhật dependencies

```bash
pip install --upgrade -r requirements.txt
```

## Gỡ cài đặt

### Xóa virtual environment

```bash
deactivate  # Nếu đang activated
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Xóa dữ liệu benchmark

```bash
rm -rf benchmarks/data/
```

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra `docs/FAQ.md` cho câu hỏi thường gặp
2. Kiểm tra `docs/TROUBLESHOOTING.md` cho hướng dẫn khắc phục
3. Xem log output từ benchmark scripts
4. Kiểm tra Python và NumPy version
