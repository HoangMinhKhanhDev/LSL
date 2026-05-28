# Living Synapse LLM: Detailed Plan and Success Criteria

Kế hoạch này xác định các bước nâng cấp PoC Living Synapse Language Model để chứng minh rõ cơ chế “trọng số sống”, đo định lượng, và mở đường tiến gần hơn tới kiến trúc LLM/Transformer mà không biến nó thành LoRA/adapter.

## Nguyên tắc bất biến

- **Không phải LoRA/adapter**: Không coi `W_live` là adapter học sẵn; `W_live` phải thay đổi real-time trong inference bằng luật cục bộ.
- **Không backprop trong inference**: Không dùng optimizer, gradient graph, hoặc fine-tune toàn model khi model đang quan sát dữ liệu mới.
- **Ba tiêu chí bắt buộc**:
  - Synapse thật thay đổi: có tensor trọng số nội tại bị cập nhật in-place trong inference.
  - Trọng số hiệu dụng động: `W_effective(t)` thay đổi do `W_live`, fatigue/excitation/inhibition/state.
  - Circuit/routing thay đổi: đường truyền/top-k neuron hoặc attention-path thay đổi real-time theo input và state.
- **Góc nhìn sinh học trước**: Ưu tiên plasticity, hệ động lực, neuromodulation, memory consolidation, predictive coding.

## Trạng thái hiện tại

- **Core đã có**: `lsl/model.py`, `lsl/synapse.py`, `lsl/router.py`, `lsl/neuromod.py`, `lsl/memory.py`.
- **Demo đã có**:
  - `demo_association.py`: học liên kết tức thì, reset `W_live` làm mất trí nhớ ngắn hạn.
  - `demo_stability.py`: chống drift tương đối khi gặp random transitions.
  - `demo_consolidation.py`: replay + consolidation chuyển một phần plastic memory sang slow weights.
  - `demo_mini_tokens.py`: mini-token vocab 16.
- **Tài liệu đã có**: `README.md`, `theory.md`.
- **Test đã có**: `test_lsl.py` chạy pass.

## Mục tiêu phase tiếp theo

1. **Chứng minh cơ chế sống rõ ràng hơn**
   - Log được `W_live` thay đổi theo từng observation.
   - Log được `W_effective` khác nhau trước/sau fatigue, reset, consolidation.
   - Log được router chọn đường khác nhau theo input/state.

2. **Định lượng kết quả**
   - Đo `P(target|input)` trước/sau observe, sau noise, sau reset, sau consolidation.
   - Đo `live_norm`, `slow_norm`, số synapse consolidated, số neuron routed.
   - Đo chi phí update theo thời gian và số phép cập nhật gần đúng.
   - Đo drift/retention: giữ được bao nhiêu phần association sau nhiễu và sau reset.

3. **Tiến gần LLM/Transformer hơn**
   - Thiết kế phiên bản attention-like nhỏ: query/key/value toy với living synapses hoặc dynamic routing.
   - Giữ PoC nhỏ, NumPy/CPU, dễ quan sát.
   - Không chuyển sang LoRA; nếu có delta weight thì delta phải được cập nhật online theo activity.

## Công việc chi tiết

### 1. Instrumentation: làm cơ chế nhìn thấy được

- **Thêm metrics API**: tạo phương thức trả về snapshot gồm `live_norm`, `slow_norm`, fatigue mean/max, router usage, top active neurons.
- **Thêm trace mode**: khi chạy demo có thể in từng bước: prediction error, modulator, novelty, reward, consolidated count.
- **Thêm W_eff diff**: đo `||W_effective_after - W_effective_before||` để chứng minh trọng số hiệu dụng đổi thật.
- **Thêm router trace**: lưu hoặc in top-k mask theo từng token để chứng minh circuit/routing đổi real-time.

### 2. Benchmark định lượng tối thiểu

- **Association benchmark**:
  - Task: nhiều cặp `A->B`, `C->D`, `E->A`.
  - Metric: delta probability, top-1 accuracy, số observation cần để vượt ngưỡng.
- **Stability benchmark**:
  - Task: học cặp chính rồi bơm random/noisy transitions.
  - Metric: retention ratio = post_noise_score / post_learning_score.
- **Consolidation benchmark**:
  - Task: học, replay, consolidate, reset live.
  - Metric: slow retention = score_after_reset_consolidated - score_after_reset_control.
- **Compute benchmark**:
  - Metric: thời gian trung bình mỗi `observe()`, số synapse cập nhật, memory footprint của `W_live`.

### 3. Tiêu chí thành công định lượng

- **Instant learning**:
  - `P(target|input)` tăng ít nhất 20% tương đối sau online observations trong cùng phiên.
  - `live_norm > 0` sau learning và `live_norm == 0` sau `reset_live()`.
- **Living weight proof**:
  - `W_live` thay đổi in-place ở ít nhất output layer và một hidden layer.
  - `||W_effective_after - W_effective_before|| > 0` khi fatigue/plasticity thay đổi.
  - Router mask khác nhau cho ít nhất 2 input/state khác nhau.
- **Self-stability**:
  - Sau noise, association chính giữ ít nhất 70% mức tăng đã học.
  - `live_norm` không vượt giới hạn clipping trong stress test.
- **Consolidation**:
  - Sau replay + consolidation + `reset_live()`, score cao hơn control không consolidation.
  - Có ít nhất một số synapse được transfer từ `W_live` sang `W_slow`.
- **Compute**:
  - Demo chạy CPU dưới vài giây.
  - Không sử dụng gradient/backprop/optimizer.

### 4. Attention-like prototype gần Transformer

- **Toy attention layer**:
  - Dùng Q/K/V nhỏ trên mini-token sequence.
  - Routing top-k attention heads hoặc key-path thay đổi theo state.
- **Living attention synapses**:
  - Cho phép một phần `W_q`, `W_k`, `W_v` hoặc attention bias sống thay đổi online.
  - Update bằng local rule dựa trên query activation, key activation, prediction surprise.
- **Success criteria**:
  - Attention path đổi sau exposure.
  - Model tăng khả năng recall một token-pair/sequence mới.
  - Reset live làm mất phần thích nghi ngắn hạn.

### 5. Tài liệu nghiên cứu

- **Cập nhật theory.md**:
  - Thêm định nghĩa formal `W_effective(t) = gate(t) * bio_state(t) * (W_slow + W_live(t))`.
  - Thêm bảng mapping sinh học ↔ thành phần code.
  - Thêm phần phân biệt LoRA, RAG, external memory, fast weights.
- **Cập nhật README.md**:
  - Thêm lệnh chạy benchmark.
  - Thêm bảng kết quả mẫu.
  - Thêm tiêu chí thành công.

## Rủi ro kỹ thuật

- **Learning quá yếu**: tăng reward/modulator/lr hoặc dùng task dễ hơn, nhưng vẫn giữ luật cục bộ.
- **Learning quá mạnh gây drift**: tăng decay, clipping, fatigue, inhibition hoặc chaos guard.
- **Consolidation không rõ**: giảm threshold, tăng replay, hoặc đo trực tiếp transfer norm thay vì chỉ xác suất output.
- **Toy quá xa LLM**: thêm attention-like prototype nhưng vẫn giữ quy mô nhỏ.

## Deliverables mong muốn

- **Benchmark scripts**: `benchmark_association.py`, `benchmark_stability.py`, `benchmark_consolidation.py`, `benchmark_compute.py`.
- **Trace script**: `trace_living_weights.py` để chứng minh 3 tiêu chí living weights.
- **Attention prototype**: `lsl/attention.py` và `demo_attention_living.py` nếu phase này mở rộng sang transformer-like.
- **Updated docs**: `README.md`, `theory.md` có tiêu chí và kết quả mẫu.
- **Tests**: test bảo đảm `W_live` đổi, reset hoạt động, consolidation đổi `W_slow`, router mask thay đổi.

## Định nghĩa hoàn thành phase

Phase được xem là hoàn thành khi có thể chạy một bộ lệnh duy nhất để tạo báo cáo ngắn cho thấy:

- **Trọng số sống thật**: `W_live`, `W_effective`, routing đều thay đổi real-time.
- **Học được trong phiên**: xác suất/accuracy tăng sau observe.
- **Không phải LoRA**: không có training adapter/offline fine-tune; không dùng gradient/optimizer.
- **Tự ổn định**: sau noise không sụp hoàn toàn, norm không bùng nổ.
- **Có trí nhớ đa tầng**: consolidation giúp giữ một phần thông tin sau `reset_live()`.
