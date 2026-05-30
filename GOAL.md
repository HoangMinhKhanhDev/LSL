# LSL Extreme Strict Goal Contract

`python benchmark_goal_strict.py` is the claim-bearing strict gate. It must
return non-zero on any failed target and can emit machine-readable evidence via
`--json-output`.

The legacy 18-goal suite is retained only as a smoke profile:

```powershell
python benchmark_goal_strict.py --profile smoke
```

## Strict Core Constraints

- No backpropagation, `.backward()`, autograd, optimizer state, Adam, SGD,
  RMSprop, or momentum in the strict model path.
- No GPU requirement and no PyTorch, TensorFlow, or JAX dependency in strict
  runtime.
- No attention matrix, Q/K/V attention, self-attention, cross-attention, or
  all-pairs token interaction.
- No global hidden error signal; prediction errors and weight updates must be
  local.
- All learning updates must be online, per token, and local to active synapses.
- No external API, hardcoded eval answer, full-history scan, or batch retrain
  inside the strict path.

## Public Benchmark Interface

```powershell
python benchmark_goal_strict.py
python benchmark_goal_strict.py --json-output results/lsl_extreme_strict.json
python benchmarks/energy/measure_native_sparse_energy.py --output results/energy_evidence.json
python benchmark_goal_strict.py --require-real-energy --energy-evidence results/energy_evidence.json
```

Strict mode requires the native sparse C extension `lsl._sparse_native`; build
it with:

```powershell
python setup.py build_ext --inplace
```

Energy evidence may be JSON or CSV with these fields: `method`, `device`,
`timestamp`, `tokens`, `dense_watts`, `sparse_watts`,
`dense_joule_per_token`, and `sparse_joule_per_token`. Real watt evidence
passes only when sparse watts are `<=20` and sparse joule/token saves at least
`98%` versus dense. Without `--require-real-energy`, strict success claims
proxy energy savings only. The measurement helper reads a real Windows power
sensor through Intel Power Gadget `EnergyLib64.dll` when available, or the
Windows `Power Meter` performance counter as a watts-integrating fallback. It
does not emit synthetic evidence.

## Phase 1: Sparse Distributed Representations

- G1.1 semantic overlap ratio `>=30x`.
- G1.2 exact capacity `log2(C(100000,40)) >= 500`.
- G1.3 store `100000` sparse patterns with recall `>=99%`.
- G1.4 one-shot recognition `>=99%` with up to `40%` noisy or missing active
  bits.
- G1.5 sparse Hopfield-style completion restores `>=95%` from only `20%`
  active bits.
- G1.6 native CPU sparse primitive achieves `>=500x` wall-clock speedup and
  matching ops/cache proxy versus dense baseline.

## Phase 2: Predictive Coding

- G2.1 local error drops `>=99%` in all tracked layers.
- G2.2 adaptive suppression `>=95%`.
- G2.3 source/runtime scan reports `0` locality violations.
- G2.4 online loss `<=2.0` within `10` epochs.
- G2.5 static-context compute savings `>=98%` by proxy, plus real-energy
  evidence when supplied.
- G2.6 multi-step cause-effect probability `>=0.90`, with false links `<=0.10`.

## Phase 3: Cortical Column Sequence Memory

- G3.1 deep ambiguous-context prediction accuracy `>=95%`.
- G3.2 active-state suppression `>=98%`.
- G3.3 complex grammar emergence `>=95%`.
- G3.4 topic coherence `>=0.90` over `20000` generated tokens.
- G3.5 per-token latency max/min ratio `<=1.20` across long contexts.
- G3.6 continual learning across `50` domains with old-domain retention
  `>=99%`.

## Phase 4-8 Scale And Agent Targets

- 1M semantic SDR scale: collision `<=0.1%`, recovery `>=90%`.
- 128k context retrieval: sparse lookup accuracy `>=75%`, no attention matrix,
  and no full-history scan.
- TinyStories and WikiText-2: subword tokenizer, real corpus run, CPU NumPy
  Transformer/SSM baselines, LSL inference speed `>=20x`.
- Open generation: loop rate `<=3%`, UNK rate `<=0.3%`.
- Reasoning workspace: multi-step logic trace accuracy `>=80%`.
- Entity-event graph: works at `100k` and `1M` events with no full scan and
  latency ratio `<=2.0`.
- BioComputeAgent dialogue generation: Phase 9 reports multi-domain dialogue
  training scale, tokens/sec, ms/token, loop rate, UNK rate, domain coherence,
  and no-full-scan diagnostics.

## Target Registry

Every strict target is registered in `benchmarks/strict/target_registry.py` with
`id`, `phase`, `tier`, `metric`, `threshold`, `measurement`, and runtime
`status`. A target cannot support a claim unless it has an executable benchmark
result and `status=PASS`.
