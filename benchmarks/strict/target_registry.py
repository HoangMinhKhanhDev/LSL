"""Machine-readable target registry for the LSL strict gate."""
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class TargetSpec:
    id: str
    phase: str
    name: str
    tier: str
    metric: str
    threshold: str
    measurement: str


@dataclass
class TargetResult:
    id: str
    phase: str
    name: str
    tier: str
    metric: str
    threshold: str
    measurement: str
    value: object
    status: str
    detail: str = ""

    @property
    def success(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


STRICT_TARGETS: List[TargetSpec] = [
    TargetSpec("G1.1", "phase1", "semantic overlap", "strict", "related/random overlap ratio", ">=30x", "proxy"),
    TargetSpec("G1.2", "phase1", "combinatorial capacity", "strict", "log2 C(100000,40)", ">=500", "structural"),
    TargetSpec("G1.3", "phase1", "interference-free storage", "strict", "100000 pattern recall", ">=99%", "proxy"),
    TargetSpec("G1.4", "phase1", "one-shot recognition", "strict", "recognition with 40% missing/noisy active bits", ">=99%", "proxy"),
    TargetSpec("G1.5", "phase1", "pattern completion", "strict", "completion from 20% active bits", ">=95%", "proxy"),
    TargetSpec("G1.6", "phase1", "native sparse compute", "strict", "native sparse CPU wall/ops/cache speedup", ">=500x", "real"),
    TargetSpec("G2.1", "phase2", "local error convergence", "strict", "per-layer local error drop", ">=99%", "proxy"),
    TargetSpec("G2.2", "phase2", "adaptive suppression", "strict", "learned-signal suppression", ">=95%", "proxy"),
    TargetSpec("G2.3", "phase2", "zero locality violations", "strict", "forbidden structural/runtime violations", "0", "structural"),
    TargetSpec("G2.4", "phase2", "online loss convergence", "strict", "online transition loss within 10 epochs", "<=2.0", "proxy"),
    TargetSpec("G2.5", "phase2", "static-context energy savings", "strict", "proxy savings plus optional watt evidence", ">=98%", "proxy"),
    TargetSpec("G2.6", "phase2", "multi-step cause-effect proof", "strict", "true causal probability / false link probability", ">=0.90 / <=0.10", "proxy"),
    TargetSpec("G3.1", "phase3", "deep ambiguous context", "strict", "branching context prediction accuracy", ">=95%", "proxy"),
    TargetSpec("G3.2", "phase3", "active-state suppression", "strict", "predicted/silent token processing", ">=98%", "proxy"),
    TargetSpec("G3.3", "phase3", "complex grammar emergence", "strict", "complex grammar sequence accuracy", ">=95%", "proxy"),
    TargetSpec("G3.4", "phase3", "long topic coherence", "strict", "topic coherence over 20000 generated tokens", ">=0.90", "proxy"),
    TargetSpec("G3.5", "phase3", "real-time latency stability", "strict", "per-token latency max/min ratio", "<=1.20", "real"),
    TargetSpec("G3.6", "phase3", "50-domain retention", "strict", "old-domain retention after 50 domains", ">=99%", "proxy"),
    TargetSpec("G4.1", "phase4_5", "1M semantic scale", "strict", "collision / recovery at 1M vocab", "<=0.1% / >=90%", "proxy"),
    TargetSpec("G4.2", "phase4_5", "128k sparse context", "strict", "128k retrieval accuracy with no full scan", ">=75%", "proxy"),
    TargetSpec("G4.3", "phase4_5", "real-corpus CPU baselines", "strict", "TinyStories/WikiText-2 subword speedup", ">=20x", "real"),
    TargetSpec("G6.1", "phase6_8", "open generation quality", "strict", "loop rate / UNK rate", "<=3% / <=0.3%", "proxy"),
    TargetSpec("G7.1", "phase6_8", "reasoning workspace", "strict", "multi-step trace accuracy", ">=80%", "proxy"),
    TargetSpec("G8.1", "phase6_8", "entity-event graph scale", "strict", "100k and 1M event accuracy, no scan, latency ratio", ">=target / 0 scan / <=2.0", "proxy"),
    TargetSpec("Structural", "strict", "strict scanner", "strict", "forbidden implementation patterns", "0", "structural"),
]


def registry_by_id(targets: Iterable[TargetSpec] = STRICT_TARGETS) -> Dict[str, TargetSpec]:
    return {target.id: target for target in targets}


def result_for(spec: TargetSpec, ok: bool, value: object, detail: str = "", status: Optional[str] = None) -> TargetResult:
    return TargetResult(
        id=spec.id,
        phase=spec.phase,
        name=spec.name,
        tier=spec.tier,
        metric=spec.metric,
        threshold=spec.threshold,
        measurement=spec.measurement,
        value=value,
        status=status or ("PASS" if ok else "FAIL"),
        detail=detail,
    )

