"""Results storage system with full metadata tracking.

Stores all runs in results/ with comprehensive metadata including:
- Configuration parameters
- Random seed
- Timestamp
- Git commit hash
- System information
- Metrics
- Samples
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess


@dataclass
class RunMetadata:
    """Metadata for a single run."""
    benchmark_name: str
    timestamp: str
    seed: int
    git_hash: Optional[str] = None
    git_branch: Optional[str] = None
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunConfig:
    """Configuration parameters for a run."""
    dataset: str
    vocab_size: int
    max_tokens: int
    lsl_profile: str
    candidate_cap: int
    tokenizer: str = "subword"
    # Additional config fields can be added dynamically
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        base = asdict(self)
        base.update(base.pop('extra', {}))
        del base['extra']
        return base


@dataclass
class RunMetrics:
    """Performance metrics for a run."""
    tokens: float
    elapsed_seconds: float
    us_per_token: float
    tokens_per_second: float
    vocab_size: int
    seen_tokens: float = 0.0
    # Additional metrics can be added dynamically
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        base = asdict(self)
        base.update(base.pop('extra', {}))
        del base['extra']
        return base


@dataclass
class RunResult:
    """Complete result for a single run."""
    metadata: RunMetadata
    config: RunConfig
    metrics: RunMetrics
    success: bool
    error: Optional[str] = None
    sample_prompt: Optional[str] = None
    sample: Optional[str] = None
    checkpoint_path: Optional[str] = None
    corpus_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "metadata": self.metadata.to_dict(),
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "error": self.error,
            "sample_prompt": self.sample_prompt,
            "sample": self.sample,
            "checkpoint_path": self.checkpoint_path,
            "corpus_path": self.corpus_path,
        }


class ResultsStorage:
    """Storage system for benchmark and training results."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """Initialize results storage.
        
        Args:
            results_dir: Directory to store results. Defaults to 'results/' in project root.
        """
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        (self.results_dir / "samples").mkdir(exist_ok=True)
        (self.results_dir / "configs").mkdir(exist_ok=True)
    
    def get_git_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get git commit hash and branch.
        
        Returns:
            Tuple of (git_hash, git_branch) or (None, None) if not a git repo
        """
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            return git_hash, git_branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, None
    
    def create_metadata(self, benchmark_name: str, seed: int) -> RunMetadata:
        """Create metadata for a run.
        
        Args:
            benchmark_name: Name of the benchmark
            seed: Random seed used
            
        Returns:
            RunMetadata instance
        """
        import sys
        import platform as plat
        
        git_hash, git_branch = self.get_git_info()
        
        return RunMetadata(
            benchmark_name=benchmark_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            seed=seed,
            git_hash=git_hash,
            git_branch=git_branch,
            python_version=sys.version,
            platform=plat.platform(),
            hostname=plat.node(),
        )
    
    def generate_run_id(self, benchmark_name: str, seed: int, config: RunConfig) -> str:
        """Generate unique run ID based on benchmark, seed, and config.
        
        Args:
            benchmark_name: Name of the benchmark
            seed: Random seed
            config: Run configuration
            
        Returns:
            Unique run ID string
        """
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        hash_input = f"{benchmark_name}_{seed}_{config_str}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"{benchmark_name}_{seed}_{hash_hex}"
    
    def save_result(self, result: RunResult, run_id: Optional[str] = None) -> str:
        """Save a complete run result.
        
        Args:
            result: RunResult to save
            run_id: Optional run ID. If None, generates one automatically.
            
        Returns:
            Run ID used for saving
        """
        if run_id is None:
            run_id = self.generate_run_id(
                result.metadata.benchmark_name,
                result.metadata.seed,
                result.config
            )
        
        # Save full result as JSON
        result_path = self.results_dir / f"{run_id}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save metrics separately for easy aggregation
        metrics_path = self.results_dir / "metrics" / f"{run_id}_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result.metrics.to_dict(), f, indent=2)
        
        # Save config separately
        config_path = self.results_dir / "configs" / f"{run_id}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(result.config.to_dict(), f, indent=2)
        
        # Save sample if available
        if result.sample:
            sample_path = self.results_dir / "samples" / f"{run_id}_sample.txt"
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(result.sample)
        
        return run_id
    
    def load_result(self, run_id: str) -> Optional[RunResult]:
        """Load a run result by ID.
        
        Args:
            run_id: Run ID to load
            
        Returns:
            RunResult if found, None otherwise
        """
        result_path = self.results_dir / f"{run_id}.json"
        if not result_path.exists():
            return None
        
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return RunResult(
            metadata=RunMetadata(**data['metadata']),
            config=RunConfig(**data['config']),
            metrics=RunMetrics(**data['metrics']),
            success=data['success'],
            error=data.get('error'),
            sample_prompt=data.get('sample_prompt'),
            sample=data.get('sample'),
            checkpoint_path=data.get('checkpoint_path'),
            corpus_path=data.get('corpus_path'),
        )
    
    def list_results(self, benchmark_name: Optional[str] = None) -> List[str]:
        """List all result IDs, optionally filtered by benchmark.
        
        Args:
            benchmark_name: Optional benchmark name filter
            
        Returns:
            List of run IDs
        """
        all_results = []
        for path in self.results_dir.glob("*.json"):
            if path.name.startswith("_"):
                continue
            run_id = path.stem
            
            # Filter by benchmark name if specified
            if benchmark_name is not None:
                result = self.load_result(run_id)
                if result and result.metadata.benchmark_name != benchmark_name:
                    continue
            
            all_results.append(run_id)
        
        return sorted(all_results)
    
    def aggregate_metrics(self, benchmark_name: str, metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Aggregate metrics across all runs for a benchmark.
        
        Args:
            benchmark_name: Benchmark name to aggregate
            metric_keys: Specific metric keys to aggregate. If None, aggregates all numeric metrics.
            
        Returns:
            Dictionary with aggregated statistics (mean, std, min, max, count)
        """
        run_ids = self.list_results(benchmark_name)
        if not run_ids:
            return {}
        
        all_metrics = []
        for run_id in run_ids:
            result = self.load_result(run_id)
            if result and result.success:
                all_metrics.append(result.metrics.to_dict())
        
        if not all_metrics:
            return {}
        
        # Determine which keys to aggregate
        if metric_keys is None:
            # Find all numeric keys
            metric_keys = []
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)):
                    metric_keys.append(key)
        
        aggregated = {}
        for key in metric_keys:
            values = [m.get(key, 0) for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                import numpy as np
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        
        return aggregated
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs side by side.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "runs": [],
            "configs": {},
            "metrics": {},
        }
        
        for run_id in run_ids:
            result = self.load_result(run_id)
            if result:
                comparison["runs"].append(run_id)
                comparison["configs"][run_id] = result.config.to_dict()
                comparison["metrics"][run_id] = result.metrics.to_dict()
        
        return comparison
    
    def cleanup_old_results(self, benchmark_name: str, keep_latest: int = 10) -> int:
        """Remove old results, keeping only the latest N runs.
        
        Args:
            benchmark_name: Benchmark name to clean up
            keep_latest: Number of latest runs to keep
            
        Returns:
            Number of results removed
        """
        run_ids = self.list_results(benchmark_name)
        if len(run_ids) <= keep_latest:
            return 0
        
        # Sort by timestamp (from metadata)
        runs_with_time = []
        for run_id in run_ids:
            result = self.load_result(run_id)
            if result:
                runs_with_time.append((run_id, result.metadata.timestamp))
        
        # Sort by timestamp descending
        runs_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old runs
        to_remove = [run_id for run_id, _ in runs_with_time[keep_latest:]]
        removed = 0
        for run_id in to_remove:
            # Remove all associated files
            for pattern in ["*.json", "metrics/*", "configs/*", "samples/*"]:
                for path in self.results_dir.glob(pattern):
                    if path.stem.startswith(run_id):
                        path.unlink()
                        removed += 1
        
        return removed


def create_storage(results_dir: Optional[str] = None) -> ResultsStorage:
    """Factory function to create a ResultsStorage instance.
    
    Args:
        results_dir: Directory to store results
        
    Returns:
        ResultsStorage instance
    """
    return ResultsStorage(results_dir=results_dir)
