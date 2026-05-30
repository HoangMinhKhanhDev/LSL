"""Hierarchical Abstraction Mechanism for Phase 4.

Implements multi-level abstraction and routing to enable:
- Hierarchical concept representation
- Compositional binding across levels
- Top-down and bottom-up information flow
- Abstraction hierarchy (low-level → high-level concepts)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class HierarchicalRouter:
    """Routes information between abstraction levels."""
    
    def __init__(self, num_levels: int, dim_per_level: int, seed: int = 42):
        self.num_levels = num_levels
        self.dim_per_level = dim_per_level
        self.total_dim = num_levels * dim_per_level
        
        rng = np.random.default_rng(seed)
        
        # Routing weights (upward: low→high, downward: high→low)
        self.W_up = rng.standard_normal((num_levels - 1, dim_per_level, dim_per_level)) * 0.02
        self.W_down = rng.standard_normal((num_levels - 1, dim_per_level, dim_per_level)) * 0.02
        
        # Level activation (which levels are active)
        self.level_active = np.ones(num_levels, dtype=bool)
        
        # Level importance (higher = more abstract)
        self.level_importance = np.linspace(0.5, 1.0, num_levels)
    
    def route_upward(self, level: int, x: np.ndarray) -> np.ndarray:
        """Route information from level to higher level."""
        if level >= self.num_levels - 1:
            return np.zeros(self.dim_per_level)
        
        # Linear projection to next level
        return x @ self.W_up[level]
    
    def route_downward(self, level: int, x: np.ndarray) -> np.ndarray:
        """Route information from level to lower level."""
        if level <= 0:
            return np.zeros(self.dim_per_level)
        
        # Linear projection to previous level
        return x @ self.W_down[level - 1]
    
    def aggregate_upward(self, states: List[np.ndarray]) -> np.ndarray:
        """Aggregate information from all lower levels to top level."""
        if len(states) == 0:
            return np.zeros(self.dim_per_level)
        
        # Weighted sum by level importance
        aggregated = np.zeros(self.dim_per_level)
        for i, state in enumerate(states):
            if i < len(self.level_importance):
                aggregated += self.level_importance[i] * state
        
        return aggregated / len(states)
    
    def distribute_downward(self, top_state: np.ndarray) -> List[np.ndarray]:
        """Distribute top-level information to all lower levels."""
        states = []
        for i in range(self.num_levels):
            # Top state influences all levels, scaled by importance
            influence = self.level_importance[i] * top_state
            states.append(influence)
        return states


class HierarchicalMemory:
    """Multi-level memory system with abstraction hierarchy."""
    
    def __init__(self, num_levels: int = 3, dim_per_level: int = 256, 
                 capacity_per_level: int = 1000, seed: int = 42):
        self.num_levels = num_levels
        self.dim_per_level = dim_per_level
        self.capacity_per_level = capacity_per_level
        
        self.router = HierarchicalRouter(num_levels, dim_per_level, seed)
        
        # Memory at each level
        self.memories = [
            np.zeros((capacity_per_level, dim_per_level))
            for _ in range(num_levels)
        ]
        
        # Usage tracking
        self.usage = [0 for _ in range(num_levels)]
        
        # Level-specific patterns
        self.level_patterns = [[] for _ in range(num_levels)]
    
    def store(self, x: np.ndarray, level: int = 0) -> int:
        """Store pattern at specified level."""
        if level < 0 or level >= self.num_levels:
            level = 0
        
        # Store at current level
        idx = self.usage[level] % self.capacity_per_level
        self.memories[level][idx] = x
        self.usage[level] += 1
        self.level_patterns[level].append(idx)
        
        # Propagate upward to higher levels (abstraction)
        if level < self.num_levels - 1:
            abstracted = self.router.route_upward(level, x)
            self.store(abstracted, level + 1)
        
        return idx
    
    def retrieve(self, query: np.ndarray, level: int = 0, k: int = 5) -> List[int]:
        """Retrieve similar patterns from specified level and above."""
        if level < 0 or level >= self.num_levels:
            level = 0
        
        # Retrieve from current level
        similarities = []
        for i in range(min(self.usage[level], self.capacity_per_level)):
            sim = np.dot(query, self.memories[level][i])
            similarities.append((sim, i, level))
        
        # Also retrieve from higher levels (more abstract)
        for l in range(level + 1, self.num_levels):
            # Project query to this level
            projected = self.router.route_upward(level, query)
            for i in range(min(self.usage[l], self.capacity_per_level)):
                sim = np.dot(projected, self.memories[l][i])
                similarities.append((sim, i, l))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [(idx, lvl) for _, idx, lvl in similarities[:k]]
    
    def hierarchical_query(self, query: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
        """Query across all levels with hierarchical routing."""
        results = {}
        
        # Bottom-up: start from lowest level
        current = query
        for level in range(self.num_levels):
            # Retrieve at this level
            retrieved = self.retrieve(current, level, k=3)
            results[level] = retrieved
            
            # Move to next level
            if level < self.num_levels - 1:
                current = self.router.route_upward(level, current)
        
        # Top-down: refine with top-level context
        top_level = self.num_levels - 1
        if self.usage[top_level] > 0:
            top_context = self.memories[top_level][0]  # Use first as context
            distributed = self.router.distribute_downward(top_context)
            
            # Refine with top-down context
            for level in range(self.num_levels):
                refined_query = query + distributed[level]
                refined = self.retrieve(refined_query, level, k=2)
                results[f"{level}_refined"] = refined
        
        return results


class LearnedHierarchicalMemory:
    """Count-based token -> phrase -> topic hierarchy with sparse routing."""

    def __init__(self, route_cap: int = 3):
        self.route_cap = max(1, int(route_cap))
        self.phrase_counts: Dict[Tuple[int, ...], Dict[int, float]] = {}
        self.topic_counts: Dict[int, Dict[int, float]] = {}
        self.token_topics: Dict[int, Dict[int, float]] = {}

    def observe(self, tokens: List[int], topic: int) -> None:
        topic = int(topic)
        items = [int(t) for t in tokens]
        for token in items:
            self.token_topics.setdefault(token, {})
            self.token_topics[token][topic] = self.token_topics[token].get(topic, 0.0) + 1.0
        for width in (2, 3):
            for i in range(0, max(0, len(items) - width + 1)):
                phrase = tuple(items[i:i + width])
                self.phrase_counts.setdefault(phrase, {})
                self.phrase_counts[phrase][topic] = self.phrase_counts[phrase].get(topic, 0.0) + float(width)
        self.topic_counts.setdefault(topic, {})
        for token in items:
            self.topic_counts[topic][token] = self.topic_counts[topic].get(token, 0.0) + 1.0

    def route(self, tokens: List[int]) -> List[int]:
        scores: Dict[int, float] = {}
        items = [int(t) for t in tokens]
        for width in (3, 2):
            for i in range(0, max(0, len(items) - width + 1)):
                phrase = tuple(items[i:i + width])
                for topic, count in self.phrase_counts.get(phrase, {}).items():
                    scores[topic] = scores.get(topic, 0.0) + float(width) * float(count)
        for token in items:
            for topic, count in self.token_topics.get(token, {}).items():
                scores[topic] = scores.get(topic, 0.0) + 0.25 * float(count)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [int(topic) for topic, _ in ranked[: self.route_cap]]

    def predict_topic(self, tokens: List[int]) -> Optional[int]:
        routes = self.route(tokens)
        return routes[0] if routes else None


def test_hierarchical_abstraction() -> Dict[str, float]:
    """Test hierarchical abstraction mechanism."""
    print("Testing Hierarchical Abstraction")
    print("=" * 80)
    
    # Create hierarchical memory
    hmem = HierarchicalMemory(
        num_levels=3,
        dim_per_level=64,
        capacity_per_level=100,
        seed=42,
    )
    
    # Store hierarchical patterns
    rng = np.random.default_rng(42)
    
    # Level 0: concrete patterns
    for i in range(20):
        pattern = rng.standard_normal(64)
        hmem.store(pattern, level=0)
    
    # Level 1: mid-level patterns
    for i in range(15):
        pattern = rng.standard_normal(64)
        hmem.store(pattern, level=1)
    
    # Level 2: abstract patterns
    for i in range(10):
        pattern = rng.standard_normal(64)
        hmem.store(pattern, level=2)
    
    print(f"Stored patterns: Level 0={hmem.usage[0]}, Level 1={hmem.usage[1]}, Level 2={hmem.usage[2]}")
    
    # Test retrieval
    query = rng.standard_normal(64)
    results = hmem.hierarchical_query(query)
    
    print(f"\nHierarchical query results:")
    for level, retrieved in results.items():
        if isinstance(level, int):
            print(f"  Level {level}: {len(retrieved)} matches")
        else:
            print(f"  {level}: {len(retrieved)} matches")
    
    # Test compositional binding
    print(f"\nTesting compositional binding...")
    pattern_a = rng.standard_normal(64)
    pattern_b = rng.standard_normal(64)
    
    idx_a = hmem.store(pattern_a, level=0)
    idx_b = hmem.store(pattern_b, level=0)
    
    # Query with combination
    combined_query = 0.5 * pattern_a + 0.5 * pattern_b
    combined_results = hmem.retrieve(combined_query, level=0, k=5)
    
    print(f"  Stored A at index {idx_a}, B at index {idx_b}")
    print(f"  Combined query retrieved {len(combined_results)} matches")
    
    # Check if original patterns are in results
    found_a = any(idx == idx_a for idx, _ in combined_results)
    found_b = any(idx == idx_b for idx, _ in combined_results)
    
    print(f"  Found original A: {found_a}, Found original B: {found_b}")
    
    return {
        "total_stored": sum(hmem.usage),
        "retrieval_levels": len(results),
        "compositional_found": float(found_a or found_b),
    }


if __name__ == "__main__":
    results = test_hierarchical_abstraction()
    print("\n" + "=" * 80)
    print("Hierarchical Abstraction Test Complete")
    print("=" * 80)
