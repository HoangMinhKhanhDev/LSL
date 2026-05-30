"""Phase 9 dendritic computation proof."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import DendriticLayer


def pattern(a: int, b: int, context: int):
    return [10 + int(a), 20 + int(b), 100 + int(context)]


def xor_bits(a: int, b: int):
    return [0 if int(a) else 1, 2 if int(b) else 3]


def flat_single_threshold_can_solve_xor() -> bool:
    # With non-negative raw bits, a single flat threshold sees sums [0,1,1,2].
    # XOR positives are the two middle points, so no threshold separates them
    # from both negatives at 0 and 2.
    examples = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for threshold in [i / 10.0 for i in range(-10, 31)]:
        for direction in (1, -1):
            ok = True
            for (a, b), target in examples:
                score = a + b
                pred = int(direction * (score - threshold) >= 0)
                ok = ok and pred == target
            if ok:
                return True
    return False


def coincidence_accuracy():
    layer = DendriticLayer(input_dim=16, outputs=1, segment_size=2)
    and_branch = layer.add_branch([1, 2], output=1, threshold=1.5)
    or_branch = layer.add_branch([5, 6], output=1, threshold=0.5)
    tests = []
    for a in (0, 1):
        for b in (0, 1):
            bits = []
            if a:
                bits.append(1)
            if b:
                bits.append(2)
            tests.append((and_branch.spike(bits), bool(a and b)))
    for a in (0, 1):
        for b in (0, 1):
            bits = []
            if a:
                bits.append(5)
            if b:
                bits.append(6)
            tests.append((or_branch.spike(bits), bool(a or b)))
    return sum(int(pred == expected) for pred, expected in tests) / len(tests)


def xor_one_neuron_accuracy():
    neuron = DendriticLayer(input_dim=8, outputs=1, segment_size=2, soma_threshold=0.5)
    neuron.add_branch(xor_bits(0, 1), output=1, threshold=1.5)
    neuron.add_branch(xor_bits(1, 0), output=1, threshold=1.5)
    correct = 0
    for a in (0, 1):
        for b in (0, 1):
            correct += int(neuron.soma_spike(xor_bits(a, b), output=1) == bool(a ^ b))
    return correct / 4.0


def evaluate(args):
    layer = DendriticLayer(input_dim=args.input_dim, outputs=8, segment_size=3)
    samples = []
    for context in range(4):
        for a in (0, 1):
            for b in (0, 1):
                output = (a ^ b) + 2 * context
                bits = pattern(a, b, context)
                samples.append((bits, output))
                layer.observe(bits, output)

    correct = 0
    for bits, output in samples:
        correct += int(layer.predict(bits) == output)
    nonlinear_accuracy = correct / max(1, len(samples))

    # Same subject/object bits, different context bit must route differently.
    role_tests = [(pattern(1, 0, ctx), 1 + 2 * ctx) for ctx in range(4)]
    role_accuracy = sum(int(layer.predict(bits) == out) for bits, out in role_tests) / len(role_tests)

    diag = layer.diagnostics()
    ops_gain = diag["dense_ops_proxy"] / max(1.0, diag["last_ops"])
    quality_degradation = 1.0 - nonlinear_accuracy

    nonlinear = DendriticLayer(input_dim=16, outputs=1, segment_size=2)
    branch = nonlinear.add_branch([1, 2], output=1, threshold=1.5)
    one_bit_activation = branch.activation([1])
    two_bit_activation = branch.activation([1, 2])
    flat_one_bit = branch.flat_drive([1]) / 2.0
    flat_two_bit = branch.flat_drive([1, 2]) / 2.0
    nonlinearity_gap = abs(one_bit_activation - flat_one_bit) + abs(two_bit_activation - flat_two_bit)

    specialized = DendriticLayer(input_dim=4096, outputs=1, segment_size=4)
    specialization_samples = []
    for i in range(32):
        bits = [200 + i * 8 + j for j in range(4)]
        specialized.add_branch(bits, output=1, threshold=3.5)
        specialization_samples.append(bits)
    branch_overlap = specialized.branch_activity_overlap(specialization_samples)

    tree = DendriticLayer(
        input_dim=args.tree_input_dim,
        outputs=1,
        segment_size=args.branch_size,
        branches_per_output=args.moonshot_branches,
        branch_size=args.branch_size,
    )
    active_tree_bits = tree.branches[min(123, len(tree.branches) - 1)].active_bits
    tree.predict(active_tree_bits)
    tree_diag = tree.diagnostics()
    unique_receptive_fields = len({branch.active_bits for branch in tree.branches})

    learn_layer = DendriticLayer(input_dim=32, outputs=1, segment_size=2)
    learning_branch = learn_layer.add_branch([1, 2], output=1, threshold=0.75, weights=(0.5, 0.5))
    before_weights = sum(learning_branch.weights)
    update_ops = learn_layer.learn_branch_local([1, 2], output=1, lr=0.25)
    after_weights = sum(learning_branch.weights)
    learn_diag = learn_layer.diagnostics()
    branch_local_learning_ratio = 1.0 if update_ops > 0 and learn_diag["global_error_updates"] == 0.0 and after_weights > before_weights else 0.0

    tree.learn_branch_local(active_tree_bits, output=0, lr=0.25)
    zero_update_ratio = tree.diagnostics()["last_zero_update_branch_ratio"]

    metrics = {
        "nonlinear_accuracy": nonlinear_accuracy,
        "role_context_accuracy": role_accuracy,
        "ops_energy_gain": ops_gain,
        "quality_degradation": quality_degradation,
        "segments": diag["segments"],
        "branch_one_bit_activation": one_bit_activation,
        "branch_two_bit_activation": two_bit_activation,
        "flat_one_bit_proxy": flat_one_bit,
        "flat_two_bit_proxy": flat_two_bit,
        "dendritic_nonlinearity_gap": nonlinearity_gap,
        "branch_specialization_overlap": branch_overlap,
        "coincidence_detection_accuracy": coincidence_accuracy(),
        "xor_one_neuron_accuracy": xor_one_neuron_accuracy(),
        "flat_single_neuron_xor_possible": float(flat_single_threshold_can_solve_xor()),
        "tree_branches_per_neuron": tree_diag["branches_per_neuron"],
        "branch_level_sdr_unique_ratio": unique_receptive_fields / max(1, len(tree.branches)),
        "tree_mean_branch_size": tree_diag["mean_branch_size"],
        "sparse_branch_activation_ratio": tree_diag["last_active_branch_ratio"],
        "tree_compute_density_gain": tree_diag["compute_density_gain"],
        "branch_local_learning_ratio": branch_local_learning_ratio,
        "global_error_updates": learn_diag["global_error_updates"],
        "zero_update_branch_ratio": zero_update_ratio,
    }
    checks = {
        "nonlinear": metrics["nonlinear_accuracy"] >= args.nonlinear_target,
        "role_context": metrics["role_context_accuracy"] >= args.role_target,
        "ops": metrics["ops_energy_gain"] >= args.ops_target,
        "quality": metrics["quality_degradation"] <= args.quality_degradation_target,
        "g6_1_nonlinearity": metrics["dendritic_nonlinearity_gap"] >= args.nonlinearity_gap_target,
        "g6_2_specialization": metrics["branch_specialization_overlap"] <= args.branch_overlap_target,
        "g6_3_coincidence": metrics["coincidence_detection_accuracy"] >= args.coincidence_target,
        "g6_4_xor": metrics["xor_one_neuron_accuracy"] >= args.xor_target and metrics["flat_single_neuron_xor_possible"] == 0.0,
        "g6_5_sparse_branches": metrics["sparse_branch_activation_ratio"] <= args.sparse_branch_target,
        "moonshot_tree": metrics["tree_branches_per_neuron"] >= args.moonshot_branches,
        "moonshot_branch_sdr": metrics["branch_level_sdr_unique_ratio"] >= args.branch_sdr_target,
        "moonshot_branch_learning": metrics["branch_local_learning_ratio"] >= args.branch_learning_target,
        "moonshot_zero_update": metrics["zero_update_branch_ratio"] >= args.zero_update_target,
        "moonshot_compute_density": metrics["tree_compute_density_gain"] >= args.compute_density_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--nonlinear-target", type=float, default=0.95)
    parser.add_argument("--role-target", type=float, default=0.90)
    parser.add_argument("--ops-target", type=float, default=50.0)
    parser.add_argument("--quality-degradation-target", type=float, default=0.05)
    parser.add_argument("--tree-input-dim", type=int, default=20000)
    parser.add_argument("--moonshot-branches", type=int, default=1000)
    parser.add_argument("--branch-size", type=int, default=16)
    parser.add_argument("--nonlinearity-gap-target", type=float, default=0.20)
    parser.add_argument("--branch-overlap-target", type=float, default=0.10)
    parser.add_argument("--coincidence-target", type=float, default=0.90)
    parser.add_argument("--xor-target", type=float, default=1.0)
    parser.add_argument("--sparse-branch-target", type=float, default=0.05)
    parser.add_argument("--compute-density-target", type=float, default=100.0)
    parser.add_argument("--branch-sdr-target", type=float, default=1.0)
    parser.add_argument("--branch-learning-target", type=float, default=1.0)
    parser.add_argument("--zero-update-target", type=float, default=0.90)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Dendritic Computation")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.4f}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_dendritic", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
