"""Phase 7 heldout compositional and OOD-symbol generalization."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import EntityEventGraph, ReasoningWorkspace, RoleBindingMemory


def evaluate(args):
    graph = EntityEventGraph()
    workspace = ReasoningWorkspace()
    color_rel, shape_rel, owner_rel = 1, 2, 3
    colors = list(range(100, 110))
    shapes = list(range(200, 210))
    entities = list(range(1000, 1000 + args.items))
    heldout_correct = 0
    heldout_total = 0
    for idx, entity in enumerate(entities):
        color = colors[idx % len(colors)]
        shape = shapes[(idx * 3) % len(shapes)]
        if idx % 5 != 0:
            graph.observe_event(entity, color_rel, color, episode_id=idx, evidence_id=idx)
            graph.observe_event(entity, shape_rel, shape, episode_id=idx, evidence_id=idx)
        else:
            graph.observe_event(entity, color_rel, color, episode_id=idx, evidence_id=idx)
            graph.observe_event(entity, shape_rel, shape, episode_id=idx, evidence_id=idx)
            pred_color = graph.query(entity, color_rel)
            pred_shape = graph.query(entity, shape_rel)
            heldout_correct += int(pred_color == color and pred_shape == shape)
            heldout_total += 1
        workspace.bind_pair(entity, owner_rel, entity + 50000)

    role = RoleBindingMemory()
    swap_correct = 0
    for i in range(args.items):
        subject = 10000 + i
        obj = 20000 + i
        receiver = 30000 + i
        role.observe_event(subject, 77, obj)
        role.observe_event(receiver, 78, obj)
        swap_correct += int(role.predict_object(subject, 77) == obj and role.predict_subject(78, obj) == receiver)

    ood_correct = 0
    for i in range(args.items):
        symbol = 900000 + i * 17
        graph.observe_event(symbol, 91, symbol + 1, episode_id=700000 + i, evidence_id=i)
        graph.observe_event(symbol + 1, 91, symbol + 2, episode_id=700000 + i, evidence_id=i)
        ood_correct += int(graph.query_chain(symbol, [91, 91]) == symbol + 2)

    metrics = {
        "heldout_accuracy": heldout_correct / max(1, heldout_total),
        "swap_accuracy": swap_correct / max(1, args.items),
        "ood_accuracy": ood_correct / max(1, args.items),
        "candidate_count": graph.diagnostics()["last_candidate_count"],
        "full_scan": graph.diagnostics()["last_full_scan"],
        "workspace_bindings": workspace.diagnostics()["bindings"],
    }
    checks = {
        "heldout": metrics["heldout_accuracy"] >= args.heldout_target,
        "swap": metrics["swap_accuracy"] >= args.swap_target,
        "ood": metrics["ood_accuracy"] >= 1.0 - args.ood_drop_target,
        "no_scan": metrics["full_scan"] == 0.0,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--heldout-target", type=float, default=0.80)
    parser.add_argument("--swap-target", type=float, default=0.90)
    parser.add_argument("--ood-drop-target", type=float, default=0.15)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    m = result["metrics"]
    ok = bool(result["success"])
    print("Phase 7: Heldout Generalization")
    print("=" * 88)
    print(f"Heldout compositional: {m['heldout_accuracy']:.2%}")
    print(f"Role swap:             {m['swap_accuracy']:.2%}")
    print(f"OOD symbols:           {m['ood_accuracy']:.2%}")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase7_generalization_heldout", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
