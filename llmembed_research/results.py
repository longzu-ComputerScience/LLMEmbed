"""Recompute R52 test metrics from the archived predictions using Python alone."""
import argparse
from collections import Counter
import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev


METHODS = ("exp4_ctrl_mlp", "exp4", "imbalance_flat_logit_adjustment")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1", "rare_macro_f1")
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def scores(rows, class_ids, rare_ids):
    actual = Counter(int(row["label"]) for row in rows)
    predicted = Counter(int(row["prediction"]) for row in rows)
    correct = Counter(int(row["label"]) for row in rows if row["label"] == row["prediction"])
    if set(actual) != set(class_ids) or not set(predicted) <= set(class_ids):
        raise ValueError("Unexpected test label space.")
    f1 = {label: 2 * correct[label] / (actual[label] + predicted[label]) for label in class_ids}
    return dict(zip(METRICS, (
        sum(correct.values()) / len(rows),
        mean(f1.values()),
        mean(f1[label] for label in rare_ids),
    ), strict=True))


def recompute(root=ROOT):
    audit = root / "research" / "audit"
    confirmation = audit / "kaggle_downstream_v9" / "study"
    data = read_json(confirmation / "data_audit.json")
    classes = range(len(data["classes"]))
    rare = [index for index, row in enumerate(data["classes"]) if 1 <= row["train"] <= 9]
    removed = set(data["exact_train_test_duplicate_ids"])
    if len(classes) != 52 or len(rare) != 18 or len(removed) != 6:
        raise ValueError("Unexpected dataset metadata.")
    archived = {}
    for row in read_csv(audit / "r52_confirmation" / "study_metrics.csv"):
        key = row["experiment"], int(row["seed"]), row["subset"]
        if key in archived:
            raise ValueError(f"Duplicate archived result for {key}.")
        archived[key] = row

    expected = {(name, seed, split) for name in METHODS for seed in SEEDS for split in ("source", "filtered")}
    evidence = {}
    label_order = None
    for file in sorted((confirmation / "results").glob("*/result.json")):
        run = read_json(file)
        method, seed = run["experiment"], run["seed"]
        rows = read_csv(file.with_name("predictions.csv"))
        order = [(row["id"], row["label"]) for row in rows]
        ids = {sample_id for sample_id, _ in order}
        if len(rows) != data["split_sizes"]["test"] or len(ids) != len(rows) or not removed <= ids:
            raise ValueError(f"Unexpected example IDs for {method}, seed {seed}.")
        if label_order is not None and order != label_order:
            raise ValueError("Test order or labels differ between runs.")
        label_order = order
        if run["evaluation_split"] != "test" or run["data_fingerprint"] != data["fingerprint"]:
            raise ValueError("Experiment split or fingerprint mismatch.")
        filtered = [row for row in rows if row["id"] not in removed]
        for subset, examples in (("source", rows), ("filtered", filtered)):
            key = method, seed, subset
            if key not in expected or key in evidence:
                raise ValueError(f"Unexpected or duplicate result for {key}.")
            observed = scores(examples, classes, rare)
            if any(not isclose(observed[metric], float(archived[key][metric]), abs_tol=1e-12)
                   for metric in METRICS):
                raise ValueError(f"Metrics disagree for {key}.")
            evidence[key] = observed
    if set(evidence) != expected or set(archived) != expected:
        raise ValueError("The archive must contain all nine runs on both test subsets.")
    return evidence


def aggregate(evidence, subset):
    rows = []
    primary = "exp4"
    for method in METHODS:
        row = {"experiment": method, "subset": subset, "n": len(SEEDS)}
        for metric in METRICS:
            values = [evidence[method, seed, subset][metric] for seed in SEEDS]
            deltas = [value - evidence[primary, seed, subset][metric] for value, seed in zip(values, SEEDS, strict=True)]
            row.update({
                f"{metric}_mean": mean(values), f"{metric}_sample_sd": stdev(values),
                f"{metric}_delta_vs_agcmf": mean(deltas),
                f"{metric}_paired_sample_sd": stdev(deltas),
            })
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write mean, sample SD, and paired deltas to a CSV.")
    args = parser.parse_args()
    evidence = recompute()
    summary = [row for subset in ("source", "filtered") for row in aggregate(evidence, subset)]
    for row in summary:
        values = " | ".join(f"{100 * row[f'{key}_mean']:.2f} +/- {100 * row[f'{key}_sample_sd']:.2f}"
                            for key in METRICS)
        print(f"{row['subset']} | {row['experiment']} | {values}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


if __name__ == "__main__":
    main()
