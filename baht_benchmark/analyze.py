"""Statistical analysis for BAHT benchmark results.

Loads Sacred/TensorBoard logs, computes means/stds, runs significance tests
with Holm-Bonferroni correction for multiple comparisons.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_sacred_results(results_dir: str) -> Dict[str, List[dict]]:
    """Load results from Sacred FileStorageObserver directories.

    Returns:
        Dict mapping run_name -> list of per-seed metric dicts
    """
    results = defaultdict(list)
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue

        info_path = run_dir / "info.json"
        config_path = run_dir / "config.json"

        if not info_path.exists():
            continue

        with open(info_path) as f:
            info = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        # Extract key metrics
        metrics = {}
        for m in ["test_return_mean", "test_return_std", "byz_detection_recall",
                   "byz_detection_f1", "byz_detection_accuracy", "contribution_loss"]:
            if m in info:
                v = info[m]
                if isinstance(v, dict):
                    # Sacred stores time-indexed dicts; get last value
                    vals = [vv for vv in v.values() if isinstance(vv, (int, float))]
                    if vals:
                        metrics[m] = vals[-1]
                elif isinstance(v, (int, float)):
                    metrics[m] = v
                elif isinstance(v, list) and v:
                    metrics[m] = v[-1] if isinstance(v[-1], (int, float)) else None

        # Identify the run
        env = config.get("env", "unknown")
        alg = config.get("name", config.get("label", "unknown"))
        byz_type = config.get("byzantine_type", "unknown")
        run_name = f"{env}_{alg}_{byz_type}"

        metrics["seed"] = config.get("seed", 0)
        metrics["config"] = config
        results[run_name].append(metrics)

    return dict(results)


def compute_statistics(results: Dict[str, List[dict]], metric: str = "test_return_mean") -> dict:
    """Compute mean, std, and confidence intervals for each method."""
    stats = {}
    for name, runs in results.items():
        values = [r[metric] for r in runs if metric in r and r[metric] is not None]
        if not values:
            continue
        stats[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "n": len(values),
            "ci95": 1.96 * np.std(values) / np.sqrt(len(values)) if len(values) > 1 else float("inf"),
            "values": values,
        }
    return stats


def pairwise_tests(stats: dict, method: str = "welch") -> List[dict]:
    """Run pairwise significance tests between all method pairs.

    Returns list of test results with Holm-Bonferroni corrected p-values.
    """
    if not SCIPY_AVAILABLE:
        print("scipy not installed — skipping significance tests")
        return []

    names = list(stats.keys())
    tests = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = stats[names[i]], stats[names[j]]
            if a["n"] < 2 or b["n"] < 2:
                continue

            if method == "welch":
                t_stat, p_val = scipy_stats.ttest_ind(a["values"], b["values"], equal_var=False)
            elif method == "mannwhitney":
                t_stat, p_val = scipy_stats.mannwhitneyu(a["values"], b["values"], alternative="two-sided")
            else:
                raise ValueError(f"Unknown test method: {method}")

            tests.append({
                "method_a": names[i],
                "method_b": names[j],
                "mean_a": a["mean"],
                "mean_b": b["mean"],
                "t_stat": t_stat,
                "pvalue_raw": p_val,
                "significant_raw": p_val < 0.05,
            })

    # Holm-Bonferroni correction
    if tests:
        tests.sort(key=lambda x: x["pvalue_raw"])
        n_tests = len(tests)
        for rank, test in enumerate(tests):
            corrected = test["pvalue_raw"] * (n_tests - rank)
            test["pvalue_corrected"] = min(corrected, 1.0)
            test["significant_corrected"] = test["pvalue_corrected"] < 0.05

    return tests


def print_results_table(stats: dict, metric: str = "test_return_mean"):
    """Print a formatted results table."""
    print(f"\n{'Method':<40s} {'Mean':>10s} {'Std':>10s} {'CI95':>10s} {'N':>5s}")
    print("-" * 80)
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["mean"]):
        print(f"{name:<40s} {s['mean']:>10.3f} {s['std']:>10.3f} {s['ci95']:>10.3f} {s['n']:>5d}")


def print_significance_tests(tests: List[dict]):
    """Print significance test results."""
    if not tests:
        return
    print(f"\n{'Comparison':<60s} {'p-raw':>8s} {'p-corr':>8s} {'Sig?':>5s}")
    print("-" * 85)
    for t in tests:
        sig = "*" if t["significant_corrected"] else ""
        comp = f"{t['method_a']} vs {t['method_b']}"
        print(f"{comp:<60s} {t['pvalue_raw']:>8.4f} {t['pvalue_corrected']:>8.4f} {sig:>5s}")


def analyze(results_dir: str, metric: str = "test_return_mean"):
    """Full analysis pipeline."""
    print(f"Loading results from: {results_dir}")
    results = load_sacred_results(results_dir)

    if not results:
        print("No results found.")
        return

    print(f"Found {sum(len(v) for v in results.values())} runs across {len(results)} configurations")

    stats = compute_statistics(results, metric)
    print_results_table(stats, metric)

    tests = pairwise_tests(stats)
    print_significance_tests(tests)

    return stats, tests


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to Sacred results directory")
    parser.add_argument("--metric", default="test_return_mean", help="Metric to analyze")
    args = parser.parse_args()
    analyze(args.results_dir, args.metric)
