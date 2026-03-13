"""Held-out evaluation protocol for the BAHT benchmark.

Strong eval design requires two kinds of generalization:
1. TEAMMATE GENERALIZATION: Train with some teammate policies, test with
   held-out policies the agent has never seen. This validates zero-shot
   coordination — the core AHT claim.

2. BYZANTINE GENERALIZATION: Train against some Byzantine types, test against
   held-out types. This validates that detection is based on outcome impact
   (CVC's claim), not memorized behavioral patterns.

Usage:
    from baht_benchmark.eval_protocol import (
        split_population, default_byzantine_splits, EvalProtocol
    )

    # Split teammates into train/test
    train_manifest, test_manifest = split_population(
        "populations/mpe-pp/manifest.json",
        train_fraction=0.7,
        strategy="by_algorithm",
    )

    # Get held-out Byzantine splits
    byz_splits = default_byzantine_splits()
    # byz_splits.train_types = ["random", "freeze"]
    # byz_splits.test_types = ["flip", "offset"]
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ByzantineSplit:
    """Defines which Byzantine types are used for training vs testing."""
    train_types: List[str]
    test_types: List[str]
    name: str = ""

    def summary(self) -> str:
        return (f"Byzantine split '{self.name}': "
                f"train={self.train_types}, test={self.test_types}")


def default_byzantine_splits() -> List[ByzantineSplit]:
    """Standard held-out Byzantine type evaluation.

    Split strategy: train on the easiest types, test on harder types.
    This tests whether the method generalizes to unseen adversarial behaviors.
    """
    return [
        ByzantineSplit(
            name="easy_to_hard",
            train_types=["random", "freeze"],
            test_types=["flip", "offset"],
        ),
        ByzantineSplit(
            name="random_only",
            train_types=["random"],
            test_types=["freeze", "flip", "offset"],
        ),
        ByzantineSplit(
            name="all_train",
            train_types=["random", "freeze", "flip", "offset"],
            test_types=["random", "freeze", "flip", "offset"],
        ),
    ]


def split_population(
    manifest_path: str,
    train_fraction: float = 0.7,
    strategy: str = "by_algorithm",
    seed: int = 42,
) -> Tuple[dict, dict]:
    """Split a teammate population manifest into train/test sets.

    Args:
        manifest_path: Path to manifest.json from pretrain step.
        train_fraction: Fraction of policies for training.
        strategy: How to split:
            - "by_algorithm": Hold out entire algorithms (strongest test).
              E.g., train with IQL/IPPO/MAPPO, test with QMIX/POAM.
            - "by_seed": Hold out seeds within each algorithm.
              E.g., train with seed 42, test with seed 43.
            - "random": Random split across all policies.

    Returns:
        (train_manifest, test_manifest): Two manifest dicts with disjoint policies.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    policies = manifest["policies"]
    rng = random.Random(seed)

    if strategy == "by_algorithm":
        train_policies, test_policies = _split_by_algorithm(
            policies, train_fraction, rng)
    elif strategy == "by_seed":
        train_policies, test_policies = _split_by_seed(
            policies, train_fraction, rng)
    elif strategy == "random":
        train_policies, test_policies = _split_random(
            policies, train_fraction, rng)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Build train/test manifests
    base = {k: v for k, v in manifest.items() if k != "policies"}

    train_manifest = {**base, "policies": train_policies,
                      "n_policies": len(train_policies), "split": "train"}
    test_manifest = {**base, "policies": test_policies,
                     "n_policies": len(test_policies), "split": "test"}

    return train_manifest, test_manifest


def _split_by_algorithm(policies, train_fraction, rng):
    """Hold out entire algorithms — strongest generalization test."""
    algos = sorted(set(p["algorithm"] for p in policies))
    rng.shuffle(algos)

    n_train = max(1, int(len(algos) * train_fraction))
    train_algos = set(algos[:n_train])
    test_algos = set(algos[n_train:])

    # Ensure at least one test algorithm
    if not test_algos:
        test_algos = {algos[-1]}
        train_algos.discard(algos[-1])

    train = [p for p in policies if p["algorithm"] in train_algos]
    test = [p for p in policies if p["algorithm"] in test_algos]
    return train, test


def _split_by_seed(policies, train_fraction, rng):
    """Hold out seeds within each algorithm."""
    seeds = sorted(set(p["seed"] for p in policies))
    rng.shuffle(seeds)

    n_train = max(1, int(len(seeds) * train_fraction))
    train_seeds = set(seeds[:n_train])

    train = [p for p in policies if p["seed"] in train_seeds]
    test = [p for p in policies if p["seed"] not in train_seeds]
    return train, test


def _split_random(policies, train_fraction, rng):
    """Random split."""
    shuffled = list(policies)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    return shuffled[:n_train], shuffled[n_train:]


def save_split_manifests(
    manifest_path: str,
    output_dir: str,
    train_fraction: float = 0.7,
    strategy: str = "by_algorithm",
    seed: int = 42,
) -> Tuple[str, str]:
    """Split population and save train/test manifests to disk.

    Returns paths to (train_manifest.json, test_manifest.json).
    """
    train_manifest, test_manifest = split_population(
        manifest_path, train_fraction, strategy, seed)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_manifest.json")
    test_path = os.path.join(output_dir, "test_manifest.json")

    with open(train_path, "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_manifest, f, indent=2)

    print(f"Train manifest: {train_path} ({train_manifest['n_policies']} policies)")
    print(f"  Algorithms: {sorted(set(p['algorithm'] for p in train_manifest['policies']))}")
    print(f"Test manifest: {test_path} ({test_manifest['n_policies']} policies)")
    print(f"  Algorithms: {sorted(set(p['algorithm'] for p in test_manifest['policies']))}")

    return train_path, test_path


@dataclass
class EvalProtocol:
    """Complete evaluation protocol for a BAHT experiment.

    Combines teammate generalization (train/test split) with
    Byzantine generalization (train/test types) for rigorous evaluation.
    """
    # Teammate population split
    train_population: dict = field(default_factory=dict)
    test_population: dict = field(default_factory=dict)
    teammate_split_strategy: str = "by_algorithm"

    # Byzantine type splits
    byzantine_splits: List[ByzantineSplit] = field(default_factory=default_byzantine_splits)

    # Evaluation parameters
    eval_seeds: int = 3
    eval_episodes: int = 100  # per condition

    @staticmethod
    def from_manifest(
        manifest_path: str,
        train_fraction: float = 0.7,
        strategy: str = "by_algorithm",
    ) -> "EvalProtocol":
        """Create an eval protocol from a population manifest."""
        train, test = split_population(manifest_path, train_fraction, strategy)
        return EvalProtocol(
            train_population=train,
            test_population=test,
            teammate_split_strategy=strategy,
        )

    def training_conditions(self) -> List[dict]:
        """Generate all training conditions (train teammates × train byz types)."""
        conditions = []
        for byz_split in self.byzantine_splits:
            for byz_type in byz_split.train_types:
                conditions.append({
                    "population": self.train_population,
                    "byzantine_type": byz_type,
                    "split_name": byz_split.name,
                    "phase": "train",
                })
        return conditions

    def evaluation_conditions(self) -> List[dict]:
        """Generate all evaluation conditions.

        Returns conditions for:
        1. In-distribution: train teammates + train byz types (sanity check)
        2. Teammate generalization: test teammates + train byz types
        3. Byzantine generalization: train teammates + test byz types
        4. Full generalization: test teammates + test byz types (hardest)
        """
        conditions = []
        for byz_split in self.byzantine_splits:
            for population_name, population in [
                ("train_teammates", self.train_population),
                ("test_teammates", self.test_population),
            ]:
                for byz_phase, byz_types in [
                    ("train_byz", byz_split.train_types),
                    ("test_byz", byz_split.test_types),
                ]:
                    for byz_type in byz_types:
                        conditions.append({
                            "population": population,
                            "byzantine_type": byz_type,
                            "split_name": byz_split.name,
                            "teammate_source": population_name,
                            "byz_phase": byz_phase,
                            "phase": "eval",
                        })
        return conditions

    def summary(self) -> dict:
        n_train = self.train_population.get("n_policies", 0)
        n_test = self.test_population.get("n_policies", 0)
        return {
            "teammate_split": self.teammate_split_strategy,
            "train_policies": n_train,
            "test_policies": n_test,
            "byzantine_splits": [s.name for s in self.byzantine_splits],
            "n_training_conditions": len(self.training_conditions()),
            "n_eval_conditions": len(self.evaluation_conditions()),
        }


def main():
    """CLI for creating eval splits."""
    import argparse
    parser = argparse.ArgumentParser(description="Create held-out evaluation splits")
    parser.add_argument("manifest", help="Path to population manifest.json")
    parser.add_argument("--output_dir", default="eval_splits",
                        help="Output directory for split manifests")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--strategy", choices=["by_algorithm", "by_seed", "random"],
                        default="by_algorithm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path, test_path = save_split_manifests(
        args.manifest, args.output_dir,
        args.train_fraction, args.strategy, args.seed,
    )

    # Show Byzantine splits
    print("\nByzantine type splits:")
    for split in default_byzantine_splits():
        print(f"  {split.summary()}")


if __name__ == "__main__":
    main()
