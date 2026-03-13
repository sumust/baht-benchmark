"""Teammate and Byzantine diversity specification for the BAHT benchmark.

This is the core of what makes BAHT a proper benchmark. A benchmark without
diversity is just self-play with noise — it tells you nothing about robustness.

Diversity has two orthogonal axes:
  1. COOPERATIVE diversity: who are the non-Byzantine teammates?
  2. BYZANTINE diversity: what do the adversarial agents do?

Both must be varied systematically for the benchmark to be meaningful.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# COOPERATIVE TEAMMATE DIVERSITY
# ═══════════════════════════════════════════════════════════════════════

class TeammateAlgorithm(Enum):
    """Training algorithms that produce behaviorally distinct policies."""
    # Value-based (fundamentally different from policy gradient)
    IQL = "iql"              # Independent Q-Learning
    QMIX = "qmix"           # Value decomposition via monotonic mixing
    VDN = "vdn"             # Additive value decomposition

    # Policy gradient (different optimization landscapes)
    IPPO = "ippo"            # Independent PPO
    MAPPO = "mappo"          # Centralized critic PPO
    MADDPG = "maddpg"       # Continuous → discrete, different exploration
    PAC = "pac"             # Policy-augmented actor-critic

    # Teammate-aware (produces more cooperative behavior)
    POAM = "poam"           # With teammate modeling
    SHAPLEY = "shapley"     # With contribution estimation


class TeammateSkillLevel(Enum):
    """Skill levels from checkpoints at different training stages."""
    NOVICE = "novice"           # 25% of training
    INTERMEDIATE = "intermediate"  # 50% of training
    COMPETENT = "competent"     # 75% of training
    EXPERT = "expert"           # 100% of training


class ArchitectureVariant(Enum):
    """Network architecture variants that produce different behaviors."""
    RNN_SHARED = "rnn"           # Parameter sharing across agents
    RNN_NONSHARED = "rnn_ns"     # Independent parameters per agent
    RNN_NORM = "rnn_norm"        # With layer normalization
    RNN_NORM_NS = "rnn_norm_ns"  # Non-shared + layer norm


@dataclass
class TeammateSpec:
    """Specification for a single teammate policy in the population."""
    algorithm: TeammateAlgorithm
    skill_level: TeammateSkillLevel
    architecture: ArchitectureVariant = ArchitectureVariant.RNN_SHARED
    seed: int = 42
    # Hyperparameter variation
    entropy_coef: float = 0.01
    lr: float = 0.0005
    # Metadata
    checkpoint_fraction: float = 1.0  # What fraction of t_max to train


@dataclass
class TeammatePopulationSpec:
    """Full specification for a diverse teammate population.

    The goal: produce teammates with BEHAVIORALLY DISTINCT policies,
    not just different random seeds of the same algorithm.

    Diversity comes from:
    1. Algorithm: Value-based vs policy-gradient vs teammate-aware
    2. Skill level: Novice to expert (checkpoint at 25/50/75/100%)
    3. Architecture: Shared vs non-shared parameters
    4. Hyperparameters: Different entropy, LR → different exploration
    5. Seeds: Multiple seeds per configuration
    """
    specs: List[TeammateSpec] = field(default_factory=list)

    # Sampling strategy during training
    sampling: str = "algorithm_balanced"  # uniform, skill_balanced, algorithm_balanced, curriculum

    # Curriculum learning: start with similar policies, gradually diversify
    use_curriculum: bool = True
    curriculum_warmup_steps: int = 100_000

    @staticmethod
    def standard(seeds_per_config: int = 2) -> "TeammatePopulationSpec":
        """Standard population for the BAHT benchmark.

        Produces: 5 algorithms × 4 skill levels × 2 seeds = 40 policies.
        This is the minimum for a proper benchmark.
        """
        specs = []
        algorithms = [
            # Value-based family
            TeammateAlgorithm.IQL,
            TeammateAlgorithm.QMIX,
            # Policy gradient family
            TeammateAlgorithm.IPPO,
            TeammateAlgorithm.MAPPO,
            # Teammate-aware
            TeammateAlgorithm.POAM,
        ]
        skill_levels = [
            (TeammateSkillLevel.NOVICE, 0.25),
            (TeammateSkillLevel.INTERMEDIATE, 0.50),
            (TeammateSkillLevel.COMPETENT, 0.75),
            (TeammateSkillLevel.EXPERT, 1.0),
        ]

        for algo in algorithms:
            for skill, frac in skill_levels:
                for seed in range(seeds_per_config):
                    specs.append(TeammateSpec(
                        algorithm=algo,
                        skill_level=skill,
                        seed=42 + seed,
                        checkpoint_fraction=frac,
                    ))

        return TeammatePopulationSpec(
            specs=specs,
            sampling="algorithm_balanced",
            use_curriculum=True,
        )

    @staticmethod
    def minimal(seeds_per_config: int = 1) -> "TeammatePopulationSpec":
        """Minimal population for quick experiments.

        3 algorithms × 2 skill levels × 1 seed = 6 policies.
        """
        specs = []
        for algo in [TeammateAlgorithm.IQL, TeammateAlgorithm.IPPO, TeammateAlgorithm.MAPPO]:
            for skill, frac in [(TeammateSkillLevel.NOVICE, 0.25), (TeammateSkillLevel.EXPERT, 1.0)]:
                specs.append(TeammateSpec(
                    algorithm=algo,
                    skill_level=skill,
                    seed=42,
                    checkpoint_fraction=frac,
                ))

        return TeammatePopulationSpec(
            specs=specs,
            sampling="uniform",
            use_curriculum=False,
        )

    @staticmethod
    def extended(seeds_per_config: int = 3) -> "TeammatePopulationSpec":
        """Extended population with architecture and hyperparameter variation.

        7 algorithms × 4 skill levels × 2 architectures × 3 seeds = 168 policies.
        For thorough evaluation only.
        """
        specs = []
        algorithms = [
            TeammateAlgorithm.IQL,
            TeammateAlgorithm.QMIX,
            TeammateAlgorithm.VDN,
            TeammateAlgorithm.IPPO,
            TeammateAlgorithm.MAPPO,
            TeammateAlgorithm.PAC,
            TeammateAlgorithm.POAM,
        ]
        architectures = [ArchitectureVariant.RNN_SHARED, ArchitectureVariant.RNN_NONSHARED]
        skill_levels = [
            (TeammateSkillLevel.NOVICE, 0.25),
            (TeammateSkillLevel.INTERMEDIATE, 0.50),
            (TeammateSkillLevel.COMPETENT, 0.75),
            (TeammateSkillLevel.EXPERT, 1.0),
        ]

        for algo in algorithms:
            for arch in architectures:
                for skill, frac in skill_levels:
                    for seed in range(seeds_per_config):
                        specs.append(TeammateSpec(
                            algorithm=algo,
                            skill_level=skill,
                            architecture=arch,
                            seed=42 + seed,
                            checkpoint_fraction=frac,
                        ))

        return TeammatePopulationSpec(
            specs=specs,
            sampling="algorithm_balanced",
            use_curriculum=True,
        )

    @property
    def n_policies(self):
        return len(self.specs)

    def summary(self):
        algos = set(s.algorithm.value for s in self.specs)
        skills = set(s.skill_level.value for s in self.specs)
        archs = set(s.architecture.value for s in self.specs)
        seeds = set(s.seed for s in self.specs)
        return {
            "n_policies": self.n_policies,
            "algorithms": sorted(algos),
            "skill_levels": sorted(skills),
            "architectures": sorted(archs),
            "seeds": sorted(seeds),
        }


# ═══════════════════════════════════════════════════════════════════════
# BYZANTINE DIVERSITY
# ═══════════════════════════════════════════════════════════════════════

class ByzantineType(Enum):
    """Byzantine behavior types, ordered by detection difficulty.

    Implemented types (in ByzantineRunner):
        random, freeze, flip, offset

    Not yet implemented (require trained adversary policies):
        adversary, mixed, stealth, stealth_independent
    """
    # Tier 1: Trivially detectable (implemented)
    RANDOM = "random"       # Uniform random actions
    FROZEN = "freeze"       # Always action 0

    # Tier 2: Requires behavioral analysis (implemented)
    FLIP = "flip"           # Inverted cooperative policy
    OFFSET = "offset"       # Action = (cooperative_action + k) % n_actions

    # Tier 3: Requires learned detection (NOT YET IMPLEMENTED)
    ADVERSARY = "adversary"     # Reward-minimizing (trained policy)
    MIXED = "mixed"             # 50% adversary, 50% random

    # Tier 4: Detection-aware (NOT YET IMPLEMENTED)
    STEALTH = "stealth"                   # Adversarial + evasive (trained)
    STEALTH_INDEPENDENT = "stealth_independent"  # Heuristic stealth (env-specific)


@dataclass
class ByzantineSpec:
    """Specification for Byzantine behavior in experiments."""
    # Which types to include
    types: List[ByzantineType] = field(default_factory=lambda: [
        ByzantineType.RANDOM,
        ByzantineType.FROZEN,
        ByzantineType.FLIP,
    ])

    # Budget: how many Byzantines per episode
    budget_min: int = 0
    budget_max: int = 1

    # Per-episode type mixing: each Byzantine can be a different type
    type_mixing: bool = True

    # Type sampling weights (None = uniform)
    type_weights: Optional[List[float]] = None

    @staticmethod
    def easy() -> "ByzantineSpec":
        """Easy: only trivially detectable types."""
        return ByzantineSpec(
            types=[ByzantineType.RANDOM, ByzantineType.FROZEN],
            budget_min=1, budget_max=1,
            type_mixing=False,
        )

    @staticmethod
    def medium() -> "ByzantineSpec":
        """Medium: includes behavioral analysis types."""
        return ByzantineSpec(
            types=[ByzantineType.RANDOM, ByzantineType.FROZEN,
                   ByzantineType.FLIP, ByzantineType.OFFSET],
            budget_min=1, budget_max=1,
            type_mixing=True,
        )

    @staticmethod
    def hard() -> "ByzantineSpec":
        """Hard: all implemented types with higher budget."""
        return ByzantineSpec(
            types=[ByzantineType.RANDOM, ByzantineType.FROZEN,
                   ByzantineType.FLIP, ByzantineType.OFFSET],
            budget_min=1, budget_max=2,
            type_mixing=True,
        )

    @staticmethod
    def adversarial() -> "ByzantineSpec":
        """Adversarial: detection-aware Byzantines.

        NOTE: Requires trained adversary policies. Currently falls back
        to the hard preset since adversary/stealth types are not yet
        implemented in ByzantineRunner.
        """
        # TODO: enable once adversary types are implemented
        return ByzantineSpec.hard()

    @staticmethod
    def full() -> "ByzantineSpec":
        """Full spectrum of implemented types, variable budget."""
        return ByzantineSpec(
            types=[ByzantineType.RANDOM, ByzantineType.FROZEN,
                   ByzantineType.FLIP, ByzantineType.OFFSET],
            budget_min=0, budget_max=2,
            type_mixing=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkProtocol:
    """Complete benchmark protocol specifying diversity on both axes.

    A proper BAHT benchmark must specify:
    1. What cooperative teammates look like (behavioral diversity)
    2. What Byzantines look like (adversarial diversity)
    3. How to evaluate (train/test split, cross-eval)
    """
    name: str
    teammate_population: TeammatePopulationSpec
    byzantine_spec: ByzantineSpec

    # Domain randomization (noise on top of diversity)
    obs_noise: bool = True
    action_noise: bool = True

    # Evaluation protocol
    train_test_split: float = 0.7  # Fraction of algorithms for training
    cross_evaluate: bool = True    # NxN eval matrix across conditions

    # Methods to benchmark
    methods: List[str] = field(default_factory=lambda: ["shapley", "poam", "ippo"])

    # Replication
    seeds: int = 5

    @staticmethod
    def quick() -> "BenchmarkProtocol":
        """Quick protocol for debugging (~1 hour)."""
        return BenchmarkProtocol(
            name="quick",
            teammate_population=TeammatePopulationSpec.minimal(),
            byzantine_spec=ByzantineSpec.easy(),
            obs_noise=False,
            action_noise=False,
            cross_evaluate=False,
            methods=["shapley", "poam"],
            seeds=2,
        )

    @staticmethod
    def standard() -> "BenchmarkProtocol":
        """Standard protocol for papers (~12 hours)."""
        return BenchmarkProtocol(
            name="standard",
            teammate_population=TeammatePopulationSpec.standard(),
            byzantine_spec=ByzantineSpec.medium(),
            obs_noise=True,
            action_noise=True,
            cross_evaluate=True,
            methods=["shapley", "poam", "ippo"],
            seeds=5,
        )

    @staticmethod
    def full() -> "BenchmarkProtocol":
        """Full protocol for thorough evaluation (~48 hours)."""
        return BenchmarkProtocol(
            name="full",
            teammate_population=TeammatePopulationSpec.extended(),
            byzantine_spec=ByzantineSpec.full(),
            obs_noise=True,
            action_noise=True,
            cross_evaluate=True,
            methods=["shapley", "poam", "ippo"],
            seeds=5,
        )

    def summary(self):
        pop = self.teammate_population.summary()
        n_byz = len(self.byzantine_spec.types)
        n_methods = len(self.methods)
        return {
            "protocol": self.name,
            "teammate_policies": pop["n_policies"],
            "teammate_algorithms": pop["algorithms"],
            "teammate_skill_levels": pop["skill_levels"],
            "byzantine_types": [t.value for t in self.byzantine_spec.types],
            "byzantine_budget": f"{self.byzantine_spec.budget_min}-{self.byzantine_spec.budget_max}",
            "methods": self.methods,
            "seeds": self.seeds,
            "domain_randomization": {
                "obs_noise": self.obs_noise,
                "action_noise": self.action_noise,
                "type_mixing": self.byzantine_spec.type_mixing,
            },
            "total_training_runs": n_methods * self.seeds * n_byz,
        }


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT-SPECIFIC BYZANTINE HEURISTICS
# ═══════════════════════════════════════════════════════════════════════

# These are "stealth_independent" heuristics: harm the team without
# being obviously detectable, designed per-environment.

ENV_STEALTH_HEURISTICS: Dict[str, dict] = {
    "dsse": {
        "description": "Move normally but never search — wastes coverage",
        "rule": "Allow movement actions (0-7) but never action 8 (search)",
    },
    "lbf": {
        "description": "Move to food but never pickup — blocks collection",
        "rule": "Allow movement (0-4) but never action 5 (load)",
    },
    "mpe-pp": {
        "description": "Chase prey but with 20% position jitter — almost catches but doesn't",
        "rule": "With 20% prob, replace action with random adjacent action",
    },
    "overcooked": {
        "description": "Move to ingredients but never interact — looks busy, does nothing",
        "rule": "Allow movement (0-4) but never action 5 (interact)",
    },
    "advisor-trust": {
        "description": "Strategic lying: correct 20% of the time to avoid detection",
        "rule": "adversarial_mode='strategic' with 0.2 honest probability",
    },
    "smac-3s5z": {
        "description": "Attack wrong target — splits team fire, looks active",
        "rule": "Select attack action but target random enemy instead of focus fire",
    },
}
