# BAHT Benchmark Suite

A unified benchmark suite for **Byzantine Ad Hoc Teamwork (BAHT)** — evaluating RL agents that must cooperate with unknown teammates, some of whom may be adversarial (Byzantine).

## Why This Benchmark Exists

Most MARL benchmarks test agents in **self-play** with fixed teammates. This is meaningless for BAHT because:
- Self-play lets agents trivially detect Byzantines via "acts like me = cooperative"
- Fixed Byzantine types don't test generalization
- No behavioral diversity = no real ad hoc teamwork

This benchmark provides **genuine diversity on both axes**:
1. **Cooperative teammates**: Different algorithms × skill levels × architectures
2. **Byzantine adversaries**: From random noise to learned stealth adversaries

## Environments

| Environment | Agents | Actions | Tier | Why BAHT |
|-------------|--------|---------|------|----------|
| **MPE Predator-Prey** | 4 | 5 | Core | Physical coordination, collision |
| **DSSE (Drone Swarm)** | 4 | 9 | Core | Coverage waste, spatial coordination |
| **LBF (Foraging)** | 3 | 6 | Core | Cooperation required for collection |
| **Advisor Trust** | 5-7 | 2 | Core | Detection directly = voting weights |
| **Matrix Games** | 2 | 2 | Core | Simplest testbed, debugging |
| **Overcooked** | 2 | 6 | Extended | 2-agent edge case |
| **SMAC 3s5z** | 8 | 14 | Extended | Large team, multiple Byzantines |
| **SMAC 5v6** | 5 | 12 | Extended | Asymmetric, hard |

## Teammate Diversity

The benchmark pre-trains diverse teammate populations:

| Axis | Values | Why |
|------|--------|-----|
| **Algorithm** | IQL, QMIX, IPPO, MAPPO, POAM | Value-based vs policy-gradient vs teammate-aware |
| **Skill level** | Novice (25%), Intermediate (50%), Competent (75%), Expert (100%) | Checkpoints at different training stages |
| **Architecture** | Shared params, Non-shared params | Different parameter sharing → different specialization |
| **Seeds** | 2-3 per config | Different convergence points |

**Standard population**: 5 algorithms × 4 skill levels × 2 seeds = **40 distinct policies**

## Byzantine Diversity

Byzantine types ordered by detection difficulty:

| Tier | Type | Behavior | Detectability |
|------|------|----------|---------------|
| 1 (Easy) | `random` | Uniform random actions | Trivial |
| 1 (Easy) | `frozen` | Always action 0 | Trivial |
| 2 (Medium) | `flip` | Inverted cooperative policy | Behavioral analysis |
| 2 (Medium) | `offset` | Action shifted by k | Behavioral analysis |
| 3 (Hard) | `adversary` | Learned reward-minimizer | Requires modeling |
| 3 (Hard) | `mixed` | 50% adversary, 50% random | Inconsistent signal |
| 4 (Adversarial) | `stealth` | Adversary + detection evasion | Detection-aware |
| 4 (Adversarial) | `stealth_independent` | Env-specific heuristic stealth | Designed to evade |

Variable budget: 0 to F_max Byzantines per episode, with per-agent type mixing.

## Quick Start

```bash
# Install
pip install -e .

# Step 1: Pre-train diverse teammate population (~4 hours on 2 GPUs)
python -m baht_benchmark.pretrain --env mpe-pp --protocol standard --gpus 2

# Step 2: Run benchmark
python -m baht_benchmark.run --env mpe-pp --method all --seeds 5

# Quick smoke test
python -m baht_benchmark.pretrain --env mpe-pp --protocol minimal
python -m baht_benchmark.run --env mpe-pp --method shapley --seeds 2 --t_max 50000

# Full suite
python -m baht_benchmark.pretrain --env all --protocol standard --gpus 2
python -m baht_benchmark.run --suite core --method all --seeds 5 --gpus 2
```

## Benchmark Protocols

```python
from baht_benchmark import BenchmarkProtocol

# Quick (debugging): 6 teammates, easy Byzantines, 2 seeds
proto = BenchmarkProtocol.quick()

# Standard (papers): 40 teammates, medium Byzantines, 5 seeds
proto = BenchmarkProtocol.standard()

# Full (thorough): 168 teammates, all Byzantine types, 5 seeds
proto = BenchmarkProtocol.full()

print(proto.summary())
```

## Integration with Shapley-AHT

```bash
# From shapley-aht repo
pip install -e ../baht-benchmark

# Use in code
from baht_benchmark import ENVIRONMENTS, get_env_config, BenchmarkProtocol
config = get_env_config("dsse")
proto = BenchmarkProtocol.standard()
```

## Architecture

```
baht_benchmark/
  __init__.py          # Public API
  diversity.py         # Teammate + Byzantine diversity specifications
  registry.py          # Environment registry with per-env configs
  pretrain.py          # Pre-train diverse teammate populations
  run.py               # CLI entry point for running experiments
  analyze.py           # Statistical analysis (Holm-Bonferroni)
  envs/                # Environment wrappers (MultiAgentEnv interface)
    base.py            # MultiAgentEnv base class
    dsse.py, lbf.py, advisor_trust.py, overcooked.py, smac.py
    gymma.py           # Gym-to-MultiAgentEnv adapter
  configs/             # Per-environment YAML configs
```

## Adding a New Environment

1. Create a wrapper in `baht_benchmark/envs/` implementing `MultiAgentEnv`
2. Register in `baht_benchmark/registry.py` with noise calibration
3. Add env-specific stealth heuristic in `diversity.py:ENV_STEALTH_HEURISTICS`
4. Add YAML config in `baht_benchmark/configs/`
5. Add pre-training defaults in `pretrain.py:ENV_DEFAULTS`
