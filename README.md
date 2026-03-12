# BAHT Benchmark Suite

A unified benchmark suite for **Byzantine Ad Hoc Teamwork (BAHT)** — evaluating RL agents that must cooperate with unknown teammates, some of whom may be adversarial (Byzantine).

## Environments

| Environment | Agents | Actions | Interface | Install | Status |
|-------------|--------|---------|-----------|---------|--------|
| **MPE Predator-Prey** | 4 | 5 | Native MultiAgentEnv | `./3rdparty/mpe` | Core |
| **DSSE (Drone Swarm Search)** | 4 | 9 | PettingZoo → MultiAgentEnv | `pip install DSSE` | Core |
| **LBF (Level-Based Foraging)** | 3 | 6 | Gymnasium → GymmaWrapper | `pip install lbforaging` | Core |
| **Advisor Trust** | 5-7 | 2 | Native MultiAgentEnv | Built-in | Core |
| **Matrix Games** | 2 | 2 | Gym → GymmaWrapper | `pip install matrix-games` | Core |
| **Overcooked** | 2 | 6 | Gym → GymmaWrapper | `pip install overcooked-ai` | Extended |
| **SMAC (StarCraft)** | 3-11 | 7-14 | Native MultiAgentEnv | `pip install pysc2` | Extended |

## Quick Start

```bash
# Install
pip install -e .

# Run a single environment benchmark
python -m baht_benchmark.run --env mpe-pp --method shapley --seeds 3 --t_max 250000

# Run the full core suite
python -m baht_benchmark.run --suite core --method all --seeds 5

# Run with specific Byzantine type
python -m baht_benchmark.run --env dsse --method shapley --byz_type adversary --seeds 3

# List available environments
python -m baht_benchmark.run --list-envs
```

## Integration with Shapley-AHT

This benchmark is designed as a companion to the [shapley-aht](../shapley-aht) codebase:

```bash
# From shapley-aht repo, install benchmark as dependency
pip install -e ../baht-benchmark

# Use in experiments
from baht_benchmark import ENVIRONMENTS, get_env_config
config = get_env_config("dsse")
```

## Architecture

```
baht_benchmark/
  __init__.py          # Registry and public API
  envs/                # Environment wrappers (MultiAgentEnv interface)
    base.py            # MultiAgentEnv base class
    mpe_pp.py          # MPE Predator-Prey (from 3rdparty/mpe)
    dsse.py            # Drone Swarm Search
    lbf.py             # Level-Based Foraging
    advisor_trust.py   # Advisor Trust
    matrix_games.py    # Matrix Games (via gymma)
    overcooked.py      # Overcooked-AI
    smac.py            # StarCraft Multi-Agent Challenge
    gymma.py           # Gym-to-MultiAgentEnv adapter
  configs/             # Per-environment YAML configs
  run.py               # CLI entry point
  registry.py          # Environment registry with metadata
  pretrain.py          # Pre-train diverse teammate populations
  evaluate.py          # Cross-evaluation pipeline
  analyze.py           # Statistical analysis with Holm-Bonferroni
```

## Byzantine Types

All environments support these Byzantine agent behaviors:
- **random**: Uniform random actions
- **frozen**: Fixed action (no-op or action 0)
- **flip**: Inverted cooperative policy
- **adversary**: Reward-minimizing policy (trained)
- **stealth**: Adversarial + evasive (hardest to detect)

## Adding a New Environment

1. Create a wrapper in `baht_benchmark/envs/` implementing `MultiAgentEnv`
2. Register it in `baht_benchmark/registry.py`
3. Add a config YAML in `baht_benchmark/configs/`
4. Add install instructions to `setup.py` extras
