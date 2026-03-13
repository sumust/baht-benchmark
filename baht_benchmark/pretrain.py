"""Pre-train diverse teammate populations for BAHT benchmark environments.

This is the most important step in the benchmark. Without genuine behavioral
diversity, BAHT reduces to self-play where detection is trivial.

Diversity comes from 4 axes:
  1. Algorithm: IQL, QMIX, IPPO, MAPPO, POAM (different learning paradigms)
  2. Skill level: Checkpoints at 25/50/75/100% of training (novice to expert)
  3. Architecture: Shared vs non-shared parameters
  4. Seeds: Multiple random initializations per configuration

Usage:
    # Standard population (5 algos × 4 skill levels × 2 seeds = 40 policies)
    python -m baht_benchmark.pretrain --env mpe-pp --protocol standard

    # Minimal for quick experiments (3 algos × 2 skill levels × 1 seed = 6 policies)
    python -m baht_benchmark.pretrain --env mpe-pp --protocol minimal

    # Custom: specific algorithms only
    python -m baht_benchmark.pretrain --env mpe-pp --algorithms ippo mappo qmix --seeds 3
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from baht_benchmark.registry import get_env_config, list_environments, EnvConfig
from baht_benchmark.diversity import (
    TeammatePopulationSpec, TeammateSpec, TeammateAlgorithm,
    TeammateSkillLevel, ArchitectureVariant,
)


# Algorithm → shapley-aht config mapping
ALGO_CONFIGS: Dict[str, dict] = {
    "iql": {
        "alg_config": "mpe/iql" if os.path.exists("src/config/algs/mpe/iql.yaml") else "iql",
        "agent": "rnn",
        "learner": "q_learner",
        "on_policy": False,
    },
    "qmix": {
        "alg_config": "qmix",
        "agent": "rnn",
        "learner": "q_learner",
        "on_policy": False,
    },
    "vdn": {
        "alg_config": "vdn",
        "agent": "rnn",
        "learner": "q_learner",
        "on_policy": False,
    },
    "ippo": {
        "alg_config": "mpe/ippo",
        "agent": "rnn_norm",
        "learner": "ppo_learner",
        "on_policy": True,
    },
    "mappo": {
        "alg_config": "mpe/mappo",
        "agent": "rnn_norm",
        "learner": "ppo_learner",
        "on_policy": True,
    },
    "maddpg": {
        "alg_config": "sc2/maddpg",
        "agent": "rnn",
        "learner": "maddpg_learner",
        "on_policy": False,
    },
    "pac": {
        "alg_config": "sc2/pac_ns",
        "agent": "rnn",
        "learner": "pac_learner",
        "on_policy": True,
    },
    "poam": {
        "alg_config": "mpe/poam",
        "agent": "rnn_poam",
        "learner": "poam_learner",
        "on_policy": True,
    },
    "shapley": {
        "alg_config": "mpe/shapley",
        "agent": "rnn_shapley",
        "learner": "shapley_learner",
        "on_policy": True,
    },
}

# Default env configs for shapley-aht main.py
ENV_DEFAULTS: Dict[str, dict] = {
    "mpe-pp": {
        "default_config": "default/default_mpe_pp",
        "env_config": "mpe",
        "env_args_key": "mpe:SimpleTag-v0",
    },
    "dsse": {
        "default_config": "default/default_mpe_pp",
        "env_config": "gymma",
        "env_args_key": "DSSE-v0",
        "note": "Uses MPE-PP defaults; create default_dsse.yaml for env-specific settings",
    },
    "lbf": {
        "default_config": "default/default_mpe_pp",
        "env_config": "gymma",
        "env_args_key": "lbforaging:Foraging-8x8-3p-2f-coop-v3",
        "note": "Uses MPE-PP defaults; create default_lbf.yaml for env-specific settings",
    },
    "matrix-games": {
        "default_config": "default/default_matrix_games",
        "env_config": "gymma",
        "env_args_key": "matrixgames:pdilemma-nostate-v0",
    },
}


def find_shapley_root() -> Path:
    """Find the shapley-aht codebase root."""
    candidates = [
        Path(__file__).parent.parent.parent / "shapley-aht",
        Path.home() / "Downloads" / "shapley-aht",
        Path.cwd(),
        Path.cwd().parent / "shapley-aht",
    ]
    if "SHAPLEY_AHT_ROOT" in os.environ:
        candidates.insert(0, Path(os.environ["SHAPLEY_AHT_ROOT"]))
    for c in candidates:
        if (c / "src" / "main.py").exists():
            return c
    raise FileNotFoundError(
        "Cannot find shapley-aht. Set SHAPLEY_AHT_ROOT env var."
    )


def find_python_exe(shapley_root: Path) -> str:
    """Find the correct Python executable for subprocesses."""
    candidates = [
        os.path.expanduser("~/miniconda/envs/baht/bin/python"),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "python"),
        os.path.join(str(shapley_root), ".venv", "bin", "python"),
        sys.executable,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return sys.executable


def get_checkpoint_fractions(t_max: int) -> Dict[str, int]:
    """Get checkpoint steps for each skill level."""
    return {
        "novice": max(t_max // 4, 1000),
        "intermediate": max(t_max // 2, 2000),
        "competent": max(3 * t_max // 4, 3000),
        "expert": t_max,
    }


def pretrain_single(
    shapley_root: Path,
    python_exe: str,
    env_config: EnvConfig,
    algorithm: str,
    seed: int,
    t_max: int,
    architecture: str = "rnn",
    output_dir: Path = None,
    gpu_id: int = 0,
    save_intermediate: bool = True,
) -> subprocess.Popen:
    """Launch a single pre-training run.

    When save_intermediate=True, saves checkpoints at 25/50/75/100%
    of training for skill-level diversity.
    """
    if algorithm not in ALGO_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGO_CONFIGS.keys())}")

    algo_cfg = ALGO_CONFIGS[algorithm]
    env_defaults = ENV_DEFAULTS.get(env_config.name, {})

    # Determine save intervals for skill-level checkpoints
    if save_intermediate:
        save_interval = max(t_max // 4, 1000)
    else:
        save_interval = t_max

    results_dir = output_dir or (shapley_root / "populations" / env_config.name / algorithm / f"seed_{seed}")
    results_dir = Path(results_dir)

    # Build command
    default_config = env_defaults.get("default_config", "default/default_mpe_pp")
    cmd = [
        python_exe, str(shapley_root / "src" / "main.py"),
        f"--config={default_config}",
        f"--alg-config={algo_cfg['alg_config']}",
        "with",
        f"t_max={t_max}",
        f"seed={seed}",
        f"local_results_path={results_dir}",
        f"save_model=True",
        f"save_model_interval={save_interval}",
        f"test_interval={max(t_max // 10, 1000)}",
        f"log_interval={max(t_max // 10, 1000)}",
        f"use_wandb=False",
    ]

    # Architecture override
    if architecture == "rnn_ns":
        cmd.append("agent=rnn_ns")
    elif architecture == "rnn_norm":
        cmd.append("agent=rnn_norm")
    elif architecture == "rnn_norm_ns":
        cmd.append("agent=rnn_norm_ns")

    # Environment-specific args
    if env_defaults.get("env_args_key"):
        cmd.append(f"env_args.key={env_defaults['env_args_key']}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{shapley_root}/src:{shapley_root}/3rdparty/mpe:{env.get('PYTHONPATH', '')}"

    log_path = results_dir.parent / f"pretrain_{algorithm}_s{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=str(shapley_root),
    )
    return proc


def build_manifest(pop_dir: Path, env_config: EnvConfig, t_max: int) -> dict:
    """Scan population directory and build manifest.json with all policies."""
    policies = []

    # Find all checkpoints
    for agent_file in sorted(pop_dir.rglob("agent.th")):
        # Parse path: populations/{env}/{algo}/seed_{s}/models/{logname}/{step}/agent.th
        parts = agent_file.relative_to(pop_dir).parts
        if len(parts) < 4:
            continue

        algorithm = parts[0]  # e.g., "ippo"
        seed_dir = parts[1]   # e.g., "seed_42"
        seed = int(seed_dir.split("_")[1]) if "_" in seed_dir else 0

        # Determine step from directory name
        step_dir = agent_file.parent.name  # e.g., "50000" or "best"
        if step_dir == "best":
            step = t_max  # Assume best ≈ full training
            skill_level = "expert"
        elif step_dir.isdigit():
            step = int(step_dir)
            frac = step / t_max
            if frac <= 0.3:
                skill_level = "novice"
            elif frac <= 0.6:
                skill_level = "intermediate"
            elif frac <= 0.85:
                skill_level = "competent"
            else:
                skill_level = "expert"
        else:
            continue

        model_path = str(agent_file.parent.parent)  # Up to logname level

        # Check for Sacred config — search for any run's config.json
        sacred_dir = model_path.replace("models", "sacred")
        has_config = False
        if os.path.isdir(sacred_dir):
            for run_dir in sorted(Path(sacred_dir).iterdir(), reverse=True):
                if (run_dir / "config.json").exists():
                    has_config = True
                    break

        policies.append({
            "path": model_path,
            "algorithm": algorithm,
            "seed": seed,
            "skill_level": skill_level,
            "training_steps": step,
            "has_sacred_config": has_config,
        })

    # Deduplicate by (algorithm, seed, skill_level)
    seen = set()
    unique_policies = []
    for p in policies:
        key = (p["algorithm"], p["seed"], p["skill_level"])
        if key not in seen:
            seen.add(key)
            unique_policies.append(p)

    manifest = {
        "env": env_config.name,
        "n_agents": env_config.n_agents,
        "n_policies": len(unique_policies),
        "t_max": t_max,
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": sorted(set(p["algorithm"] for p in unique_policies)),
        "skill_levels": sorted(set(p["skill_level"] for p in unique_policies)),
        "seeds": sorted(set(p["seed"] for p in unique_policies)),
        "policies": unique_policies,
    }

    with open(pop_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def pretrain_population(
    shapley_root: Path,
    env_config: EnvConfig,
    population_spec: TeammatePopulationSpec,
    t_max: int = None,
    gpus: int = 1,
    max_parallel: int = 4,
) -> dict:
    """Pre-train a full teammate population according to spec."""
    t_max = t_max or env_config.pretrain_t_max
    pop_dir = shapley_root / "populations" / env_config.name
    python_exe = find_python_exe(shapley_root)

    # Check if already done
    manifest_path = pop_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Found existing population: {manifest['n_policies']} policies")
        print(f"  Algorithms: {manifest.get('algorithms', [])}")
        print(f"  Skill levels: {manifest.get('skill_levels', [])}")
        return manifest

    pop_dir.mkdir(parents=True, exist_ok=True)

    # Group specs by (algorithm, seed) — each run produces all skill levels
    run_groups: Dict[tuple, List[TeammateSpec]] = {}
    for spec in population_spec.specs:
        key = (spec.algorithm.value, spec.seed, spec.architecture.value)
        if key not in run_groups:
            run_groups[key] = []
        run_groups[key].append(spec)

    # Deduplicate: one training run per (algorithm, seed, architecture)
    unique_runs = list(run_groups.keys())

    print(f"\n{'='*70}")
    print(f"  PRE-TRAINING DIVERSE TEAMMATE POPULATION")
    print(f"{'='*70}")
    print(f"  Environment: {env_config.name}")
    print(f"  Population spec: {population_spec.n_policies} total policies")
    print(f"  Unique training runs: {len(unique_runs)}")
    print(f"  t_max per run: {t_max}")
    print(f"  GPUs: {gpus}, max parallel: {max_parallel}")
    summary = population_spec.summary()
    print(f"  Algorithms: {summary['algorithms']}")
    print(f"  Skill levels: {summary['skill_levels']}")
    print(f"  Architectures: {summary['architectures']}")
    print()

    # Launch runs in batches
    active = []
    completed = 0
    failed = 0

    for i, (algo, seed, arch) in enumerate(unique_runs):
        # Wait if at max parallel
        while len(active) >= max_parallel:
            for j, (proc, info) in enumerate(active):
                if proc.poll() is not None:
                    rc = proc.returncode
                    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                    print(f"  [{completed+failed+1}/{len(unique_runs)}] {info}: {status}")
                    if rc == 0:
                        completed += 1
                    else:
                        failed += 1
                    active.pop(j)
                    break
            else:
                time.sleep(5)

        gpu_id = i % gpus
        output_dir = pop_dir / algo / f"seed_{seed}"

        print(f"  Launching: {algo} seed={seed} arch={arch} on GPU {gpu_id}")
        proc = pretrain_single(
            shapley_root, python_exe, env_config,
            algorithm=algo, seed=seed, t_max=t_max,
            architecture=arch, output_dir=output_dir, gpu_id=gpu_id,
            save_intermediate=True,
        )
        active.append((proc, f"{algo}_s{seed}_{arch}"))
        time.sleep(2)

    # Wait for remaining
    for proc, info in active:
        proc.wait()
        rc = proc.returncode
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        if rc == 0:
            completed += 1
        else:
            failed += 1
        print(f"  [{completed+failed}/{len(unique_runs)}] {info}: {status}")

    print(f"\n  Completed: {completed}/{len(unique_runs)}, Failed: {failed}")

    # Build manifest from what was actually saved
    manifest = build_manifest(pop_dir, env_config, t_max)
    print(f"  Population: {manifest['n_policies']} policies saved")
    print(f"  Algorithms: {manifest['algorithms']}")
    print(f"  Skill levels: {manifest['skill_levels']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train diverse teammate populations for BAHT benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard population (40 policies, ~4 hours on 2 GPUs)
  python -m baht_benchmark.pretrain --env mpe-pp --protocol standard --gpus 2

  # Minimal for debugging (6 policies, ~30 min)
  python -m baht_benchmark.pretrain --env mpe-pp --protocol minimal

  # Custom algorithms
  python -m baht_benchmark.pretrain --env mpe-pp --algorithms ippo mappo qmix iql --seeds 3

  # All core environments
  python -m baht_benchmark.pretrain --env all --protocol standard --gpus 2
        """,
    )
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name or 'all' for all core envs")
    parser.add_argument("--protocol", type=str, default="standard",
                        choices=["minimal", "standard", "extended"],
                        help="Population protocol (default: standard)")
    parser.add_argument("--algorithms", nargs="+", type=str, default=None,
                        help="Override: specific algorithms to train")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override: seeds per config")
    parser.add_argument("--t_max", type=int, default=None,
                        help="Override: training steps per run")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--shapley_root", type=str, default=None)
    args = parser.parse_args()

    # Find shapley-aht
    shapley_root = Path(args.shapley_root) if args.shapley_root else find_shapley_root()
    print(f"Using shapley-aht at: {shapley_root}")

    # Build population spec
    if args.algorithms:
        # Custom spec from CLI
        seeds = args.seeds or 2
        specs = []
        for algo_name in args.algorithms:
            algo = TeammateAlgorithm(algo_name)
            for skill, frac in [
                (TeammateSkillLevel.NOVICE, 0.25),
                (TeammateSkillLevel.INTERMEDIATE, 0.50),
                (TeammateSkillLevel.COMPETENT, 0.75),
                (TeammateSkillLevel.EXPERT, 1.0),
            ]:
                for seed in range(seeds):
                    specs.append(TeammateSpec(
                        algorithm=algo, skill_level=skill,
                        seed=42 + seed, checkpoint_fraction=frac,
                    ))
        population_spec = TeammatePopulationSpec(specs=specs)
    else:
        # Protocol-based spec
        if args.protocol == "minimal":
            population_spec = TeammatePopulationSpec.minimal(args.seeds or 1)
        elif args.protocol == "standard":
            population_spec = TeammatePopulationSpec.standard(args.seeds or 2)
        elif args.protocol == "extended":
            population_spec = TeammatePopulationSpec.extended(args.seeds or 3)

    # Get environments
    if args.env == "all":
        envs = list_environments(tier="core")
    else:
        envs = [get_env_config(args.env)]

    for env_config in envs:
        pretrain_population(
            shapley_root, env_config, population_spec,
            t_max=args.t_max,
            gpus=args.gpus,
            max_parallel=args.max_parallel,
        )


if __name__ == "__main__":
    main()
