"""CLI entry point for the BAHT benchmark suite.

Usage:
    python -m baht_benchmark.run --env mpe-pp --method shapley --seeds 3 --t_max 250000
    python -m baht_benchmark.run --suite core --method all --seeds 5
    python -m baht_benchmark.run --list-envs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from baht_benchmark.registry import ENVIRONMENTS, get_env_config, get_suite, EnvConfig


METHODS = {
    "shapley": {
        "alg_config": "mpe/shapley",
        "description": "Shapley-AHT: LOO contribution estimation",
    },
    "poam": {
        "alg_config": "mpe/poam_byz",
        "description": "POAM baseline: no contribution estimation",
    },
    "ippo": {
        "alg_config": "mpe/ippo",
        "description": "IPPO baseline: no teammate modeling",
    },
}


def find_shapley_aht_root() -> Path:
    """Find the shapley-aht codebase root."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent.parent / "shapley-aht",
        Path.home() / "Downloads" / "shapley-aht",
        Path.cwd(),
        Path.cwd().parent / "shapley-aht",
    ]
    for c in candidates:
        if (c / "src" / "main.py").exists():
            return c
    raise FileNotFoundError(
        "Cannot find shapley-aht codebase. Set SHAPLEY_AHT_ROOT env var or run from shapley-aht directory."
    )


def generate_config(env_config: EnvConfig, method: str, seed: int,
                    byz_type: str = "random", byz_budget: int = None,
                    output_dir: str = None) -> dict:
    """Generate a run config dict for shapley-aht's main.py."""
    byz_budget = byz_budget or env_config.recommended_byzantine_budget

    config = {
        "env": env_config.env_key,
        "env_args": env_config.env_kwargs.copy(),
        "n_agents": env_config.n_agents,
        "n_actions": env_config.n_actions,
        "episode_limit": env_config.episode_limit,
        "t_max": env_config.default_t_max,
        "batch_size": env_config.default_batch_size,
        "buffer_size": env_config.default_buffer_size,
        "batch_size_run": env_config.default_batch_size_run,
        "hidden_dim": env_config.default_hidden_dim,
        "seed": seed,
        "byzantine_budget": byz_budget,
        "byzantine_type": byz_type,
        "runner": "byzantine",
        "mac": "open_train_mac",
        "use_cuda": True,
        "use_tensorboard": True,
    }
    return config


def run_experiment(shapley_root: Path, env_config: EnvConfig, method: str,
                   seed: int, byz_type: str, gpu_id: int = 0,
                   t_max: int = None, use_wandb: bool = False,
                   wandb_project: str = "baht-benchmark",
                   extra_args: dict = None):
    """Launch a single experiment as a subprocess."""
    method_info = METHODS[method]
    alg_config = method_info["alg_config"]

    sacred_args = [
        f"seed={seed}",
        f"byzantine_type={byz_type}",
        f"byzantine_budget={env_config.recommended_byzantine_budget}",
    ]

    if t_max:
        sacred_args.append(f"t_max={t_max}")

    if use_wandb:
        sacred_args.append("use_wandb=True")
        sacred_args.append(f"wandb_project={wandb_project}")

    if extra_args:
        for k, v in extra_args.items():
            sacred_args.append(f"{k}={v}")

    cmd = [
        sys.executable, str(shapley_root / "src" / "main.py"),
        f"--config={alg_config}",
        "with",
    ] + sacred_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{shapley_root}/src:{shapley_root}/3rdparty/mpe:{env.get('PYTHONPATH', '')}"

    run_name = f"{env_config.name}_{method}_{byz_type}_s{seed}"
    print(f"  Launching: {run_name}")
    print(f"    CMD: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(shapley_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ), run_name


def list_envs():
    """Print available environments."""
    print("\nBAHT Benchmark Environments")
    print("=" * 80)

    for tier in ["core", "extended"]:
        envs = [e for e in ENVIRONMENTS.values() if e.tier == tier]
        if not envs:
            continue
        print(f"\n{tier.upper()} ({len(envs)} environments):")
        print("-" * 60)
        for e in envs:
            status = "built-in" if e.install == "built-in" else e.install
            print(f"  {e.name:<20s} {e.n_agents} agents, {e.n_actions} actions, ep={e.episode_limit}")
            print(f"  {'':20s} {e.description}")
            print(f"  {'':20s} Install: {status}")
            print()


def main():
    parser = argparse.ArgumentParser(description="BAHT Benchmark Suite")
    parser.add_argument("--env", type=str, help="Environment name (e.g., mpe-pp, dsse)")
    parser.add_argument("--suite", type=str, help="Run a suite: core, extended, all, quick, standard")
    parser.add_argument("--method", type=str, default="all", help="Method: shapley, poam, ippo, all")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--t_max", type=int, default=None, help="Override training steps")
    parser.add_argument("--byz_type", type=str, default="random", help="Byzantine type")
    parser.add_argument("--byz_budget", type=int, default=None, help="Byzantine budget")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--use_wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="baht-benchmark")
    parser.add_argument("--list-envs", action="store_true", help="List available environments")
    parser.add_argument("--shapley_root", type=str, default=None, help="Path to shapley-aht repo")

    args = parser.parse_args()

    if args.list_envs:
        list_envs()
        return

    if not args.env and not args.suite:
        parser.error("Must specify --env or --suite (or --list-envs)")

    # Find shapley-aht
    if args.shapley_root:
        shapley_root = Path(args.shapley_root)
    else:
        try:
            shapley_root = find_shapley_aht_root()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    print(f"Using shapley-aht at: {shapley_root}")

    # Get environments to run
    if args.suite:
        envs = get_suite(args.suite)
    else:
        envs = [get_env_config(args.env)]

    # Get methods to run
    if args.method == "all":
        methods = list(METHODS.keys())
    else:
        methods = [args.method]

    # Build run list
    runs = []
    for env_config in envs:
        for method in methods:
            for seed in range(1, args.seeds + 1):
                runs.append((env_config, method, seed))

    print(f"\nTotal runs: {len(runs)} ({len(envs)} envs x {len(methods)} methods x {args.seeds} seeds)")
    print(f"Byzantine type: {args.byz_type}")
    print()

    # Launch runs with GPU round-robin
    processes = []
    for i, (env_config, method, seed) in enumerate(runs):
        gpu_id = i % args.gpus
        t_max = args.t_max or env_config.default_t_max
        proc, name = run_experiment(
            shapley_root, env_config, method, seed,
            args.byz_type, gpu_id, t_max,
            args.use_wandb, args.wandb_project,
        )
        processes.append((proc, name))

    # Wait for all to complete
    print(f"\nWaiting for {len(processes)} runs to complete...")
    results = {}
    for proc, name in processes:
        returncode = proc.wait()
        status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        results[name] = status
        print(f"  {name}: {status}")

    # Summary
    n_ok = sum(1 for v in results.values() if v == "OK")
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok}/{len(results)} succeeded, {n_fail} failed")

    if n_fail > 0:
        print("\nFailed runs:")
        for name, status in results.items():
            if status != "OK":
                print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
