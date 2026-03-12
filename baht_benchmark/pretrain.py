"""Pre-train diverse teammate populations for BAHT benchmark environments.

Creates populations/ directory with diverse policies for each environment.
Used with OpenTrainMAC for real ad hoc teamwork (not self-play).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from baht_benchmark.registry import get_env_config, list_environments, EnvConfig


def pretrain_env(shapley_root: Path, env_config: EnvConfig, seeds: int = 3,
                 t_max: int = None, gpu_id: int = 0) -> Path:
    """Pre-train diverse teammates for a single environment."""
    t_max = t_max or env_config.pretrain_t_max
    pop_dir = shapley_root / "populations" / f"{env_config.name}-teammates"
    pop_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPre-training teammates for {env_config.name}")
    print(f"  Seeds: {seeds}, t_max: {t_max}")
    print(f"  Output: {pop_dir}")

    # Use the shapley-aht pretrain script if available
    pretrain_script = shapley_root / "scripts" / "pretrain_teammates.py"
    if pretrain_script.exists():
        cmd = [
            sys.executable, str(pretrain_script),
            "--env", env_config.name,
            "--seeds", str(seeds),
            "--t_max", str(t_max),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{shapley_root}/src:{shapley_root}/3rdparty/mpe:{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(cmd, env=env, cwd=str(shapley_root))
        if proc.returncode != 0:
            print(f"  WARNING: Pre-training failed for {env_config.name}")
            return pop_dir
    else:
        # Fall back to running main.py directly for each seed
        for seed in range(1, seeds + 1):
            print(f"  Training seed {seed}/{seeds}...")
            cmd = [
                sys.executable, str(shapley_root / "src" / "main.py"),
                f"--config=mpe/ippo",
                "with",
                f"seed={seed}",
                f"t_max={t_max}",
                f"save_model=True",
                f"save_model_interval={t_max}",
                f"local_results_path=populations/{env_config.name}-teammates/seed_{seed}",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONPATH"] = f"{shapley_root}/src:{shapley_root}/3rdparty/mpe:{env.get('PYTHONPATH', '')}"
            subprocess.run(cmd, env=env, cwd=str(shapley_root))

    # Write manifest
    manifest = {
        "env": env_config.name,
        "n_agents": env_config.n_agents,
        "seeds": seeds,
        "t_max": t_max,
    }
    with open(pop_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Done: {pop_dir}")
    return pop_dir


def main():
    parser = argparse.ArgumentParser(description="Pre-train diverse teammate populations")
    parser.add_argument("--env", type=str, help="Environment name (or 'all' for all core envs)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--shapley_root", type=str, default=None)
    args = parser.parse_args()

    # Find shapley-aht
    if args.shapley_root:
        shapley_root = Path(args.shapley_root)
    else:
        from baht_benchmark.run import find_shapley_aht_root
        shapley_root = find_shapley_aht_root()

    if args.env == "all":
        envs = list_environments(tier="core")
    else:
        envs = [get_env_config(args.env)]

    for i, env_config in enumerate(envs):
        gpu_id = i % args.gpus
        pretrain_env(shapley_root, env_config, args.seeds, args.t_max, gpu_id)


if __name__ == "__main__":
    main()
