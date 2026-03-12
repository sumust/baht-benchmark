"""Monitor running BAHT benchmark experiments via wandb.

Usage:
    python -m baht_benchmark.monitor                          # latest runs
    python -m baht_benchmark.monitor --project baht-benchmark # specific project
    python -m baht_benchmark.monitor --watch 60               # poll every 60s
"""

import argparse
import sys
import time
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_runs(project: str, max_runs: int = 50):
    """Fetch runs from wandb."""
    api = wandb.Api()
    try:
        runs = api.runs(project, per_page=max_runs)
        return list(runs)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []


def format_run(run) -> dict:
    """Extract key metrics from a wandb run."""
    summary = run.summary
    info = {
        "name": run.name,
        "state": run.state,
        "runtime": summary.get("_runtime", 0),
    }

    # Key BAHT metrics
    for key in ["test_return_mean", "test_return_std",
                "byz_detection_recall", "byz_detection_f1",
                "byz_detection_accuracy", "byz_detection_fpr",
                "contribution_loss", "loss", "t_env"]:
        if key in summary:
            val = summary[key]
            if isinstance(val, (int, float)):
                info[key] = val

    return info


def print_dashboard(runs, project: str):
    """Print a formatted dashboard of all runs."""
    if not runs:
        print(f"No runs found in project '{project}'")
        return

    # Group by state
    by_state = defaultdict(list)
    for r in runs:
        by_state[r.state].append(r)

    print(f"\n{'═'*90}")
    print(f"  BAHT BENCHMARK MONITOR — {project}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*90}")
    print(f"  Running: {len(by_state.get('running', []))}  |  "
          f"Finished: {len(by_state.get('finished', []))}  |  "
          f"Failed: {len(by_state.get('failed', []))}  |  "
          f"Crashed: {len(by_state.get('crashed', []))}")
    print()

    # Active runs
    active = by_state.get("running", [])
    if active:
        print(f"  ACTIVE RUNS ({len(active)})")
        print(f"  {'Name':<45s} {'t_env':>8s} {'Return':>10s} {'Det.Recall':>10s} {'Det.F1':>8s}")
        print(f"  {'-'*85}")
        for r in active:
            info = format_run(r)
            t_env = info.get("t_env", "")
            ret = info.get("test_return_mean", "")
            recall = info.get("byz_detection_recall", "")
            f1 = info.get("byz_detection_f1", "")
            print(f"  {info['name']:<45s} "
                  f"{str(int(t_env)) if t_env != '' else '—':>8s} "
                  f"{f'{ret:.2f}' if ret != '' else '—':>10s} "
                  f"{f'{recall:.3f}' if recall != '' else '—':>10s} "
                  f"{f'{f1:.3f}' if f1 != '' else '—':>8s}")
        print()

    # Completed runs
    finished = by_state.get("finished", [])
    if finished:
        print(f"  COMPLETED RUNS ({len(finished)})")
        print(f"  {'Name':<45s} {'Return':>10s} {'Det.Recall':>10s} {'Det.F1':>8s} {'Runtime':>10s}")
        print(f"  {'-'*85}")
        for r in finished:
            info = format_run(r)
            ret = info.get("test_return_mean", "")
            recall = info.get("byz_detection_recall", "")
            f1 = info.get("byz_detection_f1", "")
            runtime = info.get("runtime", 0)
            hrs = int(runtime // 3600)
            mins = int((runtime % 3600) // 60)
            print(f"  {info['name']:<45s} "
                  f"{f'{ret:.2f}' if ret != '' else '—':>10s} "
                  f"{f'{recall:.3f}' if recall != '' else '—':>10s} "
                  f"{f'{f1:.3f}' if f1 != '' else '—':>8s} "
                  f"{f'{hrs}h{mins:02d}m':>10s}")

        # Summary stats
        returns = [format_run(r).get("test_return_mean") for r in finished]
        returns = [v for v in returns if isinstance(v, (int, float))]
        if returns:
            import numpy as np
            print(f"\n  Return: mean={np.mean(returns):.2f} ± {np.std(returns):.2f} "
                  f"(min={min(returns):.2f}, max={max(returns):.2f})")
        print()

    # Failed runs
    failed = by_state.get("failed", []) + by_state.get("crashed", [])
    if failed:
        print(f"  FAILED/CRASHED ({len(failed)})")
        for r in failed:
            runtime = r.summary.get("_runtime", 0)
            print(f"  {r.name:<45s} {r.state:>10s}  (ran {int(runtime)}s)")
        print()


def main():
    if not WANDB_AVAILABLE:
        print("ERROR: wandb not installed. Run: pip install wandb")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Monitor BAHT benchmark runs")
    parser.add_argument("--project", type=str, default="baht-benchmark",
                        help="wandb project name")
    parser.add_argument("--watch", type=int, default=0,
                        help="Poll interval in seconds (0 = one-shot)")
    parser.add_argument("--max_runs", type=int, default=50)
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Watching project '{args.project}' every {args.watch}s (Ctrl+C to stop)")
        while True:
            try:
                runs = get_runs(args.project, args.max_runs)
                # Clear screen
                print("\033[2J\033[H", end="")
                print_dashboard(runs, args.project)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        runs = get_runs(args.project, args.max_runs)
        print_dashboard(runs, args.project)


if __name__ == "__main__":
    main()
