#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# BAHT Benchmark: Pre-train + Train + Analyze
# ═══════════════════════════════════════════════════════════════════════
#
# ONE SCRIPT to run the entire BAHT benchmark pipeline.
#
# Repo layout (sister repos):
#   ~/Downloads/baht-benchmark/    ← this repo (benchmark suite)
#   ~/Downloads/shapley-aht/       ← training codebase
#
# Usage:
#   ./run_full_benchmark.sh                              # defaults
#   ENV=mpe-pp GPUS=2 USE_WANDB=1 ./run_full_benchmark.sh
#   PROTOCOL=minimal SEEDS=2 T_MAX=50000 ./run_full_benchmark.sh
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration (override with env vars) ────────────────────────────

ENV="${ENV:-mpe-pp}"
PROTOCOL="${PROTOCOL:-standard}"       # minimal | standard | extended
METHODS="${METHODS:-shapley poam ippo}"
BYZ_TYPES="${BYZ_TYPES:-random frozen flip}"
SEEDS="${SEEDS:-3}"
T_MAX="${T_MAX:-250000}"
PRETRAIN_T_MAX="${PRETRAIN_T_MAX:-200000}"
GPUS="${GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-baht-benchmark}"

# ── Auto-detect paths ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$SCRIPT_DIR"

# Find shapley-aht as a sister repo
find_shapley() {
    # Explicit override
    if [ -n "${SHAPLEY_AHT_ROOT:-}" ] && [ -f "$SHAPLEY_AHT_ROOT/src/main.py" ]; then
        echo "$SHAPLEY_AHT_ROOT"; return
    fi
    # Sister repo (../shapley-aht relative to this script)
    local sister="$SCRIPT_DIR/../shapley-aht"
    if [ -f "$sister/src/main.py" ]; then
        echo "$(cd "$sister" && pwd)"; return
    fi
    # Common locations
    for d in "$HOME/Downloads/shapley-aht" "$HOME/workspace/lain/shapley-aht" "$HOME/shapley-aht" "$HOME/code/shapley-aht" "/workspace/lain/shapley-aht"; do
        if [ -f "$d/src/main.py" ]; then
            echo "$d"; return
        fi
    done
    return 1
}

SHAPLEY_ROOT=$(find_shapley) || {
    echo "ERROR: Cannot find shapley-aht."
    echo "  Expected at: $SCRIPT_DIR/../shapley-aht/"
    echo "  Or set: SHAPLEY_AHT_ROOT=/path/to/shapley-aht"
    exit 1
}

# Find Python (venv > conda > system)
find_python() {
    if [ -n "${PYTHON_EXE:-}" ] && [ -f "$PYTHON_EXE" ]; then
        echo "$PYTHON_EXE"; return
    fi
    # venv inside shapley-aht
    if [ -f "$SHAPLEY_ROOT/.venv/bin/python" ]; then
        echo "$SHAPLEY_ROOT/.venv/bin/python"; return
    fi
    # conda env named "baht"
    if [ -f "$HOME/miniconda/envs/baht/bin/python" ]; then
        echo "$HOME/miniconda/envs/baht/bin/python"; return
    fi
    if [ -f "$HOME/miniconda3/envs/baht/bin/python" ]; then
        echo "$HOME/miniconda3/envs/baht/bin/python"; return
    fi
    # Active conda env
    if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"; return
    fi
    echo "$(which python3)"
}
PY=$(find_python)

# PYTHONPATH: shapley-aht src + mpe + benchmark root (for baht_benchmark imports)
export PYTHONPATH="$SHAPLEY_ROOT/src:$SHAPLEY_ROOT/3rdparty/mpe:$BENCHMARK_ROOT:${PYTHONPATH:-}"

# ── Logging ───────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$BENCHMARK_ROOT/logs/$ENV/$TIMESTAMP"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/pipeline.log"; }

log "═══════════════════════════════════════════════════════════════"
log "  BAHT BENCHMARK PIPELINE"
log "═══════════════════════════════════════════════════════════════"
log "  Environment:    $ENV"
log "  Protocol:       $PROTOCOL"
log "  Methods:        $METHODS"
log "  Byzantine:      $BYZ_TYPES"
log "  Seeds:          $SEEDS"
log "  t_max:          $T_MAX  (pretrain: $PRETRAIN_T_MAX)"
log "  GPUs:           $GPUS"
log "  wandb:          $([ "$USE_WANDB" = "1" ] && echo "$WANDB_PROJECT" || echo "off")"
log "  Shapley root:   $SHAPLEY_ROOT"
log "  Benchmark root: $BENCHMARK_ROOT"
log "  Python:         $PY"
log "  Logs:           $LOG_DIR"
log ""

# Sanity check: can Python import baht_benchmark?
$PY -c "from baht_benchmark import ENVIRONMENTS; print(f'  Benchmark loaded: {len(ENVIRONMENTS)} environments')" 2>&1 | tee -a "$LOG_DIR/pipeline.log" || {
    log "ERROR: Cannot import baht_benchmark. Run: pip install -e $BENCHMARK_ROOT"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-train diverse teammate population
# ═══════════════════════════════════════════════════════════════════════

POPULATION_DIR="$SHAPLEY_ROOT/populations/$ENV"

if [ -f "$POPULATION_DIR/manifest.json" ]; then
    N_POLICIES=$($PY -c "import json; print(json.load(open('$POPULATION_DIR/manifest.json'))['n_policies'])")
    log "PHASE 1: SKIP — found existing population ($N_POLICIES policies)"
else
    log "PHASE 1: Pre-training diverse teammates ($PROTOCOL protocol)"
    log "  This takes a while. Logs in $LOG_DIR/pretrain.log"

    $PY -m baht_benchmark.pretrain \
        --env "$ENV" \
        --protocol "$PROTOCOL" \
        --t_max "$PRETRAIN_T_MAX" \
        --gpus "$GPUS" \
        --max_parallel "$MAX_PARALLEL" \
        --shapley_root "$SHAPLEY_ROOT" \
        2>&1 | tee "$LOG_DIR/pretrain.log"

    if [ ! -f "$POPULATION_DIR/manifest.json" ]; then
        log "ERROR: Pre-training failed — no manifest.json created"
        exit 1
    fi
    N_POLICIES=$($PY -c "import json; print(json.load(open('$POPULATION_DIR/manifest.json'))['n_policies'])")
    log "PHASE 1: Done — $N_POLICIES policies trained"
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Train ego agents with Byzantine injection
# ═══════════════════════════════════════════════════════════════════════

log ""
log "PHASE 2: Training BAHT methods against diverse teammates"

WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS="use_wandb=True wandb_project=$WANDB_PROJECT"
fi

RESULTS_DIR="$SHAPLEY_ROOT/results/baht_benchmark/$ENV/$TIMESTAMP"
PIDS=()
RUN_NAMES=()
GPU_IDX=0

# Generate uncntrl_agents Sacred args from manifest (once, shared by all runs)
UNCNTRL_ARGS=$($PY -c "
import json, sys
manifest = json.load(open('$POPULATION_DIR/manifest.json'))
# Get n_agents from manifest or env config
n_agents = manifest.get('n_agents', None)
if n_agents is None:
    # Fall back to env registry
    try:
        from baht_benchmark import get_env_config
        n_agents = get_env_config('$ENV').n_agents
    except:
        n_agents = 4  # safe default for MPE-PP
parts = []
for i, p in enumerate(manifest['policies']):
    name = f'agent_{i}'
    path = p.get('path', '')
    parts.append(f'uncntrl_agents.{name}.agent_loader=rnn_eval_agent_loader')
    parts.append(f'uncntrl_agents.{name}.agent_path={path}')
    parts.append(f'uncntrl_agents.{name}.load_step=best')
    parts.append(f'uncntrl_agents.{name}.n_agents_to_populate={n_agents - 1}')
    parts.append(f'uncntrl_agents.{name}.test_mode=True')
print(' '.join(parts))
" 2>&1) || {
    log "ERROR: Failed to parse manifest.json"
    log "  Contents:"
    cat "$POPULATION_DIR/manifest.json" 2>/dev/null | head -20 | while read -r line; do log "    $line"; done
    exit 1
}

if [ -z "$UNCNTRL_ARGS" ]; then
    log "ERROR: manifest.json has no policies"
    exit 1
fi

log "  Teammate population: $N_POLICIES policies loaded from manifest"

# Determine default config based on environment
case $ENV in
    mpe-pp)       DEFAULT_CONFIG="default/default_mpe_pp_baht" ;;
    matrix-games) DEFAULT_CONFIG="default/default_matrix_games" ;;
    *)            DEFAULT_CONFIG="default/default_mpe_pp_baht" ;;
esac

for METHOD in $METHODS; do
    for BYZ in $BYZ_TYPES; do
        for SEED in $(seq 1 "$SEEDS"); do
            RUN_NAME="${METHOD}_${BYZ}_s${SEED}"
            RUN_DIR="$RESULTS_DIR/$METHOD/$BYZ/seed_$SEED"
            GPU_ID=$((GPU_IDX % GPUS))
            GPU_IDX=$((GPU_IDX + 1))

            log "  Launching: $RUN_NAME on GPU $GPU_ID"

            # Determine alg config
            case $METHOD in
                shapley) ALG_CONFIG="mpe/shapley" ;;
                poam)    ALG_CONFIG="mpe/poam_byz" ;;
                ippo)    ALG_CONFIG="mpe/ippo" ;;
                iql)     ALG_CONFIG="mpe/iql" ;;
                qmix)    ALG_CONFIG="mpe/qmix" ;;
                mappo)   ALG_CONFIG="mpe/mappo" ;;
                *)       ALG_CONFIG="mpe/$METHOD" ;;
            esac

            CUDA_VISIBLE_DEVICES=$GPU_ID $PY "$SHAPLEY_ROOT/src/main.py" \
                --config="$DEFAULT_CONFIG" \
                --alg-config="$ALG_CONFIG" \
                with \
                seed=$SEED \
                t_max=$T_MAX \
                byzantine_type="$BYZ" \
                byzantine_budget=1 \
                runner=byzantine \
                mac=open_train_mac \
                local_results_path="$RUN_DIR" \
                save_model=True \
                save_model_interval=$T_MAX \
                $UNCNTRL_ARGS \
                $WANDB_ARGS \
                > "$LOG_DIR/${RUN_NAME}.log" 2>&1 &

            PIDS+=($!)
            RUN_NAMES+=("$RUN_NAME")
        done
    done
done

N_RUNS=${#PIDS[@]}
log ""
log "  $N_RUNS runs launched. Waiting..."
if [ "$USE_WANDB" = "1" ]; then
    log "  Monitor: python -m baht_benchmark.monitor --project $WANDB_PROJECT --watch 60"
fi

# ── Wait and report ───────────────────────────────────────────────────

N_OK=0
N_FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    RC=$?
    if [ $RC -eq 0 ]; then
        STATUS="OK"
        N_OK=$((N_OK + 1))
    else
        STATUS="FAILED (exit $RC)"
        N_FAIL=$((N_FAIL + 1))
    fi
    log "  [$(($N_OK + $N_FAIL))/$N_RUNS] ${RUN_NAMES[$i]}: $STATUS"
done

log ""
log "PHASE 2: Done — $N_OK/$N_RUNS succeeded, $N_FAIL failed"

if [ $N_FAIL -gt 0 ]; then
    log ""
    log "Failed run logs:"
    for i in "${!PIDS[@]}"; do
        if grep -q "Traceback\|Error\|FAILED" "$LOG_DIR/${RUN_NAMES[$i]}.log" 2>/dev/null; then
            log "  $LOG_DIR/${RUN_NAMES[$i]}.log"
            tail -5 "$LOG_DIR/${RUN_NAMES[$i]}.log" 2>/dev/null | while read -r line; do
                log "    $line"
            done
        fi
    done
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Analyze results
# ═══════════════════════════════════════════════════════════════════════

log ""
log "PHASE 3: Analyzing results"

$PY -m baht_benchmark.analyze "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/analysis.log"

log ""
log "═══════════════════════════════════════════════════════════════"
log "  BENCHMARK COMPLETE"
log "═══════════════════════════════════════════════════════════════"
log "  Results: $RESULTS_DIR"
log "  Logs:    $LOG_DIR"
log "  Analyze: python -m baht_benchmark.analyze $RESULTS_DIR"
if [ "$USE_WANDB" = "1" ]; then
    log "  Monitor: python -m baht_benchmark.monitor --project $WANDB_PROJECT"
fi
