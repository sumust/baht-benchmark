#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# BAHT Benchmark: Pre-train + Train + Analyze
# ═══════════════════════════════════════════════════════════════════════
#
# ONE SCRIPT to run the entire BAHT benchmark pipeline:
#   Phase 1: Pre-train diverse teammate populations
#   Phase 2: Train ego agents (Shapley / POAM / IPPO) against diverse
#            teammates with Byzantine injection
#   Phase 3: Analyze results
#
# Usage:
#   # Standard benchmark on MPE-PP (2 GPUs, ~6 hours total)
#   ./run_full_benchmark.sh
#
#   # Custom: specific env, protocol, GPUs
#   ENV=dsse PROTOCOL=minimal GPUS=1 SEEDS=2 ./run_full_benchmark.sh
#
#   # On lovelace (2 GPUs, wandb logging)
#   ENV=mpe-pp GPUS=2 USE_WANDB=1 ./run_full_benchmark.sh
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

# Find shapley-aht
if [ -n "${SHAPLEY_AHT_ROOT:-}" ]; then
    SHAPLEY_ROOT="$SHAPLEY_AHT_ROOT"
elif [ -d "$SCRIPT_DIR/../shapley-aht/src/main.py" ]; then
    SHAPLEY_ROOT="$(cd "$SCRIPT_DIR/../shapley-aht" && pwd)"
elif [ -d "$HOME/Downloads/shapley-aht/src/main.py" ]; then
    SHAPLEY_ROOT="$HOME/Downloads/shapley-aht"
else
    # Search common locations
    for d in "$HOME/Downloads/shapley-aht" "$HOME/shapley-aht" "$HOME/code/shapley-aht"; do
        if [ -f "$d/src/main.py" ]; then
            SHAPLEY_ROOT="$d"
            break
        fi
    done
fi

if [ -z "${SHAPLEY_ROOT:-}" ] || [ ! -f "$SHAPLEY_ROOT/src/main.py" ]; then
    echo "ERROR: Cannot find shapley-aht. Set SHAPLEY_AHT_ROOT=/path/to/shapley-aht"
    exit 1
fi

# Find Python
if [ -n "${PYTHON_EXE:-}" ]; then
    PY="$PYTHON_EXE"
elif [ -f "$HOME/miniconda/envs/baht/bin/python" ]; then
    PY="$HOME/miniconda/envs/baht/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
elif [ -f "$SHAPLEY_ROOT/.venv/bin/python" ]; then
    PY="$SHAPLEY_ROOT/.venv/bin/python"
else
    PY="$(which python3)"
fi

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
log "  t_max:          $T_MAX"
log "  GPUs:           $GPUS"
log "  Shapley root:   $SHAPLEY_ROOT"
log "  Python:         $PY"
log "  Logs:           $LOG_DIR"
log ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-train diverse teammate population
# ═══════════════════════════════════════════════════════════════════════

POPULATION_DIR="$SHAPLEY_ROOT/populations/$ENV"

if [ -f "$POPULATION_DIR/manifest.json" ]; then
    N_POLICIES=$(python3 -c "import json; print(json.load(open('$POPULATION_DIR/manifest.json'))['n_policies'])")
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
        log "ERROR: Pre-training failed — no manifest.json"
        exit 1
    fi
    N_POLICIES=$(python3 -c "import json; print(json.load(open('$POPULATION_DIR/manifest.json'))['n_policies'])")
    log "PHASE 1: Done — $N_POLICIES policies trained"
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Train ego agents with Byzantine injection
# ═══════════════════════════════════════════════════════════════════════

log ""
log "PHASE 2: Training BAHT methods against diverse teammates"

WANDB_FLAG=""
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG="--use_wandb --wandb_project $WANDB_PROJECT"
fi

RESULTS_DIR="$SHAPLEY_ROOT/results/baht_benchmark/$ENV/$TIMESTAMP"
PIDS=()
RUN_NAMES=()
GPU_IDX=0

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
                *)       ALG_CONFIG="mpe/$METHOD" ;;
            esac

            # Build Sacred args from manifest
            UNCNTRL_ARGS=$($PY -c "
import json
manifest = json.load(open('$POPULATION_DIR/manifest.json'))
# Build uncntrl_agents Sacred override
parts = []
for i, p in enumerate(manifest['policies']):
    name = f'agent_{i}'
    parts.append(f'uncntrl_agents.{name}.agent_loader=rnn_eval_agent_loader')
    parts.append(f'uncntrl_agents.{name}.agent_path={p[\"path\"]}')
    parts.append(f'uncntrl_agents.{name}.load_step=best')
    parts.append(f'uncntrl_agents.{name}.n_agents_to_populate=${manifest[\"n_agents\"]- 1}')
    parts.append(f'uncntrl_agents.{name}.test_mode=True')
print(' '.join(parts))
" 2>/dev/null || echo "")

            CUDA_VISIBLE_DEVICES=$GPU_ID $PY "$SHAPLEY_ROOT/src/main.py" \
                --config="default/default_mpe_pp_baht" \
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
                $WANDB_FLAG \
                > "$LOG_DIR/${RUN_NAME}.log" 2>&1 &

            PIDS+=($!)
            RUN_NAMES+=("$RUN_NAME")
        done
    done
done

N_RUNS=${#PIDS[@]}
log ""
log "  $N_RUNS runs launched. Waiting..."

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
        # Check if process failed by looking at log for errors
        if grep -q "Traceback\|Error\|FAILED" "$LOG_DIR/${RUN_NAMES[$i]}.log" 2>/dev/null; then
            log "  $LOG_DIR/${RUN_NAMES[$i]}.log"
            tail -5 "$LOG_DIR/${RUN_NAMES[$i]}.log" 2>/dev/null | while read line; do
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
    log "  W&B:     https://wandb.ai/$WANDB_PROJECT"
fi
