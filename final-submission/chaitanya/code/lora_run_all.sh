#!/bin/bash
# Run all 30 LoRA combos as isolated subprocesses.
# Each combo gets a fresh Python process, so memory is fully reclaimed between combos.
# Cached combos are skipped (the runner checks for tuned_*.npy before running).

set -u
PY="/Users/chaitanyaparwatkar/Desktop/CSCI535-SER-Final/my_code/.venv/bin/python"
RUNNER="/Users/chaitanyaparwatkar/Desktop/CSCI535-SER-Final/my_code/lora_run_one.py"
LOG_DIR="/tmp/lora_combos"
PROGRESS="/tmp/lora_combos_progress.txt"
mkdir -p "$LOG_DIR"

MODELS=("wav2vec2" "HuBERT" "Whisper" "WavLM" "MERT" "w2v-BERT")
# Order: smallest -> largest so first failures are obvious
DATASETS=("EmoDB" "SAVEE" "AESDD" "MESD" "RAVDESS")

mark() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$PROGRESS"; }

mark "=== LoRA combo runner started ==="
done_count=0
skip_count=0
fail_count=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        log="$LOG_DIR/${model}_${dataset}.log"
        mark "START $model / $dataset"
        "$PY" "$RUNNER" --model "$model" --dataset "$dataset" > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            # Check if it was a cache-hit or a real run
            if grep -q "CACHED" "$log"; then
                mark "CACHE $model / $dataset (skipped, already done)"
                skip_count=$((skip_count+1))
            else
                acc=$(grep "^Acc:" "$log" | tail -1 | sed 's/Acc: //')
                mark "DONE  $model / $dataset (acc=$acc)"
                done_count=$((done_count+1))
            fi
        else
            mark "FAIL  $model / $dataset (exit $rc), see $log"
            fail_count=$((fail_count+1))
        fi
    done
done

mark "=== Done: $done_count new, $skip_count cached, $fail_count failed ==="
