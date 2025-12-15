#!/bin/bash
# Run baseline experiments on 50-query dev set with CURRENT fixed code
# Usage: Run in tmux for persistence

set -e
cd /home/yxiao/Personal_Folder/Research/Multimodal-RAG-Impact-Assessment

# Activate conda environment
source ~/mambaforge/etc/profile.d/conda.sh
conda activate harvey-rag

# Force CPU to avoid GPU issues
export CUDA_VISIBLE_DEVICES=""

echo "=============================================="
echo "DEV SET BASELINE EXPERIMENTS (50 queries)"
echo "Using CURRENT fixed code"
echo "Started: $(date)"
echo "=============================================="

# 1. Text-Only Baseline
echo ""
echo "[1/2] Running Text-Only baseline..."
echo "Started: $(date)"
python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/dev_baseline_text_only.json \
  --name dev_baseline_text_only \
  --no_visual --no_captions

echo "Completed: $(date)"

# 2. Text+Caption (with temporal context improvements)
echo ""
echo "[2/2] Running Text+Caption (improved)..."
echo "Started: $(date)"
python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/dev_baseline_text_caption.json \
  --name dev_baseline_text_caption \
  --no_visual

echo "Completed: $(date)"

# Summary
echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results:"
for f in data/experiments/dev_baseline_*.json; do
  if [ -f "$f" ]; then
    name=$(basename "$f" .json)
    extent=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metadata',{}).get('summary_stats',{}).get('extent_mae','?'))")
    damage=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metadata',{}).get('summary_stats',{}).get('damage_mae','?'))")
    echo "$name: Extent MAE=$extent%, Damage MAE=$damage%"
  fi
done
echo ""
echo "Compare with: python scripts/compare_experiments.py data/experiments/dev_baseline_*.json"


# 3. Full Multimodal (Text + Visual Analysis)
echo ""
echo "[3/3] Running Multimodal (Text + Visual)..."
echo "Started: $(date)"
python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/dev_baseline_multimodal.json \
  --name dev_baseline_multimodal

echo "Completed: $(date)"
