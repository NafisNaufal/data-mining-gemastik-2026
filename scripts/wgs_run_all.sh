#!/usr/bin/env bash
# Driver: run wgs_one.sh over all accessions in parallel inside the tbprof env.
# Usage: wgs_run_all.sh   (reads accessions.tsv; CONC concurrent samples, TBP_THREADS each)
set -uo pipefail
ROOT="/mnt/nas-hpg9/adhi/projects/gemastik/data-mining/data/indonesia_wgs"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONC="${CONC:-12}"
export TBP_THREADS="${TBP_THREADS:-4}"
source /mnt/extended-home/adhi/miniconda3/etc/profile.d/conda.sh
conda activate tbprof
echo "TB-Profiler $(tb-profiler version 2>&1 | head -1) | CONC=$CONC threads=$TBP_THREADS | $(date -Is)"
tail -n +2 "$ROOT/accessions.tsv" | cut -f1 \
  | xargs -P "$CONC" -I{} bash "$HERE/wgs_one.sh" {}
echo "=== all samples attempted; collating $(date -Is) ==="
cd "$ROOT/results"
tb-profiler collate -d "$ROOT/results" -p "$ROOT/indonesia_collate" 2>&1 | tail -5
echo "DONE $(date -Is). Results: $(ls $ROOT/results/*.results.json 2>/dev/null | wc -l) samples"
