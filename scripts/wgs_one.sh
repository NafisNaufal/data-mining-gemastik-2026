#!/usr/bin/env bash
# Process ONE SRA run: download -> TB-Profiler -> cleanup. Resumable & space-frugal.
# Usage: wgs_one.sh <SRR_accession>
# Env: must be run inside the `tbprof` conda env. Threads via TBP_THREADS (default 4).
set -uo pipefail
RUN="$1"
ROOT="/mnt/nas-hpg9/adhi/projects/gemastik/data-mining/data/indonesia_wgs"
RES="$ROOT/results"
FQ="$ROOT/fastq/$RUN"
SRA="$ROOT/sra/$RUN"
LOG="$ROOT/logs/$RUN.log"
T="${TBP_THREADS:-4}"
mkdir -p "$RES" "$ROOT/logs"

# Resume: skip if TB-Profiler result already exists.
if [ -f "$RES/$RUN.results.json" ]; then echo "[$RUN] done (skip)"; exit 0; fi

{
  echo "=== $RUN start $(date -Is) ==="
  mkdir -p "$FQ" "$SRA"
  # 1) download .sra
  if ! prefetch "$RUN" -O "$SRA" --max-size 30g; then echo "[$RUN] prefetch FAIL"; rm -rf "$FQ" "$SRA"; exit 11; fi
  # 2) extract fastq
  SRAFILE=$(find "$SRA" -name "$RUN.sra" -o -name "$RUN.sralite" | head -1)
  if ! fasterq-dump "${SRAFILE:-$RUN}" -O "$FQ" --split-files -e "$T" -t "$FQ"; then echo "[$RUN] fasterq FAIL"; rm -rf "$FQ" "$SRA"; exit 12; fi
  rm -rf "$SRA"
  # 3) detect paired vs single
  R1="$FQ/${RUN}_1.fastq"; R2="$FQ/${RUN}_2.fastq"; RS="$FQ/${RUN}.fastq"
  if [ -f "$R1" ] && [ -f "$R2" ]; then
     ARGS="-1 $R1 -2 $R2"
  elif [ -f "$RS" ]; then
     ARGS="-1 $RS"
  elif [ -f "$R1" ]; then
     ARGS="-1 $R1"
  else echo "[$RUN] no fastq produced"; rm -rf "$FQ"; exit 13; fi
  # 4) TB-Profiler (writes to <dir>/results/, <dir>/bam/, <dir>/vcf/)
  if ! tb-profiler profile $ARGS -p "$RUN" -d "$ROOT" --threads "$T" --txt --csv; then
     echo "[$RUN] tb-profiler FAIL"; rm -rf "$FQ"; exit 14; fi
  # 5) cleanup fastq + this sample's bam/vcf to save space (keep results json/txt/csv).
  #    Per-sample (not whole dir) so parallel samples are not disturbed.
  rm -rf "$FQ" "$ROOT/bam/$RUN"* "$ROOT/vcf/$RUN"* 2>/dev/null
  echo "=== $RUN OK $(date -Is) ==="
} >>"$LOG" 2>&1
ec=$?
[ -f "$RES/$RUN.results.json" ] && { echo "[$RUN] OK"; exit 0; } || { echo "[$RUN] FAILED (see $LOG)"; exit $ec; }
