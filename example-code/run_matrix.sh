#!/bin/bash
set -euo pipefail

#run_matrix.sh
CPU_LIST=(1 2 4 8)
MEM_LIST=(2 4 8 16)
# CPU_LIST=(1)
# MEM_LIST=(2)
GPU_LIST=("off")

for CPU in "${CPU_LIST[@]}"; do
  for MEM in "${MEM_LIST[@]}"; do
    for GPU in "${GPU_LIST[@]}"; do
      ./run_case.sh "$CPU" "$MEM" "$GPU"
    done
  done
done