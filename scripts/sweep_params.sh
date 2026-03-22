#!/bin/bash
# Parameter sweep script for Lenia pattern discovery
# Scans mu/sigma parameter space and records which configurations
# produce interesting (non-trivial) dynamics
# Usage: ./scripts/sweep_params.sh [build_dir]

set -euo pipefail

BUILD_DIR="${1:-build}"
BENCH_BIN="${BUILD_DIR}/lenia_bench"
OUTPUT_DIR="patterns/sweep"
ITERATIONS=500
SIZE_2D=256
SIZE_3D=64

if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Build the project first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Lenia Parameter Sweep"
echo "============================================"
echo ""

# 2D sweep
echo "dimension,grid_size,mu,sigma,radius,T,final_mass,classification" \
    > "${OUTPUT_DIR}/sweep_2d.csv"

for MU in 0.10 0.12 0.14 0.15 0.16 0.18 0.20 0.25 0.30; do
    for SIGMA in 0.010 0.012 0.014 0.015 0.016 0.018 0.020 0.025; do
        echo -n "  2D mu=${MU} sigma=${SIGMA}... "
        RESULT=$("$BENCH_BIN" --dim 2 --size $SIZE_2D \
            --mu "$MU" --sigma "$SIGMA" \
            --iterations $ITERATIONS --threads 4 --seed 42 2>&1 | tail -1)
        echo "$RESULT"
        echo "2,$SIZE_2D,$MU,$SIGMA,13,10,$RESULT,unknown" \
            >> "${OUTPUT_DIR}/sweep_2d.csv"
    done
done

echo ""
echo "Sweep results in ${OUTPUT_DIR}/"
echo "Interesting patterns have final_mass > 0 and < total_cells"
