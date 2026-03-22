#!/bin/bash
# Lenia Benchmark Script
# Runs performance benchmarks for 2D and 3D Lenia simulations
# Usage: ./scripts/benchmark.sh [build_dir]

set -euo pipefail

BUILD_DIR="${1:-build}"
BENCH_BIN="${BUILD_DIR}/lenia_bench"
OUTPUT_DIR="benchmarks"
ITERATIONS=100

if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Build the project first."
    echo "  cmake -B build && cmake --build build"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Lenia Performance Benchmark Suite"
echo "============================================"
echo ""

# 2D Benchmarks
echo "--- 2D Benchmarks ---"
for SIZE in 128 256 512; do
    echo "Running 2D ${SIZE}x${SIZE}..."
    "$BENCH_BIN" --dim 2 --size "$SIZE" --threads 1,2,4,8 \
        --iterations "$ITERATIONS" --seed 42 \
        > "${OUTPUT_DIR}/bench_2d_${SIZE}.csv" 2>&1
done

echo ""
echo "--- 3D Benchmarks ---"
for SIZE in 32 64 128; do
    echo "Running 3D ${SIZE}^3..."
    "$BENCH_BIN" --dim 3 --size "$SIZE" --threads 1,2,4,8 \
        --iterations "$ITERATIONS" --seed 42 \
        > "${OUTPUT_DIR}/bench_3d_${SIZE}.csv" 2>&1
done

echo ""
echo "Benchmarks complete. Results in ${OUTPUT_DIR}/"
echo "Generate plots with: gnuplot scripts/plot_performance.gp"
