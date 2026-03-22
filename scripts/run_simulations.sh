#!/bin/bash
# Run headless simulations for all 6 multi-kernel patterns.
# Collects mass logs and final grid snapshots for figure generation.
#
# Usage: ./scripts/run_simulations.sh [BUILD_DIR]
#   BUILD_DIR defaults to build-full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${1:-${PROJECT_DIR}/build-full}"
LENIA="${BUILD_DIR}/lenia_viz"

if [ ! -x "$LENIA" ]; then
    echo "Error: $LENIA not found or not executable" >&2
    echo "Build first: cmake --build $BUILD_DIR -j\$(nproc)" >&2
    exit 1
fi

DATA_DIR="${PROJECT_DIR}/data"
CELLS_DIR="${DATA_DIR}/cells"
mkdir -p "$DATA_DIR"

PRESETS=(fish butterfly ghost divide exotic protoeel)
GRID_SIZE=128
STEPS=1000
THREADS=4

echo "=== Running headless simulations ==="
echo "  Grid: ${GRID_SIZE}^3, Steps: $STEPS, Threads: $THREADS"
echo ""

for preset in "${PRESETS[@]}"; do
    cells="${CELLS_DIR}/${preset}_cells.bin"
    if [ ! -f "$cells" ]; then
        echo "Warning: $cells not found, skipping $preset" >&2
        continue
    fi

    echo "--- $preset ---"
    "$LENIA" --headless \
        --preset "$preset" \
        --cells "$cells" \
        --size "$GRID_SIZE" \
        --steps "$STEPS" \
        --threads "$THREADS" \
        --mass-log "${DATA_DIR}/${preset}_mass.csv" \
        --snapshot "${DATA_DIR}/${preset}_state.bin"
    echo ""
done

echo "=== All simulations complete ==="
echo "Mass logs: ${DATA_DIR}/*_mass.csv"
echo "Snapshots: ${DATA_DIR}/*_state.bin"
