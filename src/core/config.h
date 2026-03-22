#pragma once

#include "lenia.h"

#include <string>

// Parse command-line arguments into a LeniaConfig.
// Supported flags:
//   --dim 2|3          Dimension (default: 2)
//   --size N           Grid side length (default: 256)
//   --radius R         Kernel radius (default: 13)
//   --mu F             Growth center (default: 0.15)
//   --sigma F          Growth width (default: 0.015)
//   --beta-center F    Kernel ring center (default: 0.5)
//   --beta-width F     Kernel ring width (default: 0.15)
//   --T N              Time resolution, dt=1/T (default: 10)
//   --threads N        Number of OpenMP threads (default: 1)
//   --seed N           Random seed (default: 42)
//   --beta "1,0.75"    Comma-separated beta weights for primary kernel
//   --preset P         Load a 312 preset: fish|divide|embryo
//   --pattern P        Pattern: orbium|random|blob (default: orbium)
//   --steps N          Number of simulation steps (default: 1000)
//   --benchmark        Run in benchmark mode (no rendering)
//   --headless         Run without display
//   --help             Print usage and exit
LeniaConfig parse_args(int argc, char* argv[]);

// Apply a Chan 312-format preset (3D, 1 channel, 2 kernels)
void apply_312_preset(LeniaConfig& cfg, const std::string& name);

// Load configuration from a simple key=value text file.
// Lines starting with '#' are comments. Blank lines are skipped.
// Unknown keys are ignored. Example file:
//
//   # Lenia simulation config
//   dim=2
//   size=256
//   radius=13
//   mu=0.15
//   sigma=0.015
//   T=10
//   threads=4
//   seed=42
LeniaConfig load_config_file(const std::string& path);

// Print current configuration to stdout.
void print_config(const LeniaConfig& config);
