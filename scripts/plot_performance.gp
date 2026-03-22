# Gnuplot script for Lenia performance analysis
# Usage: gnuplot scripts/plot_performance.gp

set terminal pngcairo size 1200,800 enhanced font 'Arial,12'
set style data linespoints
set style line 1 lc rgb '#0060ad' lw 2 pt 7 ps 1.5
set style line 2 lc rgb '#dd181f' lw 2 pt 5 ps 1.5
set style line 3 lc rgb '#00a651' lw 2 pt 9 ps 1.5
set style line 4 lc rgb '#ff8c00' lw 2 pt 13 ps 1.5
set grid

# --- Thread Scaling (2D) ---
set output 'thesis/figures/scaling_2d.png'
set title 'Thread Scaling — Lenia 2D'
set xlabel 'Number of Threads'
set ylabel 'Steps per Second'
set key top left

# Expected CSV format: dim,grid_size,threads,iterations,total_ms,avg_ms,steps_per_sec
plot 'benchmarks/bench_2d_128.csv' using 3:7 title '128x128' ls 1, \
     'benchmarks/bench_2d_256.csv' using 3:7 title '256x256' ls 2, \
     'benchmarks/bench_2d_512.csv' using 3:7 title '512x512' ls 3

# --- Thread Scaling (3D) ---
set output 'thesis/figures/scaling_3d.png'
set title 'Thread Scaling — Lenia 3D'
set xlabel 'Number of Threads'
set ylabel 'Steps per Second'

plot 'benchmarks/bench_3d_32.csv' using 3:7 title '32^3' ls 1, \
     'benchmarks/bench_3d_64.csv' using 3:7 title '64^3' ls 2, \
     'benchmarks/bench_3d_128.csv' using 3:7 title '128^3' ls 3

# --- Speedup (2D) ---
set output 'thesis/figures/speedup_2d.png'
set title 'Speedup — Lenia 2D (512x512)'
set xlabel 'Number of Threads'
set ylabel 'Speedup'
set key top left

# Compute speedup relative to 1-thread baseline
# Assumes first row is 1 thread
T1 = 0
plot 'benchmarks/bench_2d_512.csv' using 3:(T1 = ($3==1 ? $7 : T1), $7 > 0 ? (T1/$7 > 0 ? $3*1.0 : 1) : 1) title 'Ideal' ls 4 dashtype 2, \
     '' using 3:($7 > 0 ? (T1/$7) : 1) title 'Measured' ls 2

# --- Speedup (3D) ---
set output 'thesis/figures/speedup_3d.png'
set title 'Speedup — Lenia 3D (128^3)'
set xlabel 'Number of Threads'
set ylabel 'Speedup'

plot 'benchmarks/bench_3d_128.csv' using 3:(T1 = ($3==1 ? $7 : T1), $7 > 0 ? (T1/$7 > 0 ? $3*1.0 : 1) : 1) title 'Ideal' ls 4 dashtype 2, \
     '' using 3:($7 > 0 ? (T1/$7) : 1) title 'Measured' ls 2

print "Plots saved to thesis/figures/"
