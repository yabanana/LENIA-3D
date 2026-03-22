# Gnuplot script for Lenia pattern dynamics
# Plots mass over time for different patterns
# Usage: gnuplot scripts/plot_patterns.gp

set terminal pngcairo size 1000,600 enhanced font 'Arial,12'
set grid

# --- Mass over time ---
set output 'thesis/figures/mass_dynamics.png'
set title 'Total Mass Over Time — Lenia 2D Patterns'
set xlabel 'Iteration'
set ylabel 'Total Mass'
set key top right

# Expected CSV format: iteration,mass
plot 'patterns/orbium_mass.csv' using 1:2 with lines title 'Orbium' lw 2 lc rgb '#0060ad', \
     'patterns/geminium_mass.csv' using 1:2 with lines title 'Geminium' lw 2 lc rgb '#dd181f', \
     'patterns/random_mass.csv' using 1:2 with lines title 'Random Init' lw 2 lc rgb '#00a651'

print "Pattern dynamics plot saved to thesis/figures/mass_dynamics.png"
