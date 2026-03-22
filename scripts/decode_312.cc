#include "core/config.h"
#include "core/lenia.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>

struct CellData {
    int nz, nr, nc;
    std::vector<float> data;
};

CellData load_cells(const char* path) {
    CellData cd;
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&cd.nz), 4);
    fin.read(reinterpret_cast<char*>(&cd.nr), 4);
    fin.read(reinterpret_cast<char*>(&cd.nc), 4);
    cd.data.resize(cd.nz * cd.nr * cd.nc);
    fin.read(reinterpret_cast<char*>(cd.data.data()), cd.data.size() * sizeof(float));
    return cd;
}

void place_cells(Grid<3>& grid, int N, const CellData& cd) {
    grid.fill(0.0f);
    int od = (N - cd.nz) / 2;
    int orow = (N - cd.nr) / 2;
    int oc = (N - cd.nc) / 2;
    for (int d = 0; d < cd.nz; ++d)
        for (int r = 0; r < cd.nr; ++r)
            for (int c = 0; c < cd.nc; ++c) {
                float v = cd.data[d * cd.nr * cd.nc + r * cd.nc + c];
                if (v > 0.0f)
                    grid.at(od + d, orow + r, oc + c) = v;
            }
}

void test(const char* label, const char* cells_path,
          int R0, const char* b0_str, float mu0, float s0,
          int R1, const char* b1_str, float mu1, float s1,
          int N, int steps) {
    auto cd = load_cells(cells_path);
    std::printf("\n=== %s N=%d cells=(%d,%d,%d) ===\n", label, N, cd.nz, cd.nr, cd.nc);

    LeniaConfig cfg;
    cfg.dimension = 3; cfg.grid_size = N; cfg.num_threads = 4; cfg.T = 10;

    // Parse beta strings (simplified: handle "1", "1,3/4,1/12", "1,1,0", etc.)
    auto parse_beta = [](const char* s) -> std::vector<float> {
        std::vector<float> v;
        const char* p = s;
        while (*p) {
            // Try to parse fraction a/b or float
            char* end;
            float num = std::strtof(p, &end);
            if (end > p && *end == '/') {
                float den = std::strtof(end + 1, &end);
                v.push_back(num / den);
            } else {
                v.push_back(num);
            }
            p = end;
            if (*p == ',') ++p;
        }
        return v;
    };

    cfg.kernel = {R0, parse_beta(b0_str), 4, KernelCoreFunc::Exponential};
    cfg.growth = {mu0, s0};
    KernelGrowthPair k1;
    k1.kernel = {R1, parse_beta(b1_str), 4, KernelCoreFunc::Exponential};
    k1.growth = {mu1, s1};
    cfg.extra_kernels.push_back(k1);

    Lenia lenia(cfg);
    place_cells(lenia.grid_3d_mut(), N, cd);

    double m0 = lenia.total_mass();
    double mmin = m0, mmax = m0;

    for (int s = 1; s <= steps; ++s) {
        lenia.step();
        double m = lenia.total_mass();
        if (m < mmin) mmin = m;
        if (m > mmax) mmax = m;
        if (s <= 5 || s % 200 == 0)
            std::printf("  step %4d  mass %9.1f  ratio %.4f%s\n",
                        s, m, m/m0,
                        m/m0 < 0.05 ? " DEAD" : m/m0 > 3.0 ? " EXPLODING" : "");
        if (m < m0 * 0.01) { std::printf("  DEAD at step %d\n", s); return; }
    }
    double r = lenia.total_mass() / m0;
    std::printf("  range [%.3f, %.3f]  final=%.4f  %s\n",
                mmin/m0, mmax/m0, r,
                r > 3.0 ? "EXPLODING" : r < 0.05 ? "DEAD" : "ALIVE");
}

int main() {
    // Fish: R=20, b="1", m=0.13 s=0.025 / R=20 b="1,3/4,1/12" m=0.24 s=0.046
    test("fish", "/tmp/fish_cells.bin",
         20, "1", 0.13f, 0.025f,
         20, "1,0.75,0.0833", 0.24f, 0.046f,
         64, 1000);
    test("fish", "/tmp/fish_cells.bin",
         20, "1", 0.13f, 0.025f,
         20, "1,0.75,0.0833", 0.24f, 0.046f,
         128, 1000);

    // Divide: R=16, same mu/s as fish
    test("divide", "/tmp/divide_cells.bin",
         16, "1", 0.13f, 0.025f,
         16, "1,0.75,0.0833", 0.24f, 0.046f,
         64, 1000);

    // Embryo: R=15, b="1" m=0.1 s=0.0239 / R=15 b="1,1,0" m=0.235 s=0.051
    test("embryo", "/tmp/embryo_cells.bin",
         15, "1", 0.1f, 0.0239f,
         15, "1,1,0", 0.235f, 0.051f,
         64, 1000);

    return 0;
}
