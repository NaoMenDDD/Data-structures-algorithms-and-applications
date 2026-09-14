// ===========================================================================
// 04_matmul_cache_order.cpp
// 矩阵乘 C = A * B (方阵, 行主序连续存储), 对比三种循环次序的 cache 行为与耗时:
//   i-j-k : 对 B 的访问跨列 (stride = n), 最差
//   i-k-j : 对 B、C 都顺序访问, 最好
//   j-k-i : 对 C、A 都跨行 (stride = n), 也差
//
// 编译运行:
//   g++ -std=c++17 -O2 04_matmul_cache_order.cpp -o 04_matmul_cache_order
//   ./04_matmul_cache_order
// ===========================================================================
#include <chrono>
#include <cmath>        // std::abs
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Mat = std::vector<double>;

// volatile 汇总口, 防止 -O2 把"结果没人用"的整个循环删掉
static volatile double g_sink = 0.0;

inline std::size_t at(int i, int j, int n) {
    return std::size_t(i) * std::size_t(n) + std::size_t(j);
}

// ---------------------------- 三种循环次序 ----------------------------
// i-j-k: 内层 k 上 A[i][k] 顺序, B[k][j] 跨步 n, C[i][j] 驻留寄存器
void matmul_ijk(const Mat& A, const Mat& B, Mat& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += A[at(i, k, n)] * B[at(k, j, n)];
            C[at(i, j, n)] = s;
        }
}

// i-k-j: 内层 j 上 C[i][j] 与 B[k][j] 都顺序, A[i][k] 是标量; 最佳
void matmul_ikj(const Mat& A, const Mat& B, Mat& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const double a = A[at(i, k, n)];
            for (int j = 0; j < n; ++j)
                C[at(i, j, n)] += a * B[at(k, j, n)];
        }
}

// j-k-i: 内层 i 上 C[i][j] 与 A[i][k] 都跨步 n; 差
void matmul_jki(const Mat& A, const Mat& B, Mat& C, int n) {
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            const double b = B[at(k, j, n)];
            for (int i = 0; i < n; ++i)
                C[at(i, j, n)] += A[at(i, k, n)] * b;
        }
}

static void fillRandom(Mat& M, unsigned seed) {
    std::size_t s = seed;
    for (auto& x : M) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // LCG
        x = double((s >> 33) & 0xFFFF) / 65535.0;                  // [0,1)
    }
}

static double checksum(const Mat& M) {
    double s = 0.0;
    for (double x : M) s += x;
    return s;
}

static double timeIt(void (*fn)(const Mat&, const Mat&, Mat&, int),
                     const Mat& A, const Mat& B, Mat& C, int n) {
    C.assign(std::size_t(n) * n, 0.0);
    auto t0 = Clock::now();
    fn(A, B, C, n);
    auto t1 = Clock::now();
    g_sink += checksum(C);   // 触碰结果, 阻止优化器删除
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const int n = 768;   // 768^3 * 2 ~= 9.1e8 FLOP, 各次序耗时差异明显
    std::cout << "n = " << n << " (double, 行主序连续存储)\n\n";

    Mat A(std::size_t(n) * n), B(std::size_t(n) * n), C(std::size_t(n) * n);
    fillRandom(A, 12345);
    fillRandom(B, 67890);

    // 预热缓存并让 CPU 跑到高频
    { Mat W(std::size_t(n) * n, 0.0); matmul_ikj(A, B, W, n); g_sink += checksum(W); }

    // 先算基准 (ikj) 的结果用于正确性比对
    Mat ref(std::size_t(n) * n, 0.0);
    matmul_ikj(A, B, ref, n);

    struct Row { const char* name; void (*fn)(const Mat&, const Mat&, Mat&, int); };
    Row rows[] = {
        {"i-j-k (较差)", matmul_ijk},   // 内层 k：B 按列走，stride = n
        {"i-k-j (最佳)", matmul_ikj},   // 内层 j：B、C 都顺序
        {"j-k-i (最差)", matmul_jki},   // 内层 i：C、A 都跨行
    };

    // 先把 ikj 单独测一次作为「相对基准」，否则第一行会除以尚未赋值的 0。
    const double t_ikj = timeIt(matmul_ikj, A, B, C, n);

    std::cout << std::left << std::setw(16) << "循环次序"
              << std::right << std::setw(12) << "耗时(s)"
              << std::setw(14) << "相对 ikj" << "   正确?\n";
    for (auto& r : rows) {
        double t = timeIt(r.fn, A, B, C, n);
        // 与 ikj 参考解比对 (浮点允许 1e-6 相对误差)
        bool ok = true;
        for (std::size_t p = 0; p < ref.size(); p += 9973)
            if (std::abs(C[p] - ref[p]) > 1e-6 * (1.0 + std::abs(ref[p]))) { ok = false; break; }
        std::cout << std::left << std::setw(16) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << t
                  << std::setw(13) << (t / t_ikj) << "x   "
                  << (ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << "\n(汇总和为 " << g_sink << ", 仅用于阻止优化)\n";
    std::cout << "结论: 三种次序的浮点运算次数完全相同 (都是 2n^3),\n"
                 "      差异纯粹来自内存访问的局部性 (cache line 命中率)。\n";
    return 0;
}
