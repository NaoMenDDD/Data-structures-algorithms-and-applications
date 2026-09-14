// ===========================================================================
// 04_matrix_rowcol.cpp
// 行主序 (row-major) 与列主序 (column-major) 的内存映射与地址计算验证。
//
// 编译运行:
//   g++ -std=c++17 -O2 04_matrix_rowcol.cpp -o 04_matrix_rowcol && ./04_matrix_rowcol
// ===========================================================================
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// 1. 通用 n 维地址映射 (0-based)
// ---------------------------------------------------------------------------
// row-major (C / C++ / PyTorch / NumPy 默认):
//   offset = ((i0*d1 + i1)*d2 + i2)*d3 + ... + i_{n-1}
//   等价地: stride[k] = d_{k+1} * d_{k+2} * ... * d_{n-1},  offset = sum_k i_k*stride[k]
std::size_t offsetRowMajor(const std::vector<std::size_t>& dims,
                           const std::vector<std::size_t>& idx) {
    std::size_t off = 0;
    for (std::size_t k = 0; k < dims.size(); ++k)
        off = off * dims[k] + idx[k];   // 霍纳 (Horner) 法则展开
    return off;
}

// column-major (Fortran / MATLAB / BLAS / Julia 默认):
//   offset = i0 + d0*(i1 + d1*(i2 + ...))
//   stride[k] = d0 * d1 * ... * d_{k-1}
std::size_t offsetColMajor(const std::vector<std::size_t>& dims,
                           const std::vector<std::size_t>& idx) {
    std::size_t off = 0;
    for (std::size_t k = dims.size(); k-- > 0;)   // 从最后一维往回卷
        off = off * dims[k] + idx[k];
    return off;
}

// 行主序的 stride 向量: stride[n-1] = 1, stride[k] = stride[k+1]*dims[k+1]
std::vector<std::size_t> stridesRowMajor(const std::vector<std::size_t>& dims) {
    std::vector<std::size_t> s(dims.size(), 1);
    for (std::size_t k = dims.size() - 1; k-- > 0;)
        s[k] = s[k + 1] * dims[k + 1];
    return s;
}

// 列主序的 stride 向量: stride[0] = 1, stride[k] = stride[k-1]*dims[k-1]
std::vector<std::size_t> stridesColMajor(const std::vector<std::size_t>& dims) {
    std::vector<std::size_t> s(dims.size(), 1);
    for (std::size_t k = 1; k < dims.size(); ++k)
        s[k] = s[k - 1] * dims[k - 1];
    return s;
}

// 用 stride 计算偏移, 与上面的霍纳式应完全一致 (验证两种定义等价)
std::size_t offsetByStride(const std::vector<std::size_t>& stride,
                           const std::vector<std::size_t>& idx) {
    std::size_t off = 0;
    for (std::size_t k = 0; k < idx.size(); ++k)
        off += idx[k] * stride[k];
    return off;
}

void printVec(const std::string& name, const std::vector<std::size_t>& v) {
    std::cout << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << v[i] << (i + 1 < v.size() ? ", " : "");
    std::cout << "]\n";
}

// ---------------------------------------------------------------------------
// 2. 二维: 把 3x4 矩阵分别按行主序/列主序装进一维缓冲区并回读
// ---------------------------------------------------------------------------
void demo2D() {
    std::cout << "===== 2D: 3 行 x 4 列 =====\n";
    const std::size_t R = 3, C = 4;
    // 逻辑矩阵 M[i][j] = i*10 + j
    std::cout << "逻辑矩阵 M (i=行, j=列):\n";
    for (std::size_t i = 0; i < R; ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < C; ++j)
            std::cout << std::setw(4) << i * 10 + j;
        std::cout << "\n";
    }

    // --- 行主序物理布局 ---
    std::vector<int> rowMajor(R * C);
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            rowMajor[offsetRowMajor({R, C}, {i, j})] = int(i * 10 + j);
    std::cout << "行主序物理内存: ";
    for (int x : rowMajor) std::cout << std::setw(4) << x;
    std::cout << "   (每行 4 个连续)\n";

    // --- 列主序物理布局 ---
    std::vector<int> colMajor(R * C);
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            colMajor[offsetColMajor({R, C}, {i, j})] = int(i * 10 + j);
    std::cout << "列主序物理内存: ";
    for (int x : colMajor) std::cout << std::setw(4) << x;
    std::cout << "   (每列 3 个连续)\n";

    // 验证: M[1][2] 在两个布局里都取得到, 但物理偏移不同
    std::size_t oR = offsetRowMajor({R, C}, {1, 2});
    std::size_t oC = offsetColMajor({R, C}, {1, 2});
    std::cout << "M[1][2] = 12, 行主序偏移 = " << oR
              << ", 列主序偏移 = " << oC << "\n";
    assert(rowMajor[oR] == 12 && colMajor[oC] == 12);
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 3. 三维: 用 PyTorch 风格的 stride 观察 NCHW 与 NHWC
// ---------------------------------------------------------------------------
void demo3D() {
    std::cout << "===== 3D: PyTorch 风格 stride, 形状 N x C x H x W =====\n";
    // 4 维更贴近张量; 这里用 N, C, HW 三维做示意:
    // shape = (N=2, C=3, L=4)
    std::vector<std::size_t> dims = {2, 3, 4};
    auto sRM = stridesRowMajor(dims);
    auto sCM = stridesColMajor(dims);
    printVec("shape (N,C,L)          ", dims);
    printVec("row-major stride (C,CHW)", sRM);
    printVec("col-major stride       ", sCM);
    std::cout << "解释: row-major 下最后一维 L 的 stride=1 (最内层连续),\n"
                 "      col-major 下第一维 N 的 stride=1。\n\n";

    // 用霍纳式与 stride 式互相验证: 遍历所有下标
    std::size_t total = 1;
    for (auto d : dims) total *= d;
    for (std::size_t flat = 0; flat < total; ++flat) {
        // 由 flat 反解下标 (row-major 反解)
        std::vector<std::size_t> idx(3);
        std::size_t t = flat;
        for (std::size_t k = 3; k-- > 0;) { idx[k] = t % dims[k]; t /= dims[k]; }
        assert(offsetRowMajor(dims, idx) == offsetByStride(sRM, idx));
        assert(offsetColMajor(dims, idx) == offsetByStride(sCM, idx));
    }
    std::cout << "[OK] " << total << " 个下标: 霍纳式与 stride 点积结果完全一致\n\n";
}

// ---------------------------------------------------------------------------
// 4. 1-based 与 0-based 换算: "第 k 行第 j 列"
// ---------------------------------------------------------------------------
void demoIndexBase() {
    std::cout << "===== 1-based 与 0-based =====\n";
    const std::size_t R = 5, C = 6;
    // 数学书上常写 A(k, j) 表示第 k 行第 j 列, k,j 从 1 开始。
    // 转成 0-based 是 (k-1, j-1), 行主序偏移 = (k-1)*C + (j-1)。
    for (std::size_t k = 1; k <= R; ++k) {
        for (std::size_t j = 1; j <= C; ++j) {
            std::size_t off = (k - 1) * C + (j - 1);
            std::size_t off0 = offsetRowMajor({R, C}, {k - 1, j - 1});
            assert(off == off0);
        }
    }
    std::cout << "A(3,5) (1-based, 5 行 6 列) -> 0-based (2,4) -> 偏移 "
              << (3 - 1) * C + (5 - 1) << "\n";
    std::cout << "[OK] 1-based -> 0-based -> 行主序偏移 换算全部正确\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    demo2D();
    demo3D();
    demoIndexBase();
    std::cout << "全部测试通过。\n";
    return 0;
}
