// ===========================================================================
// 04_sparse_csr.cpp
// 稀疏矩阵的三元组 (triple) 与 CSR 表示、互相转换、稀疏转置、稀疏矩阵-向量乘。
//
// 编译运行:
//   g++ -std=c++17 -O2 04_sparse_csr.cpp -o 04_sparse_csr && ./04_sparse_csr
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------
struct Triple {           // 三元组: (行, 列, 值), 行主序有序
    int row, col;
    double val;
};

struct CSR {              // Compressed Sparse Row
    int rows = 0, cols = 0;
    std::vector<int> rowPtr;    // 长度 rows+1, rowPtr[i]..rowPtr[i+1] 是第 i 行的非零区间
    std::vector<int> colIdx;    // 长度 nnz, 每行内的列号 (升序)
    std::vector<double> vals;   // 长度 nnz
    int nnz() const { return (int)vals.size(); }
};

// 密集存储 (行主序) -> 用于对拍
using Dense = std::vector<double>;
Dense toDense(const CSR& A) {
    Dense M(std::size_t(A.rows) * A.cols, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k)
            M[std::size_t(i) * A.cols + A.colIdx[k]] = A.vals[k];
    return M;
}

// ---------------------------------------------------------------------------
// 三元组 -> CSR: 计数排序 (counting sort by row), O(nnz + rows)
// ---------------------------------------------------------------------------
CSR triplesToCSR(int rows, int cols, std::vector<Triple> t) {
    std::stable_sort(t.begin(), t.end(),
                     [](const Triple& a, const Triple& b) {
                         return a.row != b.row ? a.row < b.row : a.col < b.col;
                     });
    CSR A;
    A.rows = rows; A.cols = cols;
    A.rowPtr.assign(rows + 1, 0);
    for (const auto& e : t) A.rowPtr[e.row + 1]++;
    for (int i = 0; i < rows; ++i) A.rowPtr[i + 1] += A.rowPtr[i];   // 前缀和
    A.colIdx.resize(t.size());
    A.vals.resize(t.size());
    std::vector<int> next = A.rowPtr;   // 每行写入游标
    for (const auto& e : t) {
        int p = next[e.row]++;
        A.colIdx[p] = e.col;
        A.vals[p] = e.val;
    }
    return A;
}

// ---------------------------------------------------------------------------
// CSR -> 三元组
// ---------------------------------------------------------------------------
std::vector<Triple> csrToTriples(const CSR& A) {
    std::vector<Triple> t;
    t.reserve(A.nnz());
    for (int i = 0; i < A.rows; ++i)
        for (int k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k)
            t.push_back({i, A.colIdx[k], A.vals[k]});
    return t;
}

// ---------------------------------------------------------------------------
// 稀疏转置 (三元组版): 交换 row/col 后按 (row,col) 排序
// ---------------------------------------------------------------------------
std::vector<Triple> transposeTriples(std::vector<Triple> t) {
    for (auto& e : t) std::swap(e.row, e.col);
    std::stable_sort(t.begin(), t.end(),
                     [](const Triple& a, const Triple& b) {
                         return a.row != b.row ? a.row < b.row : a.col < b.col;
                     });
    return t;
}

// ---------------------------------------------------------------------------
// 稀疏转置 (CSR 快速转置 / counting sort)
//   colCount[c] = 第 c 列非零个数 (即 A^T 第 c 行的非零个数)
//   对列做前缀和得到 A^T 的 rowPtr, 再散射填入
//   复杂度 O(nnz + cols)
// ---------------------------------------------------------------------------
CSR transposeCSR(const CSR& A) {
    CSR T;
    T.rows = A.cols; T.cols = A.rows;
    T.rowPtr.assign(A.cols + 1, 0);
    for (int c : A.colIdx) T.rowPtr[c + 1]++;          // 统计每列
    for (int c = 0; c < A.cols; ++c) T.rowPtr[c + 1] += T.rowPtr[c];
    T.colIdx.resize(A.nnz());
    T.vals.resize(A.nnz());
    std::vector<int> next = T.rowPtr;
    for (int i = 0; i < A.rows; ++i)
        for (int k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k) {
            int c = A.colIdx[k];
            int p = next[c]++;
            T.colIdx[p] = i;        // 原来的行号变成列号
            T.vals[p] = A.vals[k];
        }
    return T;   // 结果自然按 (行=原列, 列=原行) 升序
}

// ---------------------------------------------------------------------------
// 稀疏矩阵 - 稠密向量乘: y = A x, 复杂度 O(nnz), 远优于 O(rows*cols)
// ---------------------------------------------------------------------------
std::vector<double> spmv(const CSR& A, const std::vector<double>& x) {
    std::vector<double> y(A.rows, 0.0);
    for (int i = 0; i < A.rows; ++i) {
        double s = 0.0;
        for (int k = A.rowPtr[i]; k < A.rowPtr[i + 1]; ++k)
            s += A.vals[k] * x[A.colIdx[k]];
        y[i] = s;
    }
    return y;
}

// 密集矩阵-向量乘, 仅用于对拍
std::vector<double> denseMatVec(const Dense& M, int r, int c,
                                const std::vector<double>& x) {
    std::vector<double> y(r, 0.0);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            y[i] += M[std::size_t(i) * c + j] * x[j];
    return y;
}

// ---------------------------------------------------------------------------
// 打印 & 自测
// ---------------------------------------------------------------------------
void printCSR(const CSR& A, const char* name) {
    std::cout << name << " (" << A.rows << "x" << A.cols << ", nnz=" << A.nnz() << ")\n";
    std::cout << "  rowPtr = [";
    for (std::size_t i = 0; i < A.rowPtr.size(); ++i)
        std::cout << A.rowPtr[i] << (i + 1 < A.rowPtr.size() ? ", " : "]\n");
    std::cout << "  colIdx = [";
    for (std::size_t i = 0; i < A.colIdx.size(); ++i)
        std::cout << A.colIdx[i] << (i + 1 < A.colIdx.size() ? ", " : "]\n");
    std::cout << "  vals   = [";
    for (std::size_t i = 0; i < A.vals.size(); ++i)
        std::cout << A.vals[i] << (i + 1 < A.vals.size() ? ", " : "]\n");
}

int main() {
    // 4x5 稀疏矩阵, 7 个非零元
    // [ 1  0  0  2  0 ]
    // [ 0  3  0  0  0 ]
    // [ 0  0  0  4  5 ]
    // [ 6  0  7  0  0 ]
    const int R = 4, C = 5;
    std::vector<Triple> t = {
        {0,0,1},{0,3,2},{1,1,3},{2,3,4},{2,4,5},{3,0,6},{3,2,7}
    };
    std::cout << std::fixed << std::setprecision(2);

    CSR A = triplesToCSR(R, C, t);
    std::cout << "===== 三元组 -> CSR =====\n";
    printCSR(A, "A");

    std::cout << "\n===== CSR -> 三元组 (往返) =====\n";
    auto back = csrToTriples(A);
    std::sort(back.begin(), back.end(), [](const Triple& a, const Triple& b) {
        return a.row != b.row ? a.row < b.row : a.col < b.col; });
    std::sort(t.begin(), t.end(), [](const Triple& a, const Triple& b) {
        return a.row != b.row ? a.row < b.row : a.col < b.col; });
    assert(back.size() == t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        assert(back[i].row == t[i].row && back[i].col == t[i].col &&
               back[i].val == t[i].val);
    }
    std::cout << "[OK] CSR -> 三元组 与原始三元组一致\n";

    std::cout << "\n===== 稀疏转置 (CSR 快速转置) =====\n";
    CSR At = transposeCSR(A);
    printCSR(At, "A^T");
    // 用三元组转置对拍
    auto atT = transposeTriples(csrToTriples(A));
    auto atC = csrToTriples(At);
    assert(atT.size() == atC.size());
    for (std::size_t i = 0; i < atT.size(); ++i)
        assert(atT[i].row == atC[i].row && atT[i].col == atC[i].col &&
               atT[i].val == atC[i].val);
    std::cout << "[OK] CSR 快速转置 与 三元组转置 结果一致\n";

    std::cout << "\n===== 稀疏矩阵-向量乘 y = A x =====\n";
    std::vector<double> x = {1, 2, 3, 4, 5};
    auto ySparse = spmv(A, x);
    auto yDense = denseMatVec(toDense(A), R, C, x);
    for (int i = 0; i < R; ++i) {
        std::cout << "  y[" << i << "] sparse=" << ySparse[i]
                  << "  dense=" << yDense[i] << "\n";
        assert(std::abs(ySparse[i] - yDense[i]) < 1e-9);
    }
    std::cout << "[OK] 稀疏 SpMV 与稠密结果一致 (复杂度 O(nnz) 而非 O(rows*cols))\n";
    std::cout << "\n全部测试通过。\n";
    return 0;
}
