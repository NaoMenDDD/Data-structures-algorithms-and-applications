// ===========================================================================
// 14_divide_conquer.cpp
// 分治（Divide and Conquer）三件套演示：
//   1) 二分查找 lower_bound / upper_bound（统计比较次数，对照 O(log n)）
//   2) 快速幂（朴素 O(n) vs 二进制分解 O(log n)）+ 矩阵快速幂求斐波那契
//   3) Karatsuba 大整数乘法（4 次子乘法 -> 3 次）
//      用「十进制大整数」（vector<int> 存每位）实测乘法次数，
//      与教科书竖式 O(n^2) 对照，验证 O(n^{log2 3}) ≈ O(n^1.585)。
//
// 说明：大整数用 digit 向量表示（低位在前），故可以做到上千位而不溢出——
//       这正是需要「大整数乘法」的真实场景（RSA、π 计算、多项式乘法）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 14_divide_conquer.cpp -o 14_divide_conquer && ./14_divide_conquer
// ===========================================================================
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace dnc {

// ===========================================================================
// 1. 二分查找：在有序数组中找 target 的 [lo, hi) 区间
//    循环不变式 —— 答案始终落在 [lo, hi) 内
// ===========================================================================

// 返回第一个 >= target 的下标（等价 std::lower_bound）；比较次数记在 comps
int lowerBound(const std::vector<int>& a, int target, long long& comps) {
    comps = 0;
    int lo = 0, hi = static_cast<int>(a.size());          // 区间 [lo, hi)
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;               // 防溢出写法
        ++comps;
        if (a[static_cast<std::size_t>(mid)] < target) lo = mid + 1;  // mid 及其左侧排除
        else                                          hi = mid;       // mid 可能是答案，保留
    }
    return lo;                                            // lo == hi 即答案
}

// 返回第一个 > target 的下标（等价 std::upper_bound）
int upperBound(const std::vector<int>& a, int target, long long& comps) {
    comps = 0;
    int lo = 0, hi = static_cast<int>(a.size());
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        ++comps;
        if (a[static_cast<std::size_t>(mid)] <= target) lo = mid + 1;
        else                                           hi = mid;
    }
    return lo;
}

// 朴素线性查找，用于对比
int linearSearch(const std::vector<int>& a, int target, long long& comps) {
    comps = 0;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        ++comps;
        if (a[static_cast<std::size_t>(i)] == target) return i;
    }
    return -1;
}

// ===========================================================================
// 2. 快速幂（Exponentiation by Squaring）
//    a^n = (a^{n/2})^2            n 偶
//    a^n = a * (a^{(n-1)/2})^2    n 奇
//    每次规模减半 -> T(n) = T(n/2) + O(1) = O(log n)
// ===========================================================================

long long powNaive(long long a, long long n) {            // 朴素 O(n) 次乘法
    long long r = 1;
    for (long long i = 0; i < n; ++i) r *= a;
    return r;
}

long long powFast(long long a, long long n) {             // 递归二分
    if (n == 0) return 1;
    const long long half = powFast(a, n / 2);
    const long long sq = half * half;
    return (n & 1) ? sq * a : sq;                         // 奇数额外乘一次
}

long long powMod(long long a, long long n, long long mod) {  // 取模版（密码学常用）
    long long r = 1 % mod;
    a %= mod;
    while (n > 0) {
        if (n & 1) r = static_cast<long long>(static_cast<__int128>(r) * a % mod);
        a = static_cast<long long>(static_cast<__int128>(a) * a % mod);
        n >>= 1;
    }
    return r;
}

// ---- 矩阵快速幂：让「线性递推」也能 O(log n) ----
// 斐波那契满足 [F(n+1), F(n)]^T = M^n [F(1), F(0)]^T, M = [[1,1],[1,0]]
struct Mat2 {
    unsigned long long m[2][2];
};

Mat2 matMul(const Mat2& a, const Mat2& b) {
    Mat2 c{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            unsigned long long s = 0;
            for (int k = 0; k < 2; ++k) s += a.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                                              b.m[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
            c.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s;
        }
    return c;
}

Mat2 matPow(Mat2 base, unsigned long long e) {
    Mat2 r{{{1, 0}, {0, 1}}};                             // 单位阵
    while (e > 0) {
        if (e & 1) r = matMul(r, base);
        base = matMul(base, base);
        e >>= 1;
    }
    return r;
}

unsigned long long fibFast(int n) {                       // 矩阵快速幂 O(log n)
    if (n == 0) return 0;
    const Mat2 M{{{1, 1}, {1, 0}}};
    return matPow(M, static_cast<unsigned long long>(n)).m[0][1];
}

unsigned long long fibNaive(int n) {                      // 朴素 O(n)
    unsigned long long a = 0, b = 1;
    for (int i = 0; i < n; ++i) { const unsigned long long t = a + b; a = b; b = t; }
    return a;
}

// ===========================================================================
// 3. Karatsuba 大整数乘法
//    把 n 位数 x, y 按 10^{n/2} 拆成 x = x1*B + x0, y = y1*B + y0
//      x*y = z2 * B^2 + z1 * B + z0
//      z2 = x1*y1,  z0 = x0*y0,  z1 = (x1+x0)(y1+y0) - z2 - z0
//    只需 3 次子乘法（而非教科书 4 次），T(n) = 3 T(n/2) + O(n)
//    Master 定理 -> O(n^{log2 3}) ≈ O(n^1.585)
// ===========================================================================

using Big = std::vector<int>;        // 十进制大整数：低位在前，每一位 0..9
long long g_mul = 0;                 // 统计「单位数乘法」次数
const int KARATSUBA_THRESHOLD = 16;  // 规模 <= 阈值就用竖式打底

Big bigFromString(const std::string& s) {
    Big r;
    r.reserve(s.size());
    for (auto it = s.rbegin(); it != s.rend(); ++it) r.push_back(*it - '0');
    return r;
}

std::string bigToString(const Big& a) {
    std::string s;
    s.reserve(a.size());
    for (auto it = a.rbegin(); it != a.rend(); ++it) s.push_back(static_cast<char>('0' + *it));
    return s;
}

Big bigTrim(Big a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
    return a;
}

Big bigAdd(const Big& a, const Big& b) {
    Big r;
    const std::size_t n = std::max(a.size(), b.size());
    r.reserve(n + 1);
    int carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        r.push_back(s % 10);
        carry = s / 10;
    }
    if (carry) r.push_back(carry);
    return r;
}

Big bigSub(const Big& a, const Big& b) {                   // 要求 a >= b
    Big r;
    r.reserve(a.size());
    int borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        int s = a[i] - borrow - (i < b.size() ? b[i] : 0);
        if (s < 0) { s += 10; borrow = 1; } else { borrow = 0; }
        r.push_back(s);
    }
    return bigTrim(std::move(r));
}

Big bigShift(const Big& a, int zeros) {                    // 乘 10^zeros（左移十进制位）
    if (a.size() == 1 && a[0] == 0) return a;
    Big r(static_cast<std::size_t>(zeros), 0);
    r.insert(r.end(), a.begin(), a.end());
    return r;
}

Big schoolMul(const Big& a, const Big& b) {                // 竖式 O(n*m)
    Big r(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        int carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            ++g_mul;                                       // 每一位相乘计一次
            const int cur = r[i + j] + a[i] * b[j] + carry;
            r[i + j] = cur % 10;
            carry = cur / 10;
        }
        r[i + b.size()] += carry;
    }
    return bigTrim(std::move(r));
}

Big karatsuba(const Big& x, const Big& y) {
    if (std::min(x.size(), y.size()) <=
        static_cast<std::size_t>(KARATSUBA_THRESHOLD)) {
        return schoolMul(x, y);                            // 小规模打底
    }
    const std::size_t n = std::max(x.size(), y.size());
    const std::size_t half = n / 2;

    Big xp = x, yp = y;                                    // 对齐到同长，便于对称拆分
    xp.resize(n, 0);
    yp.resize(n, 0);
    Big x0(xp.begin(), xp.begin() + static_cast<std::ptrdiff_t>(half));
    Big x1(xp.begin() + static_cast<std::ptrdiff_t>(half), xp.end());
    Big y0(yp.begin(), yp.begin() + static_cast<std::ptrdiff_t>(half));
    Big y1(yp.begin() + static_cast<std::ptrdiff_t>(half), yp.end());

    const Big z2 = karatsuba(x1, y1);
    const Big z0 = karatsuba(x0, y0);
    // z1 = (x1+x0)(y1+y0) - z2 - z0：复用 z2、z0，只要第 3 次子乘法
    const Big z1 = bigSub(bigSub(karatsuba(bigAdd(x1, x0), bigAdd(y1, y0)), z2), z0);

    return bigAdd(bigAdd(bigShift(z2, static_cast<int>(2 * half)),
                         bigShift(z1, static_cast<int>(half))),
                  z0);
}

} // namespace dnc

// ===========================================================================
// 测试与打印
// ===========================================================================
static void testBinarySearch() {
    std::cout << "===== 1) 二分查找：比较次数 vs O(log n) =====\n";
    std::mt19937 rng(12345);
    for (int n : {16, 128, 1024, 16384, 262144}) {
        std::vector<int> a(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = i * 2;  // 有序
        long long sumB = 0, sumL = 0;
        const int trials = 200;
        for (int t = 0; t < trials; ++t) {
            const int target = static_cast<int>(rng() % static_cast<unsigned>(2 * n));
            long long c1 = 0, c2 = 0;
            dnc::lowerBound(a, target, c1);
            dnc::linearSearch(a, target, c2);
            sumB += c1;
            sumL += c2;
        }
        std::cout << "  n=" << n << "\t二分=" << (sumB / trials)
                  << "\t线性=" << (sumL / trials) << "\tlog2(n)="
                  << std::fixed << std::setprecision(0) << std::log2(static_cast<double>(n))
                  << "\n";
    }
    std::vector<int> a = {0, 2, 2, 2, 5, 7, 7, 9};
    for (int t : {-1, 0, 2, 3, 7, 8, 9, 10}) {
        long long c = 0;
        const int lb = dnc::lowerBound(a, t, c);
        const int ub = dnc::upperBound(a, t, c);
        assert(lb == static_cast<int>(std::lower_bound(a.begin(), a.end(), t) - a.begin()));
        assert(ub == static_cast<int>(std::upper_bound(a.begin(), a.end(), t) - a.begin()));
    }
    std::cout << "  [断言通过] lower_bound/upper_bound 与 std 一致\n\n";
}

static void testFastPow() {
    std::cout << "===== 2) 快速幂：朴素 O(n) vs 二分 O(log n) =====\n";
    for (long long n : {10, 20, 30, 40}) {
        const long long v1 = dnc::powNaive(2, n);
        const long long v2 = dnc::powFast(2, n);
        std::cout << "  2^" << n << " = " << v1 << "\t快速幂一致: " << (v1 == v2 ? "yes" : "NO") << "\n";
        assert(v1 == v2);
    }
    const long long p = 1000000007LL;
    const long long r = dnc::powMod(123456789, p - 1, p);
    std::cout << "  费马小定理: 123456789^(p-1) mod p = " << r << " (应为 1) -> "
              << (r == 1 ? "OK" : "FAIL") << "\n";
    assert(r == 1);

    std::cout << "  ---- 矩阵快速幂求斐波那契：O(log n) vs 朴素 O(n) ----\n";
    for (int n : {10, 50, 90}) {
        const unsigned long long f1 = dnc::fibFast(n);
        const unsigned long long f2 = dnc::fibNaive(n);
        std::cout << "  F(" << n << ") = " << f1 << "\t一致性: " << (f1 == f2 ? "yes" : "NO") << "\n";
        assert(f1 == f2);
    }
    std::cout << "  -> 「线性递推」用矩阵幂只花 O(log n) 次矩阵乘法，指数/递推都能二分。\n\n";
}

static void testKaratsuba() {
    std::cout << "===== 3) Karatsuba：乘法次数对照（十进制大整数） =====\n";
    std::mt19937_64 rng(999);
    auto randDigits = [&](int d) {
        dnc::Big a(static_cast<std::size_t>(d));
        for (int i = 0; i < d; ++i) a[static_cast<std::size_t>(i)] = static_cast<int>(rng() % 10);
        a[static_cast<std::size_t>(d - 1)] = 1 + static_cast<int>(rng() % 9);   // 最高位非 0
        return a;
    };
    std::cout << "  (竖式阈值打底 = " << dnc::KARATSUBA_THRESHOLD << " 位)\n";
    for (int d : {16, 32, 64, 128, 256, 512, 1024}) {
        const dnc::Big x = randDigits(d), y = randDigits(d);

        dnc::g_mul = 0;
        const dnc::Big rs = dnc::schoolMul(x, y);
        const long long cs = dnc::g_mul;

        dnc::g_mul = 0;
        const dnc::Big rk = dnc::karatsuba(x, y);
        const long long ck = dnc::g_mul;

        assert(rs == rk);
        std::cout << "  d=" << d << "\t竖式=" << cs << "\tKaratsuba=" << ck
                  << "\t加速比=" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(cs) / static_cast<double>(ck))
                  << "\td^1.585=" << std::setprecision(0)
                  << std::pow(static_cast<double>(d), 1.585) << "\t一致=yes\n";
    }
    // 已知值交叉验证：111111111^2 = 12345678987654321
    const dnc::Big ones = dnc::bigFromString("111111111");
    assert(dnc::bigToString(dnc::karatsuba(ones, ones)) == "12345678987654321");
    // 小规模穷举
    for (int x = 0; x < 60; ++x)
        for (int y = 0; y < 60; ++y) {
            const dnc::Big a = dnc::bigFromString(std::to_string(x));
            const dnc::Big b = dnc::bigFromString(std::to_string(y));
            assert(dnc::bigToString(dnc::karatsuba(a, b)) == std::to_string(x * y));
        }
    std::cout << "  [断言通过] 与竖式结果一致；111111111^2=12345678987654321 校验通过\n\n";
}

int main() {
    std::cout << "########## 分治算法演示 ##########\n\n";
    testBinarySearch();
    testFastPow();
    testKaratsuba();
    std::cout << "全部测试通过。\n";
    return 0;
}
