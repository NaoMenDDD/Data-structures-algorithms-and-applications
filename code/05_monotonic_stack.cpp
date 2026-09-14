// ===========================================================================
// 05_monotonic_stack.cpp
// 单调栈 (monotonic stack) 的两个经典应用:
//   1) 下一个更大元素 (Next Greater Element, NGE)
//   2) 柱状图中最大矩形 (Largest Rectangle in Histogram)
// 两者都给出 O(n^2) 暴力版本用于随机对拍, 保证结论正确。
//
// 编译运行:
//   g++ -std=c++17 -O2 -Wall -Wextra 05_monotonic_stack.cpp -o 05_monotonic_stack
//   ./05_monotonic_stack
// ===========================================================================
#include <algorithm>   // std::max, std::min
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>      // std::mt19937
#include <stack>
#include <vector>

// ---------------------------------------------------------------------------
// 1. 下一个更大元素 (Next Greater Element)
// ---------------------------------------------------------------------------
// 对每个位置 i, 求最小的 j > i 使 a[j] > a[i]; 若不存在记为 -1。
//
// 单调栈做法: 维护一个"对应值严格递减"的下标栈。
//   * 扫描到 a[i] 时, 反复弹出栈顶值 < a[i] 的下标 —— a[i] 正是它们的答案。
//   * 弹完后把 i 压栈, 保持栈内值递减。
// 每个下标至多进栈一次、出栈一次 -> 时间 O(n), 空间 O(n)。
std::vector<int> nextGreater(const std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<int> res(static_cast<std::size_t>(n), -1);
    std::stack<int> st;                 // 存下标, 其值自底向顶严格递减
    for (int i = 0; i < n; ++i) {
        while (!st.empty() && a[st.top()] < a[i]) {
            res[static_cast<std::size_t>(st.top())] = a[i];
            st.pop();
        }
        st.push(i);
    }
    return res;
}

// 暴力版 O(n^2), 仅用于对拍
std::vector<int> nextGreaterBrute(const std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<int> res(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (a[j] > a[i]) {
                res[static_cast<std::size_t>(i)] = a[j];
                break;
            }
    return res;
}

// 上一个更大元素 (Previous Greater Element): 求最小的 j < i 使 a[j] > a[i]。
// 与上面镜像: 正序扫描, 先弹掉值 <= a[i] 的栈顶, 此时栈顶即为答案。
std::vector<int> previousGreater(const std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<int> res(static_cast<std::size_t>(n), -1);
    std::stack<int> st;
    for (int i = 0; i < n; ++i) {
        while (!st.empty() && a[st.top()] <= a[i]) st.pop();
        if (!st.empty()) res[static_cast<std::size_t>(i)] = st.top();
        st.push(i);
    }
    return res;
}

// ---------------------------------------------------------------------------
// 2. 柱状图中最大矩形 (Largest Rectangle in Histogram)
// ---------------------------------------------------------------------------
// 给定每个柱子的高度 h[i] (宽度均为 1), 求能框住的最大矩形面积。
//
// 关键观察: 以 h[i] 为高的矩形, 其左右边界分别是"左边第一个更矮的柱子"与
// "右边第一个更矮的柱子"。单调递增栈在弹出 i 时, 恰好同时确定了这两个边界:
//   弹栈时刻的下标 j 是右边第一个更矮者, 弹完后新的栈顶是左边第一个更矮者。
//
// 技巧: 在末尾追加一个高度 0 的哨兵 (或循环到 i == n 用 0 触发), 保证结束时
// 栈中所有元素都被弹出结算。
//
// 时间 O(n), 空间 O(n)。返回类型用 long long 防止 height * width 溢出。
long long largestRectangleArea(const std::vector<int>& h) {
    const int n = static_cast<int>(h.size());
    std::stack<int> st;                 // 存下标, 对应高度单调递增
    long long best = 0;
    for (int i = 0; i <= n; ++i) {
        const int cur = (i == n) ? 0 : h[static_cast<std::size_t>(i)];  // 哨兵 0
        while (!st.empty() && h[static_cast<std::size_t>(st.top())] > cur) {
            const int height = h[static_cast<std::size_t>(st.top())];
            st.pop();
            // 左边界: 弹出后栈顶 + 1; 若栈空则从 0 开始
            const int left = st.empty() ? 0 : st.top() + 1;
            const long long width = i - left;
            best = std::max(best, static_cast<long long>(height) * width);
        }
        st.push(i);
    }
    return best;
}

// 暴力版 O(n^2): 枚举左端点, 向右扩展时维护区间最小值
long long largestRectangleBrute(const std::vector<int>& h) {
    const int n = static_cast<int>(h.size());
    long long best = 0;
    for (int i = 0; i < n; ++i) {
        int mn = h[static_cast<std::size_t>(i)];
        for (int j = i; j < n; ++j) {
            mn = std::min(mn, h[static_cast<std::size_t>(j)]);
            best = std::max(best,
                            static_cast<long long>(mn) * (j - i + 1));
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// 打印辅助
// ---------------------------------------------------------------------------
static void printVec(const char* name, const std::vector<int>& v) {
    std::cout << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << v[i] << (i + 1 < v.size() ? ", " : "");
    std::cout << "]\n";
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------
static void testNextGreater() {
    std::cout << "===== 下一个更大元素 =====\n";
    std::vector<int> a = {2, 1, 2, 4, 3};
    auto got = nextGreater(a);
    printVec("a               ", a);
    printVec("nextGreater     ", got);
    std::vector<int> expect = {4, 2, 4, -1, -1};
    assert(got == expect);
    std::cout << "[OK] 手算用例通过\n";

    // 严格递减: 全部无更大元素
    std::vector<int> desc = {5, 4, 3, 2, 1};
    assert(nextGreater(desc) == std::vector<int>(5, -1));
    std::cout << "[OK] 严格递减数组全部返回 -1\n";

    auto pg = previousGreater(a);
    printVec("previousGreater ", pg);
    std::vector<int> pgExpect = {-1, 0, -1, -1, 3};
    assert(pg == pgExpect);
    std::cout << "[OK] 上一个更大元素 (下标版) 正确\n\n";
}

static void testHistogram() {
    std::cout << "===== 柱状图最大矩形 =====\n";
    struct Case { std::vector<int> h; long long want; };
    Case cases[] = {
        {{2, 1, 5, 6, 2, 3}, 10},
        {{2, 4}, 4},
        {{}, 0},
        {{5}, 5},
        {{1, 1, 1, 1}, 4},
        {{6, 2, 5, 4, 5, 1, 6}, 12},
    };
    for (const auto& c : cases) {
        long long got = largestRectangleArea(c.h);
        std::cout << "  面积 = " << got << "  (期望 " << c.want << ")   高度 = [";
        for (std::size_t i = 0; i < c.h.size(); ++i)
            std::cout << c.h[i] << (i + 1 < c.h.size() ? ", " : "");
        std::cout << "]\n";
        assert(got == c.want);
    }
    std::cout << "[OK] 含空数组/单柱/全等的 6 组用例通过\n\n";
}

static void testRandomAgainstBrute() {
    std::cout << "===== 随机对拍 (固定种子 mt19937) =====\n";
    std::mt19937 rng(20240913u);
    std::uniform_int_distribution<int> distVal(-5, 20);
    std::uniform_int_distribution<int> distLen(0, 40);

    std::size_t ngeChecked = 0, rectChecked = 0;
    for (int trial = 0; trial < 500; ++trial) {
        const int len = distLen(rng);
        std::vector<int> a(static_cast<std::size_t>(len));
        for (int& x : a) x = distVal(rng);

        assert(nextGreater(a) == nextGreaterBrute(a));
        ++ngeChecked;

        // 直方图高度取非负, 面积才有意义
        std::vector<int> h(static_cast<std::size_t>(len));
        for (int& x : h) x = distVal(rng) % 10;
        for (int& x : h) if (x < 0) x = -x;   // 取模可能得负, 再取绝对值
        assert(largestRectangleArea(h) == largestRectangleBrute(h));
        ++rectChecked;
    }
    std::cout << "[OK] " << ngeChecked << " 组 NGE 与 " << rectChecked
              << " 组直方图, 单调栈结果与 O(n^2) 暴力完全一致\n\n";
}

int main() {
    testNextGreater();
    testHistogram();
    testRandomAgainstBrute();
    std::cout << "全部测试通过。\n";
    return 0;
}
