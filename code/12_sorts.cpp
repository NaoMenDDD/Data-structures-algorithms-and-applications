// 12_sorts.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 12 章 排序》
// 主题：八大经典排序 —— 插入 / 选择 / 冒泡（O(n^2)）、归并 / 快排 / 堆排
//       （O(n log n)），以及稳定性的实测与快排最坏情况的旋钮
//
//   排序是算法的「万能前置步骤」：二分查找要先有序、去重先有序、
//   很多贪心（如区间调度、Kruskal 的边排序）先有序。
//
//   本文件做四件事：
//     ① **正确性对拍**：每种排序的随机测试结果 == std::sort；
//     ② **稳定性实测**：给每个元素带一个 tag，看相等键的相对次序是否保持；
//     ③ **快排最坏情况**：同样 2000 个「已升序」的数，
//        固定取首个元素做 pivot 会退化到 O(n^2)，随机 pivot 则稳定在 O(n log n)；
//     ④ **耗时基准**：O(n^2) 组与 O(n log n) 组分别在合适规模上跑，直观感受量级差。
//
//   稳定性口诀：
//     稳定：插入、冒泡、归并、计数、基数；
//     不稳定：选择、快排（经典版）、堆排。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 12_sorts.cpp -o 12_sorts && ./12_sorts
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 12_sorts.cpp -o sorts_san && ./sorts_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::sort, std::is_sorted, std::swap
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>        // std::move
#include <vector>

// 带 tag 的元素：只按 key 比较，tag 用来检测排序是否稳定。
struct Item {
    int key;
    int tag;
    friend bool operator<(const Item& a, const Item& b) { return a.key < b.key; }
};

// ===========================================================================
// O(n^2) 三兄弟
// ===========================================================================

// 插入排序：像打扑克理牌——把第 i 张牌插进左边已有序的部分。
// 最好 O(n)（已有序），最坏 / 平均 O(n^2)；**稳定**。对「近乎有序」的数据极快。
template <class T>
long long insertionSort(std::vector<T>& a) {
    long long cmp = 0;
    for (std::size_t i = 1; i < a.size(); ++i) {
        T key = std::move(a[i]);
        std::size_t j = i;
        while (j > 0 && (++cmp, key < a[j - 1])) { a[j] = std::move(a[j - 1]); --j; }
        a[j] = std::move(key);
    }
    return cmp;                                  // 返回「比较次数」，便于最坏情况分析
}

// 选择排序：每一轮从未排序部分选出最小值，放到已排序部分的末尾。
// 无论输入如何都是 O(n^2)（比较次数固定 n(n-1)/2）；**不稳定**（交换会跳位）。
template <class T>
void selectionSort(std::vector<T>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::size_t m = i;
        for (std::size_t j = i + 1; j < n; ++j) if (a[j] < a[m]) m = j;
        if (m != i) std::swap(a[i], a[m]);
    }
}

// 冒泡排序：相邻两两比较、大数一路「冒」到右边；带「本趟无交换即已有序」的提前退出。
// 稳定；最好 O(n)（已有序 + 提前退出），最坏 / 平均 O(n^2)。教学价值 > 实用价值。
template <class T>
bool bubbleSort(std::vector<T>& a) {
    const std::size_t n = a.size();
    bool anySwap = false;
    for (std::size_t pass = 0; pass + 1 < n; ++pass) {
        bool swapped = false;
        for (std::size_t j = 0; j + 1 < n - pass; ++j)
            if (a[j + 1] < a[j]) { std::swap(a[j], a[j + 1]); swapped = true; anySwap = true; }
        if (!swapped) break;                     // 已经完全有序，提前收工
    }
    return anySwap;
}

// ===========================================================================
// O(n log n) 三巨头
// ===========================================================================

// 归并排序（自顶向下分治）：分到长度 1，再两两合并。**稳定**，需 O(n) 额外空间。
template <class T>
void mergeSortImpl(std::vector<T>& a, std::vector<T>& buf, std::size_t lo, std::size_t hi) {
    if (hi - lo <= 1) return;
    const std::size_t mid = lo + (hi - lo) / 2;
    mergeSortImpl(a, buf, lo, mid);
    mergeSortImpl(a, buf, mid, hi);
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) buf[k++] = (a[j] < a[i]) ? a[j++] : a[i++];  // 相等取左 -> 稳定
    while (i < mid) buf[k++] = a[i++];
    while (j < hi)  buf[k++] = a[j++];
    for (std::size_t t = lo; t < hi; ++t) a[t] = buf[t];
}
template <class T>
void mergeSort(std::vector<T>& a) {
    if (a.size() < 2) return;
    std::vector<T> buf(a.size());
    mergeSortImpl(a, buf, 0, a.size());
}

// 快速排序（三路划分 + 随机 pivot）：把数组切成「< pivot / == pivot / > pivot」三段。
// 三路划分让「大量重复元素」也保持 O(n log n)（经典两路版会退化）。
// 每轮只递归较小的那半、较大的那半改循环 -> 递归栈深 O(log n)。**不稳定**。
template <class T>
void quickSort3(std::vector<T>& a, std::size_t lo, std::size_t hi, std::mt19937& rng, long long& cmp) {
    while (lo + 1 < hi) {
        const std::size_t p = lo + rng() % (hi - lo);
        const T pivot = a[p];
        std::size_t lt = lo, i = lo, gt = hi;
        while (i < gt) {
            ++cmp;
            if (a[i] < pivot)        std::swap(a[lt++], a[i++]);   // 归入「< pivot」
            else if (pivot < a[i])   std::swap(a[i], a[--gt]);     // 归入「> pivot」
            else                     ++i;                          // 与 pivot 相等，留在中间
        }
        // [lo,lt) < pivot, [lt,gt) == pivot, [gt,hi) > pivot
        if (lt - lo < hi - gt) { quickSort3(a, lo, lt, rng, cmp); lo = gt; }   // 递归小侧
        else                   { quickSort3(a, gt, hi, rng, cmp); hi = lt; }   // 递归小侧
    }
}
template <class T>
long long quickSort(std::vector<T>& a, std::mt19937& rng) {
    long long cmp = 0;
    if (a.size() > 1) quickSort3(a, 0, a.size(), rng, cmp);
    return cmp;
}

// 「反面教材」快排：永远取首元素做 pivot（Lomuto 划分）。在已排序输入上会退化到 O(n^2)。
template <class T>
void quickSortFirstPivot(std::vector<T>& a, std::size_t lo, std::size_t hi, long long& cmp) {
    if (lo + 1 >= hi) return;
    const T pivot = a[lo];
    std::size_t i = lo;
    for (std::size_t j = lo + 1; j < hi; ++j) { ++cmp; if (a[j] < pivot) { ++i; std::swap(a[i], a[j]); } }
    std::swap(a[lo], a[i]);
    quickSortFirstPivot(a, lo, i, cmp);
    quickSortFirstPivot(a, i + 1, hi, cmp);
}

// 堆排序：先建大顶堆，再不断把堆顶（最大值）换到末尾、堆缩小、下沉调整。**不稳定**，原地。
template <class T>
static void siftDown(std::vector<T>& a, std::size_t n, std::size_t i) {
    while (true) {
        const std::size_t l = 2 * i + 1, r = 2 * i + 2;
        std::size_t m = i;
        if (l < n && a[m] < a[l]) m = l;
        if (r < n && a[m] < a[r]) m = r;
        if (m == i) break;
        std::swap(a[i], a[m]);
        i = m;
    }
}
template <class T>
void heapSort(std::vector<T>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = n / 2; i-- > 0; ) siftDown(a, n, i);            // 自底向上建堆 O(n)
    for (std::size_t e = n; e-- > 1; ) { std::swap(a[0], a[e]); siftDown(a, e, 0); }  // 逐个取出
}

// ===========================================================================
// 工具
// ===========================================================================

// 稳定性检测：若存在「相邻且 key 相等但 tag 逆序」，则不稳定。
static bool isStable(const std::vector<Item>& v) {
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i - 1].key == v[i].key && v[i - 1].tag > v[i].tag) return false;
    return true;
}

static std::vector<Item> makeItems(std::mt19937& rng, std::size_t n, int keyRange) {
    std::vector<Item> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i].key = static_cast<int>(rng() % static_cast<unsigned>(keyRange)); v[i].tag = static_cast<int>(i); }
    return v;
}

template <class T>
static bool sortedEq(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] < b[i] || b[i] < a[i]) return false;
    return true;
}

static double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    std::cout << "======== 12 经典排序：正确性 · 稳定性 · 基准 ========\n\n";

    // ---------- 1. ★ 正确性对拍：每种排序 == std::sort ----------
    std::cout << "=== 1. ★ 随机对拍：八种排序结果 == std::sort ===\n";
    {
        std::mt19937 rng(20260913);
        const int trials = 400;
        for (int t = 0; t < trials; ++t) {
            const std::size_t n = rng() % 120;
            const int range = 1 + static_cast<int>(rng() % 40);      // 故意制造大量重复键
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng() % static_cast<unsigned>(range));
            std::vector<int> ref = base;
            std::sort(ref.begin(), ref.end());

            std::vector<int> v;
            v = base; insertionSort(v);            assert(sortedEq(v, ref));
            v = base; selectionSort(v);            assert(sortedEq(v, ref));
            v = base; bubbleSort(v);               assert(sortedEq(v, ref));
            v = base; mergeSort(v);                assert(sortedEq(v, ref));
            std::mt19937 r2(12345 + static_cast<unsigned>(t));
            v = base; quickSort(v, r2);            assert(sortedEq(v, ref));
            v = base; heapSort(v);                 assert(sortedEq(v, ref));
            v = base; { long long c = 0; quickSortFirstPivot(v, 0, v.size(), c); } assert(sortedEq(v, ref));
        }
        std::cout << "  " << trials << " 组随机数组（含大量重复键）：七种排序全部 == std::sort ✓\n";
    }

    // ---------- 2. ★ 稳定性实测 ----------
    std::cout << "\n=== 2. ★ 稳定性：相等键的相对次序是否保持 ===\n";
    {
        std::mt19937 rng(20260914);
        const std::size_t n = 40;
        const int keyRange = 5;                                     // 只有 5 种键 -> 大量相等
        auto run = [&](const char* name, auto fn, bool expect) {
            auto v = makeItems(rng, n, keyRange);
            fn(v);
            const bool stable = isStable(v);
            std::cout << "     " << name << "\t稳定?\t" << (stable ? "是" : "否")
                      << "   \t（" << (stable == expect ? "符合预期" : "!! 与预期不符") << "）\n";
        };
        std::cout << "     排序\t\t是否稳定\n";
        run("插入排序        ", [](std::vector<Item>& v){ insertionSort(v); }, true);
        run("冒泡排序        ", [](std::vector<Item>& v){ bubbleSort(v); }, true);
        run("归并排序        ", [](std::vector<Item>& v){ mergeSort(v); }, true);
        run("选择排序        ", [](std::vector<Item>& v){ selectionSort(v); }, false);
        run("快速排序(三路)  ", [](std::vector<Item>& v){ std::mt19937 r(1); quickSort(v, r); }, false);
        run("堆排序          ", [](std::vector<Item>& v){ heapSort(v); }, false);
        std::cout << "  -> 需要「相等键保持原序」（如多级排序）时，必须选稳定排序（插入 / 归并）。\n";
    }

    // ---------- 3. ★ 快排最坏情况：固定首元 pivot vs 随机 pivot ----------
    std::cout << "\n=== 3. ★ 已升序输入上，快排的两种 pivot 策略 ===\n";
    {
        const std::size_t n = 2000;
        std::vector<int> sorted(n);
        for (std::size_t i = 0; i < n; ++i) sorted[i] = static_cast<int>(i);

        std::vector<int> a = sorted;
        long long cmpFixed = 0;
        quickSortFirstPivot(a, 0, a.size(), cmpFixed);
        assert(std::is_sorted(a.begin(), a.end()));

        std::vector<int> b = sorted;
        std::mt19937 rng(20260913);
        const long long cmpRandom = quickSort(b, rng);
        assert(std::is_sorted(b.begin(), b.end()));

        std::cout << "  已升序 n = " << n << "：\n";
        std::cout << "     固定首元 pivot ：比较次数 = " << cmpFixed
                  << "（≈ n^2/2 = " << (static_cast<long long>(n) * (n - 1) / 2) << "，退化！）\n";
        std::cout << "     随机   pivot ：比较次数 = " << cmpRandom
                  << "（≈ n·log2 n = " << static_cast<long long>(n) * 11 << "，正常）\n";
        std::cout << "  -> 固定 pivot 遇到「已有序 / 逆序」会一路切出空区间，深度 O(n)、比较 O(n^2)；\n";
        std::cout << "     **随机化 pivot**（或三数取中）把最坏情况变成「几乎不可能碰到」，期望 O(n log n)。\n";
        assert(cmpFixed > cmpRandom * 10);                         // 退化远多于随机
    }

    // ---------- 4. 插入排序：近乎有序是它的主场 ----------
    std::cout << "\n=== 4. 插入排序的「最好情况」：近乎有序极快 ===\n";
    {
        const std::size_t n = 5000;
        std::mt19937 rng(20260913);
        std::vector<int> randomArr(n);
        for (std::size_t i = 0; i < n; ++i) randomArr[i] = static_cast<int>(rng());
        std::vector<int> sortedArr = randomArr;
        std::sort(sortedArr.begin(), sortedArr.end());
        std::vector<int> nearly = sortedArr;                       // 近乎有序：只随机换 10 对
        for (int k = 0; k < 10; ++k) std::swap(nearly[rng() % n], nearly[rng() % n]);

        std::vector<int> v;
        v = randomArr; const long long cRand = insertionSort(v);
        v = sortedArr; const long long cSorted = insertionSort(v);
        v = nearly;    const long long cNear = insertionSort(v);
        assert(std::is_sorted(v.begin(), v.end()));
        std::cout << "  n = " << n << " 的插入排序比较次数：\n";
        std::cout << "     随机    ： " << cRand   << "\n";
        std::cout << "     已升序  ： " << cSorted << "（每轮只比较 1 次 -> O(n)）\n";
        std::cout << "     近乎有序： " << cNear   << "（略高于已升序，远低于随机）\n";
        std::cout << "  -> 这就是「小数组 / 近乎有序用插入排序」的底气，也是工程快排的常见收尾优化。\n";
        assert(cSorted < cRand / 100);
    }

    // ---------- 5. ★ 耗时基准 ----------
    std::cout << "\n=== 5. ★ 耗时基准（随机 int）===\n";
    {
        std::mt19937 rng(20260913);

        // (a) O(n^2) 组：n = 5000
        {
            const std::size_t n = 5000;
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng());
            std::vector<int> v;
            v = base; auto t0 = std::chrono::steady_clock::now(); insertionSort(v); std::cout << "  n = 5000  插入排序   " << msSince(t0) << " ms\n";
            v = base; t0 = std::chrono::steady_clock::now(); selectionSort(v);     std::cout << "  n = 5000  选择排序   " << msSince(t0) << " ms\n";
            v = base; t0 = std::chrono::steady_clock::now(); bubbleSort(v);        std::cout << "  n = 5000  冒泡排序   " << msSince(t0) << " ms\n";
        }
        // (b) O(n log n) 组：n = 200000
        {
            const std::size_t n = 200000;
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng());
            std::vector<int> v;
            std::mt19937 r1(7), r2(8);
            v = base; auto t0 = std::chrono::steady_clock::now(); mergeSort(v);          std::cout << "  n = 200000 归并排序   " << msSince(t0) << " ms\n";
            v = base; t0 = std::chrono::steady_clock::now(); quickSort(v, r1);           std::cout << "  n = 200000 快排(随机) " << msSince(t0) << " ms\n";
            v = base; t0 = std::chrono::steady_clock::now(); heapSort(v);               std::cout << "  n = 200000 堆排序     " << msSince(t0) << " ms\n";
            v = base; t0 = std::chrono::steady_clock::now(); std::sort(v.begin(), v.end()); std::cout << "  n = 200000 std::sort  " << msSince(t0) << " ms（内省排序，工程最强）\n";
            (void)r2;
        }
        std::cout << "  -> O(n^2) 只能撑到几千；O(n log n) 上 20 万仍毫秒级。量级差距一目了然。\n";
    }

    std::cout << "\n=== 6. 复杂度与选择 ===\n"
                 "  * 插入：最好 O(n) / 平均 O(n^2) / 稳定 / 近乎有序首选；\n"
                 "  * 选择：恒 O(n^2) / 不稳定 / 交换次数最少（<= n）；\n"
                 "  * 冒泡：最好 O(n)(带提前退出) / 不稳定不划算，仅教学；\n"
                 "  * 归并：恒 O(n log n) / **稳定** / 需 O(n) 额外空间（外排序、链表排序首选）；\n"
                 "  * 快排：平均 O(n log n) / 最坏 O(n^2) / 不稳定 / 原地、常数小；随机化 + 三路划分是标配；\n"
                 "  * 堆排：恒 O(n log n) / 不稳定 / 原地、最坏也有保证；\n"
                 "  * 比较排序的下界是 Ω(n log n)（决策树）；想更快只能「不比较」——计数 / 基数排序（见另一文件）。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
