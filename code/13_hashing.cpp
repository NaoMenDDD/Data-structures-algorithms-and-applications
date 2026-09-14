// 13_hashing.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 13 章 哈希与散列》
// 主题：散列表（哈希表）——哈希函数、冲突处理（链地址法 / 开放定址）、
//       负载因子与再哈希（rehash）
//
//   数组的下标访问是 O(1)：`a[i]` 直接就拿到了。散列表的全部野心就是——
//   **把「任意键 k」换算成「数组下标 i」，从而让「按键查找」也变成 O(1)。**
//
//     键 k  --哈希函数 h-->  下标 i = h(k) % 桶数  --直接访问-->  值
//
//   理想情况「一键一位、互不冲突」。但键空间远大于桶空间（鸽巢原理），
//   **冲突（collision）不可避免**，于是有两套解法：
//
//     ① **链地址法（separate chaining）**：每个桶挂一条链表，冲突的键串在链上。
//        简单、删改方便；负载因子可以 > 1。
//     ② **开放定址法（open addressing）**：冲突了就「换个座位」——按某个探测序列
//        往后找空位。线性探测 / 二次探测 / 双重散列是三种探测方式。
//        省指针、缓存友好；但负载因子必须 < 1，且**删除要留墓碑（tombstone）**。
//
//   核心指标：**负载因子 α = 元素数 / 桶数**。
//     * 链地址法：查找代价 ≈ 1 + α（平均链长）；
//     * 线性探测：查找代价 ≈ (1 + 1/(1-α)^2)/2（α 越大越陡）；
//   所以**再哈希（扩容 + 重新分布）**是维持性能的关键：α 超过阈值就翻倍桶数。
//
//   本文件：哈希函数对比、两种冲突处理、负载因子实测、与 std::unordered_map 对拍 + 基准。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 13_hashing.cpp -o 13_hashing && ./13_hashing
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 13_hashing.cpp -o hash_san && ./hash_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// 哈希函数：把字符串映射成 size_t
// ===========================================================================

// 反面教材：把字符码简单相加。**字母异位词会撞在一起**（"ab" 与 "ba" 同哈希）。
static std::size_t hashSumOfChars(const std::string& s) {
    std::size_t h = 0;
    for (unsigned char c : s) h += c;
    return h;
}

// djb2（Daniel J. Bernstein）：h = h*33 + c。经典、极简、分布尚可。
static std::size_t hashDjb2(const std::string& s) {
    std::size_t h = 5381;
    for (unsigned char c : s) h = h * 33 + c;
    return h;
}

// FNV-1a：先 xor 再乘一个素数。分布比 djb2 更均匀，常用于字符串。
static std::size_t hashFnv1a(const std::string& s) {
    std::size_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// ===========================================================================
// 冲突处理 A：链地址法（separate chaining）
//   每个桶一条链表（这里用 vector 当链）。负载因子可 > 1。删除最简单。
// ===========================================================================
class ChainingTable {
public:
    explicit ChainingTable(std::size_t cap = 8, double maxLoad = 0.75)
        : buckets_(cap), maxLoad_(maxLoad) {}

    void put(int k, int v) {
        auto& chain = buckets_[idx(k)];
        for (auto& p : chain) if (p.first == k) { p.second = v; return; }
        chain.push_back({k, v});
        ++size_;
        if (loadFactor() > maxLoad_) rehash(buckets_.size() * 2);
    }

    // 查找：返回是否命中，probes 记录「扫描过的链结点数」。
    bool get(int k, int& out, long long& probes) const {
        probes = 0;
        const auto& chain = buckets_[idx(k)];
        for (const auto& p : chain) { ++probes; if (p.first == k) { out = p.second; return true; } }
        return false;
    }

    bool erase(int k) {
        auto& chain = buckets_[idx(k)];
        for (std::size_t i = 0; i < chain.size(); ++i)
            if (chain[i].first == k) { chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i)); --size_; return true; }
        return false;
    }

    double loadFactor() const { return static_cast<double>(size_) / static_cast<double>(buckets_.size()); }
    std::size_t size() const { return size_; }
    std::size_t buckets() const { return buckets_.size(); }

    // 统计：最长链长度、平均（非空）链长度。
    std::size_t maxChain() const {
        std::size_t m = 0;
        for (const auto& c : buckets_) m = std::max(m, c.size());
        return m;
    }
    double avgChain() const {
        std::size_t nonEmpty = 0, total = 0;
        for (const auto& c : buckets_) if (!c.empty()) { ++nonEmpty; total += c.size(); }
        return nonEmpty ? static_cast<double>(total) / static_cast<double>(nonEmpty) : 0.0;
    }

private:
    std::size_t idx(int k) const {
        return static_cast<std::size_t>(std::hash<int>{}(k)) & (buckets_.size() - 1);   // 桶数恒为 2 的幂
    }
    void rehash(std::size_t newCap) {
        std::vector<std::vector<std::pair<int,int>>> nb(newCap);
        for (auto& chain : buckets_)
            for (auto& p : chain) nb[static_cast<std::size_t>(std::hash<int>{}(p.first)) & (newCap - 1)].push_back(p);
        buckets_.swap(nb);
    }
    std::vector<std::vector<std::pair<int,int>>> buckets_;
    std::size_t size_ = 0;
    double maxLoad_;
};

// ===========================================================================
// 冲突处理 B：开放定址法（open addressing）
//   探测序列由 probe 策略决定：线性 / 双重散列。删除留墓碑（TOMB）。
// ===========================================================================
enum class ProbeKind { Linear, DoubleHash };

class OpenTable {
public:
    explicit OpenTable(std::size_t cap = 16, ProbeKind kind = ProbeKind::Linear, double maxLoad = 0.75)
        : keys_(cap, 0), vals_(cap, 0), state_(cap, EMPTY_S), kind_(kind), maxLoad_(maxLoad) {}

    void put(int k, int v) {
        std::size_t firstFree = NPOS;                                // 记下第一个墓碑，供最后复用
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            const std::size_t pos = probe(k, i);
            if (state_[pos] == USED) {
                if (keys_[pos] == k) { vals_[pos] = v; return; }     // 已存在 -> 覆盖
            } else if (state_[pos] == EMPTY_S) {
                const std::size_t t = (firstFree == NPOS) ? pos : firstFree;
                keys_[t] = k; vals_[t] = v; state_[t] = USED; ++size_;
                if (loadFactor() > maxLoad_) rehash(keys_.size() * 2);
                return;
            } else {                                                 // 墓碑：继续找，别急着插
                if (firstFree == NPOS) firstFree = pos;
            }
        }
        rehash(keys_.size() * 2);                                    // 表里已无空位，强制扩容
        put(k, v);
    }

    bool get(int k, int& out, long long& probes) const {
        probes = 0;
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            const std::size_t pos = probe(k, i);
            ++probes;
            if (state_[pos] == EMPTY_S) return false;                // 撞到空位 -> 一定不存在
            if (state_[pos] == USED && keys_[pos] == k) { out = vals_[pos]; return true; }
        }
        return false;
    }

    bool erase(int k) {
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            const std::size_t pos = probe(k, i);
            if (state_[pos] == EMPTY_S) return false;
            if (state_[pos] == USED && keys_[pos] == k) { state_[pos] = TOMB_S; --size_; return true; }
        }
        return false;
    }

    double loadFactor() const { return static_cast<double>(size_) / static_cast<double>(keys_.size()); }
    std::size_t size() const { return size_; }
    std::size_t buckets() const { return keys_.size(); }

private:
    static constexpr unsigned char EMPTY_S = 0, USED = 1, TOMB_S = 2;
    static constexpr std::size_t NPOS = std::numeric_limits<std::size_t>::max();

    // 第 i 次探测的位置：线性 = (h1 + i) % cap；双重散列 = (h1 + i*h2) % cap。
    std::size_t probeCap(int k, std::size_t i, std::size_t cap) const {
        const std::size_t h1 = static_cast<std::size_t>(std::hash<int>{}(k));
        if (kind_ == ProbeKind::Linear) return (h1 + i) & (cap - 1);
        const std::size_t h2 = (static_cast<std::size_t>(std::hash<int>{}(k * 2654435761u)) | 1u);   // 取奇数
        return (h1 + i * h2) & (cap - 1);
    }
    std::size_t probe(int k, std::size_t i) const { return probeCap(k, i, keys_.size()); }

    void rehash(std::size_t newCap) {
        std::vector<int> nk(newCap, 0), nv(newCap, 0);
        std::vector<unsigned char> ns(newCap, EMPTY_S);              // 扩容后没有墓碑了
        for (std::size_t j = 0; j < keys_.size(); ++j) {
            if (state_[j] != USED) continue;
            std::size_t i = 0;
            while (true) {
                const std::size_t pos = probeCap(keys_[j], i, newCap);
                if (ns[pos] == EMPTY_S) { nk[pos] = keys_[j]; nv[pos] = vals_[j]; ns[pos] = USED; break; }
                ++i;
            }
        }
        keys_.swap(nk);
        vals_.swap(nv);
        state_.swap(ns);
    }

    std::vector<int> keys_, vals_;
    std::vector<unsigned char> state_;
    ProbeKind kind_;
    double maxLoad_;
    std::size_t size_ = 0;
};

static double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    std::cout << "======== 13 哈希表：冲突处理与负载因子 ========\n\n";

    // ---------- 1. 哈希函数：好与坏的差别 ----------
    std::cout << "=== 1. ★ 哈希函数：字母异位词会不会撞车？ ===\n";
    {
        const std::vector<std::string> words = {"ab", "ba", "abc", "acb", "bac", "cab", "cba"};
        auto show = [&](const char* name, std::size_t(*hf)(const std::string&)) {
            std::cout << "     " << name << " -> ";
            for (const auto& w : words) std::cout << w << ":" << (hf(w) % 100) << " ";
            std::cout << "\n";
        };
        show("求和哈希(坏)", hashSumOfChars);
        show("djb2       ", hashDjb2);
        show("FNV-1a     ", hashFnv1a);
        std::cout << "  -> 求和哈希把 \"ab\" 和 \"ba\" 映到同一个值（异位词必然冲突）；\n";
        std::cout << "     djb2 / FNV-1a 引入「位置权重」，异位词几乎不会撞。\n";
        assert(hashSumOfChars("ab") == hashSumOfChars("ba"));
        assert(hashDjb2("ab") != hashDjb2("ba"));
    }

    // ---------- 2. 链地址法：负载因子对链长的影响 ----------
    std::cout << "\n=== 2. ★ 链地址法：负载因子 α 决定平均链长 ===\n";
    {
        std::mt19937 rng(20260913);
        const int n = 100000;
        for (double maxLoad : {0.5, 1.0, 4.0}) {
            ChainingTable t(16, maxLoad);
            for (int i = 0; i < n; ++i) t.put(static_cast<int>(rng()), i);
            long long probes = 0, total = 0;
            std::mt19937 q(1);
            for (int i = 0; i < 20000; ++i) { int out; t.get(static_cast<int>(q()), out, probes); total += probes; }
            std::cout << "     maxLoad=" << maxLoad << "  α=" << t.loadFactor()
                      << "  桶数=" << t.buckets()
                      << "  最长链=" << t.maxChain()
                      << "  平均查找探测=" << static_cast<double>(total) / 20000 << "\n";
        }
        std::cout << "  -> 查找代价 ≈ 1 + α：α 越大桶越挤、链越长。再哈希把 α 压在阈值下，\n";
        std::cout << "     换来的代价是「扩容 + 重新分布」那一下的开销（均摊仍 O(1)）。\n";
    }

    // ---------- 3. 开放定址：探测次数随 α 陡增 ----------
    std::cout << "\n=== 3. ★ 开放定址：α 越大，线性探测的「扎堆」越明显 ===\n";
    {
        std::mt19937 rng(20260914);
        const int n = 100000;
        std::cout << "     负载因子阈值   探测方式   实际α     平均查找探测\n";
        for (double maxLoad : {0.5, 0.75, 0.9}) {
            for (ProbeKind kind : {ProbeKind::Linear, ProbeKind::DoubleHash}) {
                const char* name = (kind == ProbeKind::Linear) ? "线性探测" : "双重散列";
                OpenTable t(16, kind, maxLoad);
                for (int i = 0; i < n; ++i) t.put(static_cast<int>(rng()), i);
                long long probes = 0, total = 0;
                std::mt19937 q(2);
                for (int i = 0; i < 50000; ++i) { int out; t.get(static_cast<int>(q()), out, probes); total += probes; }
                std::cout << "     " << maxLoad << "       \t" << name << "   \t" << t.loadFactor()
                          << "   \t" << static_cast<double>(total) / 50000 << "\n";
            }
        }
        std::cout << "  -> α 小的时候两者差不多；α 越大，线性探测越差\n";
        std::cout << "     （连续占用的「一次簇聚」让后续键排长队），双重散列把簇打散，优势明显。\n";
    }

    // ---------- 4. ★ 对拍：与 std::unordered_map 行为一致 ----------
    std::cout << "\n=== 4. ★ 随机操作对拍：与 std::unordered_map 完全一致 ===\n";
    {
        std::mt19937 rng(20260915);
        ChainingTable chain;
        OpenTable open(16, ProbeKind::Linear);
        std::unordered_map<int,int> ref;
        int ops = 0;
        for (int t = 0; t < 200000; ++t) {
            const int k = static_cast<int>(rng() % 5000);
            const int action = static_cast<int>(rng() % 3);
            if (action == 0) { const int v = static_cast<int>(rng()); ref[k] = v; chain.put(k, v); open.put(k, v); }
            else if (action == 1) { const bool e = ref.erase(k) > 0; assert(chain.erase(k) == e); assert(open.erase(k) == e); }
            else {
                int co = 0, oo = 0; long long p = 0;
                const bool cf = chain.get(k, co, p);
                const bool of = open.get(k, oo, p);
                auto it = ref.find(k);
                assert(cf == (it != ref.end()) && of == (it != ref.end()));
                if (cf) { assert(co == it->second && oo == it->second); }
                if (cf) ++ops;
            }
        }
        std::cout << "  20 万次随机「插入 / 删除 / 查找」：链地址法与开放定址都与 unordered_map 一致 ✓\n";
    }

    // ---------- 5. ★ 基准：三种实现的插入 + 查找耗时 ----------
    std::cout << "\n=== 5. ★ 基准：n = 500,000 随机 int，插入 + 100 万次查找 ===\n";
    {
        const int n = 500000;
        std::mt19937 rng(20260913);
        std::vector<int> keys(n);
        for (int i = 0; i < n; ++i) keys[i] = static_cast<int>(rng());
        std::vector<int> queries(1000000);
        for (int i = 0; i < 1000000; ++i) queries[i] = static_cast<int>(rng());
        long long sink = 0;

        {   ChainingTable t(16, 0.75);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) t.put(keys[i], i);
            const double ti = msSince(t0);
            t0 = std::chrono::steady_clock::now(); long long pr = 0;
            for (int q : queries) { int out; if (t.get(q, out, pr)) sink += out; }
            std::cout << "     链地址法      插入 " << ti << " ms  查找 " << msSince(t0) << " ms\n";
        }
        {   OpenTable t(16, ProbeKind::Linear);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) t.put(keys[i], i);
            const double ti = msSince(t0);
            t0 = std::chrono::steady_clock::now(); long long pr = 0;
            for (int q : queries) { int out; if (t.get(q, out, pr)) sink += out; }
            std::cout << "     开放定址(线性)插入 " << ti << " ms  查找 " << msSince(t0) << " ms\n";
        }
        {   OpenTable t(16, ProbeKind::DoubleHash);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) t.put(keys[i], i);
            const double ti = msSince(t0);
            t0 = std::chrono::steady_clock::now(); long long pr = 0;
            for (int q : queries) { int out; if (t.get(q, out, pr)) sink += out; }
            std::cout << "     开放定址(双重)插入 " << ti << " ms  查找 " << msSince(t0) << " ms\n";
        }
        {   std::unordered_map<int,int> m;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) m[keys[i]] = i;
            const double ti = msSince(t0);
            t0 = std::chrono::steady_clock::now();
            for (int q : queries) { auto it = m.find(q); if (it != m.end()) sink += it->second; }
            std::cout << "     std::unordered_map  插入 " << ti << " ms  查找 " << msSince(t0) << " ms\n";
        }
        std::cout << "  -> 开放定址「一整块连续内存」-> 缓存友好，查找常胜；链地址法指针跳转多，但删改灵活。\n";
        assert(sink >= 0);
    }

    std::cout << "\n=== 6. 复杂度与要点 ===\n"
                 "  * 平均：插入 / 查找 / 删除都是 O(1)（负载因子被再哈希压住时）；\n"
                 "  * 最坏：所有键撞到一个桶 -> O(n)；好的哈希函数 + 再哈希让这几乎不发生；\n"
                 "  * 链地址法：查找 ≈ 1 + α；α 可 > 1；删除最简单；指针多、缓存差；\n"
                 "  * 开放定址：查找 ≈ 1/(1-α)（线性探测更陡）；省指针、缓存友好；\n"
                 "    α 必须 < 1，**删除要留墓碑**（否则会切断探测链，误判「不存在」）；\n"
                 "  * 再哈希：α 超阈值就「桶数翻倍 + 全部重新分布」；虽然单次 O(n)，\n"
                 "    但均摊到每次插入仍是 O(1)（和 vector 扩容同理）；\n"
                 "  * 桶数取「2 的幂」可用位运算取模；用「素数」则对坏哈希更鲁棒 —— 各有取舍。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
