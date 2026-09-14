// 13_bloom_lsh.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 13 章 哈希与散列》
// 主题：用哈希做「近似」——布隆过滤器（Bloom filter）与局部敏感哈希（LSH）
//
//   精确哈希表追求「键 -> 唯一位置」。但有些问题**不需要精确**：
//
//   ① 布隆过滤器（Bloom filter）：「这个元素**可能**在集合里吗？」
//      用 m 个比特 + k 个哈希函数：插入时把 k 个位置都置 1；
//      查询时只要有一个位置是 0 -> **一定不在**；全是 1 -> **可能在**。
//      * 优点：**极省空间**（几比特/元素），查询 O(k)；
//      * 特点：**没有假阴性**（在的一定说在），**有假阳性**（不在的可能说在）；
//      * 假阳性率 ≈ (1 - e^{-kn/m})^k，最优 k = (m/n)·ln2。
//      用处：缓存穿透防护、爬虫 URL 去重、数据库「先查布隆再看磁盘」、区块链 SPV。
//
//   ② 局部敏感哈希（LSH）：让「**相似**的键落到**同一桶**」。
//      与普通哈希（追求「不同键分散」）相反，LSH 追求「**相似键聚集**」。
//      * MinHash：用「最小哈希签名」估计两个集合的 **Jaccard 相似度**；
//      * 分带（banding）：把签名切成 b 段，任一段完全相同 -> 判为「候选近邻」，
//        从而把「找近重复」从 O(n^2) 降到近线性。
//      用处：文档去重、相似图片 / 视频检索、推荐召回、十亿级向量检索。
//
//   一句话：**布隆答「在不在」，LSH 答「像不像」——都是用哈希换空间的近似术。**
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 13_bloom_lsh.cpp -o 13_bloom_lsh && ./13_bloom_lsh
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 13_bloom_lsh.cpp -o bl_san && ./bl_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>          // std::pow, std::exp
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ===========================================================================
// 布隆过滤器
//   用「双重散列」模拟 k 个独立哈希：第 i 个位置 = (h1 + i*h2) % m。
// ===========================================================================
class BloomFilter {
public:
    BloomFilter(std::size_t m, int k)
        : bits_((m + 63) / 64, 0), m_(m), k_(k) {}

    void add(std::size_t h1, std::size_t h2) {
        for (int i = 0; i < k_; ++i) set((h1 + static_cast<std::size_t>(i) * h2) % m_);
    }
    bool query(std::size_t h1, std::size_t h2) const {
        for (int i = 0; i < k_; ++i)
            if (!get((h1 + static_cast<std::size_t>(i) * h2) % m_)) return false;   // 有一位是 0 -> 一定不在
        return true;                                                                // 全是 1 -> 可能在
    }
    // 被置 1 的比特数（用于计算实际填充率）。
    std::size_t popcount() const {
        std::size_t c = 0;
        for (std::uint64_t w : bits_) c += static_cast<std::size_t>(__builtin_popcountll(w));
        return c;
    }
    double fillRatio() const { return static_cast<double>(popcount()) / static_cast<double>(m_); }

private:
    void set(std::size_t i) { bits_[i >> 6] |= (std::uint64_t(1) << (i & 63)); }
    bool get(std::size_t i) const { return (bits_[i >> 6] >> (i & 63)) & 1u; }
    std::vector<std::uint64_t> bits_;
    std::size_t m_;
    int k_;
};

// 把 int 键变成两个独立的哈希值（h2 取奇数，避免与 2 的幂容量不互素）。
static void hashPair(int x, std::size_t& h1, std::size_t& h2) {
    h1 = std::hash<int>{}(x);
    h2 = (std::hash<int>{}(x * 2654435761u) | 1u);
}

// splitmix64：把 64 位整数「充分打乱」的强混合函数（一个双射）。
//   MinHash 要的是「近似随机置换」，用它对 (seed + x) 混合最稳；
//   若改用仿射哈希 (a*x+b) mod p，在「连续小整数」这种有结构的数据上会有偏差。
static std::uint64_t splitmix64(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main() {
    std::cout << "======== 13 布隆过滤器 与 局部敏感哈希 ========\n\n";

    // ---------- 1. ★ 布隆过滤器：没有假阴性，只有假阳性 ----------
    std::cout << "=== 1. ★ 布隆过滤器：在的必说在，不在的可能说在 ===\n";
    {
        const std::size_t m = 100000;                // 10 万比特
        const int k = 7;
        const int n = 10000;                         // 插入 1 万个
        BloomFilter bf(m, k);
        for (int x = 0; x < n; ++x) { std::size_t h1, h2; hashPair(x, h1, h2); bf.add(h1, h2); }

        // (a) 零假阴性：所有插入过的元素，查询必须为真
        int falseNeg = 0;
        for (int x = 0; x < n; ++x) { std::size_t h1, h2; hashPair(x, h1, h2); if (!bf.query(h1, h2)) ++falseNeg; }
        // (b) 假阳性：查询从未插入的元素
        const int trials = 100000;
        int falsePos = 0;
        for (int x = 1000000; x < 1000000 + trials; ++x) { std::size_t h1, h2; hashPair(x, h1, h2); if (bf.query(h1, h2)) ++falsePos; }

        const double theory = std::pow(1.0 - std::exp(-static_cast<double>(k) * n / static_cast<double>(m)), k);
        std::cout << "     m=" << m << " 比特, k=" << k << ", 插入 n=" << n << " 个元素\n";
        std::cout << "     假阴性数（在却说不在）= " << falseNeg << "（必须为 0）\n";
        std::cout << "     实测假阳性率 = " << static_cast<double>(falsePos) / trials
                  << "   理论值 (1-e^{-kn/m})^k = " << theory << "\n";
        std::cout << "     填充率（置 1 的比例）= " << bf.fillRatio() << "\n";
        assert(falseNeg == 0);
    }

    // ---------- 2. ★ 最优哈希函数个数 k* = (m/n)·ln2 ----------
    std::cout << "\n=== 2. ★ 哈希函数个数 k 对假阳性率的影响 ===\n";
    {
        const std::size_t m = 100000;
        const int n = 10000;                          // m/n = 10 -> 理论最优 k* = 10·ln2 ≈ 6.93
        const int trials = 200000;
        std::cout << "     k      实测假阳性率      理论假阳性率\n";
        int bestK = 0; double bestRate = 1e9;
        for (int k : {1, 2, 4, 6, 7, 8, 10, 12, 16}) {
            BloomFilter bf(m, k);
            for (int x = 0; x < n; ++x) { std::size_t h1, h2; hashPair(x, h1, h2); bf.add(h1, h2); }
            int fp = 0;
            for (int x = 1000000; x < 1000000 + trials; ++x) { std::size_t h1, h2; hashPair(x, h1, h2); if (bf.query(h1, h2)) ++fp; }
            const double rate = static_cast<double>(fp) / trials;
            const double theory = std::pow(1.0 - std::exp(-static_cast<double>(k) * n / static_cast<double>(m)), k);
            std::cout << "     " << k << "\t" << rate << "        \t" << theory << "\n";
            if (rate < bestRate) { bestRate = rate; bestK = k; }
        }
        std::cout << "  -> 实测最优 k ≈ " << bestK << "，理论最优 k* = (m/n)·ln2 ≈ "
                  << 10.0 * 0.693147 << "。k 太少 -> 冲突多；k 太多 -> 比特被填满，两者都会升。\n";
    }

    // ---------- 3. ★ 布隆过滤器的省空间：对比精确哈希集合 ----------
    std::cout << "\n=== 3. ★ 空间对比：布隆 vs 精确集合 ===\n";
    {
        const int n = 1000000;
        const std::size_t m = 10 * static_cast<std::size_t>(n);     // 约 10 比特/元素
        const int k = 7;
        std::cout << "     存 " << n << " 个元素：\n";
        std::cout << "       精确哈希集合（存键）：约 " << (static_cast<long long>(n) * 4 / 1000000.0) << " MB（按 4 字节/键算）\n";
        std::cout << "       布隆过滤器（" << m << " 比特）：" << (static_cast<double>(m) / 8.0 / 1000000.0) << " MB，"
                  << "且假阳性率约 " << std::pow(1.0 - std::exp(-static_cast<double>(k) * n / static_cast<double>(m)), k) << "\n";
        std::cout << "  -> 布隆用「约 1/40 的空间 + 一点点假阳性」换来「判断在不在」。\n";
        std::cout << "     若需要「删除」或「精确」，布隆不行——可以用计数布隆（counting Bloom）。\n";
    }

    // ---------- 4. ★ MinHash：估计集合的 Jaccard 相似度 ----------
    std::cout << "\n=== 4. ★ MinHash：用签名估计 Jaccard 相似度 ===\n";
    {
        // 造两个集合，共用一部分元素
        const int universe = 500;
        std::vector<int> a, b;
        for (int x = 0; x < 200; ++x) a.push_back(x);            // A = {0..199}
        for (int x = 100; x < 300; ++x) b.push_back(x);          // B = {100..299}
        // 真 Jaccard = |A∩B| / |A∪B| = 100 / 300 = 0.3333
        auto trueJaccard = [](const std::vector<int>& x, const std::vector<int>& y) {
            std::size_t i = 0, j = 0, inter = 0;
            while (i < x.size() && j < y.size()) {
                if (x[i] == y[j]) { ++inter; ++i; ++j; }
                else if (x[i] < y[j]) ++i; else ++j;
            }
            return static_cast<double>(inter) / static_cast<double>(x.size() + y.size() - inter);
        };
        (void)universe;
        std::cout << "    真 Jaccard(A,B) = " << trueJaccard(a, b) << "（|交|=100, |并|=300）\n";

        // k 个哈希函数 h_i(x) = splitmix64(seed_i + x)（近似随机置换）
        std::mt19937_64 rng(20260913);
        std::cout << "     签名长度 k   估计值      误差\n";
        for (int k : {16, 64, 256, 1024}) {
            std::vector<std::uint64_t> seed(k);
            for (int i = 0; i < k; ++i) seed[i] = rng();
            auto sig = [&](const std::vector<int>& s) {
                std::vector<std::uint64_t> sg(k, ~0ULL);
                for (int x : s)
                    for (int i = 0; i < k; ++i) {
                        const std::uint64_t h = splitmix64(seed[i] + static_cast<std::uint64_t>(x));
                        if (h < sg[i]) sg[i] = h;
                    }
                return sg;
            };
            const auto sa = sig(a), sb = sig(b);
            int match = 0;
            for (int i = 0; i < k; ++i) if (sa[i] == sb[i]) ++match;
            const double est = static_cast<double>(match) / k;
            const double truth = trueJaccard(a, b);
            std::cout << "     " << k << "\t\t" << est << "    " << (est - truth) << "\n";
        }
        std::cout << "  -> 签名越长，估计越准（误差 ~ O(1/sqrt(k))）。MinHash 的妙处：\n";
        std::cout << "     P(min h_i(A) == min h_i(B)) = Jaccard(A,B) —— 签名相同的比例就是相似度。\n";
    }

    // ---------- 5. ★ LSH 分带：把「找近重复」从 O(n^2) 降到近线性 ----------
    std::cout << "\n=== 5. ★ LSH 分带：在文档集合里找近重复 ===\n";
    {
        // 造一批文档：30 个「基础文档」，每个再复制一份并扰动，形成 30 对近重复；
        // 另加一批随机噪声文档。
        const int U = 300;                     // 元素宇宙大小
        std::mt19937 rng(20260915);
        auto randSet = [&](int size) {
            std::vector<int> s;
            for (int i = 0; i < size; ++i) s.push_back(static_cast<int>(rng() % static_cast<unsigned>(U)));
            std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end());
            return s;
        };
        std::vector<std::vector<int>> docs;
        std::vector<int> truthPartner;         // 每个文档的「真近重复」伙伴下标（-1 表示无）
        const int pairs = 30;
        for (int p = 0; p < pairs; ++p) {
            auto base = randSet(60);
            auto dup = base;
            for (int e = 0; e < 8; ++e) dup.push_back(static_cast<int>(rng() % static_cast<unsigned>(U)));   // 加噪
            std::sort(dup.begin(), dup.end()); dup.erase(std::unique(dup.begin(), dup.end()), dup.end());
            const int i = static_cast<int>(docs.size()); docs.push_back(base);
            const int j = static_cast<int>(docs.size()); docs.push_back(dup);
            truthPartner.push_back(j); truthPartner.push_back(i);
        }
        for (int d = 0; d < 100; ++d) { docs.push_back(randSet(60)); truthPartner.push_back(-1); }
        const int M = static_cast<int>(docs.size());

        // MinHash 签名（k=128），再分带 b=32 段、每段 r=4 行
        const int K = 128, B = 32, R = 4;
        assert(B * R == K);
        std::vector<std::uint64_t> seed(K);
        for (int i = 0; i < K; ++i) seed[i] = rng();
        std::vector<std::vector<std::uint64_t>> sig(M, std::vector<std::uint64_t>(K));
        for (int d = 0; d < M; ++d) {
            for (int i = 0; i < K; ++i) {
                std::uint64_t mn = ~0ULL;
                for (int x : docs[static_cast<std::size_t>(d)]) {
                    const std::uint64_t h = splitmix64(seed[i] + static_cast<std::uint64_t>(x));
                    if (h < mn) mn = h;
                }
                sig[static_cast<std::size_t>(d)][static_cast<std::size_t>(i)] = mn;
            }
        }
        // 用「每段一个哈希桶」找候选对
        std::vector<std::vector<int>> cand(M);
        for (int band = 0; band < B; ++band) {
            std::vector<std::pair<std::uint64_t,int>> keys;
            for (int d = 0; d < M; ++d) {
                std::uint64_t h = 1469598103934665603ULL;
                for (int r = 0; r < R; ++r) { h ^= sig[static_cast<std::size_t>(d)][static_cast<std::size_t>(band * R + r)]; h *= 1099511628211ULL; }
                keys.push_back({h, d});
            }
            std::sort(keys.begin(), keys.end());
            for (std::size_t t = 1; t < keys.size(); ++t)
                if (keys[t].first == keys[t - 1].first) {                       // 同段同哈希 -> 候选
                    const int x = std::min(keys[t].second, keys[t - 1].second);
                    const int y = std::max(keys[t].second, keys[t - 1].second);
                    if (cand[static_cast<std::size_t>(x)].end() == std::find(cand[static_cast<std::size_t>(x)].begin(), cand[static_cast<std::size_t>(x)].end(), y))
                        cand[static_cast<std::size_t>(x)].push_back(y);
                }
        }
        // 统计：真近重复对是否被找回（召回），候选里有多少是假阳（精度）
        int trueDup = 0, found = 0, candTotal = 0, candTrue = 0;
        for (int x = 0; x < M; ++x) {
            for (int y : cand[static_cast<std::size_t>(x)]) {
                ++candTotal;
                if (truthPartner[static_cast<std::size_t>(x)] == y) { ++candTrue; }
            }
        }
        for (int x = 0; x < M; ++x) if (truthPartner[static_cast<std::size_t>(x)] > x) ++trueDup;
        for (int x = 0; x < M; ++x) {
            const int y = truthPartner[static_cast<std::size_t>(x)];
            if (y > x && std::find(cand[static_cast<std::size_t>(x)].begin(), cand[static_cast<std::size_t>(x)].end(), y) != cand[static_cast<std::size_t>(x)].end()) ++found;
        }
        std::cout << "     文档数 M = " << M << "（含 " << pairs << " 对近重复 + 噪声），签名 k=" << K
                  << "，分带 b=" << B << " 段 × r=" << R << " 行\n";
        std::cout << "     真近重复对 = " << trueDup << "，被找回 = " << found
                  << "  召回率 = " << (trueDup ? static_cast<double>(found) / trueDup : 0.0) << "\n";
        std::cout << "     候选对总数 = " << candTotal << "，其中真近重复 = " << candTrue
                  << "  精度 = " << (candTotal ? static_cast<double>(candTrue) / candTotal : 0.0) << "\n";
        std::cout << "  -> 只比较「候选对」就把 O(M^2)≈" << (static_cast<long long>(M) * (M - 1) / 2)
                  << " 次两两比较砍到 " << candTotal << " 次，且近重复基本没漏。\n";
        std::cout << "     b 段 r 行决定「相似度阈值」：候选概率 = 1 - (1 - s^r)^b，是条 S 形曲线。\n";
        assert(found == trueDup);            // 这批数据里近重复应全部被找回
    }

    std::cout << "\n=== 6. 要点 ===\n"
                 "  * 布隆过滤器：m 比特 + k 哈希；**无假阴性、有假阳性**；\n"
                 "    假阳性率 ≈ (1-e^{-kn/m})^k，最优 k* = (m/n)ln2；不支持删除（除非计数布隆）；\n"
                 "  * MinHash：用「最小哈希签名」估计 **Jaccard 相似度**，签名相同比例 ≈ 相似度；\n"
                 "  * LSH 分带：b 段 × r 行，任一段撞上即候选；把「找近邻」从 O(n^2) 降到近线性；\n"
                 "  * 共同哲学：**放弃精确、换取空间与速度**——是「近似算法」的入门范式；\n"
                 "  * AI 落点：去重 / 召回 / 缓存穿透防护 / 十亿级向量检索（PQ + LSH）/ 推荐系统。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
