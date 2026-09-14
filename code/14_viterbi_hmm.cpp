// ===========================================================================
// 14_viterbi_hmm.cpp
// 隐马尔可夫模型（HMM）中的动态规划：维特比解码 + 前向算法
//   - 维特比（Viterbi）：给定观测序列，求「最可能的隐藏状态序列」（求 max）
//   - 前向（Forward）：给定观测序列，求「它的总概率」（求 sum）
//   两者都是沿着时间轴做 DP，把 K^T 条路径归纳成 T*K 个状态。
//
// 为什么与 AI / 世界模型相关：
//   语音识别、手写识别、词性标注、生物序列比对、以及早期「序列世界模型」，
//   都用 HMM 描述「隐藏状态随时间转移、每步产生一个观测」。
//   维特比就是「序列上的 argmax」，前向/后向就是「序列上的求和」——
//   这两件事在今天的 CRF、CTC、乃至带状态空间的序列模型里一模一样地出现
//   （维特比也叫 max-product / Viterbi 解码，前向就是 sum-product / 置信传播的特例）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 14_viterbi_hmm.cpp -o 14_viterbi_hmm && ./14_viterbi_hmm
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace hmm {

struct HMM {
    int K = 0;                                              // 隐藏状态数
    int M = 0;                                              // 观测符号数
    std::vector<double> start;                              // 初始分布 start[k]
    std::vector<std::vector<double>> trans;                 // trans[i][j]
    std::vector<std::vector<double>> emit;                  // emit[k][o]
};

// ---- 暴力：枚举所有 K^T 条状态路径，取最大概率（指数，仅用于小规模对拍）----
double bruteDecode(const HMM& h, const std::vector<int>& obs,
                   std::vector<int>* bestPath = nullptr) {
    const int T = static_cast<int>(obs.size());
    double bestLog = -std::numeric_limits<double>::infinity();
    std::vector<int> path(static_cast<std::size_t>(T), 0);
    std::vector<int> cur(static_cast<std::size_t>(T), 0);
    long long total = 1;
    for (int t = 0; t < T; ++t) total *= h.K;
    for (long long code = 0; code < total; ++code) {
        long long c = code;
        for (int t = 0; t < T; ++t) { cur[static_cast<std::size_t>(t)] = static_cast<int>(c % h.K); c /= h.K; }
        double lp = std::log(h.start[static_cast<std::size_t>(cur[0])]) +
                    std::log(h.emit[static_cast<std::size_t>(cur[0])][static_cast<std::size_t>(obs[0])]);
        for (int t = 1; t < T; ++t)
            lp += std::log(h.trans[static_cast<std::size_t>(cur[static_cast<std::size_t>(t - 1)])][static_cast<std::size_t>(cur[static_cast<std::size_t>(t)])]) +
                  std::log(h.emit[static_cast<std::size_t>(cur[static_cast<std::size_t>(t)])][static_cast<std::size_t>(obs[static_cast<std::size_t>(t)])]);
        if (lp > bestLog) { bestLog = lp; path = cur; }
    }
    if (bestPath) *bestPath = path;
    return bestLog;
}

// ---- 给定一条状态路径，算它的对数概率（用于校验维特比结果的概率）----
double pathLogProb(const HMM& h, const std::vector<int>& obs, const std::vector<int>& path) {
    const int T = static_cast<int>(obs.size());
    double lp = std::log(h.start[static_cast<std::size_t>(path[0])]) +
                std::log(h.emit[static_cast<std::size_t>(path[0])][static_cast<std::size_t>(obs[0])]);
    for (int t = 1; t < T; ++t)
        lp += std::log(h.trans[static_cast<std::size_t>(path[static_cast<std::size_t>(t - 1)])][static_cast<std::size_t>(path[static_cast<std::size_t>(t)])]) +
              std::log(h.emit[static_cast<std::size_t>(path[static_cast<std::size_t>(t)])][static_cast<std::size_t>(obs[static_cast<std::size_t>(t)])]);
    return lp;
}

// ---- 维特比：DP 求「最可能的状态序列」，O(T*K^2) ----
// dp[t][k] = 到 t 时刻、处于状态 k 的「最优路径」对数概率
std::vector<int> viterbi(const HMM& h, const std::vector<int>& obs, double& bestLog) {
    const int T = static_cast<int>(obs.size());
    const int K = h.K;
    std::vector<std::vector<double>> dp(static_cast<std::size_t>(T), std::vector<double>(static_cast<std::size_t>(K), 0.0));
    std::vector<std::vector<int>> bp(static_cast<std::size_t>(T), std::vector<int>(static_cast<std::size_t>(K), -1));   // 回溯指针
    for (int k = 0; k < K; ++k)
        dp[0][static_cast<std::size_t>(k)] = std::log(h.start[static_cast<std::size_t>(k)]) +
                                             std::log(h.emit[static_cast<std::size_t>(k)][static_cast<std::size_t>(obs[0])]);
    for (int t = 1; t < T; ++t)
        for (int j = 0; j < K; ++j) {
            double best = -std::numeric_limits<double>::infinity();
            int arg = 0;
            for (int i = 0; i < K; ++i) {
                const double v = dp[static_cast<std::size_t>(t - 1)][static_cast<std::size_t>(i)] +
                                 std::log(h.trans[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
                if (v > best) { best = v; arg = i; }
            }
            dp[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)] =
                best + std::log(h.emit[static_cast<std::size_t>(j)][static_cast<std::size_t>(obs[static_cast<std::size_t>(t)])]);
            bp[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)] = arg;
        }
    int last = 0;
    bestLog = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k)
        if (dp[static_cast<std::size_t>(T - 1)][static_cast<std::size_t>(k)] > bestLog) {
            bestLog = dp[static_cast<std::size_t>(T - 1)][static_cast<std::size_t>(k)];
            last = k;
        }
    std::vector<int> path(static_cast<std::size_t>(T), 0);
    int k = last;
    for (int t = T - 1; t >= 0; --t) { path[static_cast<std::size_t>(t)] = k; k = bp[static_cast<std::size_t>(t)][static_cast<std::size_t>(k)]; }
    return path;
}

// ---- 暴力：把所有路径的「概率」相加（线性尺度，仅小规模）----
double bruteProb(const HMM& h, const std::vector<int>& obs) {
    const int T = static_cast<int>(obs.size());
    double sum = 0.0;
    long long total = 1;
    for (int t = 0; t < T; ++t) total *= h.K;
    std::vector<int> cur(static_cast<std::size_t>(T), 0);
    for (long long code = 0; code < total; ++code) {
        long long c = code;
        for (int t = 0; t < T; ++t) { cur[static_cast<std::size_t>(t)] = static_cast<int>(c % h.K); c /= h.K; }
        double p = h.start[static_cast<std::size_t>(cur[0])] * h.emit[static_cast<std::size_t>(cur[0])][static_cast<std::size_t>(obs[0])];
        for (int t = 1; t < T; ++t)
            p *= h.trans[static_cast<std::size_t>(cur[static_cast<std::size_t>(t - 1)])][static_cast<std::size_t>(cur[static_cast<std::size_t>(t)])] *
                 h.emit[static_cast<std::size_t>(cur[static_cast<std::size_t>(t)])][static_cast<std::size_t>(obs[static_cast<std::size_t>(t)])];
        sum += p;
    }
    return sum;
}

// ---- 前向算法：DP 求「观测序列的总概率」，O(T*K^2) ----
double forward(const HMM& h, const std::vector<int>& obs) {
    const int T = static_cast<int>(obs.size());
    const int K = h.K;
    std::vector<double> a(static_cast<std::size_t>(K), 0.0), na(static_cast<std::size_t>(K), 0.0);
    for (int k = 0; k < K; ++k)
        a[static_cast<std::size_t>(k)] = h.start[static_cast<std::size_t>(k)] * h.emit[static_cast<std::size_t>(k)][static_cast<std::size_t>(obs[0])];
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < K; ++j) {
            double s = 0.0;
            for (int i = 0; i < K; ++i) s += a[static_cast<std::size_t>(i)] * h.trans[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            na[static_cast<std::size_t>(j)] = s * h.emit[static_cast<std::size_t>(j)][static_cast<std::size_t>(obs[static_cast<std::size_t>(t)])];
        }
        a.swap(na);
    }
    double total = 0.0;
    for (int k = 0; k < K; ++k) total += a[static_cast<std::size_t>(k)];
    return total;
}

} // namespace hmm

// 经典的「天气 / 活动」HMM
static hmm::HMM weatherHMM() {
    hmm::HMM h;
    h.K = 2;                                                 // 0=Rainy, 1=Sunny
    h.M = 3;                                                 // 0=Walk, 1=Shop, 2=Clean
    h.start = {0.6, 0.4};
    h.trans = {{0.7, 0.3}, {0.4, 0.6}};
    h.emit = {{0.1, 0.4, 0.5}, {0.6, 0.3, 0.1}};
    return h;
}

static void testClassic() {
    std::cout << "===== 1) 经典 HMM：天气与活动 =====\n";
    const hmm::HMM h = weatherHMM();
    const std::vector<std::string> stateName = {"Rainy", "Sunny"};
    const std::vector<std::string> obsName = {"Walk", "Shop", "Clean"};
    const std::vector<int> obs = {0, 1, 2, 0, 2};            // Walk, Shop, Clean, Walk, Clean
    std::cout << "  观测序列 = Walk Shop Clean Walk Clean\n";
    double bl = 0.0;
    const std::vector<int> path = hmm::viterbi(h, obs, bl);
    std::cout << "  维特比最优状态序列 = ";
    for (std::size_t t = 0; t < path.size(); ++t) std::cout << stateName[static_cast<std::size_t>(path[t])] << (t + 1 < path.size() ? " " : "");
    std::cout << "\n  最优对数概率 = " << bl << "\n";
    // 与暴力核对
    std::vector<int> bp;
    const double bruteLog = hmm::bruteDecode(h, obs, &bp);
    assert(std::abs(bruteLog - bl) < 1e-9);
    assert(std::abs(hmm::pathLogProb(h, obs, path) - bl) < 1e-9);   // 维特比给的路径确实达到该概率
    const double fw = hmm::forward(h, obs);
    const double bt = hmm::bruteProb(h, obs);
    std::cout << "  前向算法总概率 = " << fw << "\t暴力求和 = " << bt << "\n";
    assert(std::abs(fw - bt) < 1e-12);
    std::cout << "  -> 维特比 = 序列上的「max」（最可能路径）；前向 = 序列上的「sum」（总概率）。\n";
    std::cout << "  [断言通过] 维特比 == 暴力最优；前向总概率 == 暴力求和\n\n";
}

static void testRandom() {
    std::cout << "===== 2) 随机 HMM 对拍（K=2..4, T=1..8） =====\n";
    std::mt19937 rng(20260913);
    std::uniform_real_distribution<double> u(0.05, 1.0);
    auto normalize = [](std::vector<double>& v) {
        double s = 0; for (double x : v) s += x;
        for (double& x : v) x /= s;
    };
    int tested = 0;
    for (int t = 0; t < 2000; ++t) {
        hmm::HMM h;
        h.K = 2 + static_cast<int>(rng() % 3);
        h.M = 2 + static_cast<int>(rng() % 3);
        h.start.resize(static_cast<std::size_t>(h.K));
        for (auto& x : h.start) x = u(rng);
        normalize(h.start);
        h.trans.assign(static_cast<std::size_t>(h.K), std::vector<double>(static_cast<std::size_t>(h.K)));
        h.emit.assign(static_cast<std::size_t>(h.K), std::vector<double>(static_cast<std::size_t>(h.M)));
        for (auto& row : h.trans) { for (auto& x : row) x = u(rng); normalize(row); }
        for (auto& row : h.emit) { for (auto& x : row) x = u(rng); normalize(row); }
        const int T = 1 + static_cast<int>(rng() % 8);
        std::vector<int> obs(static_cast<std::size_t>(T));
        for (int i = 0; i < T; ++i) obs[static_cast<std::size_t>(i)] = static_cast<int>(rng() % static_cast<unsigned>(h.M));

        double bl = 0.0;
        const std::vector<int> path = hmm::viterbi(h, obs, bl);
        assert(std::abs(hmm::bruteDecode(h, obs) - bl) < 1e-9);
        assert(std::abs(hmm::pathLogProb(h, obs, path) - bl) < 1e-9);
        assert(std::abs(hmm::forward(h, obs) - hmm::bruteProb(h, obs)) < 1e-10);
        ++tested;
    }
    std::cout << "  [断言通过] " << tested << " 组随机 HMM：维特比最优 == 暴力；前向总概率 == 暴力求和\n\n";
}

int main() {
    std::cout << "########## HMM：维特比解码 与 前向算法 ##########\n\n";
    testClassic();
    testRandom();
    std::cout << "全部测试通过。\n";
    return 0;
}
