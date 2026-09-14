// 08_huffman.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 08 章 优先级队列与堆》
// 主题：Huffman 编码 —— 贪心 + 最小堆的教科书范例
//
//   给定每个符号的出现频率，为每个符号分配一串 0/1 变长码，使**平均码长最短**，
//   同时保证「前缀码」性质（任何码都不是另一个码的前缀），从而可以无歧义解码。
//
//   算法（贪心）：
//     ① 每个符号建一个叶子结点，权 = 频率，全部放进**最小堆**；
//     ② 反复取出堆中**权最小的两个**结点，合并成一个新结点（权 = 两者之和）
//        放回堆；直到堆里只剩一个结点 —— 它就是 Huffman 树的根；
//     ③ 从根出发，向左记 0、向右记 1，走到叶子即得该符号的码。
//
//   为什么最优？核心是「频率低的符号更深」——合并次数最多的（最早被合并的）
//   永远是频率最小的，于是它们被放到最深处，码最长。这就是贪心的正确性。
//
//   本文件还验证 Huffman 编码与信息论熵的关系：H ≤ L_huffman < H + 1。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 08_huffman.cpp -o 08_huffman && ./08_huffman
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 08_huffman.cpp -o hf_san && ./hf_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Huffman 树结点
// ---------------------------------------------------------------------------
struct HNode {
    long long              freq;          // 频率（权重）
    int                    symbol;        // >= 0 表示叶子，对应符号；-1 表示内部结点
    HNode *left = nullptr, *right = nullptr;
    HNode(long long f, int s) : freq(f), symbol(s) {}
};

struct Cmp {
    // std::priority_queue 默认用 std::less 得到"最大"在顶；这里要**最小**在顶，
    // 所以返回「a 的优先级低于 b」= a->freq > b->freq。
    bool operator()(HNode* a, HNode* b) const { return a->freq > b->freq; }
};

// 由「符号 -> 频率」构造 Huffman 树，返回根（调用方负责 destroy）
static HNode* buildHuffman(const std::vector<long long>& freq) {
    std::priority_queue<HNode*, std::vector<HNode*>, Cmp> pq;
    for (std::size_t i = 0; i < freq.size(); ++i)
        if (freq[i] > 0) pq.push(new HNode(freq[i], static_cast<int>(i)));

    if (pq.empty()) return nullptr;
    if (pq.size() == 1) {                       // 只有一个符号的退化情形
        HNode* only = pq.top(); pq.pop();
        HNode* root = new HNode(only->freq, -1);
        root->left = only;
        return root;
    }
    while (pq.size() > 1) {
        HNode* a = pq.top(); pq.pop();           // 最小
        HNode* b = pq.top(); pq.pop();           // 次小
        HNode* m = new HNode(a->freq + b->freq, -1);
        m->left = a; m->right = b;               // a、b 顺序不影响平均码长
        pq.push(m);
    }
    return pq.top();
}

static void destroy(HNode* r) {
    if (!r) return;
    destroy(r->left);
    destroy(r->right);
    delete r;
}

// 从根出发走一遍，把每个叶子的码写进 codes[symbol]
static void assignCodes(HNode* r, std::string& path, std::vector<std::string>& codes) {
    if (!r) return;
    if (r->symbol >= 0) { codes[static_cast<std::size_t>(r->symbol)] = path; return; }
    path.push_back('0'); assignCodes(r->left,  path, codes); path.pop_back();
    path.push_back('1'); assignCodes(r->right, path, codes); path.pop_back();
}

// 前缀码校验：任意两个码，不能一个是另一个的前缀
static bool isPrefixFree(const std::vector<std::string>& codes) {
    for (std::size_t i = 0; i < codes.size(); ++i) {
        if (codes[i].empty()) continue;
        for (std::size_t j = 0; j < codes.size(); ++j) {
            if (i == j || codes[j].empty()) continue;
            if (codes[i].size() <= codes[j].size() &&
                codes[j].compare(0, codes[i].size(), codes[i]) == 0)
                return false;
        }
    }
    return true;
}

// 解码：沿 Huffman 树逐位走，遇叶子输出符号并回到根（演示前缀码可无歧义解码）。
// 本文件用字母 'a'..'z'（下标 0..25），故把下标映射回字母。
static std::string decode(HNode* root, const std::string& bits, bool* ok) {
    std::string out;
    *ok = true;
    HNode* cur = root;
    for (char c : bits) {
        cur = (c == '0') ? cur->left : cur->right;
        if (!cur) { *ok = false; return out; }
        if (cur->symbol >= 0) {
            out.push_back(static_cast<char>('a' + cur->symbol));   // 下标 -> 字母
            cur = root;
        }
    }
    if (cur != root) { *ok = false; }            // 码流不完整
    return out;
}

int main() {
    std::cout << "======== 08 Huffman 编码（最小堆 + 贪心）========\n\n";

    // 英文字母的典型出现频率（%），转成整数权重
    const char letters[] = "abcdefghijklmnopqrstuvwxyz";
    const double pct[] = {
        8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 6.97, 0.15, 0.77, 4.03, 2.41,
        6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06, 2.76, 0.98, 2.36, 0.15, 1.97, 0.07
    };
    const std::size_t n = 26;
    std::vector<long long> freq(n);
    long long total = 0;
    for (std::size_t i = 0; i < n; ++i) { freq[i] = std::llround(pct[i] * 10000.0); total += freq[i]; }

    HNode* root = buildHuffman(freq);
    std::vector<std::string> codes(n);
    std::string path;
    assignCodes(root, path, codes);

    std::cout << "=== 1. 各字母的 Huffman 码 ===\n";
    std::cout << "  符号   频率%    码长  码\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "    " << letters[i] << "   " << std::fixed << std::setprecision(2)
                  << std::setw(6) << pct[i] << std::setw(6) << codes[i].size()
                  << "   " << codes[i] << "\n";
    }

    // ---------- 平均码长 vs 定长 vs 熵 ----------
    double L = 0.0, H = 0.0;                     // 平均码长 / 熵（单位：bit）
    for (std::size_t i = 0; i < n; ++i) {
        const double p = double(freq[i]) / double(total);
        L += p * double(codes[i].size());
        if (p > 0) H += -p * std::log2(p);
    }
    const double fixedLen = std::ceil(std::log2(double(n)));   // 定长编码需要 5 bit

    std::cout << "\n=== 2. 平均码长 vs 定长编码 vs 信息熵 ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  定长编码 (ceil(log2 26))      = " << fixedLen << " bit/符号\n";
    std::cout << "  Huffman 平均码长 L            = " << L << " bit/符号\n";
    std::cout << "  信息熵 H = -Σ p·log2(p)        = " << H << " bit/符号\n";
    std::cout << "  压缩比 (定长/Huffman)          = " << (fixedLen / L) << "x\n";
    std::cout << "  冗余 L - H                     = " << (L - H) << " bit/符号\n";

    // ---------- 断言：前缀码性质 + 熵的上下界 ----------
    assert(isPrefixFree(codes));
    std::cout << "\n  [OK] 前缀码性质成立（无码是另一码的前缀）\n";
    assert(H <= L + 1e-9 && L < H + 1.0 + 1e-9);      // 信息论定理：H <= L < H+1
    std::cout << "  [OK] 满足信息论界 H <= L < H+1（Huffman 最优性的体现）\n";

    // ---------- 无损解码验证 ----------
    std::string msg = "thisisaverygoodworldmodel";
    std::string bits;
    for (char c : msg) bits += codes[static_cast<std::size_t>(c - 'a')];
    bool ok = false;
    const std::string back = decode(root, bits, &ok);
    std::cout << "\n=== 3. 编解码往返验证 ===\n";
    std::cout << "  原文: " << msg << "（" << msg.size() << " 字符）\n";
    std::cout << "  编码: " << bits.size() << " bit（定长需 " << msg.size() * 5 << " bit）\n";
    std::cout << "  解码: " << back << "\n";
    assert(ok && back == msg);
    std::cout << "  [OK] 解码结果与原文一致，无损\n";

    std::cout << "\n=== 4. 直观结论 ===\n"
                 "  * 高频字母（e、t、a）拿到短码，低频字母（q、z、j）拿到长码；\n"
                 "  * 这正是「频率越低、越深、码越长」的贪心结果的直观体现；\n"
                 "  * 平均码长 L 落在 [H, H+1) 内 —— Huffman 是**最优前缀码**，\n"
                 "    与信息论下界 H 至多差 1 bit。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";

    destroy(root);
    return 0;
}
