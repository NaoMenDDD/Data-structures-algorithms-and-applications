// 06_replay_buffer.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 06 章 队列与双端队列》
// 主题：环形缓冲（ring buffer）—— 经验回放池（experience replay buffer）
//
//   这是 DQN（Mnih et al., 2013/2015）能训起来的关键组件之一。
//   智能体每一步产生一条「转移」：
//       (state, action, reward, next_state, done)
//   全部塞进一个**固定容量**的环形缓冲。超出容量时，覆盖**最旧**的那条。
//
//   为什么是队列（FIFO）而不是栈？
//     * 回放要打破「连续样本的时间相关性（temporal correlation）」。
//       若按产生顺序学，相邻样本高度相关，等价于一直朝同一方向做梯度，
//       训练会震荡/发散。随机化采样出的 mini-batch 才近似 i.i.d.。
//     * 环形缓冲的容量是「记忆的长度」：太短 -> 遗忘久远经验（灾难性遗忘），
//       太长 -> 存着大量过时策略的样本（off-policy 分布漂移）。
//
//   本文件演示：
//     1. RingBuffer<T> 泛型环形缓冲：O(1) push / 覆盖最旧 / 随机访问；
//     2. Transition 结构 + ReplayBuffer：均匀采样 mini-batch（无放回）；
//     3. 容量压力测试：push 远超容量，验证「只剩最近 capacity 条」的 FIFO 语义；
//     4. 演示「遗忘」：新策略覆盖旧数据后，采样分布的变化（off-policy 的现实问题）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 06_replay_buffer.cpp -o 06_replay_buffer
//   ./06_replay_buffer
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 06_replay_buffer.cpp -o rb_san && ./rb_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===========================================================================
// 1. 泛型环形缓冲 RingBuffer<T>
// ===========================================================================
//
// 语义：容量固定；写满后继续写会覆盖最旧的元素，size 恒为 min(pushed, capacity)。
//      - push(x)          O(1)
//      - operator[](i)    按「从旧到新」的逻辑顺序随机访问，O(1)
//      - oldest()/newest() 两端 O(1)
//
// 实现要点：底层是 std::vector<T>，head_ 指向最旧元素，size_ 是当前个数。
//   逻辑下标 i 映射到物理下标：(head_ + i) % capacity。
//   这与 06_circular_queue.cpp 的写法一致，但这里更强调「覆盖式」而不判满。
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) : buf_(capacity), head_(0), size_(0) {
        if (capacity == 0) throw std::invalid_argument("RingBuffer 容量必须 >= 1");
    }

    std::size_t size()     const { return size_; }
    std::size_t capacity() const { return buf_.size(); }
    bool full()  const { return size_ == buf_.size(); }
    bool empty() const { return size_ == 0; }

    // 写入一条；若已满，覆盖最旧并返回 true（表示发生了覆盖）
    bool push(T value) {
        const std::size_t cap = buf_.size();
        std::size_t write;
        bool overwrote = false;
        if (size_ < cap) {
            write = (head_ + size_) % cap;       // 尾部空位
            ++size_;
        } else {
            write = head_;                       // 覆盖最旧
            head_ = (head_ + 1) % cap;           // 最旧前进一格
            overwrote = true;
        }
        buf_[write] = std::move(value);
        return overwrote;
    }

    // 逻辑顺序（从旧到新）访问：i = 0 是最旧
    const T& operator[](std::size_t i) const {
        return buf_[(head_ + i) % buf_.size()];
    }
    const T& oldest() const { if (empty()) throw std::out_of_range("RingBuffer::oldest 空"); return (*this)[0]; }
    const T& newest() const { if (empty()) throw std::out_of_range("RingBuffer::newest 空"); return (*this)[size_ - 1]; }

private:
    std::vector<T> buf_;
    std::size_t head_;   // 最旧元素所在物理下标
    std::size_t size_;
};

// ===========================================================================
// 2. 转移（Transition）与经验回放池
// ===========================================================================
struct Transition {
    int   state;       // 演示用：状态用整数编号（真实里是观测张量）
    int   action;
    float reward;
    int   nextState;
    bool  done;
};

// 确定性伪随机数发生器（LCG + 简单混合），保证测试可复现
class Rng {
public:
    explicit Rng(std::uint64_t seed) : s_(seed) {}
    std::uint64_t next() {
        s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return s_;
    }
    // [0, bound)
    std::size_t below(std::size_t bound) {
        return static_cast<std::size_t>(next() >> 33) % bound;
    }
    float unit() { return static_cast<float>((next() >> 40) & 0xFFFFFF) / float(0x1000000); }
private:
    std::uint64_t s_;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(std::size_t capacity) : buf_(capacity) {}

    void push(const Transition& t) { buf_.push(t); }
    std::size_t size()     const { return buf_.size(); }
    std::size_t capacity() const { return buf_.capacity(); }

    // 均匀采样 batch 条，**无放回**（同一 batch 内不重复）。
    // 用 Fisher-Yates 的部分洗牌：只对前 batch 个位置做随机交换，O(batch)。
    std::vector<Transition> sample(std::size_t batch, Rng& rng) {
        if (batch > buf_.size()) throw std::out_of_range("sample 数量超过池内样本数");
        // 拷贝一份「逻辑下标 0..size-1」，部分洗牌
        std::vector<std::size_t> idx(buf_.size());
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        for (std::size_t i = 0; i < batch; ++i) {
            std::size_t j = i + rng.below(idx.size() - i);   // 从 [i, n) 里选
            std::swap(idx[i], idx[j]);
        }
        std::vector<Transition> out;
        out.reserve(batch);
        for (std::size_t i = 0; i < batch; ++i) out.push_back(buf_[idx[i]]);
        return out;
    }

    const Transition& operator[](std::size_t i) const { return buf_[i]; }

private:
    RingBuffer<Transition> buf_;
};

// ===========================================================================
// 3. 自测：环形语义 + 覆盖计数
// ===========================================================================
static void testRingSemantics() {
    std::cout << "=== 1. RingBuffer 的基本语义 ===\n";
    RingBuffer<int> rb(4);
    assert(rb.empty() && rb.size() == 0);

    // 未满：FIFO 累积，不覆盖
    assert(!rb.push(10) && !rb.push(20) && !rb.push(30));
    assert(rb.size() == 3 && !rb.full());
    assert(rb.oldest() == 10 && rb.newest() == 30);

    // 写满
    assert(!rb.push(40));
    assert(rb.full() && rb.size() == 4);
    std::cout << "  填入 [10,20,30,40]，size=" << rb.size()
              << " full=" << std::boolalpha << rb.full() << "\n";

    // 溢出：写 50 覆盖最旧的 10
    bool ov = rb.push(50);
    assert(ov);                                     // 返回 true 表示发生了覆盖
    assert(rb.size() == 4);                         // 容量不变
    assert(rb.oldest() == 20 && rb.newest() == 50); // 10 被挤掉
    std::cout << "  再 push 50 后：最旧 = " << rb.oldest()
              << "，最新 = " << rb.newest() << "（10 被覆盖 ✓）\n";

    // 逻辑顺序遍历：应严格是 20,30,40,50
    std::cout << "  逻辑顺序: ";
    for (std::size_t i = 0; i < rb.size(); ++i) std::cout << rb[i] << (i + 1 < rb.size() ? "," : "");
    std::cout << "\n";
    assert(rb[0] == 20 && rb[1] == 30 && rb[2] == 40 && rb[3] == 50);

    // 容量 1 的极端情况
    RingBuffer<int> one(1);
    assert(!one.push(1) && one.push(2) && one.oldest() == 2 && one.size() == 1);
    std::cout << "  容量 1：每次 push 都覆盖，size 恒为 1 ✓\n\n";
}

// ===========================================================================
// 4. 容量压力测试：push 远超容量，验证「只剩最近 capacity 条」
// ===========================================================================
static void testOverwritePressure() {
    std::cout << "=== 2. 容量压力：push 100000 条进容量 1000 的池 ===\n";
    const std::size_t cap = 1000, total = 100000;
    ReplayBuffer pool(cap);
    long long overwrites = 0;
    for (std::size_t i = 0; i < total; ++i) {
        Transition t{static_cast<int>(i), 0, 0.0f, static_cast<int>(i) + 1, false};
        if (pool.size() == pool.capacity()) ++overwrites;   // 覆盖前池已满
        pool.push(t);
    }
    // 覆盖发生次数 = total - cap
    assert(overwrites == static_cast<long long>(total - cap));
    assert(pool.size() == cap && pool.capacity() == cap);
    // 池里只剩最近 cap 条：最旧 state = total - cap，最新 state = total - 1
    assert(pool[0].state == static_cast<int>(total - cap));
    assert(pool[cap - 1].state == static_cast<int>(total - 1));
    std::cout << "  push " << total << " 条，容量 " << cap
              << " -> 覆盖 " << overwrites << " 次 (= " << total << " - " << cap << ")\n";
    std::cout << "  池中最旧 state = " << pool[0].state
              << "，最新 state = " << pool[cap - 1].state << "（FIFO 遗忘旧数据 ✓）\n\n";
}

// ===========================================================================
// 5. 均匀采样 mini-batch + 「分布漂移」的可视化
// ===========================================================================
static void testSampling() {
    std::cout << "=== 3. 均匀采样 mini-batch（无放回） ===\n";
    const std::size_t cap = 200;
    ReplayBuffer pool(cap);
    for (std::size_t i = 0; i < cap; ++i) {
        // state = i；reward 编码一个「早期 vs 后期」的差异
        pool.push(Transition{static_cast<int>(i), static_cast<int>(i % 4),
                             static_cast<float>(i), static_cast<int>(i) + 1, false});
    }

    Rng rng(20260913);
    const std::size_t batch = 32;
    std::vector<Transition> b = pool.sample(batch, rng);
    assert(b.size() == batch);

    // 无放回：batch 内 state 互不相同
    std::vector<bool> seen(cap, false);
    long long rewardSum = 0;
    for (const auto& t : b) {
        assert(!seen[static_cast<std::size_t>(t.state)]);   // 不重复
        seen[static_cast<std::size_t>(t.state)] = true;
        rewardSum += static_cast<long long>(t.reward);
    }
    std::cout << "  采样 " << batch << " 条（容量 " << cap << "）：batch 内 state 无重复 ✓\n";
    std::cout << "  该 batch 的 reward 之和 = " << rewardSum << "\n";

    // 多次采样的平均 state：应接近容量中点，说明是均匀的（而非偏向新/旧）
    const int reps = 400;
    long long total = 0, cnt = 0;
    for (int r = 0; r < reps; ++r)
        for (const auto& t : pool.sample(batch, rng)) { total += t.state; ++cnt; }
    const double mean = static_cast<double>(total) / static_cast<double>(cnt);
    std::cout << "  大量采样的平均 state = " << mean
              << "（理论中位数附近 ~" << (cap - 1) / 2.0 << "，说明均匀）\n\n";
}

// 演示 off-policy 的现实问题：数据是「旧策略」产生的，策略一变旧数据就带偏。
// 这里用「reward 均值随训练轮次漂移」来抽象：buffer 里的数据越旧，与当前策略越不匹配。
static void testDistributionShift() {
    std::cout << "=== 4. 分布漂移：旧经验 vs 新经验 ===\n";
    const std::size_t cap = 500;
    RingBuffer<Transition> buf(cap);
    Rng rng(7);

    // 阶段一：旧策略，reward 偏低（均值 0）
    for (int i = 0; i < 500; ++i)
        buf.push(Transition{i, 0, rng.unit() - 0.5f, i + 1, false});   // [-0.5, 0.5)
    auto meanRewardOld = [&] {
        double s = 0; for (std::size_t i = 0; i < buf.size(); ++i) s += buf[i].reward;
        return s / buf.size();
    };
    const double m1 = meanRewardOld();

    // 阶段二：策略改进，reward 偏高（均值 +1）。随着新数据涌入，旧数据被逐步覆盖。
    std::cout << "  旧策略数据（满池 500 条）平均 reward = " << m1 << "\n";
    for (int i = 0; i < 250; ++i)
        buf.push(Transition{i, 0, 1.0f + (rng.unit() - 0.5f), i + 1, false});
    const double m2 = meanRewardOld();
    std::cout << "  改进策略写入 250 条后（一半旧、一半新）平均 reward = " << m2 << "\n";

    for (int i = 0; i < 250; ++i)
        buf.push(Transition{i, 0, 1.0f + (rng.unit() - 0.5f), i + 1, false});
    const double m3 = meanRewardOld();
    std::cout << "  再写 250 条（旧数据全部被覆盖）平均 reward = " << m3 << "\n";
    assert(m1 < 0.2 && m2 > 0.2 && m3 > 0.8);   // 单调上升，最终收敛到新策略的 +1 附近

    std::cout << "  -> 容量决定的「记忆长度」直接控制新旧数据的混合比例：\n"
                 "     容量太大 -> 采样到大量过时(off-policy)样本，价值估计滞后；\n"
                 "     容量太小 -> 遗忘久远经验，易灾难性遗忘。这是要调的超参。\n\n";
}

int main() {
    std::cout << "======== 06 经验回放池（环形缓冲）========\n\n";
    testRingSemantics();
    testOverwritePressure();
    testSampling();
    testDistributionShift();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
