// ===========================================================================
// 06_circular_queue.cpp
// 循环队列 (circular queue / ring buffer) 的两种判满方案:
//   A) CircularQueueSacrifice<T> —— 牺牲一个存储单元区分空/满
//        空: head == tail         满: (tail + 1) % cap == head
//        可用元素数 = 缓冲区大小 - 1
//   B) CircularQueueCounted<T>   —— 额外维护 size_ 计数器
//        空: size_ == 0           满: size_ == capacity_
//        缓冲区可全部用满
// 两者都用 std::vector<T> 做底层存储 (RAII, 无需手动 new/delete)。
//
// 约定: 队首 (front) 出队, 队尾 (back) 入队。
//
// 编译运行:
//   g++ -std=c++17 -O2 -Wall -Wextra 06_circular_queue.cpp -o 06_circular_queue
//   ./06_circular_queue
// ===========================================================================
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>   // std::out_of_range
#include <string>
#include <utility>     // std::move
#include <vector>

// ===========================================================================
// 方案 A: 牺牲一个单元
// ===========================================================================
// 缓冲区实际大小 N = capacity + 1, 永远留一个空位, 从而空/满可区分:
//   * head == tail        -> 空
//   * (tail+1)%N == head  -> 满  (N = capacity + 1)
// 构造参数 capacity 表示"最多容纳的元素个数"; 底层多开一格作为判满哨兵,
// 因此这个方案对外的可用容量就是 capacity, 只是内部浪费一个格子。
template <typename T>
class CircularQueueSacrifice {
public:
    // capacity 表示队列最多能容纳的元素个数 (>= 0)
    explicit CircularQueueSacrifice(std::size_t capacity)
        : buf_(capacity + 1), head_(0), tail_(0) {}

    bool empty() const { return head_ == tail_; }
    bool full() const { return (tail_ + 1) % buf_.size() == head_; }

    // 当前元素个数
    std::size_t size() const {
        return (tail_ + buf_.size() - head_) % buf_.size();
    }
    std::size_t capacity() const { return buf_.size() - 1; }

    // 入队; 队满返回 false (不覆盖旧数据)
    bool push(const T& value) {
        if (full()) return false;
        buf_[tail_] = value;
        tail_ = (tail_ + 1) % buf_.size();
        return true;
    }

    // 出队; 空队列抛异常
    T pop() {
        if (empty()) throw std::out_of_range("CircularQueueSacrifice::pop on empty");
        T v = std::move(buf_[head_]);
        head_ = (head_ + 1) % buf_.size();
        return v;
    }

    const T& front() const {
        if (empty()) throw std::out_of_range("CircularQueueSacrifice::front on empty");
        return buf_[head_];
    }
    const T& back() const {
        if (empty()) throw std::out_of_range("CircularQueueSacrifice::back on empty");
        return buf_[(tail_ + buf_.size() - 1) % buf_.size()];
    }

private:
    std::vector<T> buf_;
    std::size_t head_;   // 队首下标
    std::size_t tail_;   // 下一个写入位置
};

// ===========================================================================
// 方案 B: 维护 size 计数器
// ===========================================================================
// 不浪费单元, 缓冲区大小即为 capacity。空/满由 size_ 唯一决定, 逻辑最直观,
// 代价是多存一个 size_t (8 字节) 并每次操作多维护一次加减。
template <typename T>
class CircularQueueCounted {
public:
    explicit CircularQueueCounted(std::size_t capacity)
        : buf_(capacity ? capacity : 1),   // 容量 0 时给 1 格占位, 永不写入
          cap_(capacity), head_(0), tail_(0), size_(0) {}

    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == cap_; }

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return cap_; }

    bool push(const T& value) {
        if (full()) return false;
        buf_[tail_] = value;
        tail_ = (tail_ + 1) % buf_.size();
        ++size_;
        return true;
    }

    T pop() {
        if (empty()) throw std::out_of_range("CircularQueueCounted::pop on empty");
        T v = std::move(buf_[head_]);
        head_ = (head_ + 1) % buf_.size();
        --size_;
        return v;
    }

    const T& front() const {
        if (empty()) throw std::out_of_range("CircularQueueCounted::front on empty");
        return buf_[head_];
    }
    const T& back() const {
        if (empty()) throw std::out_of_range("CircularQueueCounted::back on empty");
        return buf_[(head_ + size_ - 1) % buf_.size()];
    }

private:
    std::vector<T> buf_;
    std::size_t cap_;    // 逻辑容量
    std::size_t head_;
    std::size_t tail_;
    std::size_t size_;
};

// ===========================================================================
// 自测
// ===========================================================================
template <typename Q>
static void exerciseFifo(Q& q, const char* name) {
    std::cout << "----- " << name << " (capacity = " << q.capacity() << ") -----\n";
    assert(q.empty() && q.size() == 0);

    const std::size_t cap = q.capacity();
    // 1) 填满
    for (std::size_t i = 0; i < cap; ++i) {
        bool ok = q.push(static_cast<int>(i));
        assert(ok);
    }
    assert(q.full());
    assert(q.size() == cap);
    assert(!q.push(999));                       // 满时入队失败
    std::cout << "  填满后 size=" << q.size() << ", full=" << q.full()
              << ", front=" << q.front() << ", back=" << q.back() << "\n";
    assert(q.front() == 0);
    assert(q.back() == static_cast<int>(cap) - 1);

    // 2) 出队一半, 再入队 -> 触发"绕圈"(下标回绕)
    const std::size_t half = cap / 2;
    for (std::size_t i = 0; i < half; ++i)
        assert(q.pop() == static_cast<int>(i));
    for (std::size_t i = 0; i < half; ++i)
        assert(q.push(static_cast<int>(cap + i)));   // 写指针回绕
    assert(q.size() == cap);

    // 3) 全部出队, 应严格是 0 出队后再入队顺序的 FIFO
    for (std::size_t i = half; i < cap; ++i)
        assert(q.pop() == static_cast<int>(i));
    for (std::size_t i = 0; i < half; ++i)
        assert(q.pop() == static_cast<int>(cap + i));
    assert(q.empty());
    std::cout << "  [OK] 回绕后 FIFO 顺序完全正确\n";
}

static void testBothSchemes() {
    std::cout << "===== 两种判满方案对照 =====\n";
    CircularQueueSacrifice<int> a(5);    // 最多 5 个元素
    CircularQueueCounted<int>   b(5);
    assert(a.capacity() == 5 && b.capacity() == 5);
    exerciseFifo(a, "方案A 牺牲一格");
    exerciseFifo(b, "方案B 计 size");
    std::cout << "\n";

    // 边界: 容量 0 的队列任何入队都失败
    CircularQueueSacrifice<int> z0(0);
    assert(z0.empty() && z0.full() && !z0.push(1));
    std::cout << "[OK] 容量 0: 空且满, push 返回 false\n";

    // 边界: 容量 1 (牺牲一格方案底开 2 格, 刚好放 1 个)
    CircularQueueSacrifice<int> z1(1);
    assert(z1.push(7) && z1.full() && !z1.push(8));
    assert(z1.pop() == 7 && z1.empty());
    std::cout << "[OK] 容量 1: 放一个后即满, 弹出后可复用\n\n";
}

static void testStringAndExceptions() {
    std::cout << "===== 模板鲁棒性 (std::string) 与异常 =====\n";
    CircularQueueCounted<std::string> q(3);
    assert(q.push("a") && q.push("bb") && q.push("ccc"));
    assert(q.full());
    std::string s = q.pop();              // 拷贝/移动出 string 类型的元素
    assert(s == "a");
    assert(q.push("dddd"));
    assert(q.pop() == "bb" && q.pop() == "ccc" && q.pop() == "dddd");
    assert(q.empty());
    std::cout << "[OK] std::string 元素进出队正确\n";

    bool threw = false;
    try { (void)q.pop(); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    threw = false;
    try { (void)q.front(); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    std::cout << "[OK] 空队列 pop/front 抛 std::out_of_range\n\n";
}

int main() {
    testBothSchemes();
    testStringAndExceptions();
    std::cout << "全部测试通过。\n";
    return 0;
}
