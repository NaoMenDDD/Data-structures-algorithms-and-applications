// ===========================================================================
// 05_stack.cpp
// 栈 (stack) 的两种经典实现:
//   1) ArrayStack<T>  —— 动态数组实现, 容量按倍增策略扩容, push 摊还 O(1)
//   2) LinkedStack<T> —— 单链表实现, 头插/头删, 每次操作严格 O(1)
// 并给出括号匹配 (bracket matching) 的完整实现, 以及 std::stack 的对照用法。
//
// 编译运行:
//   g++ -std=c++17 -O2 -Wall -Wextra 05_stack.cpp -o 05_stack && ./05_stack
// ===========================================================================
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>     // std::unique_ptr
#include <stack>      // std::stack 对照
#include <stdexcept>  // std::out_of_range
#include <string>
#include <utility>    // std::swap
#include <vector>

// ===========================================================================
// 1. 数组实现
// ===========================================================================
// 关键点:
//   * 逻辑上栈顶在 top_ 位置, 元素存放在 data_[0..top_]
//   * 扩容采用 "倍增" (doubling): 容量不足时申请 2 倍空间并搬迁
//     -> 单次 push 最坏 O(n), 但连续 n 次 push 的总代价 O(n), 摊还 O(1)
//   * 用 unique_ptr<T[]> 管理裸数组, 天然 RAII, 不会泄漏
template <typename T>
class ArrayStack {
public:
    explicit ArrayStack(std::size_t initCap = 8)
        : data_(initCap ? std::make_unique<T[]>(initCap) : nullptr),
          cap_(initCap),
          top_(0) {}   // top_ 表示 "下一个可写位置", 也就是元素个数

    bool empty() const { return top_ == 0; }

    std::size_t size() const { return top_; }
    std::size_t capacity() const { return cap_; }

    // 返回栈顶元素的引用, 允许修改
    T& top() {
        if (empty()) throw std::out_of_range("ArrayStack::top on empty stack");
        return data_[top_ - 1];
    }
    const T& top() const {
        if (empty()) throw std::out_of_range("ArrayStack::top on empty stack");
        return data_[top_ - 1];
    }

    void push(const T& value) {
        ensureCapacity(top_ + 1);
        data_[top_++] = value;
    }
    void push(T&& value) {
        ensureCapacity(top_ + 1);
        data_[top_++] = std::move(value);
    }

    // 弹出栈顶 (返回被弹出的值); 空栈抛异常
    T pop() {
        if (empty()) throw std::out_of_range("ArrayStack::pop on empty stack");
        return std::move(data_[--top_]);
    }

    void clear() { top_ = 0; }

private:
    void ensureCapacity(std::size_t need) {
        if (need <= cap_) return;
        std::size_t newCap = cap_ ? cap_ * 2 : 1;
        while (newCap < need) newCap *= 2;         // 至少放大到 need
        auto bigger = std::make_unique<T[]>(newCap);
        for (std::size_t i = 0; i < top_; ++i)
            bigger[i] = std::move(data_[i]);       // 搬迁旧元素
        data_ = std::move(bigger);
        cap_ = newCap;
    }

    std::unique_ptr<T[]> data_;
    std::size_t cap_;   // 当前容量
    std::size_t top_;   // 当前元素个数 (也是栈顶之上第一个空位)
};

// ===========================================================================
// 2. 单链表实现
// ===========================================================================
// 栈只在 "同一端" 插入与删除, 所以单链表只需维护头指针:
//   push = 头插, pop = 头删, 均为 O(1), 无需尾指针。
// 链表实现的优势: 每次操作严格 O(1), 不依赖摊还; 缺点是每个节点多一个指针开销,
// 且内存分散 (cache 不友好) —— 这也是实际工程中栈多用数组的原因。
template <typename T>
class LinkedStack {
    struct Node {
        T value;
        std::unique_ptr<Node> next;
        explicit Node(const T& v, std::unique_ptr<Node> n = nullptr)
            : value(v), next(std::move(n)) {}
        explicit Node(T&& v, std::unique_ptr<Node> n = nullptr)
            : value(std::move(v)), next(std::move(n)) {}
    };

public:
    LinkedStack() = default;
    LinkedStack(const LinkedStack&) = delete;             // 独占所有权, 禁止拷贝
    LinkedStack& operator=(const LinkedStack&) = delete;

    bool empty() const { return !head_; }

    std::size_t size() const {
        std::size_t n = 0;
        for (const Node* p = head_.get(); p; p = p->next.get()) ++n;
        return n;   // 若想 O(1), 可额外维护 size_ 成员
    }

    T& top() {
        if (empty()) throw std::out_of_range("LinkedStack::top on empty stack");
        return head_->value;
    }

    void push(const T& value) { head_ = std::make_unique<Node>(value, std::move(head_)); }
    void push(T&& value) { head_ = std::make_unique<Node>(std::move(value), std::move(head_)); }

    T pop() {
        if (empty()) throw std::out_of_range("LinkedStack::pop on empty stack");
        T v = std::move(head_->value);
        head_ = std::move(head_->next);   // 头删, 旧头节点被 unique_ptr 自动释放
        return v;
    }

private:
    std::unique_ptr<Node> head_{};
};

// ===========================================================================
// 3. 括号匹配 (bracket matching)
// ===========================================================================
// 规则: 左括号入栈; 遇到右括号时, 栈顶必须是与之配对的左括号, 否则非法。
// 结束时栈必须为空。时间 O(n), 空间 O(n)。
bool bracketsBalanced(const std::string& s) {
    std::stack<char> st;
    auto match = [](char open, char close) {
        return (open == '(' && close == ')') ||
               (open == '[' && close == ']') ||
               (open == '{' && close == '}');
    };
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.empty() || !match(st.top(), c)) return false;
            st.pop();
        }
        // 其它字符 (字母/运算符/空白) 直接忽略
    }
    return st.empty();
}

// 括号匹配的 "错在哪里" 诊断版: 返回第一个不匹配的位置, 找不到返回 -1
int bracketsFirstError(const std::string& s) {
    std::stack<std::pair<char, std::size_t>> st;   // (括号, 下标)
    auto match = [](char open, char close) {
        return (open == '(' && close == ')') ||
               (open == '[' && close == ']') ||
               (open == '{' && close == '}');
    };
    for (std::size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '(' || c == '[' || c == '{') {
            st.push({c, i});
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.empty() || !match(st.top().first, c)) return static_cast<int>(i);
            st.pop();
        }
    }
    return st.empty() ? -1 : static_cast<int>(st.top().second);  // 有左括号没闭合
}

// ===========================================================================
// 自测
// ===========================================================================
static void testArrayStack() {
    std::cout << "===== ArrayStack (动态数组) =====\n";
    ArrayStack<int> st(2);                 // 故意给小容量, 观察扩容
    assert(st.empty() && st.size() == 0);
    for (int i = 1; i <= 10; ++i) st.push(i);
    assert(st.size() == 10);
    std::cout << "push 1..10 后: size=" << st.size()
              << ", capacity=" << st.capacity() << " (发生了倍增扩容)\n";
    assert(st.top() == 10);
    // 后进先出: 依次弹出应为 10, 9, ..., 1
    for (int expect = 10; expect >= 1; --expect) {
        assert(st.top() == expect);
        assert(st.pop() == expect);
    }
    assert(st.empty());
    std::cout << "[OK] LIFO 顺序正确, 逆序弹出 10..1\n";

    // 边界: 空栈 pop/top 应抛异常
    bool threw = false;
    try { (void)st.pop(); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    std::cout << "[OK] 空栈操作抛 std::out_of_range\n";

    // 字符串类型也能用 (模板)
    ArrayStack<std::string> ss;
    ss.push("world");
    ss.push("hello");
    assert(ss.pop() == "hello" && ss.pop() == "world");
    std::cout << "[OK] 模板对 std::string 同样工作\n\n";
}

static void testLinkedStack() {
    std::cout << "===== LinkedStack (单链表) =====\n";
    LinkedStack<int> st;
    for (int i = 1; i <= 5; ++i) st.push(i * i);
    assert(st.size() == 5 && st.top() == 25);
    assert(st.pop() == 25 && st.pop() == 16 && st.pop() == 9);
    assert(st.size() == 2 && st.top() == 4);
    std::cout << "[OK] 头插/头删实现 LIFO, pop 得到 25,16,9\n";
    // 与 ArrayStack 行为一致
    ArrayStack<int> a;
    LinkedStack<int> b;
    for (int i = 0; i < 100; ++i) { a.push(i); b.push(i); }
    while (!a.empty()) assert(a.pop() == b.pop());
    std::cout << "[OK] 100 次弹栈, 两种实现输出序列完全一致\n\n";
}

static void testBrackets() {
    std::cout << "===== 括号匹配 =====\n";
    struct Case { const char* s; bool ok; };
    Case cases[] = {
        {"()", true},
        {"()[]{}", true},
        {"(a[b]{c})", true},
        {"([{}])", true},
        {"", true},
        {"(]", false},
        {"([)]", false},         // 交错嵌套是最经典的陷阱
        {"(", false},
        {")", false},
        {"((()))", true},
        {"(()", false},
    };
    for (const auto& c : cases) {
        bool got = bracketsBalanced(c.s);
        std::cout << "  \"" << c.s << "\" -> " << (got ? "balanced" : "NOT balanced")
                  << "  (期望 " << (c.ok ? "balanced" : "NOT balanced") << ")\n";
        assert(got == c.ok);
    }
    std::cout << "[OK] 11 组用例全部正确 (含空串、交错嵌套、单独右括号)\n";
    assert(bracketsFirstError("([)]") == 2);
    assert(bracketsFirstError("(())") == -1);
    assert(bracketsFirstError("((())") == 0);
    std::cout << "[OK] 错误定位: \"([)]\" 首个错误在下标 2\n\n";
}

int main() {
    testArrayStack();
    testLinkedStack();
    testBrackets();
    std::cout << "全部测试通过。\n";
    return 0;
}
