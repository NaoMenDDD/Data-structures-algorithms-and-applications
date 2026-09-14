// 00_raii_smart_ptr.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 0 章 预备：C++ 与开发环境》
// 主题：RAII（Resource Acquisition Is Initialization，资源获取即初始化）
//       与三种智能指针 unique_ptr / shared_ptr / weak_ptr 的行为差异
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 00_raii_smart_ptr.cpp -o raii && ./raii
//
// 本文件自带 main() 与断言自测，不需要任何第三方依赖。
// 输出中的 [ctor]/[dtor]/[copy]/[move] 就是对象「一生」的完整轨迹，
// 观察这些轨迹是理解 RAII 与所有权语义最直观的方式。
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>    // unique_ptr, shared_ptr, weak_ptr, make_unique
#include <string>
#include <utility>   // std::move
#include <vector>

// ===========================================================================
// 第一部分：手写一个 RAII 类，观察构造/拷贝/移动/析构
// ===========================================================================
//
// 设计目标：管理一段「裸堆内存」，模拟像 std::vector 那样的一段缓冲区。
// 关键思想：资源的生命周期绑定在对象生命周期上 —— 对象一出作用域，
//           析构函数自动被调用，资源一定被释放。这就是 RAII。
//
// 这个类故意先写「完整版」（Rule of 5，即需要自定义析构/拷贝/移动的五个函数），
// 读者可以对照后面用 = default / = delete 的极简写法，体会二者等价。

class Buffer {
public:
    // --- 普通构造函数：获取资源（这里是 new[] 出来的内存）---
    explicit Buffer(std::size_t n, int fill = 0)
        : size_(n), data_(n ? new int[n] : nullptr) {
        for (std::size_t i = 0; i < size_; ++i) data_[i] = fill;
        std::cout << "[ctor] Buffer(" << size_ << ")\n";
    }

    // --- 拷贝构造函数：执行「深拷贝」，两块内存互不影响 ---
    // 为什么必须深拷贝？如果只是复制指针，两个对象会在析构时 delete 同一块
    // 内存，产生 double free（双重释放），属于未定义行为（UB）。
    Buffer(const Buffer& other)
        : size_(other.size_), data_(other.size_ ? new int[other.size_] : nullptr) {
        for (std::size_t i = 0; i < size_; ++i) data_[i] = other.data_[i];
        std::cout << "[copy] Buffer(const Buffer&)\n";
    }

    // --- 移动构造函数：直接「偷走」对方的指针，不复制数据 ---
    // noexcept 很重要：标准容器（如 std::vector）扩容时，
    // 只有移动构造被标记为 noexcept，才会用移动而非拷贝搬移元素。
    Buffer(Buffer&& other) noexcept
        : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;   // 把源对象置空，防止它析构时释放我们的内存
        std::cout << "[move] Buffer(Buffer&&)\n";
    }

    // --- 拷贝赋值：先释放自己的旧资源，再深拷贝对方 ---
    // 用「拷贝并交换」（copy-and-swap）惯用法天然获得强异常安全保证。
    Buffer& operator=(const Buffer& other) {
        std::cout << "[copy=] operator=(const Buffer&)\n";
        if (this != &other) {          // 自赋值检查是必须的
            Buffer tmp(other);         // 先造一份副本，若抛异常，*this 不变
            swap(tmp);                 // 与副本交换，旧的资源随 tmp 析构而释放
        }
        return *this;
    }

    // --- 移动赋值：释放自己的旧资源，接管对方的资源 ---
    Buffer& operator=(Buffer&& other) noexcept {
        std::cout << "[move=] operator=(Buffer&&)\n";
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    // --- 析构函数：释放资源。无论正常返回还是异常展开，它都一定被调用 ---
    ~Buffer() {
        std::cout << "[dtor] ~Buffer(" << size_ << ")\n";
        delete[] data_;   // 对 nullptr 调用 delete[] 是安全的（无操作）
    }

    void swap(Buffer& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    // 只读访问：const 成员函数，不修改对象状态
    std::size_t size() const { return size_; }
    int& operator[](std::size_t i) { return data_[i]; }
    const int& operator[](std::size_t i) const { return data_[i]; }

private:
    std::size_t size_;
    int* data_;
};

// 演示：函数按值返回 Buffer 时，编译器会优先使用移动（或直接构造，RVO）
Buffer makeBuffer(std::size_t n) {
    Buffer b(n, 7);
    // 返回「具名局部变量」时是 NRVO：C++17 允许但不强制编译器做拷贝消除，
    // 所以标准**保证**的只是「先尝试移动」；多数编译器实际会原地构造（零开销）。
    return b;
}

// ===========================================================================
// 第二部分：shared_ptr 引用计数与 weak_ptr 打破循环引用
// ===========================================================================

struct Node {
    std::string name;
    // 关键：用 weak_ptr 指向「父节点」，用 shared_ptr 指向「子节点」。
    // 如果父子都用 shared_ptr，就形成引用计数环：两者计数永远 >= 1，
    // 谁都释放不掉，造成内存泄漏（memory leak）。
    std::shared_ptr<Node> child;      // 强引用：我拥有孩子
    std::weak_ptr<Node>   parent;     // 弱引用：我只知道父亲是谁，但不拥有它

    explicit Node(std::string n) : name(std::move(n)) {
        std::cout << "[Node ctor] " << name << "\n";
    }
    ~Node() { std::cout << "[Node dtor] " << name << "\n"; }
};

// ===========================================================================
// 演示函数
// ===========================================================================

static void demo_unique_ptr() {
    std::cout << "\n=== unique_ptr：独占所有权 ===\n";
    // make_unique 是异常安全的创建方式（比 new 更推荐）
    auto p = std::make_unique<Buffer>(4, 1);
    std::cout << "p->size() = " << p->size() << "\n";

    // unique_ptr 不能拷贝（拷贝构造被 = delete），只能移动
    // auto q = p;              // 编译错误！
    auto q = std::move(p);      // 所有权转移，之后 p 变成 nullptr
    assert(p == nullptr);       // NOLINT(readability-container-size-empty)
    assert(q != nullptr);
    std::cout << "移动后 p 是否为空: " << std::boolalpha << (p == nullptr) << "\n";

    // 离开作用域时 q 自动析构 -> Buffer 析构 -> delete[]
}   // <-- [dtor] 在这里被调用，不需要手写 delete

static void demo_shared_ptr() {
    std::cout << "\n=== shared_ptr：共享所有权（引用计数）===\n";
    auto a = std::make_shared<Buffer>(3, 5);
    std::cout << "创建 a 后 use_count = " << a.use_count() << "\n";   // 1

    {
        auto b = a;   // 拷贝，计数 +1
        std::cout << "拷贝出 b 后 use_count = " << a.use_count() << "\n"; // 2
    }                 // b 析构，计数 -1
    std::cout << "b 离开作用域后 use_count = " << a.use_count() << "\n"; // 1

    std::weak_ptr<Buffer> w = a;   // weak_ptr 不增加引用计数
    std::cout << "weak_ptr 观察 use_count = " << a.use_count() << "\n";  // 仍是 1
    if (auto locked = w.lock()) {  // lock() 尝试提升为 shared_ptr
        std::cout << "通过 weak_ptr 成功访问, size = " << locked->size() << "\n";
    }
    // a 离开作用域，计数归零，Buffer 被销毁
}

static void demo_weak_ptr_cycle() {
    std::cout << "\n=== weak_ptr 打破循环引用 ===\n";
    auto root = std::make_shared<Node>("root");
    auto leaf = std::make_shared<Node>("leaf");

    root->child = leaf;    // root 拥有 leaf
    leaf->parent = root;   // leaf 只弱引用 root，不增计数

    assert(root.use_count() == 1);   // 只有局部变量 root 引用它
    assert(leaf.use_count() == 2);   // 局部变量 leaf + root->child

    std::cout << "root.use_count = " << root.use_count() << "\n";
    std::cout << "leaf.use_count = " << leaf.use_count() << "\n";
    std::cout << "-- 函数结束，两个 Node 都应被正确销毁 --\n";
}   // leaf 先析构（局部变量逆序），计数 -1；root->child 仍持有 leaf，
    // 所以 leaf 在这里其实还活着……真正清理发生在 root 析构之后。
    // 若把 parent 也写成 shared_ptr，则二者互不释放，最终什么都不会打印。

// ===========================================================================
// main
// ===========================================================================
int main() {
    std::cout << "########## demo: unique_ptr ##########\n";
    demo_unique_ptr();

    std::cout << "\n########## demo: shared_ptr ##########\n";
    demo_shared_ptr();

    std::cout << "\n########## demo: weak_ptr 循环引用 ##########\n";
    demo_weak_ptr_cycle();

    std::cout << "\n########## demo: Buffer 拷贝 / 移动 / RVO ##########\n";
    {
        Buffer b1(2, 9);
        Buffer b2 = b1;              // 触发 [copy]
        Buffer b3 = std::move(b1);   // 触发 [move]，b1 此后为空
        Buffer b4 = makeBuffer(3);   // C++17 拷贝消除，不触发任何拷贝/移动
        (void)b2; (void)b3; (void)b4;
    }   // 四个对象逆序析构

    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
