// ===========================================================================
// 05_expression_eval.cpp
// 表达式求值三件套:
//   1) tokenize        —— 把字符串切成 token (多位数、小数、括号、运算符)
//   2) infixToPostfix  —— 中缀转后缀 (调度场算法 Shunting-yard 的栈式简化版)
//   3) evalPostfix     —— 后缀表达式求值 (唯一的 "栈" 主战场)
// 另含: 进制转换 (十进制 -> 任意基, 用栈逆序输出) 作为热身。
//
// 编译运行:
//   g++ -std=c++17 -O2 -Wall -Wextra 05_expression_eval.cpp -o 05_expression_eval
//   ./05_expression_eval
// ===========================================================================
#include <cassert>
#include <cctype>       // std::isdigit
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>

// 运算符优先级 (数值越大越"紧"); 左括号给最低优先级, 特殊处理
static int precedence(char op) {
    switch (op) {
        case '+': case '-': return 1;
        case '*': case '/': return 2;
        case '^':           return 3;   // 幂运算, 右结合
        default:            return 0;
    }
}
static bool isRightAssoc(char op) { return op == '^'; }

// ---------------------------------------------------------------------------
// 1. 词法分析: 支持多位整数、小数、'+''-''*''/''^''(' ')'
// ---------------------------------------------------------------------------
// 注意: 不做一元负号处理 (那是 "词法 vs 语法" 的边界), 见文末陷阱说明。
std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (c == ' ' || c == '\t') { ++i; continue; }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            std::size_t j = i;
            while (j < s.size() &&
                   (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.'))
                ++j;
            out.push_back(s.substr(i, j - i));
            i = j;
        } else {
            out.push_back(std::string(1, c));
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 2. 中缀 -> 后缀 (调度场 Shunting-yard 的简化版, 只处理二元运算符)
// ---------------------------------------------------------------------------
// 规则:
//   数字       -> 直接输出
//   '('        -> 入栈
//   ')'        -> 弹栈输出直到 '(' (弹出但不输出 '(')
//   运算符 op  -> 当栈顶运算符优先级更高, 或同优先级且当前是左结合时,
//                 弹栈输出; 直到不再满足, 再压入 op
// 收尾: 把栈中剩余运算符全部弹出输出
std::vector<std::string> infixToPostfix(const std::vector<std::string>& tokens) {
    std::vector<std::string> output;
    std::stack<std::string> ops;   // 只放运算符与 '('
    for (const std::string& t : tokens) {
        if (t == "(") {
            ops.push(t);
        } else if (t == ")") {
            while (!ops.empty() && ops.top() != "(") {
                output.push_back(ops.top());
                ops.pop();
            }
            if (ops.empty())
                throw std::runtime_error("mismatched parentheses: extra ')'");
            ops.pop();   // 丢弃 '('
        } else if (t == "+" || t == "-" || t == "*" || t == "/" || t == "^") {
            char op = t[0];
            while (!ops.empty() && ops.top() != "(") {
                char top = ops.top()[0];
                if (precedence(top) > precedence(op) ||
                    (precedence(top) == precedence(op) && !isRightAssoc(op))) {
                    output.push_back(ops.top());
                    ops.pop();
                } else {
                    break;
                }
            }
            ops.push(t);
        } else {
            output.push_back(t);   // 操作数
        }
    }
    while (!ops.empty()) {
        if (ops.top() == "(") throw std::runtime_error("mismatched parentheses: extra '('");
        output.push_back(ops.top());
        ops.pop();
    }
    return output;
}

// ---------------------------------------------------------------------------
// 3. 后缀表达式求值: 单栈 + 从左到右扫描
// ---------------------------------------------------------------------------
//   遇到操作数 -> 压栈
//   遇到运算符 -> 弹出右操作数 b, 再弹左操作数 a, 计算 a op b, 结果压回
//   扫描完, 栈中唯一元素就是答案
double evalPostfix(const std::vector<std::string>& post) {
    std::stack<double> st;
    for (const std::string& t : post) {
        if (t == "+" || t == "-" || t == "*" || t == "/" || t == "^") {
            if (st.size() < 2)
                throw std::runtime_error("malformed postfix expression");
            double b = st.top(); st.pop();   // 先弹的是右操作数!
            double a = st.top(); st.pop();
            double r = 0.0;
            switch (t[0]) {
                case '+': r = a + b; break;
                case '-': r = a - b; break;   // 顺序: a - b, 不能反
                case '*': r = a * b; break;
                case '/':
                    if (b == 0.0) throw std::runtime_error("division by zero");
                    r = a / b;
                    break;
                case '^': r = std::pow(a, b); break;
            }
            st.push(r);
        } else {
            st.push(std::stod(t));
        }
    }
    if (st.size() != 1) throw std::runtime_error("trailing operands");
    return st.top();
}

// 便捷封装: 中缀字符串 -> 值
double evalInfix(const std::string& expr) {
    return evalPostfix(infixToPostfix(tokenize(expr)));
}

// ---------------------------------------------------------------------------
// 4. 进制转换: 十进制正整数 -> 任意 2..36 进制 (栈实现逆序输出)
// ---------------------------------------------------------------------------
std::string toBase(long long n, int base) {
    if (base < 2 || base > 36) throw std::invalid_argument("base out of range");
    if (n == 0) return "0";
    bool neg = n < 0;
    unsigned long long u = neg ? static_cast<unsigned long long>(-(n + 1)) + 1ULL
                               : static_cast<unsigned long long>(n);
    const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::stack<char> st;
    while (u > 0) {
        st.push(digits[u % static_cast<unsigned long long>(base)]);
        u /= static_cast<unsigned long long>(base);
    }
    std::string out = neg ? "-" : "";
    while (!st.empty()) { out.push_back(st.top()); st.pop(); }
    return out;
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------
static void testTokenize() {
    std::cout << "===== 词法分析 =====\n";
    auto tk = tokenize("12 + 3*(4 - 5)");
    std::vector<std::string> expect = {"12", "+", "3", "*", "(", "4", "-", "5", ")"};
    assert(tk == expect);
    std::cout << "  \"12 + 3*(4 - 5)\" -> [";
    for (std::size_t i = 0; i < tk.size(); ++i) std::cout << tk[i] << (i + 1 < tk.size() ? " " : "");
    std::cout << "]\n[OK] 多位数字正确合并为单个 token\n\n";
}

static void testInfixToPostfix() {
    std::cout << "===== 中缀 -> 后缀 =====\n";
    struct Case { const char* in; const char* post; };
    Case cases[] = {
        {"1+2*3",       "1 2 3 * +"},
        {"(1+2)*3",     "1 2 + 3 *"},
        {"2^3^2",       "2 3 2 ^ ^"},      // 右结合: 2^(3^2)
        {"1+2-3",       "1 2 + 3 -"},      // 左结合
        {"8/4/2",       "8 4 / 2 /"},      // 左结合: (8/4)/2
    };
    for (const auto& c : cases) {
        auto post = infixToPostfix(tokenize(c.in));
        std::string got;
        for (std::size_t i = 0; i < post.size(); ++i) got += post[i] + (i + 1 < post.size() ? " " : "");
        std::cout << "  " << c.in << "  ->  " << got << "\n";
        assert(got == c.post);
    }
    std::cout << "[OK] 含结合性 (左结合减法除法、右结合幂) 全部正确\n\n";
}

static void testEval() {
    std::cout << "===== 后缀求值 / 整式中缀求值 =====\n";
    // 后缀直接求值
    assert(std::abs(evalPostfix(tokenize("3 4 +")) - 7.0) < 1e-12);
    assert(std::abs(evalPostfix(tokenize("5 1 2 + 4 * + 3 -")) - 14.0) < 1e-12);  // 5+((1+2)*4)-3
    // 中缀一站式
    struct Case { const char* e; double v; };
    Case cases[] = {
        {"1+2*3",            7.0},
        {"(1+2)*3",          9.0},
        {"2^3^2",            512.0},      // 2^9
        {"10/4",             2.5},        // 支持小数结果
        {"3.5*2",            7.0},
        {"100 - 10*5 - 50",  0.0},        // (100-50)-50
        {"2*(3+4)*(5-1)",    56.0},
    };
    for (const auto& c : cases) {
        double got = evalInfix(c.e);
        std::cout << "  " << c.e << " = " << got << "  (期望 " << c.v << ")\n";
        assert(std::abs(got - c.v) < 1e-9);
    }
    std::cout << "[OK] 7 个整式求值全部正确\n";

    // 异常路径
    bool threw = false;
    try { evalInfix("1/0"); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    threw = false;
    try { evalInfix("(1+2"); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "[OK] 除零与括号不匹配均正确抛错\n\n";
}

static void testBase() {
    std::cout << "===== 进制转换 (栈: 先得低位, 后序输出) =====\n";
    struct Case { long long n; int base; const char* s; };
    Case cases[] = {
        {13, 2, "1101"}, {255, 16, "FF"}, {0, 10, "0"},
        {1000, 36, "RS"}, {-10, 2, "-1010"},
    };
    for (const auto& c : cases) {
        std::string got = toBase(c.n, c.base);
        std::cout << "  " << c.n << " 转 " << c.base << " 进制 = " << got
                  << "  (期望 " << c.s << ")\n";
        assert(got == c.s);
    }
    std::cout << "[OK] 含 0 与负数的进制转换全部正确\n\n";
}

int main() {
    testTokenize();
    testInfixToPostfix();
    testEval();
    testBase();
    std::cout << "全部测试通过。\n";
    return 0;
}
