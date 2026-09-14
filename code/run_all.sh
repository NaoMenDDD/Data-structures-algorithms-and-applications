#!/usr/bin/env bash
# 编译并运行 code/ 目录下的所有示例程序（每个 .cpp 都应自带 main() 与自测）。
#
# 用法：
#   ./run_all.sh          # -O2 快速验证（默认）
#   ./run_all.sh --san    # 额外开启 AddressSanitizer + UndefinedBehaviorSanitizer，
#                         # 更慢，但能抓出越界、悬垂指针、UB —— 数据结构代码的头号杀手
#
# 退出码：全部通过为 0，否则为 1。

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

SAN=0
if [[ "${1:-}" == "--san" ]]; then SAN=1; fi

FLAGS=(-std=c++17 -Wall -Wextra)
if [[ $SAN -eq 1 ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  echo "== 已开启 ASan/UBSan =="
else
  FLAGS+=(-O2)
fi

shopt -s nullglob
sources=("$DIR"/*.cpp)
shopt -u nullglob

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "在 $DIR 下没有找到 .cpp 文件。"
  exit 0
fi

pass=0
fail=0
failed=()

for src in "${sources[@]}"; do
  name="$(basename "$src" .cpp)"
  bin="$OUT/$name"

  if ! g++ "${FLAGS[@]}" "$src" -o "$bin" 2>"$OUT/$name.build.log"; then
    printf '\033[31m❌ 编译失败\033[0m  %s\n' "$name"
    head -20 "$OUT/$name.build.log" | sed 's/^/    /'
    fail=$((fail + 1)); failed+=("$name (编译)")
    continue
  fi

  if timeout 300 "$bin" >"$OUT/$name.run.log" 2>&1; then
    printf '\033[32m✅\033[0m %s\n' "$name"
    pass=$((pass + 1))
  else
    printf '\033[31m❌ 运行失败\033[0m  %s（退出码 $?）\n' "$name"
    tail -20 "$OUT/$name.run.log" | sed 's/^/    /'
    fail=$((fail + 1)); failed+=("$name (运行)")
  fi
done

echo
echo "———————————————————————————————"
printf '通过 %d / %d\n' "$pass" "$((pass + fail))"
if [[ $fail -gt 0 ]]; then
  printf '失败项：\n'
  printf '  - %s\n' "${failed[@]}"
  exit 1
fi
echo "全部通过。"
