---
name: test-runner
description: 运行测试并诊断失败原因。当任务要求"修 bug""让测试通过""pytest 失败""跑测试"时加载本 skill。
keywords: [测试, pytest, 失败, 修复, bug, 通过, test, failing]
---

# Skill: test-runner

## 何时使用
任务里出现「修 bug / 让测试通过 / pytest 失败 / 跑测试 / failing test」时加载。

## 工作流
1. **先跑一遍测试建立基线**：`run_tests()`。不要一上来就改代码——先确认失败的是哪个用例、
   报什么错。Observation 里的 `returncode` 非 0 即失败，`output` 里有断言详情。
2. **定位被测目标**：从测试文件的 `import` 与断言反推被测模块与函数。用 `read_file` 读测试
   与被测文件，**不要**凭猜测改代码。
3. **最小改动修复根因**：只改被测实现，**绝不要改测试文件**（改测试让测试变绿等于作弊）。
4. **回归验证**：重新 `run_tests()`，要求 `passed == true`。
5. **读 diff 复核**：`git_diff()` 确认改动范围符合预期、没有误改其它文件。

## 常见 bug 模式（对照检查）
- 运算符写反：`a - b` 应为 `a + b`、`<` 与 `<=` 边界、`/` 与 `//`；
- 初值与累加器：`result = 0` 应为 `1`（累乘）、循环 `range(1, n)` 应为 `range(2, n + 1)`；
- 差一错误：`range(n)` 与 `range(n + 1)`；
- 返回值缺失 / 返回了中间变量。

## 停机条件
`run_tests()` 返回 `passed == true`，且 `git_diff()` 显示改动仅限被测实现文件。
