---
name: pr-description-writer
description: 根据代码改动生成 PR 描述（背景、改动点、验证方式）。当任务要求"写 PR 描述""总结改动""生成 commit message"时加载本 skill。
keywords: [pr, 描述, 总结, commit, 改动说明, changelog]
---

# Skill: pr-description-writer

## 何时使用
任务要求写 PR 描述 / 总结改动 / 生成 commit message 时加载。

## 工作流
1. `git_diff()` 收集改动；必要时 `read_file` 补读上下文。
2. 按下面模板产出，**每一句都要能被 diff 证实**，不要写 diff 里没有的改动。

## 输出模板
```markdown
## 背景
<一句话：为什么需要这个改动，关联的 issue>

## 改动
- `path`：<改了什么、为什么这么改>

## 验证
- `run_tests()`：<通过 / 失败，关键输出摘要>

## 影响面
<是否改变公开接口、是否需要迁移、有无兼容性风险>
```

## 注意
- 标题用一句话祈使句（"修复 calculator.add 的运算符错误"），不要写成问句；
- 不要把无改动的文件列进去；不要把「重构」和「修复」混为一谈。
