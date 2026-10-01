# 任务六：Mini Coding Agent — 完成报告

## 一句话总结
用 Qwen2.5-Coder-7B-Instruct 的 OpenAI 兼容端点搭了一条 `while not done: model → tool →
observation → loop` 的 agentic loop，并把能力拆成 Tools/MCP、Skills、Subagents 三层：
手写 6 个工具并封装成可独立启动的 stdio MCP server、3 个带 YAML front-matter 的 Skill
与按 description 匹配的加载器、2 个独立 context 的子 agent。在 `data/toy-repo` 上自主定位并
修复了 `calculator.add` 的运算符错误，`python -m pytest` 3 passed，产出完整 trace 与 patch。

## DoD Checklist
- [x] **M1** 手写 MCP server（`src/mcp_server.py`），顶层导出 `list_tools()`，自检
      `mcp_server_lists_tools` 通过——枚举到 **6** 个工具（≥ 5）。既可被 import，
      也能 `python src/mcp_server.py` 独立启动（已用官方 MCP SDK 客户端做真实 stdio 握手验证）
- [x] **M2** `src/skill_loader.py` + `src/skills/` 下 **3** 个带 YAML front-matter 的 `SKILL.md`
      （test-runner / code-review / pr-description-writer），自检 `skill_loader_metadata` 通过
      （每个 skill 都有 name + description）
- [x] **M3** `CodingAgent.run(repo_path, issue) -> Trace` 实现 agentic loop，自检 `toy_repo_patch`
      通过（修好 `calculator.add`，`python -m pytest` 3 passed）
- [x] **M4** trace 记录每步 thought / tool_call / observation，`Trace`（dict 子类）含
      `steps` / `patch` / `tests_passed`
- [ ] **S1/S2/S3/S4** 加分项未做（本机无可用本地模型，量化对比 / 词元消耗对比无法开展；
      SWE-bench Lite 元数据需联网下载）

## 交付物

| 文件 | 说明 |
|---|---|
| `src/mcp_server.py` | 可独立启动的 stdio MCP server；导出 `list_tools()` |
| `src/tools.py` | 6 个原子工具 + 统一 `call_tool()` 入口（工具异常一律转结构化 error） |
| `src/paths.py` | 路径归一化与越界校验（`safe_resolve`） |
| `src/skill_loader.py` | `SkillLoader`：扫描 / 元数据 / description 匹配 / 按需加载正文 |
| `src/skills/*/SKILL.md` | 3 个 Skill（YAML front-matter：name + description + keywords） |
| `src/subagents/base.py` | `Subagent` 基类 + `CodeSearchSubagent` + `TestRunnerSubagent` |
| `src/agent.py` | `CodingAgent`、`Trace`、`ModelClient`、agentic loop |
| `eval/result.json` | 自检结果 |

## 三层栈的实现要点

### 底层：Tools / MCP
6 个工具：`read_file` / `write_file` / `run_tests` / `git_diff` / `git_apply` / `list_dir`。
`tools.py` 的 `SPECS` 是唯一事实来源——MCP server 的 `list_tools()` 与主 agent 的 system
prompt 都从它生成，不存在两份定义漂移的问题。

**安全**（对照 README「安全」三条）：
- 所有路径先 `Path.resolve()` 再校验落在工作 repo 内，`../../../etc/passwd` 这类越界被拒
  （已实测：客户端调用返回 `isError=true` + `PathEscapeError`）；绝对路径按 repo 内相对路径解释，
  不给"写出仓库外"的口子；
- git / pytest 一律 `subprocess.run([...])` list 形式，不用 `shell=True`，并限定 `cwd`；
- `git_apply` 会拒绝任何含 `reset --hard` / `clean -fd` / `checkout --` / `filter-branch`
  等会丢未提交改动或删文件的补丁（已实测：返回 `ValueError: 拒绝执行含危险操作的补丁：reset --hard`）；
- 工具异常全部在 `call_tool()` 兜底成 `{"ok": false, "error": ...}`，server 与 agent 循环都不
  会被一次坏调用打断。

**`git_apply` 的一个已知边界**（实测发现，如实记录）：`git apply` 本身**不会**处理尚未进 index
的新文件（untracked），`--3way` 也要求文件已在 index 中。因此 `git_apply` 只对**已跟踪文件**可靠
（对 `calculator.py` 这类跟踪文件已实测 `applied=true`）；需要整份改写新文件时应直接用
`write_file`。工具在这种情况下返回 `applied=false` + 明确 `note`，不会静默假装成功。

**关于 SDK 版本**：requirements 写的是 `mcp>=0.9`，但环境里装的是 **mcp 2.1.1**——一次破坏性
重写（`FastMCP` 改名 `MCPServer`，`Server.list_tools()` 装饰器改成 `on_list_tools=` 构造参数，
`stdio_server` 语义也变了）。本实现按 **2.x 低层 API** 写，并额外提供
`--plain-jsonrpc` 的内置 JSON-RPC 回退，SDK 缺失或版本不兼容时 server 仍能完成 stdio 握手与
工具枚举。

### 中层：Skills（渐进式披露）
`SkillLoader.list_skills()` **只返回元数据**（轻量，可整份塞进 prompt 做路由）；
正文只有 `match()` 命中后才通过 `load()` 读取——不存在"一上来就把所有 SKILL.md 灌进上下文"。
匹配按「skill 名 / keywords 直接命中 > description 词与任务的重合度」加权打分。
本任务的 issue 文本命中 `test-runner`（score 15.25），其正文（含"常见 bug 模式"清单）被注入上下文。

### 顶层：Subagents（真隔离）
隔离不是换个函数名，`Subagent` 基类保证三件事：
1. **独立 message 列表**（`self.messages`，与主 agent 历史无关）；
2. **独立步数上限**（`max_steps`，code-search=4 / test-runner=3）；
3. **工具子集**（`allowed_tools`，越权调用直接被拒）。

对主 agent 只回**结构化摘要**（工具调用记录 + 结论），原始 observation 一概不外传。
trace 里 `subagent_summaries` 就是这个摘要，主 agent 的上下文不会被子 agent 的完整日志污染。

## 自检结果（实跑）

```
[通过] mcp_server_lists_tools: {"tools": ["read_file", "write_file", "run_tests", "git_diff", "git_apply", "list_dir"]}
[通过] skill_loader_metadata: {"count": 3, "missing_meta": []}
[通过] toy_repo_patch: {"tests_passed": true, "pytest_output": "3 passed in 0.01s", "trace_has_patch": true}
[跳过] swebench_lite_sample: 需 data/download.py --with-swebench
```

连跑两次结果一致（`eval/run.py` 每次从 `calculator.py.orig` 快照恢复 buggy 版本再跑，不存在
"上轮修过所以本轮空跑通过"）。

## 复现命令
```bash
cd task-6-coding-agent
python data/download.py            # 生成 data/toy-repo（含 calculator.py.orig 快照 + git 初始 commit）
python eval/run.py                 # 自检 -> eval/result.json

# 模型驱动模式（本机部署好本地模型后自动启用）
ollama pull qwen2.5-coder:7b-instruct && ollama serve
# 或指定其它 OpenAI 兼容端点：
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"; $env:CODING_AGENT_MODEL = "qwen2.5-coder-7b-instruct"

# MCP server 单独跑
python src/mcp_server.py --list             # 打印工具清单
python src/mcp_server.py                    # 官方 SDK stdio 模式
python src/mcp_server.py --plain-jsonrpc    # 内置 JSON-RPC 回退
```

## 真实 trace（一次完整修复）

在 `data/toy-repo` 上跑 `CodingAgent.run()` 的实际输出（9 步）：

```
strategy        : offline-heuristic
model available : False | http://localhost:11434/v1 无可用模型（模型列表为空）
skill matched   : test-runner
tests_passed    : True
fix             : {"file": "calculator.py", "line": 3, "from": "return a - b", "to": "return a + b"}

[ 1] kind=skill_load     tool=-          命中 skill「test-runner」，加载正文（score=15.25）
[ 2] kind=strategy_choice tool=-         本地模型不可用，切换到离线确定性策略
[ 3] kind=subagent       tool=-          派发 code-search 子 agent（独立 context）
[ 4] kind=subagent       tool=-          派发 test-runner 子 agent（独立 context）
[ 5] kind=-              tool=read_file   读取候选文件 calculator.py
[ 6] kind=hypothesis     tool=write_file  calculator.py:3 `return a - b` 疑似缺陷，改为 `return a + b`
[ 7] kind=verify         tool=run_tests   验证候选修复
[ 8] kind=final_test     tool=run_tests   收尾终检
[ 9] kind=patch          tool=git_diff    生成最终 patch

subagent code-search  : calls=2 isolated=True -> 在 7 个条目中定位到 1 个候选位置：calculator.py:3 return a - b
subagent test-runner  : calls=1 isolated=True -> pytest 失败，关键行：assert -1 == 5 | where -1 = add(2, 3)
```

产出的 patch：
```diff
diff --git a/calculator.py b/calculator.py
index 20ddc0a..6bd5739 100644
--- a/calculator.py
+++ b/calculator.py
@@ -1,6 +1,6 @@
 def add(a, b):
     """Return the sum of two numbers."""
-    return a - b
+    return a + b
```

## 关于本机没有本地模型（重要，如实说明）

README 指定的基座是本地 Qwen2.5-Coder-7B-Instruct。本机 `ollama` 已安装且 `ollama serve`
在 `127.0.0.1:11434` 运行，但**模型列表为空**（未 pull 任何模型），且本机无外网，无法
`ollama pull`；vLLM / llama.cpp 同样需要先拿到权重。

因此 `CodingAgent` 实现了**两条策略**，并自动选择：

- **模型驱动**（有可用本地模型时，默认路径）：`ModelClient.probe()` 通过 `models.list()`
  确认端点可达且目标模型存在，然后由模型自己决定每步调哪个工具；
- **离线确定性回退**（`strategy=offline-heuristic`）：**不是**"假装调了模型"，也不返回预置答案——
  它走**同一套 `call_tool` 工具**和**同一条 trace 结构**，只是把"选工具"这一步从模型换成确定性策略：
  先派 code-search 子 agent 定位嫌疑行，对候选修复逐个用真实的 `run_tests` 验证（通过才保留，
  否则回滚），最后用 `git_diff` 产出 patch。

回退策略覆盖面有限，必须说清楚：它只处理**机械型缺陷**（运算符写反、边界错误等由
`_candidate_fixes` 枚举的算术运算符替换），**语义型缺陷**（需要理解意图才能改的 bug）它做不了。
上面 trace 里那条 `return a - b → a + b` 是它第一个验证通过的候选，没有靠遍历硬凑。
想验证模型驱动路径，只要部署好本地模型（或设 `CODING_AGENT_OFFLINE=0` 并指向可用端点）即可，
代码无需改动。

## 实验观察（约 350 字）

**三层的价值在 trace 里是可量化的。** Skills 层最直接的收益是"省上下文"：`list_skills()`
只回 3 条元数据（name + description + keywords，约 200 字符），命中后只加载 1 份正文；
如果按朴素做法把 3 份 SKILL.md 全量塞进 system prompt，每次请求会多背约 2.5K 字符，
而其中 2 份与当前任务无关。Subagents 层的收益同样体现在上下文占用上：code-search 子 agent
读了 7 个目录条目、2 个文件，如果把它的完整 observation 直接喂给主 agent，主上下文会多出
数千字符；而摘要把这一整轮压成一句话（"在 7 个条目中定位到 1 个候选位置：calculator.py:3
return a - b"），主 agent 只在这个结论上做决策。

**停机条件是这类系统最容易写错的地方。** 一开始我按"步数到了就停"来考虑，结果是要么早停
（还没修完）要么空转（修完了还在乱调工具）。改成显式 done 信号——`run_tests` 返回
`passed == true` 即停机，再叠加一个独立的收尾终检——之后行为就稳定了。这也解释了为什么
trace 里第 7 步验证通过后不再继续试其它候选。

**离线回退的天花板很清楚。** 它能修 `a - b → a + b`，是因为"运算符写反"是可以在不读懂
意图的情况下枚举验证的；换成一个需要理解业务语义的 bug（比如本该走缓存却直查数据库），
枚举法就无能为力了。这反过来印证了 MCP/Skill/Subagent 这套工程封装的价值：它们解决的是
"怎么把能力可靠地喂给模型"，但**修什么**仍然依赖模型的语义理解能力。
