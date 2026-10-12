# FeatureSpace self-evolution

此版本仅调整初始生成之后的修复流程。初始检索、历史分析、规划和代码生成保持原样。FeatureSpace 为单个任务保存程序元素、功能分组、依赖和行为条件，初始内容来自已有 recall/plan 产物；所有计划推断都可被后续证据修正。

## 本版变更（对照 a983c46，针对 conan-io__conan-16871 的四轮失败）

1. **历史证据可引用但显式区分新旧状态。** 上一版只接受本轮观测，第四轮诊断引用三个真实存在的旧观测 ID 被整体拒绝（`cites unavailable evidence`）。现在存储过的 `ev:` ID 都可引用；引用旧观测的条目打上 `cites_historical_evidence`，观测本身带补丁与轮次戳。
2. **同名方法重定义按名称拦截。** 上一版只拦"方法体完全相同"的重复，第一轮"同名不同体"的遮蔽式重定义漏过。现在类体内同名方法出现次数比修改前增多即拒绝。
3. **分支执行证据。** 诊断容器回收 observer 的 per-process 截断摘要，计算 `trace_coverage`；新增 `branch_outcomes`（守卫条件、实际进入的分支体、下一执行行、体内调用、判定前变量/状态，按命令与调用绑定）与 `watched_call_coverage`（受监测函数在失败测试中是否被调用；仅完整覆盖下才给 `not_called_confirmed`，否则 `not_observed_unknown`）。
4. **可验证判断闭环。** 每次结构变更必须附带 check（`hypothesis`/`expected_change`/依据）；下一轮诊断先产出 `check_reviews` 核对预期是否发生，再决定修正判断还是推进到下一个断点。修复阶段不重复已被否定的修改。每轮仍是至多两次模型调用（诊断 + edits）。

## 一轮如何工作

1. 从不可变初始镜像创建新容器，应用当前候选，执行正式测试。状态诊断另建新容器，也从初始镜像开始。测试环境恢复与网络隔离机制保持不变。
2. 从完整日志中选择真正的测试阶段，避免把前面的源码 diff 当成异常。若旧归档的评测日志已被截断，允许从本轮诊断日志恢复反馈，并标明来源。按类别和测试选择状态证据，不整类删除；每条观测都有可引用 ID、省略统计及候选指纹。
3. 诊断调用先核对上一轮 checks：把每条 `expected_change` 与本轮 `branch_outcomes`（守卫实际走的分支，由同帧下一执行行推导）、`watched_call_coverage` 和轮次测试增减对照，结果写入 `check_reviews`（observed / not_observed / not_observable）。预期未发生时重新审视该修复与判断本身；预期已发生但测试仍失败时，按执行顺序定位下一个断点。随后提出局部 `upsert`/`retire`。历史轮次的 `ev:` ID 仍可引用，但每条观测都带所属补丁与轮次戳，引用旧观测的条目会被标记 `cites_historical_evidence`，不得当作当前候选的事实。
4. 程序校验引用、代码路径和实体关系后原子合并更新。任何结构变更必须附带至少一条 check（`hypothesis` + `expected_change` + 依据），否则整个更新被拒绝。初始计划中的错误条件允许撤回；新推断仍标记为 hypothesis。候选改变后，基于旧候选的推断标记 `needs_recheck`。诊断失败保留上一版 FeatureSpace 和当前观测，继续代码生成。
5. 修复调用使用更新后的 FeatureSpace（含各 check 的 `expected_change`）、关联的执行证据和当前源码，返回协同 edits；不重复已被 check review 否定的修改。它不再单独接收原始错误日志和原始状态 packet；关联证据可保留必要的断言和状态原文。
6. 预检拒绝测试文件修改、类内同名方法重定义（无论方法体是否相同——重定义会遮蔽原方法）、仅注释/格式变化和历史候选循环。通过后导出相对于原 base 的合并补丁。下轮正式测试记录修复了哪些测试、引入哪些回归、哪些仍失败以及哪些未观测到；诊断容器同时回收每个 observer 进程的截断摘要（`observer_summaries`），据此计算 `trace_coverage`：只有完整覆盖下未被调用的受监测函数才标记 `not_called_confirmed`，否则一律 `not_observed_unknown`。FeatureSpace 关注的程序元素会补充到下一轮观测范围。

## 复用原来的初始候选

必须创建新 run-root；不要修改或恢复旧版本实验来混用两种流程。原运行和输入路径如下仅为示例，凭据路径使用部署时已有的私密配置。

```sh
python scripts/run_state_feedback.py \
  --run-root /path/to/new/featurespace-case-run \
  --inputs /path/to/original-run/generation_inputs.json \
  --reuse-initial-from /path/to/original-run \
  --prepared-repos /path/to/verified/base/repos \
  --credentials-file /path/to/private/deepseek.json \
  --same-host --bundle /data2/ymz/feabench-offline-1007 \
  --instance-id conan-io__conan-16871 \
  --feedback-mode featurespace --repair-rounds 4 --workers 1
```

省略 `--instance-id` 即使用全部 200 例；可以重复该参数选择多例。仅复制初始生成补丁、结果、plan、recall、usage 和完成标记，不复制历史修复、测试结果或参考解。程序检查任务输入/base 一致、准备仓库干净，并记录复用产物哈希。恢复仍使用 `--resume` 和原 run-root，源码以及模式和轮次预算保持冻结。

## 对照模式与预算

| 模式 | 修复输入 | 每轮逻辑模型调用 |
| --- | --- | --- |
| `basic` | 测试反馈和当前代码、修复历史 | 1 |
| `state` | 同上，加当前运行的状态证据 | 1 |
| `featurespace`（默认） | 先更新 FeatureSpace，再用其及关联证据修复 | 至多 2，诊断 + edits |

三种模式共享完整日志保存、错误选择和修复预检。`basic` 不进行额外状态诊断；另外两种模式使用相同观察器。相同四轮上限不代表相同模型或执行预算，需要同时报告调用、token、评测和诊断次数。`logical_calls`/`reported_tokens` 包含完整 workflow 的原初始成本，`new_run_logical_calls`/`new_run_reported_tokens` 排除复用的初始成本。`model_calls.json` 保存恢复前后真实调用记录；没有返回 usage 的调用另计为未知，不能按已知 token 推断其成本为零。

## 产物与限制

- `featurespace_before.json`、`featurespace_delta.json`、`featurespace_after.json`：每轮记忆及更新，可审查错误约束是否真正撤回。
- `evidence.json`：被选择的证据、来源和省略数量。完整 packet 与日志独立保留。
- `diagnosis_prompt.md` / `diagnosis_response.json`、`repair_prompt.md` / `repair_response.json`：两阶段模型输入输出。
- `candidate_before.json` / `candidate_after.json`、`outcome.json`：候选指纹、修改文件及后续测试变化。
- `diagnosis_error.json`：诊断失败时的兜底原因；`error.json`：修复未应用的原因。
- `featurespaces/<id>.json`、`repair_history.json`：包含测试结果的最新记忆和完整轮次记录。

这不是形式化程序验证：AST 指纹用于发现相同结构和循环，不能证明两个不同补丁行为不同。轨迹有边界，测试也只支持已观察行为。当前实现没有额外的按需探针模型调用；它更新下一轮的观测范围。新增本地测试和历史产物回放用于验证框架的数据流、恢复和预检，不代替真实 LLM 与离线 harness 的解决率评测。
