# 规划、生成与结构错误修复

默认 `plan-edit` 通常执行规划和代码生成两次调用，每个任务输出至多一份最终补丁。
计划是辅助信息：JSON 无效或截断、符号/历史证据校验失败、规划上下文超过预算时，
记录失败原因，回退到原来的直接生成 prompt。不能因规划失败而跳过代码生成。

新文件的依赖步骤可在计划中依次创建文件和定义声明符号；模块级常量、导出和别名
按当前 base 的真实赋值解析。测试或文档步骤不会使有效的生产代码计划整体失败。
历史证据仍限于给定的 base 祖先 commit，非法路径和臆造的已有符号仍不被认可。

编辑解析兼容完整、路径一致的 after 结束标记笔误，并支持 JSON edits。
相同编辑仅应用一次；缩进对齐基于共有源码锚点，避免新增同级方法导致整个片段移位。
包含省略占位符的编辑只在前后锚点唯一且 before/after 均保留占位符时展开真实代码，
保留未展示的源码；存在歧义时交给修复阶段，不选择任意位置。

代码输出被截断、编辑解析/定位失败、编译失败、空 diff 或 `git apply --check` 失败时，
默认至多进行两次结构修复调用。修复 prompt 包含需求、当前真实源码的连续片段、
上一轮输出和具体错误，以 JSON 返回完整 edits。所有反馈来自本次生成和当前 base；
参考 patch、测试 patch 和 harness 判定不进入 prompt。未通过预检的补丁不会提交。
这是结构检查反馈，尚不是完整的仓库功能测试反馈闭环。

```bash
python v5.py --instance conan-io__conan-16871 \
  --generation-mode plan-edit --max-repair-attempts 2 --repair-max-tokens 16384
```

`--max-repair-attempts 0` 关闭结构修复；计划失败仍使用直接生成。
其他选项保留：`--plan-max-tokens`、`--planner-model`、`--planner-json-mode`、
`--recall-only`、`LLM_THINKING_MODE`。供应商设置和实际模型由调用环境指定并记录。

每个任务保存规划输入/输出、校验和 fallback 原因、生成输入、每次原始代码响应、
结构修复 prompt 和失败记录、SDK 返回的真实 usage，以及唯一最终 `generated.patch`。
`repair_attempts`、`planning_fallback` 和 `llm_usage.calls` 明确区分所有阶段。
默认最多一次规划、一次生成、两次修复；传输重试另计。流程中不选择多个最终候选，
指标应表述为流程级 resolved rate，并报告调用预算，不能称单次模型调用。

历史独立全量实验 revision77634df：38/200=19.0%，73个无补丁；原v5为55/200=27.5%。
73个原始响应的离线审计发现10个标签错误和11个自动重缩进错误可直接恢复结构有效的
补丁。本次失败恢复实验与原实验分开保存；恢复已有失败项的结果不等同于新的独立
全量 pass@1。新版真实解决率须以实际 harness 报告为准。
