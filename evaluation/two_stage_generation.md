# 两轮调用，一份最终补丁

默认生成模式现为 `plan-edit`。一次完整求解包含一轮结构化计划、一轮代码生成。
第一轮不生成候选补丁，第二轮只返回一套 edits，随后编译成一份 `generated.patch`。
`single` 保留单轮生成用于比较。

## 两轮如何衔接

第一轮输入需求、声明定义、静态位置证据、同一 commit 的协同变更和当前代码。
它输出 JSON：行为 contracts、带文件/符号/目的/约束/证据的步骤、步骤依赖以及
`missing_context`。默认与代码生成使用同一模型、同一温度，输出上限为 2,048 tokens。

脚本检查相对路径、base 文件、当前函数/类/字段、给定历史证据和依赖环；继承方法、
重新导出及模块限定名会映射到真实源码位置。声明的新定义和明确的 `create_symbol`
用于区分已有符号与要创建的入口。计划的语义正确性仍需模型对照需求和源码判断。

随后按计划补入请求的定义、调用者/被调用者、稀有状态字段的相关位置及包导出/stub。
第二轮使用核验后的计划与当前代码，去掉旧 prompt 中要求机械复制相似方法或无条件
导出 helper 的指令。每轮 prompt 上限仍为 80,000 字符；需要的当前方法未能完整进入
第二轮代码区时，会记录失败并停止代码调用。

第一轮 JSON 无效、截断、编辑位置无效或依赖成环会直接失败，不调用模型修补计划。
无效的可选补充上下文请求会记录 warning。代码生成与补丁编译失败也不会再次请求
修复或生成其他候选。重新运行同一个输出目录时，会清除上一轮候选补丁，防止失败
后误提交旧补丁。

## 运行

安装原有 `requirements.txt` 中的依赖，沿用模型和凭据环境变量。

```bash
python v5.py --instance conan-io__conan-16871 \
  --generation-mode plan-edit --output outputs/plan_edit
```

单轮对照：

```bash
python v5.py --instance conan-io__conan-16871 \
  --generation-mode single --output outputs/single
```

选项：

- `--plan-max-tokens 4096` 调整计划输出预算；环境变量为 `LLM_PLAN_MAX_TOKENS`。
- `--planner-model ...` 指定规划模型，默认使用生成模型。
- `--planner-json-mode api` 使用 OpenAI-compatible JSON response format；不支持它的
  服务可指定 `prompt`，只靠提示约束 JSON。Anthropic 路径使用 JSON 提示与本地解析。
- `--recall-only` 不调用模型，保存静态上下文及 `planning_prompt.md`。此时没有真实
  计划或第二轮生成上下文；`prompt.md` 仍是静态对照。完成两轮后它保存最终生成 prompt。

DeepSeek JSON 模式需要 JSON response format、明确的 JSON 提示和足够输出预算；
服务也可能返回空内容，本流程会将其作为无效计划记录，不静默增加语义重试。
参见 [DeepSeek JSON 文档](https://api-docs.deepseek.com/guides/json_mode/)。

## 输出与指标

|文件|内容|
|---|---|
|`planning_prompt.md` / `plan_response.md`|第一轮输入与原始输出|
|`plan.json` / `plan_validation.json`|规范化计划、位置核验与 warnings|
|`generation_prompt.md` / `generation_recall.json`|第二轮输入、实际可见位置和补充依赖|
|`response.md` / `generated.patch`|唯一一轮代码输出及其补丁|
|`llm_usage.json` / `result.json`|两阶段用量、耗时、失败阶段及最终补丁数量|

使用 SDK 返回的真实 token 数，不以字符数猜测用量；缓存 token 明细保留在每轮
原始 usage 中。服务未返回用量时总量为 `null`，`reported_*` 仅统计已报告部分。
本流程没有假设接口的单价，不虚构货币成本。

`logical_calls` 记录逻辑阶段调用次数；`provider_attempts` 记录适配器尝试次数。
SDK 内置重试已关闭，传输/服务器错误的有限重试由接口包装层记录。一次成功
`plan-edit` 通常是两次逻辑调用，实际传输尝试可能因网络错误增加。

`pipeline_attempts=1`、`max_final_candidates=1`；计划失败仍算这次求解失败。
因此可称两阶段方法的流程级 pass@1，而非单次模型调用的 pass@1。FEA/SWE-bench
主指标建议使用每例一份最终补丁的 resolved rate，并同时报告调用次数、token 和时间。
参考补丁和测试补丁不进入两轮 prompt，也不用于挑选或重试候选。

## 当前验证

标准库测试验证两阶段调用顺序、计划核验、补充不可见的源码、唯一补丁输出、用量
记录、非法计划/截断/过大上下文失败、编译失败不重试及旧补丁清理。
SDK 适配器使用模拟响应验证 OpenAI-compatible 和 Anthropic 用量字段与重试计数。

本地真实 base 快照的静态输入召回仍为 27/27 修复文件、9/9 额外修复文件，已有
方法覆盖 5/5。它验证静态上下文的回归，不是模型生成计划之后的召回或解决率。
当前未发起实际付费 LLM 请求，也未重跑 SWE-bench；本机验证环境尚未安装供应商
SDK，真实接口兼容性和解决率仍需在配置好的运行环境中验证。
