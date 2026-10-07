# 用协同 commit 理解职责，再生成当前代码的修改

此轮在 `change_1007` 的 `ab2c6e8` 基础上增强历史信息。默认 `expanded` 模式先选择
相似的当前函数，再检索 base commit 祖先中的相关提交，将同一提交的多个文件变更
成组呈现。`v5.py` 是 `run_oracle.py` 的兼容入口，默认也使用新模式。

## 原先如何使用 commit

1. 在声明文件中按新函数名字选择相似 anchor（增强版最多 3 个）。
2. `CommitTracer` 用 `git log <base> -S 'def <anchor>' -- <file>`，选择最早的名称
   数量变化提交；大重构时尝试下一条。
3. 读取同次提交的其他 Python 文件，按目录邻近和关键词过滤，每个 anchor 最多
   3 个文件、每个 diff 最多 2,000 字符。
4. 将这些文件、diff 和部分属性改名提示放入生成 prompt；另一路搜索调用链历史。

这条链路确实使用了协同变更，但名称数量变化不一定是该类方法的引入；协同文件
可能没有共享关键词或位于其他职责目录；按文件去重会丢失提交归属。调用链历史
标成“完整提交”，实际读取的是目标文件 diff；历史信息还可能在分析预算末尾被裁掉。

## 当前如何改进

`explorer/history_patterns.py` 将一个 commit 作为一个参考单元，包含：

- 提交 ID、第一父提交、标题与消息摘要、检索依据和分数。
- 同次提交中的定义/签名、调用、状态写入、分支/异常、公开导出和配置变更。
- 历史 diff 中涉及的符号在**当前 base** 中的对应方法。
- 能解析的当前跨文件调用关系，以及历史定义与调用/导入名称的关联。

检索包括经前后 AST 验证的函数引入、相关函数的后续修改、与请求相关的提交消息。
目录和关键词用于选择相关 commit，其协同文件无需都满足原来的硬筛选条件。
默认最多 2 组参考、每组最多 6 个生产文件，最多补入 6 个历史关联文件。

普通自然语言单词不会作为跨模块历史 anchor。对限定 API，需要当前类/模块解析
证据；`conan.tools.files` 命名空间不会误匹配成别处的 `Lock.files` 属性。大规模
重构仍会过滤。新建文件或 shallow 仓库可能没有合适历史，此时继续使用静态检索。

prompt 保留完整提交职责表，按预算选择完整 diff hunk。无法容纳的大 hunk 展示
明确标记的变更行观察。历史区最多 8,000 字符，仍在原有分析和 80,000 字符总预算
内；目标包导出及 stub 优先进入代码区。完整调用/状态遍历在历史发现后只运行一次。

生成时先写简短的集成计划：将当前需求映射到入口、helper、状态/guard、导出和
配置，说明哪些历史职责适用，再给出所需 edits。历史实现可能已被替换，编辑
before 片段仍须来自当前 `<code>`。协同文件是否需要改动应由当前需求决定。

## 真实参考示例

Boto3-74 的历史 `06e4107986bf`（Added waiter actions to resources）同时涉及：

|文件|同次提交中的职责证据|
|---|---|
|`boto3/resources/model.py`|资源 waiter 的模型信息变化|
|`boto3/resources/factory.py`|动态资源方法的生成和组装变化|
|`boto3/resources/action.py`|`WaiterAction` 及其状态、调用行为变化|
|`boto3/docs.py`|对应的文档生成变化|

当前 base 的静态解析还确认了
`ResourceFactory._create_waiter -> WaiterAction.__init__`。这将历史协同变更与当前
调用联系起来，帮助解释模型信息如何经 factory 暴露为资源行为。实际变更仍应由
当前请求决定。

Sympy-14853 的 `99975e18b073`（Add discrete module, fft）同时涉及算法文件
`discrete/transforms.py`、包导出 `discrete/__init__.py` 和 `setup.py` 的集成。
该提交的职责表及当前包导出一起保留在 prompt 中。

## 验证范围

`history_patterns_17.json` 保存新模式在 17 个真实 base-tree 快照上的结果。
比较数据为 `unresolved/available_base_regression.json`（`ab2c6e8`）。

|最终 prompt 的覆盖指标|修改前|新模式|
|---|---:|---:|
|修复文件|27/27|27/27|
|额外修复文件|9/9|9/9|
|可评分的已有方法|5/5|5/5|

新模式平均 prompt 为 77,052 字符，平均可见 13.7 个文件。这里确认历史上下文没有
破坏已验证的召回，尚未证实代码正确率提升。位置评分只有 5 个已有方法。

40 个单元测试通过，覆盖跨职责目录的协同变更、当前方法映射、命名空间歧义、相同
方法名在不同类中的引入、后续修改、未来提交排除、测试文件排除、off 模式、
大重构和成组历史预算。

17 例中只有 5 个是当前 unresolved。Conan-16871 本地为不带 `.git` 的真实源码
archive，此例无法验证历史检索；Roboflow-245 的缓存是 shallow 仓库，没有筛选出
可用历史参考。本轮没有调用模型生成新补丁，也没有运行新的 SWE-bench 测试。

## 使用与消融

只做检索并保存 `prompt.md`、`recall.json`，不 checkout 或调用模型：

```bash
python run_oracle.py --instance boto__boto3-74 \
  --dataset-path "$FEABENCH_DATA_PATH" \
  --repo-dir /path/to/base-tree-with-git-history \
  --recall-only --history-mode patterns --output outputs/history
```

`recall.json.history_patterns` 保存提交组、职责、关系及 diff；
`visible_history_commits` 记录实际进入 prompt 的组，`diagnostics.history` 记录
可用性、查询数量和超时。Git 搜索起点均为本例 base，不使用 HEAD 或 `--all`；
每个选中提交还做祖先检查。查询缓存与时间预算控制探索成本。

`--history-mode patterns` 为默认新模式；`legacy` 保留旧逻辑作比较；`off` 关闭
两条历史上下文路径。`--recall-mode v5` 默认保留 legacy，可作为旧检索对照。

召回评估也支持 `--history-mode`。公平的生成实验应固定模型、温度、输出 token
预算和数据集合，比较 `off` / `legacy` / `patterns`，再用 honest SWE-bench 分别
统计新增测试通过、旧测试回归和最终 resolved。base 之后的提交及参考 patch/test
不能进入任何一组生成输入。
