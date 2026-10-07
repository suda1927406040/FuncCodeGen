# FuncGen v5：修复位置召回

`change_1007` 将 `/data2/ymz/TOSEM2026/v5/v5.py` 的生成流程迁入
`run_oracle.py`，`v5.py` 提供兼容入口，并增加无需调用模型的召回评估。

默认 `expanded` 检索在 v5 历史共修改文件之外，加入多 anchor、跨文件调用者和
被调用者、导入别名与重新导出、状态字段生产者/消费者，以及新包的初始化和导出位置。
候选有具体函数、行号、字段写入和 if/异常分支证据；候选文件不自动成为必改文件。
检索也从请求点名的 API 出发，解析继承/成员对象调用、已有模块的传递导出，以及
代码引用的配置和模板；最终代码区为候选的首要完整方法预留空间。

完整文件先做 AST 解析，之后按完整代码块控制最终 prompt 的 80,000 字符预算，
避免原来的 20,000 字符截断让 AST 失败。编辑解析支持缺少外层结束标签的完整片段；
补丁导出和应用使用同一补丁，拒绝无法定位、歧义、语法错误及空补丁。

## 本地运行

安装 `requirements.txt` 中的依赖。数据与仓库路径通过环境变量配置，默认位于本项目：

```bash
export FEABENCH_DATA_PATH=/Users/shiroha123/projects/feabench-local/feabench-data/FEA-Bench-v1.0-Lite-Standard/test/data-00000-of-00001.arrow
export FEABENCH_REPOS_DIR=/path/to/instance-repos
export OPENAI_API_KEY=你的已有凭据
export OPENAI_BASE_URL=你的已有接口地址
export LLM_MODEL_GENERATE=deepseek-chat
```

只做检索、保存 `recall.json`、`prompt.md` 和 `explorer_context.md`：

```bash
python run_oracle.py --instance conan-io__conan-16871 \
  --recall-only --output outputs/recall
```

正常生成并导出补丁：

```bash
python v5.py --instance conan-io__conan-16871 --output outputs/v5
```

不设置 `--apply` 时只生成补丁。普通运行会按数据集中的 base commit 准备实例仓库。
若已有准确的 base-tree 源码目录，可加 `--repo-dir /path/to/base-tree` 跳过 checkout；
这个选项只允许单个实例，并禁止 `--apply`。提供的源码必须确实来自该实例的 base commit。
`--dataset-path` 可以代替数据环境变量。

检索模式：`--recall-mode v5` 使用迁入后的单 anchor 历史逻辑；`expanded` 使用增强检索。
`--recall-max-files 12` 控制最多增加的候选文件数。迁入后的 v5 模式也包含完整 AST
解析和 prompt 预算修正，因此并非原始 55/200 实验代码的逐字节重放。

## 检查与召回实验

单元测试仅需 Python 标准库和 Git：

```bash
python -m unittest discover -s tests -v
```

在准确的 base-tree 快照上比较修复文件召回，不调用模型：

```bash
python scripts/evaluate_recall.py \
  --dataset-path "$FEABENCH_DATA_PATH" \
  --repo-root /path/to/base-tree-snapshots \
  --output evaluation/recall.json
```

快照目录按实例 ID 命名。`--baseline-root /path/to/b67f93b-source` 可增加原始 main
版本的比较；`--modes expanded` 可仅运行增强检索。参考补丁仅在检索完成后用于评分，
新测试和参考补丁均不传给检索或 prompt 构造接口。

已检查的 17 个真实 base-tree 小样本、统计定义及限制见
[evaluation/README.md](evaluation/README.md)。召回率与最终解决率需要分别评测。

针对当前 honest 结果全部 **121 个 unresolved** 的逐例诊断、改动依据及验证范围见
[evaluation/unresolved/README.md](evaluation/unresolved/README.md)。本地 17 个快照中只有
5 个属于该失败集合。指定全量集合时加 `--ids-json evaluation/unresolved/ids.json
--require-complete`；`--location-labels evaluation/unresolved/location_labels.json` 可在检索
完成后评分具体已有方法是否保留在 prompt，标签不会传给检索或生成接口。

默认历史检索现按同一 commit 成组展示定义、调用、状态、导出及配置变更，并映射到
当前 base 方法。`--history-mode patterns|legacy|off` 用于比较。实现、真实历史示例
和验证结果见 [evaluation/history_patterns.md](evaluation/history_patterns.md)。
