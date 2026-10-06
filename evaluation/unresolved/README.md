# 121 个 unresolved 的诊断与改动

分析集合固定为 `/data/ymz/FEA-Bench/evaluator/SWE-bench/v5-full.V5_full_honest.json`
中的 121 个 `unresolved_ids`。原实验为 200 例，55 resolved、121 unresolved、
17 empty patch、7 error；本报告分析这 121 例，后两类没有混入分母。
`ids.json` 保存完整选择集合。没有按本地缓存是否可用来选择分析样例。

## 原始输出：文件遗漏只是其中一类问题

`original_unresolved.json` 逐例保存原 prompt 的可见文件、目标文件、事后参考补丁
修改文件以及测试通过/失败数。参考补丁只用于诊断和评分，不用于检索或构造 prompt。
文件统计限生产 `.py` / `.pyi`；不把 changelog、作者信息等混入功能代码召回率。
“可见”指文件名进入原 prompt 的 `<code>`，新建文件占位符也计为文件可见。

|指标|结果|
|---|---:|
|参考补丁修改的 Python / stub 文件|195|
|原 prompt 中可见的修复文件|155/195（79.5%）|
|声明目标以外的额外修复文件|58|
|原 prompt 中可见的额外修复文件|18/58（31.0%）|
|至少遗漏一个修复文件的实例|32/121|
|旧测试出现回归的实例|31/121|
|新增测试至少通过一个的实例|33/121|
|新增测试部分通过、部分失败的实例|31/121|

以上测试类别有重叠。其余 2 个有新增测试通过的失败样例，新增测试全部通过，但旧测试
出现回归。不能把所有 unresolved 都归因于漏文件。

原始响应的本地重放还发现 5 例 Python 语法问题，以及 19 例声明定义不在声明位置。
其中既有 MONAI-465 未创建目标文件、Pvlib-1562 未实现声明入口，也有合法的 helper
移动和可能的继承实现；19 是待检查的结构警告数，不能直接解释为 19 个必然的 API
缺失。120 例有足够的原 prompt 源码可重放，Sympy-21510 的一个基础文件缺失，不能
完整重放。详见 `structural_audit.json`。

## 已有服务器位置报告：需要召回具体方法

`existing_location_reports.json` 来自服务器已有的两个静态评估文件，只保存符号名、
文件名和位置统计，没有导出各仓库源码。这是历史诊断数据，**不是本次分支的全量验证**。
74 个 unresolved 含已有代码的修改位置，共 127 处；新函数本身不属于已有位置。

|已有静态报告的检索通道|已有修复位置命中|候选位置数|
|---|---:|---:|
|reference chain|5/127（3.9%）|38|
|原报告 FeatureSpace|31/127（24.4%）|773|
|v2 hybrid FeatureSpace|68/127（53.5%）|1225|
|v2 repository lexical|50/127（39.4%）|1406|

扩大候选类/文件会提高覆盖，也会带入大量无关位置。需要优先展示有明确请求、调用、
状态或导出证据的方法，并检查它们是否真正留在最终 prompt 的代码区。

跨仓库遗漏包括：

|实例|原 prompt 缺失的位置或关联文件|本次通用改动|
|---|---|---|
|Roboflow-245|`dataset/core.py`、`detection/utils.py`|从请求中的 `DetectionDataset.from_pascal_voc` 等 API 出发，再扩展调用|
|Roboflow-847|`Detections.with_nms`、公开导出|明确 API 检索与已有模块的导出追踪|
|Sklearn-17225|`median_absolute_error`、`utils/stats.py`|声明目标以外的 API 与 `_weighted_percentile` 调用检索|
|Sympy-21510|`logic/inference.py::satisfiable`|识别自然语言中未加反引号的、少量定义的长 API 名|
|Conan-12243|`cli/commands/graph.py::graph_info`|识别请求中的命令/API 名|
|Conan-10154、Optax-632/897|新包/上层包的 `__init__.py`|新模块初始化路径与已有传递式 re-export|
|Conan-11284|`autotools.py`、`new_v2_autotools.py`|继承/成员对象调用与字符串中的模板类引用|
|Sagemaker-3423/3432|Neuron/PyTorch/TensorFlow JSON 配置；3432 还缺 `fw_utils.py`|运行代码中的资源路径与请求中的框架名称联合检索|
|Astropy-16677|`modeling/fitting.py`|跨文件调用和继承方法检索|

这些是改动的依据，不代表上述全部样例已在本分支被重新验证。32 个缺失 Python 文件
样例的完整清单在 `summary.json` 的 `missing_source_cases`，其余样例也在逐例报告中。

## 已落实的检索与预算改动

- 从请求明确引用的函数/方法直接启动检索；对普通文本中的长 API 名仅接受少量定义
  的精确名称匹配，优先级低于带限定名或反引号的引用。
- 解析 `self` / `cls` 的继承方法，以及构造函数赋值可确认类型的 `self.worker.method()`。
- 追踪已有模块经父包导出的路径，并保留正常的 `pkg/build/...` 生产包。
- 检索目标代码实际引用的配置资源；动态目录需要与请求相关的资源文件名。模板仅在
  template/example 路径内、字符串确实引用目标类时补入。
- 大配置显示原文连续窗口；完整 Python 方法按检索优先级显示，为后续候选的首要
  方法预留空间。修正首个类方法装饰器在截取时重复或错误附着到其他方法的问题。
- 评估同时保存最终 prompt 中可见的方法名，新增完整集合检查，避免缺失快照被静默
  跳过之后仍把小样本结果当作全量结果。

## 当前验证范围

本地只有 17 个已确认的真实 base-tree 快照，其中 **5 个属于这 121 个 unresolved**：
Conan-16871/10154/11675、Roboflow-245/910；另 12 个是已解决样例，用于回归检查。
另一处本地缓存没有数据集所指的真实 base commit，未作为评估输入。

`before_available_unresolved.json` 使用本次修改前的 `ee74005` 检索/渲染代码，
`available_base_regression.json` 使用本次修改后的代码，两者使用同一批真实 base 源码。
位置标签在检索和构造 prompt 完成后才用于评分。

|5 个可用 unresolved 的指标|修改前|修改后|
|---|---:|---:|
|修复文件可见|12/12|12/12|
|额外修复文件可见|6/6|6/6|
|已有修复方法在最终代码区可见|4/5|5/5|

Conan-16871 单例已确认：旧代码显示 `compute_pid.py` 的文件名，却没有完整显示
`compute_package_id`；预算调整后该方法进入最终代码区，已有位置覆盖从 2/3 到 3/3。
这也解释了为何单看文件召回会高估上下文质量。原始服务器 v5 的另一份 prompt 本来
就曾包含此方法，但模型将 `cant_build` 与 `invalid` 的异常状态同时赋值，仍然失败；
位置覆盖本身不能保证语义正确。

全部 17 例回归仍覆盖 27/27 修复文件、9/9 额外修复文件，平均 prompt 为 76,455 字符，
平均显示 13 个文件（含声明目标和伴随导出/stub）。以上为开发集结果；位置评分只有
5 个已有方法，不能推断全量 127 个位置的覆盖。

29 个标准库单元测试覆盖请求 API、继承、委托、导出、配置、模板、完整方法预算、
装饰器和补丁一致性。没有运行新的模型求解或 SWE-bench 测试，不能宣称解决率提升。

完整 121 例的新检索回归需要服务器真实 base 源码。目前源码跨机器导出，以及上传
本地分析模块到服务器的操作被自动审批拒绝；原因是可能传输非公开代码。已有报告
的只读位置元数据分析已完成，源码导出的明确授权仍待用户答复。

## 复现

汇总已保存的 121 例诊断：

```bash
python scripts/summarize_unresolved.py
```

有完整、准确的 base-tree 快照后，对固定 unresolved 集合只做检索：

```bash
python scripts/evaluate_recall.py \
  --dataset-path "$FEABENCH_DATA_PATH" \
  --repo-root /path/to/base-tree-snapshots \
  --ids-json evaluation/unresolved/ids.json --require-complete \
  --location-labels evaluation/unresolved/location_labels.json \
  --modes expanded --output evaluation/unresolved/full_recall.json
```

`--require-complete` 缺任何请求实例时直接失败；不加它时，输出 `coverage` 明确列出未
评估的实例。随后应使用相同生成模型和预算分别生成修改前后的补丁，运行 honest
SWE-bench，比较新增测试与旧测试的最终结果。
