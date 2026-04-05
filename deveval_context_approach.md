# DevEval 上下文提取与 Prompt 构建方案

## 概述

核心思路：用 definition 模式让 LLM 生成完整函数（带 `def` + docstring + body），然后用 AST 定位替换，避免缩进问题。上下文提取参考 `run_oracle.py` 的 `ContextBuilder` 策略，并叠加 DevEval 特有的 API dependency 信息。

## 关键改动

### 1. Definition 提取（替代 body_position 替换）

从源文件用 AST 提取目标函数的签名 + docstring 作为 definition，要求 LLM 输出完整函数。

```python
def extract_definition(comp_record):
    """从源文件 AST 提取函数 definition（签名 + docstring）"""
    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(REPOS, comp_record["completion_path"])
    
    func_name = comp_record["namespace"].split(".")[-1]
    # 在 module.functions 或 module.classes[cls].methods 中找 func_name
    func_info = ...  # FunctionInfo 对象
    
    # 从 source 中截取签名行
    src_lines = module.source.splitlines()
    sig_lines = src_lines[func_info.line_start-1 : sig_end]  # def 行到冒号
    signature_text = "\n".join(sig_lines)
    
    # 截取 docstring 行
    docstring_lines = ...  # 从 sig_end 开始找 """..."""
    definition_text = signature_text + "\n" + "\n".join(docstring_lines)
    
    return {
        "definition_text": definition_text,  # def + docstring
        "line_start": func_info.line_start,  # AST 行范围（1-indexed）
        "line_end": func_info.line_end,
        "indent": len(src_lines[func_info.line_start-1]) - len(src_lines[func_info.line_start-1].lstrip()),
        "class_name": class_name,  # method 时有值
    }
```

### 2. 参考结构提取（复用 run_oracle 的 ContextBuilder 思路）

对每个目标函数，提取以下上下文：

#### 2.1 类上下文（method 时）
```python
def analyze_file(comp_record):
    # __init__ 实例变量: self.xxx = yyy
    init_vars = {}
    if "__init__" in cls.methods:
        for m in re.finditer(r'self\.(\w+)\s*=\s*(.+)', init_src):
            init_vars[m.group(1)] = m.group(2)
    
    # 类变量: TYPE_MAPPING = {...} 等
    class_vars = dict(cls_info.class_vars)
    
    # 完整 __init__ 源码
    init_body = cls.methods["__init__"].body_source
    
    # 兄弟方法（排除目标函数和 __init__，单独展示）
    siblings = {mn: mi for mn, mi in cls.methods.items() 
                if mn != func_name and mn != "__init__"}
```

#### 2.2 调用图
```python
def _find_method_callees(method_info):
    """从方法体中提取 self.xxx() 调用"""
    callees = re.findall(r'self\.([a-zA-Z_]\w*)\s*\(', method_info.body_source)
    return list(dict.fromkeys(callees))
```

#### 2.3 参考调用链（最相似兄弟方法的完整调用链）
```python
def _build_reference_call_chain(func_name, cls_info):
    """找命名最相似的兄弟方法，BFS 追踪其调用链"""
    func_words = set(re.findall(r'[a-z]+', func_name))
    # 按词重叠排序找最相似的方法
    similar.sort(key=lambda x: x.similarity, reverse=True)
    # BFS 从最相似方法开始，追踪 self.xxx() 调用链（最多5层）
```

### 3. DevEval 特有：API Dependency 解析

DevEval 数据提供 `dependency` 字段：`{intra_class: [...], intra_file: [...], cross_file: [...]}`，列出目标函数必须调用的 API。

```python
def resolve_api_source(namespace, comp_record, analyzer):
    """解析 dependency namespace 到源码"""
    target_name = namespace.split(".")[-1]
    
    # 1. 同文件查找（覆盖 intra_class + intra_file）
    module = analyzer.parse_file(REPOS, comp_record["completion_path"])
    # 在 module.functions 和 module.classes[cls].methods 中找
    
    # 2. 跨文件查找（cross_file）
    # namespace 如 "boto.regioninfo.connect" -> boto/regioninfo.py
    parts = namespace.split(".")
    for i in range(len(parts)-1, 0, -1):
        module_path = "/".join(parts[:i]) + ".py"
        candidate = REPOS / project_path / module_path
        if candidate.exists():
            mod = analyzer.parse_file(REPOS, candidate_rel)
            # 在 functions 和 classes 中找 target_name
```

### 4. Prompt 结构

```
[Requirement] 功能描述 + 参数说明

[Dependency summary] dependency 字段原始信息
[definitions] 函数签名+docstring（让 LLM 知道要补全什么）

[analysis section]
  - 类上下文（类名、基类、__init__ 变量、类变量）
  - __init__ 完整源码
  - 兄弟方法完整源码
  - 调用图
  - 参考调用链（最相似兄弟 + 其调用的方法）
  - **REQUIRED API CALLS**（解析到的 dependency API 源码）

[imports] 文件中的 import 语句

[file context] 文件源码（截断到 8000 chars）

[Instructions]
  - 必须调用 REQUIRED API CALLS 中列出的 API
  - 输出完整函数定义（包括 def、docstring、body）
  - 保持正确缩进
  - 只输出代码
```

### 5. 评估方式

用 AST 找到函数的完整范围（`func_node.lineno` 到 `func_node.end_lineno`），整体替换生成的完整函数：

```python
def eval_one(ns, data, gen_code, source_root):
    # AST 找函数范围
    func_node = _walk(tree, func_name)
    start, end = func_node.lineno, func_node.end_lineno
    
    # 缩进修正：以 def 行为基准 shift 所有行
    orig_indent = len(file_lines[start-1]) - len(file_lines[start-1].lstrip())
    gen_def_indent = 找 gen_code 中 def 行的缩进
    shift = orig_indent - gen_def_indent
    # 统一 shift 所有非空行
    
    # 替换
    new_lines = file_lines[:start-1] + [gen_code] + file_lines[end:]
```

## 适配 CoderEval 的要点

CoderEval 与 DevEval 类似，都是函数补全任务。关键区别：

1. **上下文来源**：CoderEval 可能没有 `dependency` 字段，需要看数据里有什么参考信息
2. **参考函数提取**：同文件/同类的兄弟方法、调用链逻辑可直接复用
3. **Definition 模式**：同样适用——提取签名+docstring，让 LLM 输出完整函数
4. **缩进处理**：eval 脚本中的 dedent + shift 逻辑可通用
5. **API dependency**：如果 CoderEval 有类似的依赖信息，resolve_api_source 可直接用

## 关键文件路径

| 文件 | 说明 |
|------|------|
| `/data2/ymz/TOSEM2026/run_deveval.py` | 生成脚本（definition模式 + 上下文提取 + API dependency） |
| `/data2/ymz/TOSEM2026/eval_deveval_def.py` | 评估脚本（AST整体替换 + 缩进修正 + conda DevEval 环境） |
| `/data2/ymz/TOSEM2026/run_oracle.py` | 参考的上下文提取模板（`build_analysis_section`） |
| `/data2/ymz/TOSEM2026/explorer/ast_analyzer.py` | AST 分析器（`ASTAnalyzer`，`FunctionInfo`，`ClassInfo`） |

## 环境

- 生成: `conda activate py38`（有 openai 包）
- 评估: `conda run --name DevEval --no-capture-output python -m pytest`（有项目依赖）
- 评估脚本中 subprocess 调用 pytest 时必须用 DevEval 环境
- 部分项目需要在 DevEval 环境中 `pip install -e .` 安装项目依赖
