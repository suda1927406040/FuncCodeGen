"""Run code completion on CoderEval dataset using definition-based generation.

Key design: extract the target function as a "definition" (signature + docstring),
ask the LLM to output the COMPLETE function (with body), then replace the old
function entirely using AST-based line ranges. This avoids all indentation issues.

Reference extraction follows run_oracle's strategy (sibling methods, call chains,
class context), adapted for CoderEval's data format.

Usage:
    conda run -n py38 python run_coder_eval.py --instance <id>
    conda run -n py38 python run_coder_eval.py --limit 10
    conda run -n py38 python run_coder_eval.py --level self_contained --limit 20
    conda run -n py38 python run_coder_eval.py --project neo4j/neo4j-python-driver
"""
import os
import re
import sys
import ast
import json
import argparse
import subprocess
import difflib
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import CODEREVAL_DATA, CODEREVAL_REPOS, OUTPUT_DIR
from explorer.ast_analyzer import ASTAnalyzer, FunctionInfo, ClassInfo
from generator.llm_interface import call_llm

# Paths to CoderEval evaluation infrastructure
CODEREVAL_BASE = Path("/data2/ymz/codereval_data/CoderEval-Docker/home/travis/builds")
CODEREVAL_JSON = CODEREVAL_BASE / "CoderEval4Python.json"
CODEREVAL_REPO_BASE = CODEREVAL_BASE / "repos"


# ============================================================================
# Data Loading
# ============================================================================

def load_coder_eval_json() -> Dict:
    """Load CoderEval4Python.json for test file mapping."""
    with open(CODEREVAL_JSON) as f:
        return json.load(f)


def load_coder_eval() -> List[Dict]:
    """Load CoderEval JSONL dataset."""
    records = []
    with open(CODEREVAL_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# Test File Resolution
# ============================================================================

def find_test_file(record_id: str, coder_eval_json: Dict) -> Optional[str]:
    """Find the test file path for a given CoderEval instance ID.

    Returns the absolute path to the _passk_validte.py or standalone test file.
    """
    # Find the record in CoderEval4Python.json
    target = None
    for r in coder_eval_json["RECORDS"]:
        if r["_id"] == record_id:
            target = r
            break
    if target is None:
        return None

    file_path = target["file_path"]
    project = target.get("project", "")
    name = target["name"]

    # Exception map from PythonExec.py
    dict_std_nonestd = {
        CODEREVAL_REPO_BASE / "standalone/neo4j-_meta-deprecated.py":
            CODEREVAL_REPO_BASE / "neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
        CODEREVAL_REPO_BASE / "standalone/neo4j-work-query-unit_of_work.py":
            CODEREVAL_REPO_BASE / "neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
        CODEREVAL_REPO_BASE / "standalone/krake-krake-controller-kubernetes-hooks-on.py":
            CODEREVAL_REPO_BASE / "rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py",
    }

    # Try standalone path first
    standalone_filename = file_path.replace(".py", "").replace("/", "-") + "-" + name + ".py"
    standalone_path = CODEREVAL_REPO_BASE / "standalone" / standalone_filename

    if standalone_path in dict_std_nonestd:
        mapped = dict_std_nonestd[standalone_path]
        if mapped.exists():
            return str(mapped)

    if standalone_path.exists():
        return str(standalone_path)

    # Try in-repo path
    repo_dir = CODEREVAL_REPO_BASE / project.replace("/", "---")
    if str(repo_dir) == str(CODEREVAL_REPO_BASE / "neo4j---neo4j-python-driver"):
        repo_file_str = str(repo_dir / "src" / file_path).replace(
            ".py", "_" + name + "_passk_validte.py")
    else:
        repo_file_str = str(repo_dir / file_path).replace(
            ".py", "_" + name + "_passk_validte.py")

    # Fix mozilla path
    if "mozilla---relman-auto-nag" in repo_file_str:
        repo_file_str = repo_file_str.replace(
            "/mozilla---relman-auto-nag/auto-ang",
            "/mozilla---relman-auto-nag/bugbot")

    repo_file = Path(repo_file_str)
    if repo_file in dict_std_nonestd:
        mapped = dict_std_nonestd[repo_file]
        if mapped.exists():
            return str(mapped)

    if repo_file.exists():
        return str(repo_file)

    return None


# ============================================================================
# Repo Path Resolution (for source file context)
# ============================================================================

def resolve_repo_and_file(project_path: str, completion_path: str) -> Tuple[Path, str]:
    """Resolve CoderEval project_path + completion_path to actual file on disk."""
    repo_name = project_path.replace("/", "---")
    repo_dir = CODEREVAL_REPOS / repo_name

    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo not found: {repo_dir}")

    direct = repo_dir / completion_path
    if direct.exists():
        return repo_dir, completion_path

    fname = Path(completion_path).name
    for root, dirs, files in os.walk(repo_dir):
        if fname in files:
            actual = Path(root) / fname
            rel = str(actual.relative_to(repo_dir))
            if completion_path.endswith(rel) or rel.endswith(completion_path):
                return repo_dir, rel
            comp_parts = Path(completion_path).parts
            rel_parts = Path(rel).parts
            if len(comp_parts) >= 2 and len(rel_parts) >= 2:
                if comp_parts[-2:] == rel_parts[-2:]:
                    return repo_dir, rel

    return repo_dir, completion_path


# ============================================================================
# Definition Extraction via AST
# ============================================================================

def extract_definition(repo_dir: Path, file_path: str,
                       func_name: str, comp_type: str) -> Optional[Dict]:
    """Extract the target function's definition (signature + docstring) and
    its full AST range for later replacement.

    Returns dict with:
      - signature: "def method_name(self, arg1, ...)" with correct indentation
      - docstring: the docstring text
      - definition_text: signature + docstring (what the LLM sees as starting point)
      - line_start, line_end: 1-indexed line range of the entire function
      - indent: the function's indentation level
      - class_name: parent class name (for methods)
    """
    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(repo_dir, file_path)
    if module is None:
        return None

    # Find the function/method
    func_info = None
    class_name = None

    if comp_type == "method":
        for cls_name, cls_info in module.classes.items():
            if func_name in cls_info.methods:
                func_info = cls_info.methods[func_name]
                class_name = cls_name
                break
            for nc_name, nc_info in cls_info.nested_classes.items():
                if func_name in nc_info.methods:
                    func_info = nc_info.methods[func_name]
                    class_name = nc_name
                    break
    else:
        if func_name in module.functions:
            func_info = module.functions[func_name]

    if func_info is None:
        return None

    # Extract signature from source
    src_lines = module.source.splitlines()
    sig_start = func_info.line_start  # 1-indexed
    sig_end = func_info.line_start
    # Find where signature ends (look for colon at end of line)
    for i in range(sig_start - 1, min(sig_start + 10, len(src_lines))):
        if src_lines[i].rstrip().endswith(":"):
            sig_end = i + 1
            break

    # Build signature text
    sig_lines = src_lines[sig_start - 1:sig_end]
    signature_text = "\n".join(sig_lines)

    # Get docstring
    docstring = func_info.docstring or ""

    # Build full definition = signature + docstring
    if docstring:
        # Find docstring in source after signature
        docstring_lines = []
        in_docstring = False
        for i in range(sig_end, min(sig_end + 20, len(src_lines))):
            line = src_lines[i]
            docstring_lines.append(line)
            if '"""' in line or "'''" in line:
                if in_docstring:
                    break
                in_docstring = True
                # Check if closing quotes on same line
                if line.count('"""') >= 2 or line.count("'''") >= 2:
                    break
        definition_text = signature_text + "\n" + "\n".join(docstring_lines)
    else:
        definition_text = signature_text

    # Calculate indent
    first_line = src_lines[sig_start - 1]
    indent = len(first_line) - len(first_line.lstrip())

    return {
        "signature": signature_text,
        "docstring": docstring,
        "definition_text": definition_text,
        "line_start": func_info.line_start,    # 1-indexed
        "line_end": func_info.line_end,          # 1-indexed
        "indent": indent,
        "class_name": class_name,
        "func_info": func_info,
    }


# ============================================================================
# AST Analysis — oracle-style context extraction (from run_deveval / run_oracle)
# ============================================================================

def analyze_file(repo_dir: Path, file_path: str,
                 func_name: str, comp_type: str) -> Optional[Dict]:
    """Parse the target file and extract full reference context.

    Follows run_oracle's ContextBuilder strategy: class context, __init__ vars,
    sibling methods, call graph, reference call chain.
    """
    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(repo_dir, file_path)
    if module is None:
        return None

    result = {
        "file_source": module.source,
        "imports": [],
        "functions": dict(module.functions),
        "classes": dict(module.classes),
        "class_context": None,
        "sibling_methods": {},
        "calling_patterns": {},
        "reference_call_chain": [],
    }

    # Format imports
    for imp in module.imports:
        if imp["type"] == "import":
            result["imports"].append(f"import {imp['module']}")
        elif imp["type"] == "from":
            names = ", ".join(n["name"] if isinstance(n, dict) else n
                              for n in imp.get("names", []))
            mod_name = imp.get("module", "")
            level = "." * imp.get("level", 0)
            result["imports"].append(f"from {level}{mod_name} import {names}")

    if comp_type != "method":
        return result

    # Find the class containing this method
    target_cls = None
    for cls_name, cls_info in module.classes.items():
        if func_name in cls_info.methods:
            target_cls = cls_info
            result["class_context"] = {
                "class_name": cls_name,
                "bases": cls_info.bases,
            }
            break
        for nc_name, nc_info in cls_info.nested_classes.items():
            if func_name in nc_info.methods:
                target_cls = nc_info
                result["class_context"] = {
                    "class_name": nc_name,
                    "bases": nc_info.bases,
                }
                break

    if target_cls is None:
        return result

    # Extract __init__ vars
    init_vars = {}
    if "__init__" in target_cls.methods:
        init_src = target_cls.methods["__init__"].body_source
        for m in re.finditer(r'self\.([a-zA-Z_]\w*)\s*=\s*(.+)', init_src):
            init_vars[m.group(1)] = m.group(2).strip()
    result["class_context"]["init_vars"] = init_vars
    result["class_context"]["class_vars"] = dict(target_cls.class_vars)

    # Sibling methods (excluding target)
    for mn, mi in target_cls.methods.items():
        if mn != func_name:
            result["sibling_methods"][mn] = mi

    # Calling patterns
    for mn, mi in result["sibling_methods"].items():
        callees = _find_method_callees(mi)
        if callees:
            result["calling_patterns"][mn] = callees

    # Reference call chain
    result["reference_call_chain"] = _build_reference_call_chain(
        func_name, target_cls
    )

    return result


def _find_method_callees(method_info) -> List[str]:
    """Extract self.xxx() calls from a method body."""
    body = method_info.body_source
    callees = re.findall(r'self\.([a-zA-Z_]\w*)\s*\(', body)
    return list(dict.fromkeys(callees))


def _build_reference_call_chain(func_name: str, cls_info) -> List[Dict]:
    """Find the most similar sibling method by name overlap, then BFS trace
    its call chain (up to 5 levels).
    """
    func_words = set(re.findall(r'[a-z]+', func_name))

    similar = []
    for mn, mi in cls_info.methods.items():
        if mn == func_name:
            continue
        m_words = set(re.findall(r'[a-z]+', mn))
        overlap = len(func_words & m_words)
        if overlap >= 1 and len(func_words) > 0:
            similarity = overlap / max(len(func_words), len(m_words))
            if similarity >= 0.2:
                similar.append((mn, mi, similarity))

    if not similar:
        return []

    similar.sort(key=lambda x: x[2], reverse=True)

    chain = []
    visited = set()
    queue = [similar[0][0]]

    while queue and len(chain) < 5:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in cls_info.methods:
            mi = cls_info.methods[current]
            callees = _find_method_callees(mi)
            chain.append({
                "method_name": current,
                "body_source": mi.body_source,
                "callees": callees,
                "docstring": mi.docstring,
            })
            for c in callees:
                if c not in visited:
                    queue.append(c)

    return chain


# ============================================================================
# Prompt Building — definition-style + oracle analysis (adapted from run_deveval)
# ============================================================================

def build_analysis_section(record: Dict, analysis: Dict) -> str:
    """Build analysis section following oracle/DevEval pattern."""
    parts = []
    func_name = record["signature"].replace("def ", "").split("(")[0].split(".")[0]

    # Determine comp_type from analysis
    cc = analysis.get("class_context") if analysis else None
    is_method = cc is not None

    # Class context
    if cc and is_method:
        parts.append(f"**This is a method of class `{cc['class_name']}`**.")
        if cc.get("bases"):
            parts.append(f"  Inherits from: {', '.join(cc['bases'])}")
        parts.append("")

        if cc.get("class_vars"):
            parts.append("**Class-level variables** (you may need to reference these):")
            for var_name, var_val in cc["class_vars"].items():
                parts.append(f"  {var_name}: {var_val}")
            parts.append("")

        if cc.get("init_vars"):
            parts.append("**Instance variables set in __init__** (your method may use these):")
            for var, val in cc["init_vars"].items():
                parts.append(f"  self.{var} = {val}")
            parts.append("")

        siblings = analysis.get("sibling_methods", {})
        if "__init__" in siblings:
            init_body = siblings["__init__"].body_source
            if len(init_body) > 4000:
                init_body = init_body[:4000] + "\n    # ... (truncated)"
            parts.append("**__init__ method source** (follow its initialization pattern):")
            parts.append(init_body)
            parts.append("")

    # Sibling methods
    siblings = analysis.get("sibling_methods", {}) if analysis else {}
    if siblings:
        parts.append(f"**Existing methods in the same class** ({len(siblings)} total):")
        for mn, mi in siblings.items():
            if mn == "__init__":
                continue
            parts.append(f"\nMethod `{mn}`:")
            if mi.docstring:
                parts.append(f'  Docstring: """{mi.docstring}"""')
            body = mi.body_source
            if len(body) > 3000:
                body = body[:3000] + "\n    # ... (truncated)"
            parts.append(body)
        parts.append("")

    # Call graph
    calling_patterns = analysis.get("calling_patterns", {}) if analysis else {}
    if calling_patterns:
        parts.append("**Method call graph** (which self.xxx() each method calls):")
        for caller, callees in calling_patterns.items():
            parts.append(f"  {caller} → calls: {', '.join(callees)}")
        parts.append("")

    # Reference call chain
    ref_chain = analysis.get("reference_call_chain", []) if analysis else []
    if ref_chain:
        parts.append("**REFERENCE IMPLEMENTATION** — follow this pattern closely:")
        parts.append("The most similar existing method and its full call chain are shown below.")
        parts.append(f"Your new method `{func_name}` should follow the SAME structural pattern.\n")
        for node in ref_chain:
            parts.append(f"Method `{node['method_name']}()`:")
            if node["callees"]:
                parts.append(f"  (internally calls: {', '.join(node['callees'])})")
            body = node["body_source"]
            if len(body) > 3000:
                body = body[:3000] + "\n    # ... (truncated)"
            parts.append(body)
            parts.append("")

    return "\n".join(parts)


def build_prompt(record: Dict, analysis: Optional[Dict],
                 definition: Optional[Dict]) -> str:
    """Build definition-style prompt (adapted from run_deveval).

    Instead of "complete the last line", we present the function definition
    and ask the LLM to output the COMPLETE function with body.
    """
    func_name = record["signature"].replace("def ", "").split("(")[0].split(".")[0]
    completion_path = record["completion_path"]

    # Build definition section
    def_section = ""
    if definition:
        if definition.get("class_name"):
            def_section = f"<definitions>\n[start of new definitions in {completion_path}]\n"
            def_section += f"(definition of {definition['class_name']}.{func_name}:)\n"
            def_section += definition["definition_text"] + "\n"
            def_section += f"[end of new definitions in {completion_path}]\n</definitions>\n"
        else:
            def_section = f"<definitions>\n[start of new definitions in {completion_path}]\n"
            def_section += f"(definition of {func_name}:)\n"
            def_section += definition["definition_text"] + "\n"
            def_section += f"[end of new definitions in {completion_path}]\n</definitions>\n"
    else:
        # Fallback: use record['input'] as the definition
        def_section = f"<definitions>\n[start of new definitions in {completion_path}]\n"
        def_section += f"(definition of {func_name}:)\n"
        def_section += record['input'] + "\n"
        def_section += f"[end of new definitions in {completion_path}]\n</definitions>\n"

    # Build analysis section
    analysis_section = ""
    if analysis:
        analysis_section = build_analysis_section(record, analysis)

    # Imports section
    imports_section = ""
    if analysis and analysis.get("imports"):
        imports_section = "**Imports in this file:**\n```python\n"
        for imp in analysis["imports"][:30]:
            imports_section += imp + "\n"
        imports_section += "```\n"

    # File context (surrounding code for reference)
    file_context = ""
    if analysis and analysis.get("file_source"):
        src = analysis["file_source"]
        if len(src) > 8000:
            src = src[:8000] + "\n# ... (truncated)"
        file_context = f"<code>\n[start of {completion_path}]\n{src}\n[end of {completion_path}]\n</code>"

    full_prompt = f"""You are a Python expert. Complete the function definition by implementing its body.

## Requirement
**Functionality**: {record['human_label']}

{def_section}
{analysis_section}
{imports_section}
{file_context}

## Instructions
1. Implement the function body based on the requirement, definitions, and analysis above.
2. Follow the structural pattern shown in the "REFERENCE IMPLEMENTATION" section if present.
3. Output the COMPLETE function definition including `def` line, docstring, and body.
4. Preserve the original indentation level.
5. Do not add any explanations or markdown fences — output raw Python code only.

Output the complete function:"""

    return full_prompt


# ============================================================================
# Response Parsing (definition-style)
# ============================================================================

def extract_function_from_response(response: str, definition: Optional[Dict]) -> str:
    """Extract the complete function from LLM response.

    The LLM should output a complete function starting with `def`.
    We extract it, ensuring correct indentation.
    """
    # Try code block first
    code = response.strip()
    code_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        code = max(code_blocks, key=len).strip()
    else:
        code_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            code = max(code_blocks, key=len).strip()

    # Find the def line
    lines = code.split('\n')
    def_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_start = i
            break

    if def_start is not None:
        code = '\n'.join(lines[def_start:])
    elif definition:
        # LLM only output the body (no def line) — prepend definition header
        sig = definition["definition_text"]
        body = textwrap.dedent(code)
        body = textwrap.indent(body, ' ' * (definition["indent"] + 4))
        code = sig + '\n' + body

    return code


# ============================================================================
# Execution-based Evaluation
# ============================================================================

def evaluate_by_execution(generated_code: str, test_file_path: str,
                          target_func_name: str) -> Dict:
    """Evaluate generated code by inserting it into the test file and running.

    Uses AST to find the target function and replace it entirely,
    with indent correction (same strategy as eval_deveval_def.py).
    """
    if not os.path.exists(test_file_path):
        return {"passed": False, "error": "Test file not found"}

    # Read the test file
    with open(test_file_path) as f:
        test_content = f.read()

    # Parse AST to find the target function and its location
    try:
        tree = ast.parse(test_content)
    except SyntaxError as e:
        return {"passed": False, "error": f"Test file syntax error: {e}"}

    file_lines = test_content.split("\n")
    target_node = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == target_func_name or node.name == "_" + target_func_name:
                target_node = node
                break

    if target_node is None:
        return {"passed": False, "error": f"Target function '{target_func_name}' not found in test file"}

    start_line = target_node.lineno
    end_line = target_node.end_lineno

    # Get the indentation of the target function
    first_line = file_lines[start_line - 1]
    start_indent = len(first_line) - len(first_line.lstrip())

    # Normalize generated code indentation to match target
    code_lines = generated_code.strip().split("\n")
    code_first = code_lines[0].lstrip()
    code_indent = len(code_lines[0]) - len(code_first)

    new_code_lines = []
    if code_indent < start_indent:
        pad = " " * (start_indent - code_indent)
        for cl in code_lines:
            if cl.strip():
                new_code_lines.append(pad + cl)
            else:
                new_code_lines.append(cl)
    else:
        new_code_lines = code_lines

    new_code = "\n".join(new_code_lines)

    # Build the patched file: lines before target + new code + lines after target
    patched_lines = file_lines[:start_line - 1] + [new_code] + file_lines[end_line:]
    patched_content = "\n".join(patched_lines)

    # Write to a temp file and execute
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     dir=os.path.dirname(test_file_path)) as tmp:
        tmp.write(patched_content)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(test_file_path),
        )
        passed = proc.returncode == 0
        return {
            "passed": passed,
            "return_code": proc.returncode,
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout (30s)"}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_single(record: Dict, output_base: Path,
               coder_eval_json: Dict) -> Dict:
    """Run code completion for one CoderEval instance."""
    rid = record["id"]
    out_dir = output_base / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "id": rid,
        "project": record["project_path"],
        "level": record["level"],
        "signature": record["signature"],
        "success": False,
        "passed": False,
        "similarity": 0.0,
        "exact_match": False,
        "timing": {},
    }

    import time
    t_start = time.time()

    # Step 1: Find test file for evaluation
    test_file = find_test_file(rid, coder_eval_json)
    result["test_file"] = test_file

    # Step 2: Resolve repo and file path for context
    try:
        repo_dir, actual_path = resolve_repo_and_file(
            record["project_path"], record["completion_path"]
        )
    except FileNotFoundError as e:
        result["error"] = str(e)
        return result

    full_path = repo_dir / actual_path
    if not full_path.exists():
        result["error"] = f"File not found: {actual_path}"
        return result

    file_content = full_path.read_text(errors="replace")
    func_name = record["signature"].replace("def ", "").split("(")[0].split(".")[0]

    # Step 3: Determine comp_type (method vs function)
    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(repo_dir, actual_path)
    comp_type = "function"
    if module:
        for cls_name, cls_info in module.classes.items():
            if func_name in cls_info.methods:
                comp_type = "method"
                break
            for nc_name, nc_info in cls_info.nested_classes.items():
                if func_name in nc_info.methods:
                    comp_type = "method"
                    break

    # Step 4: Extract definition via AST
    definition = extract_definition(repo_dir, actual_path, func_name, comp_type)

    # Step 5: Analyze context (oracle-style)
    analysis = analyze_file(repo_dir, actual_path, func_name, comp_type)

    # Step 6: Build prompt
    prompt = build_prompt(record, analysis, definition)
    (out_dir / "prompt.md").write_text(prompt)
    result["timing"]["analysis"] = time.time() - t_start
    result["prompt_chars"] = len(prompt)
    result["comp_type"] = comp_type
    if definition:
        result["func_range"] = (definition["line_start"], definition["line_end"])
        result["indent"] = definition["indent"]

    # Step 7: Call LLM
    t0 = time.time()
    response = call_llm(
        prompt,
        system="You are an expert Python developer. Output complete, correct Python code.",
        model="deepseek-chat",
        max_tokens=4096,
        temperature=0.0,
    )
    result["timing"]["llm_call"] = time.time() - t0
    (out_dir / "response.md").write_text(response)

    # Step 8: Extract generated code
    generated = extract_function_from_response(response, definition)
    gold = record["original_code"]
    (out_dir / "generated.py").write_text(generated)
    (out_dir / "gold.py").write_text(gold)

    # Step 9: Evaluate by execution
    eval_result = {"passed": False, "error": "no test file"}
    if test_file:
        eval_result = evaluate_by_execution(generated, test_file, func_name)

    result["passed"] = eval_result.get("passed", False)
    result["eval_detail"] = eval_result

    # Also compute similarity for reference
    similarity = difflib.SequenceMatcher(None, generated.strip(), gold.strip()).ratio()
    result["similarity"] = round(similarity, 4)
    result["exact_match"] = similarity > 0.95
    result["success"] = True
    result["timing"]["total"] = time.time() - t_start

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    status = "PASS" if result["passed"] else ("EXACT" if result["exact_match"] else f"{similarity:.0%}")
    print(f"  [{status}] {rid[:12]}... {func_name} ({record['level']}, {comp_type})")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Single instance ID")
    parser.add_argument("--limit", type=int, help="Run first N instances")
    parser.add_argument("--level", help="Filter by dependency level")
    parser.add_argument("--project", help="Filter by project path")
    parser.add_argument("--output", default=str(OUTPUT_DIR.parent / "output_coder_eval"),
                        help="Output base dir")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "sk-95fe62163fc44c258255f82e47837b5c")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("Loading CoderEval dataset...")
    records = load_coder_eval()
    coder_eval_json = load_coder_eval_json()
    print(f"Loaded {len(records)} instances")

    if args.instance:
        records = [r for r in records if r["id"] == args.instance]
    if args.level:
        records = [r for r in records if r["level"] == args.level]
    if args.project:
        records = [r for r in records if r["project_path"] == args.project]
    if args.limit:
        records = records[:args.limit]

    print(f"Running {len(records)} instances")

    results = []
    for idx, rec in enumerate(records):
        print(f"\n[{idx+1}/{len(records)}] {rec['id'][:12]}... "
              f"({rec['project_path']}, {rec['level']})")
        try:
            r = run_single(rec, output_base, coder_eval_json)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"id": rec["id"], "success": False, "error": str(e)})

    # Summary
    ok = sum(1 for r in results if r.get("success"))
    passed = sum(1 for r in results if r.get("passed"))
    exact = sum(1 for r in results if r.get("exact_match"))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok}/{len(results)} completed, {passed} passed (execution), {exact} exact match")
    pass_rate = passed / len(results) * 100 if results else 0
    print(f"Pass rate: {pass_rate:.1f}%")

    # Per-level breakdown
    levels = {}
    for r in results:
        lvl = r.get("level", "unknown")
        if lvl not in levels:
            levels[lvl] = {"count": 0, "passed": 0, "exact": 0, "sim_sum": 0, "success": 0}
        levels[lvl]["count"] += 1
        if r.get("success"):
            levels[lvl]["success"] += 1
            levels[lvl]["sim_sum"] += r.get("similarity", 0)
        if r.get("passed"):
            levels[lvl]["passed"] += 1
        if r.get("exact_match"):
            levels[lvl]["exact"] += 1

    print("\nPer-level breakdown:")
    for lvl, stats in sorted(levels.items()):
        pr = stats["passed"] / stats["count"] * 100 if stats["count"] else 0
        print(f"  {lvl:25s}: {stats['passed']:2d}/{stats['count']:2d} passed ({pr:.0f}%), "
              f"{stats['exact']} exact")

    (output_base / "summary.json").write_text(json.dumps({
        "total": len(results),
        "successful": ok,
        "passed_execution": passed,
        "pass_rate": round(pass_rate, 2),
        "exact_matches": exact,
        "per_level": levels,
        "results": results,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
