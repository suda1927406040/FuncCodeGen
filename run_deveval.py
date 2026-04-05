"""Run code completion on DevEval dataset with definition-based generation.

Key design: extract the target function as a "definition" (signature + docstring),
ask the LLM to output the COMPLETE function (with body), then replace the old
function entirely using AST-based line ranges. This avoids all indentation issues.

Reference extraction follows run_oracle's strategy (sibling methods, call chains,
class context), plus DevEval-specific API dependency resolution.

Usage:
    conda run -n py38 python run_deveval.py --limit 10
    conda run -n py38 python run_deveval.py --project Internet/boto
    conda run -n py38 python run_deveval.py --instance <namespace>
"""
import os
import re
import sys
import json
import argparse
import difflib
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR
from explorer.ast_analyzer import ASTAnalyzer, FunctionInfo, ClassInfo
from generator.llm_interface import call_llm

DEVEVAL_DATA = Path("/data2/ymz/baseline/DevEval/data.jsonl")
DEVEVAL_REPOS = Path("/data2/ymz/baseline/DevEval/Source_Code")


# ============================================================================
# Data Loading
# ============================================================================

def load_deveval() -> List[Dict]:
    records = []
    with open(DEVEVAL_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# Definition Extraction via AST
# ============================================================================

def extract_definition(comp_record: Dict) -> Optional[Dict]:
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
    comp_path = comp_record["completion_path"]
    func_name = comp_record["namespace"].split(".")[-1]
    comp_type = comp_record["type"]

    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(DEVEVAL_REPOS, comp_path)
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
    else:
        if func_name in module.functions:
            func_info = module.functions[func_name]

    if func_info is None:
        return None

    # Extract signature from source
    src_lines = module.source.splitlines()
    # Get lines from def to end of signature (may span multiple lines for decorators)
    sig_start = func_info.line_start  # 1-indexed
    sig_end = func_info.line_start    # will extend if multi-line signature
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


def extract_function_range(comp_path: str, func_name: str,
                           comp_type: str) -> Optional[Tuple[int, int]]:
    """Get the (start, end) line range of a function for replacement.

    Returns 1-indexed (line_start, line_end).
    """
    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(DEVEVAL_REPOS, comp_path)
    if module is None:
        return None

    if comp_type == "method":
        for cls_name, cls_info in module.classes.items():
            if func_name in cls_info.methods:
                mi = cls_info.methods[func_name]
                return (mi.line_start, mi.line_end)
    else:
        if func_name in module.functions:
            mi = module.functions[func_name]
            return (mi.line_start, mi.line_end)

    return None


# ============================================================================
# AST Analysis — same strategy as run_oracle's ContextBuilder
# ============================================================================

def analyze_file(comp_record: Dict) -> Optional[Dict]:
    """Parse the target file and extract full reference context."""
    comp_path = comp_record["completion_path"]
    func_name = comp_record["namespace"].split(".")[-1]
    comp_type = comp_record["type"]

    repo_file = DEVEVAL_REPOS / comp_path
    if not repo_file.exists():
        return None

    analyzer = ASTAnalyzer(max_file_chars=100000)
    module = analyzer.parse_file(DEVEVAL_REPOS, comp_path)
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
    body = method_info.body_source
    callees = re.findall(r'self\.([a-zA-Z_]\w*)\s*\(', body)
    return list(dict.fromkeys(callees))


def _build_reference_call_chain(func_name: str, cls_info) -> List[Dict]:
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
# Dependency API Resolution
# ============================================================================

def resolve_api_source(namespace: str, comp_record: Dict,
                       analyzer: ASTAnalyzer) -> Optional[str]:
    parts = namespace.split(".")
    target_name = parts[-1]
    project_path = comp_record["project_path"]
    comp_path = comp_record["completion_path"]

    # 1. Same file
    module = analyzer.parse_file(DEVEVAL_REPOS, comp_path)
    if module:
        if target_name in module.functions:
            return module.functions[target_name].body_source
        for cls_name, cls_info in module.classes.items():
            if target_name in cls_info.methods:
                return cls_info.methods[target_name].body_source
            if target_name in cls_info.class_vars:
                return f"# {target_name} = {cls_info.class_vars[target_name]}"

    # 2. Cross-file
    if len(parts) >= 2:
        for i in range(len(parts) - 1, 0, -1):
            module_path = "/".join(parts[:i]) + ".py"
            candidate_rel = f"{project_path}/{module_path}"
            candidate = DEVEVAL_REPOS / candidate_rel
            if candidate.exists():
                mod = analyzer.parse_file(DEVEVAL_REPOS, candidate_rel)
                if mod:
                    if parts[-1] in mod.functions:
                        return mod.functions[parts[-1]].body_source
                    for cn, ci in mod.classes.items():
                        if parts[-1] in ci.methods:
                            return ci.methods[parts[-1]].body_source

    # 3. Search under project subdirs
    if len(parts) >= 3:
        for i in range(len(parts) - 1, 0, -1):
            module_path = "/".join(parts[:i]) + ".py"
            project_dir = DEVEVAL_REPOS / project_path
            if project_dir.exists():
                for subdir in project_dir.iterdir():
                    if subdir.is_dir():
                        candidate_rel = f"{project_path}/{subdir.name}/{module_path}"
                        candidate = DEVEVAL_REPOS / candidate_rel
                        if candidate.exists():
                            mod = analyzer.parse_file(DEVEVAL_REPOS, candidate_rel)
                            if mod:
                                if parts[-1] in mod.functions:
                                    return mod.functions[parts[-1]].body_source
                                for cn, ci in mod.classes.items():
                                    if parts[-1] in ci.methods:
                                        return ci.methods[parts[-1]].body_source
    return None


def resolve_all_dependencies(comp_record: Dict,
                             analyzer: ASTAnalyzer) -> Dict[str, str]:
    dep = comp_record.get("dependency", {})
    resolved = {}
    for dep_type in ["intra_class", "intra_file", "cross_file"]:
        for ns in dep.get(dep_type, []):
            src = resolve_api_source(ns, comp_record, analyzer)
            if src:
                resolved[ns] = src
    return resolved


# ============================================================================
# Prompt Building — definition-style + oracle analysis + API deps
# ============================================================================

def build_analysis_section(comp_record: Dict, analysis: Dict,
                           resolved_deps: Dict[str, str]) -> str:
    """Build analysis section (same as before)."""
    parts = []
    func_name = comp_record["namespace"].split(".")[-1]
    comp_type = comp_record["type"]

    # Class context
    cc = analysis.get("class_context")
    if cc and comp_type == "method":
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
    siblings = analysis.get("sibling_methods", {})
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
    calling_patterns = analysis.get("calling_patterns", {})
    if calling_patterns:
        parts.append("**Method call graph** (which self.xxx() each method calls):")
        for caller, callees in calling_patterns.items():
            parts.append(f"  {caller} → calls: {', '.join(callees)}")
        parts.append("")

    # Reference call chain
    ref_chain = analysis.get("reference_call_chain", [])
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

    # Required API calls
    dep = comp_record.get("dependency", {})
    has_deps = any(dep.values())
    if has_deps:
        parts.append("=" * 40)
        parts.append("**REQUIRED API CALLS** — your implementation MUST call these APIs:")
        parts.append("=" * 40)
        for dep_type in ["intra_class", "intra_file", "cross_file"]:
            apis = dep.get(dep_type, [])
            if apis:
                parts.append(f"\n  [{dep_type}]")
                for api_ns in apis:
                    if api_ns in resolved_deps:
                        parts.append(f"\n  API `{api_ns}`:")
                        parts.append(f"  ```python")
                        src = resolved_deps[api_ns]
                        if len(src) > 2000:
                            src = src[:2000] + "\n  # ... (truncated)"
                        parts.append(f"  {src}")
                        parts.append(f"  ```")
                    else:
                        parts.append(f"\n  API `{api_ns}`: (source could not be resolved)")
        parts.append("")

    return "\n".join(parts)


def build_deveval_prompt(comp_record: Dict, analysis: Optional[Dict],
                         resolved_deps: Dict[str, str],
                         definition: Optional[Dict]) -> str:
    """Build definition-style prompt.

    Instead of "complete the last line", we present the function definition
    and ask the LLM to output the COMPLETE function with body.
    """
    requirement = comp_record.get("requirement", {})
    functionality = requirement.get("Functionality", "")
    arguments = requirement.get("Arguments", "")

    # Build definition section
    def_section = ""
    if definition:
        comp_path = comp_record["completion_path"]
        func_name = comp_record["namespace"].split(".")[-1]
        if definition.get("class_name"):
            def_section = f"<definitions>\n[start of new definitions in {comp_path}]\n"
            def_section += f"(definition of {definition['class_name']}.{func_name}:)\n"
            def_section += definition["definition_text"] + "\n"
            def_section += f"[end of new definitions in {comp_path}]\n</definitions>\n"
        else:
            def_section = f"<definitions>\n[start of new definitions in {comp_path}]\n"
            def_section += f"(definition of {func_name}:)\n"
            def_section += definition["definition_text"] + "\n"
            def_section += f"[end of new definitions in {comp_path}]\n</definitions>\n"

    # Build analysis section
    analysis_section = ""
    if analysis:
        analysis_section = build_analysis_section(comp_record, analysis, resolved_deps)

    # Imports section
    imports_section = ""
    if analysis and analysis.get("imports"):
        imports_section = "**Imports in this file:**\n```python\n"
        for imp in analysis["imports"][:30]:
            imports_section += imp + "\n"
        imports_section += "```\n"

    # Dependency summary
    dep = comp_record.get("dependency", {})
    dep_info = ""
    if any(dep.values()):
        dep_info = "**Dependency summary:**\n"
        for dt, items in dep.items():
            if items:
                dep_info += f"  {dt}: {items}\n"
        dep_info += "\n"

    # File context (surrounding code for reference)
    file_context = ""
    if analysis and analysis.get("file_source"):
        src = analysis["file_source"]
        # Truncate to avoid huge prompts, but show enough context
        if len(src) > 8000:
            src = src[:8000] + "\n# ... (truncated)"
        file_context = f"<code>\n[start of {comp_record['completion_path']}]\n{src}\n[end of {comp_record['completion_path']}]\n</code>"

    full_prompt = f"""You are a Python expert. Complete the function definition by implementing its body.

## Requirement
**Functionality**: {functionality}
{f'**Arguments**: {arguments}' if arguments else ''}

{dep_info}
{def_section}
{analysis_section}
{imports_section}
{file_context}

## Instructions
1. Implement the function body based on the requirement, definitions, and analysis above.
2. **CRITICAL**: The function MUST call the APIs listed in "REQUIRED API CALLS" if present.
3. Follow the structural pattern shown in the "REFERENCE IMPLEMENTATION" section.
4. Output the COMPLETE function definition including `def` line, docstring, and body.
5. Preserve the original indentation level.
6. Do not add any explanations or markdown fences — output raw Python code only.

Output the complete function:"""

    return full_prompt


# ============================================================================
# Response Parsing
# ============================================================================

def extract_complete_function(response: str, definition: Optional[Dict]) -> str:
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
# Main Pipeline
# ============================================================================

def run_single(comp_record: Dict, output_base: Path) -> Dict:
    ns = comp_record["namespace"]
    safe_name = ns.replace(".", "_").replace("/", "_")
    out_dir = output_base / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "namespace": ns,
        "project": comp_record["project_path"],
        "type": comp_record["type"],
        "success": False,
        "completion": "",
        "timing": {},
    }

    import time
    t_start = time.time()

    # Step 1: Extract definition
    definition = extract_definition(comp_record)
    if definition is None:
        result["error"] = "Could not extract definition"
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    result["func_range"] = (definition["line_start"], definition["line_end"])
    result["indent"] = definition["indent"]

    # Step 2: Analyze context (oracle-style)
    analysis = analyze_file(comp_record)

    # Step 3: Resolve API dependencies
    analyzer = ASTAnalyzer(max_file_chars=100000)
    resolved_deps = resolve_all_dependencies(comp_record, analyzer)

    dep = comp_record.get("dependency", {})
    total_deps = sum(len(v) for v in dep.values())
    result["resolved_deps"] = len(resolved_deps)
    result["total_deps"] = total_deps

    # Step 4: Build prompt
    prompt = build_deveval_prompt(comp_record, analysis, resolved_deps, definition)
    (out_dir / "prompt.md").write_text(prompt)
    result["timing"]["analysis"] = time.time() - t_start
    result["prompt_chars"] = len(prompt)

    # Step 5: Call LLM
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

    # Step 6: Extract complete function
    generated = extract_complete_function(response, definition)
    result["completion"] = generated
    result["success"] = True
    result["timing"]["total"] = time.time() - t_start

    (out_dir / "generated.py").write_text(generated)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    dep_info = f"deps={result['resolved_deps']}/{total_deps}" if total_deps else ""
    print(f"  {ns[:50]:50s} {dep_info} prompt={len(prompt)}")

    return result


def collect_output(results: List[Dict], output_path: Path):
    """Collect all completions into a single JSONL for evaluation."""
    with open(output_path, 'w') as f:
        for r in results:
            if r.get("success") and r.get("completion"):
                f.write(json.dumps({
                    "namespace": r["namespace"],
                    "completion": r["completion"],
                    "func_range": r.get("func_range"),
                    "indent": r.get("indent"),
                }) + '\n')


def _worker(args):
    idx, rec, output_base, total = args
    prefix = f"[{idx+1}/{total}]"
    try:
        r = run_single(rec, output_base)
        print(f" {prefix} OK {rec['namespace'][:50]}")
        return r
    except Exception as e:
        print(f" {prefix} ERROR {rec['namespace'][:50]}: {e}")
        return {"namespace": rec["namespace"], "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Filter by namespace substring")
    parser.add_argument("--limit", type=int, help="Run first N instances")
    parser.add_argument("--project", help="Filter by project_path")
    parser.add_argument("--type", help="Filter by type (function/method)")
    parser.add_argument("--output", default=str(OUTPUT_DIR.parent / "output_deveval"),
                        help="Output base dir")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "sk-95fe62163fc44c258255f82e47837b5c")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("Loading DevEval dataset...")
    records = load_deveval()
    print(f"Loaded {len(records)} instances")

    if args.instance:
        records = [r for r in records if args.instance in r["namespace"]]
    if args.project:
        records = [r for r in records if args.project in r["project_path"]]
    if args.type:
        records = [r for r in records if r["type"] == args.type]
    if args.limit:
        records = records[:args.limit]

    def _safe(ns): return ns.replace(".", "_").replace("/", "_")
    remaining = []
    for r in records:
        safe = _safe(r["namespace"])
        if not (output_base / safe / "result.json").exists():
            remaining.append(r)
    skipped = len(records) - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already completed, {len(remaining)} remaining")

    print(f"Running {len(remaining)} instances with {args.workers} workers")

    total = len(remaining)
    worker_args = [(i, rec, output_base, total) for i, rec in enumerate(remaining)]

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    results = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, worker_args):
            results.append(r)

    # Load previously completed results
    prev_results = []
    for d in output_base.iterdir():
        rf = d / "result.json"
        if d.is_dir() and rf.exists():
            try:
                prev_results.append(json.loads(rf.read_text()))
            except Exception:
                pass

    all_results = prev_results + results

    # Collect output
    output_jsonl = output_base / "deveval_output.jsonl"
    collect_output(all_results, output_jsonl)
    print(f"\nCollected {sum(1 for r in all_results if r.get('completion'))} completions to {output_jsonl}")

    # Summary
    ok = sum(1 for r in all_results if r.get("success"))
    dep_resolved = sum(r.get("resolved_deps", 0) for r in all_results)
    dep_total = sum(r.get("total_deps", 0) for r in all_results)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok}/{len(all_results)} successful")
    if dep_total > 0:
        print(f"Dependency resolution: {dep_resolved}/{dep_total} "
              f"({dep_resolved/dep_total*100:.0f}%)")

    (output_base / "summary.json").write_text(json.dumps({
        "total": len(all_results),
        "successful": ok,
        "dep_resolution": {"resolved": dep_resolved, "total": dep_total},
        "results": all_results,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
