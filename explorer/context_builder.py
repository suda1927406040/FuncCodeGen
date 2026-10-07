"""Context builder - assemble structured context from AST analysis and import tracing.

This is the main orchestrator for the deterministic exploration phase.
It combines AST analysis, import tracing, and pattern matching to build
a complete context dictionary for the LLM generation phase.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field

from explorer.ast_analyzer import ASTAnalyzer, ModuleInfo, ClassInfo, FunctionInfo
from explorer.import_tracer import ImportTracer, ImportNode
from repo_manager import get_file_content
from config import MAX_CONTEXT_CHARS, MAX_SIBLING_METHODS
from explorer.feature_retriever import FeatureRetriever, RecallCandidate, identifier_terms


@dataclass
class InitPyInfo:
    """Structured info about an __init__.py file."""
    file: str
    has_all: bool = False
    all_names: List[str] = field(default_factory=list)
    imports_our_module: bool = False
    source_preview: str = ""
    # What pattern do sibling modules follow?
    sibling_export_pattern: str = ""  # e.g. "from .text_util import *" or explicit list


@dataclass
class CallChainNode:
    """A method in a call chain reference."""
    method_name: str
    class_name: str
    file: str
    body_source: str
    callees: List[str] = field(default_factory=list)


@dataclass
class ComponentContext:
    """Structured context for a single new component to generate."""
    name: str
    signature: str
    comp_type: str                         # "class" or "function"
    target_file: str
    doc: str = ""

    # Resolved target info
    parent_class: Optional[str] = None
    parent_class_file: Optional[str] = None
    nested_class: Optional[str] = None

    # Gathered context
    sibling_methods: Dict[str, FunctionInfo] = field(default_factory=dict)
    class_init_vars: Dict[str, str] = field(default_factory=dict)
    class_vars: Dict[str, str] = field(default_factory=dict)
    base_classes: Dict[str, Tuple[str, ModuleInfo]] = field(default_factory=dict)
    similar_functions: List[Dict[str, Any]] = field(default_factory=list)
    calling_patterns: Dict[str, List[str]] = field(default_factory=dict)
    init_py_exports: Optional[InitPyInfo] = None
    module_imports: List[Dict[str, Any]] = field(default_factory=list)

    # Call chain: the most similar sibling's full call graph
    reference_call_chain: List[CallChainNode] = field(default_factory=list)

    # FeatureSpace: files discovered via anchor's git commit history
    feature_space_files: List[Any] = field(default_factory=list)

    # Inter-component deps
    depends_on_components: List[str] = field(default_factory=list)  # names of other new components

    # Source snippets
    target_file_source: str = ""
    relevant_snippets: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class InstanceContext:
    """Complete context for an entire instance."""
    instance_id: str
    repo_dir: Path
    problem_statement: str
    request: str
    components: List[ComponentContext] = field(default_factory=list)
    has_extra_changes: bool = False
    total_context_chars: int = 0
    recalled_files: List[RecallCandidate] = field(default_factory=list)
    recall_diagnostics: Dict[str, Any] = field(default_factory=dict)
    base_commit: str = ""
    history_patterns: List[Any] = field(default_factory=list)
    request_reference_keys: List[str] = field(default_factory=list)


class ContextBuilder:
    """Build structured context deterministically for LLM generation."""

    def __init__(self, repo_dir: Path, max_context_chars: int = MAX_CONTEXT_CHARS,
                 recall_mode: str = "expanded", max_recalled_files: int = 12,
                 history_mode: str = "patterns"):
        self.repo_dir = repo_dir
        self.analyzer = ASTAnalyzer()
        self.tracer = ImportTracer(repo_dir, self.analyzer)
        self.max_context_chars = max_context_chars
        self.recall_mode = recall_mode
        self.max_recalled_files = max_recalled_files
        self.repository_index = None
        if history_mode not in {"patterns", "legacy", "off"}:
            raise ValueError("Unknown history mode: " + history_mode)
        self.history_mode = "legacy" if recall_mode == "v5" and history_mode != "off" else history_mode

    def build_context(self, instance_id: str, problem_statement: str,
                      definitions: List[Dict], has_extra_changes: bool = False,
                      new_components_raw: Optional[List[Dict]] = None,
                      base_commit: str = "") -> InstanceContext:
        from data_loader import parse_problem_statement
        parsed = parse_problem_statement(problem_statement)

        ctx = InstanceContext(
            instance_id=instance_id,
            repo_dir=self.repo_dir,
            problem_statement=problem_statement,
            request=parsed["request"],
            has_extra_changes=has_extra_changes or parsed["has_extra_changes"],
            base_commit=base_commit,
        )

        defs = parsed["definitions"] if parsed["definitions"] else definitions
        if not defs and new_components_raw:
            defs = self._raw_components_to_defs(new_components_raw)

        for defn in defs:
            file_path = defn["file"]
            for comp in defn.get("components", []):
                comp_ctx = self._build_component_context(comp, file_path, parsed["request"])
                if comp_ctx:
                    ctx.components.append(comp_ctx)

        # Resolve inter-component deps BEFORE building call chains
        self._resolve_inter_component_deps(ctx)

        # Build reference call chains (needs inter-component info)
        for comp in ctx.components:
            self._build_reference_call_chain(comp)

        retriever = None
        if self.recall_mode == "expanded":
            retriever = FeatureRetriever(self.repo_dir, self.max_recalled_files, base_commit=base_commit)
            self.repository_index = retriever.index
            retriever.prepare_anchors(ctx)  # Full call/state traversal runs once, after history discovery.
            ctx.request_reference_keys = retriever.request_references

        history_diagnostics = {}
        if base_commit and self.history_mode == "patterns" and retriever:
            from explorer.history_patterns import CommitPatternRetriever, companion_files
            history = CommitPatternRetriever(self.repo_dir, retriever.index)
            ctx.history_patterns = history.retrieve(ctx, retriever.anchor_keys)
            history_diagnostics = history.diagnostics
            targets = {comp.target_file for comp in ctx.components}
            files = companion_files(ctx.history_patterns, targets)
            # Keep one global set; repeated components must not multiply a
            # historical file's score or duplicate its diff in the prompt.
            if ctx.components:
                ctx.components[0].feature_space_files = files

        # Every history query is limited to ancestors of the supplied base commit.
        if base_commit and self.history_mode == "legacy":
            from explorer.commit_tracer import CommitTracer
            target_files = {comp.target_file for comp in ctx.components}
            fs_tracer = CommitTracer(self.repo_dir)
            for comp in ctx.components:
                anchors = [n.method_name for n in comp.reference_call_chain[:1]]
                if retriever:
                    anchors = [retriever.index.symbols[k].name.split(".")[-1]
                               for k in retriever.anchor_keys.get(comp.name, [])]
                by_file = {}
                for anchor in dict.fromkeys(anchors):
                    for candidate in fs_tracer.trace_anchor_co_modified_files(
                            anchor, comp.target_file, base_commit, target_files,
                            new_component_name=comp.name.split(".")[-1]):
                        by_file.setdefault(candidate.file, candidate)
                comp.feature_space_files = list(by_file.values())

        if retriever:
            ctx.recalled_files = retriever.retrieve(ctx)
            ctx.recall_diagnostics = {
                "mode": self.recall_mode,
                "indexed_files": len(retriever.index.imports),
                "indexed_symbols": len(retriever.index.symbols),
                "parse_failures": retriever.index.parse_failures,
                "anchors": retriever.anchor_keys,
                "target_context_symbols": retriever.target_context_symbols,
                "request_references": retriever.request_references,
            }

        ctx.recall_diagnostics.update(history_mode=self.history_mode, history=history_diagnostics,
                                      history_commits=[pattern.commit for pattern in ctx.history_patterns])

        ctx.total_context_chars = self._estimate_context_size(ctx)
        return ctx

    def _build_component_context(self, comp: Dict, file_path: str,
                                  request: str) -> Optional[ComponentContext]:
        comp_ctx = ComponentContext(
            name=comp["name"],
            signature=comp.get("signature", ""),
            comp_type=comp.get("type", "function"),
            target_file=file_path,
            doc=comp.get("doc", ""),
        )

        target_module = self.analyzer.parse_file(self.repo_dir, file_path)
        if target_module is None:
            comp_ctx.target_file_source = ""
            self._handle_new_file_context(comp_ctx, request)
            return comp_ctx

        comp_ctx.target_file_source = target_module.source
        comp_ctx.module_imports = target_module.imports

        self._resolve_class_membership(comp_ctx, target_module)

        if comp_ctx.parent_class:
            cls = target_module.classes[comp_ctx.parent_class]
            if comp_ctx.nested_class and comp_ctx.nested_class in cls.nested_classes:
                cls = cls.nested_classes[comp_ctx.nested_class]
            comp_ctx.sibling_methods = dict(cls.methods)
            comp_ctx.class_init_vars = self.analyzer.get_class_init_vars(
                target_module, comp_ctx.parent_class, comp_ctx.nested_class
            )
            comp_ctx.class_vars = dict(cls.class_vars)

        if comp_ctx.parent_class:
            comp_ctx.base_classes = self.tracer.trace_base_class(
                comp_ctx.parent_class, target_module, file_path
            )

        comp_ctx.similar_functions = self._find_similar_functions(comp_ctx, target_module)

        # Analyze calling patterns of ALL siblings
        if comp_ctx.parent_class:
            for mname, method in comp_ctx.sibling_methods.items():
                callees = self.analyzer.find_method_callees(method)
                if callees:
                    comp_ctx.calling_patterns[mname] = callees

        comp_ctx.init_py_exports = self._check_init_py(file_path)

        # Collect relevant snippets from base classes
        for base_name, (base_file, base_module) in comp_ctx.base_classes.items():
            base_cls_name = base_name.split(".")[-1]
            if base_cls_name in base_module.classes:
                base_cls = base_module.classes[base_cls_name]
                for mname, method in base_cls.methods.items():
                    comp_words = set(re.findall(r'[a-z]+', comp_ctx.name.split(".")[-1]))
                    m_words = set(re.findall(r'[a-z]+', mname))
                    if comp_words & m_words and len(comp_words) > 0:
                        comp_ctx.relevant_snippets.append({
                            "file": base_file,
                            "snippet": method.body_source[:2000],
                            "purpose": f"Base class method '{mname}' (similar naming)",
                        })

        return comp_ctx

    def _build_reference_call_chain(self, comp: ComponentContext):
        """Build the reference call chain from the most similar sibling method.

        Key insight: if we're adding `allowed_compatible`, and `allowed` is the most
        similar sibling, we should trace what `allowed` calls (its full call chain)
        so the LLM can replicate the pattern.

        This is the "funcA calls funcB" problem the user described.
        """
        if not comp.similar_functions:
            return

        # Take the top similar function as the reference
        best_ref = comp.similar_functions[0]
        ref_method_name = best_ref["method"]
        ref_class_name = best_ref["class"]

        # BFS: trace the call chain from the reference method
        methods = comp.sibling_methods
        if not ref_class_name:
            module = self.analyzer.parse_file(self.repo_dir, comp.target_file)
            methods = module.functions if module else {}
        visited = set()
        queue = [ref_method_name]
        chain = []

        while queue and len(chain) < 5:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in methods:
                method = methods[current]
                callees = self.analyzer.find_method_callees(method)
                chain.append(CallChainNode(
                    method_name=current,
                    class_name=ref_class_name,
                    file=comp.target_file,
                    body_source=method.body_source,
                    callees=callees,
                ))
                for callee in callees:
                    if callee not in visited:
                        queue.append(callee)

        comp.reference_call_chain = chain

    def _resolve_class_membership(self, comp_ctx: ComponentContext, module: ModuleInfo):
        parts = comp_ctx.name.split(".")

        if len(parts) == 1:
            if comp_ctx.comp_type == "class":
                if parts[0] in module.classes:
                    comp_ctx.parent_class = parts[0]
            return

        if len(parts) == 2:
            if parts[0] in module.classes:
                cls = module.classes[parts[0]]
                if parts[1] in cls.methods:
                    comp_ctx.parent_class = parts[0]
                    comp_ctx.parent_class_file = comp_ctx.target_file
                elif parts[1] in cls.nested_classes:
                    comp_ctx.parent_class = parts[0]
                    comp_ctx.nested_class = parts[1]
                else:
                    # New method to add to this class
                    comp_ctx.parent_class = parts[0]
                    comp_ctx.parent_class_file = comp_ctx.target_file
            return

        if len(parts) == 3:
            if parts[0] in module.classes:
                cls = module.classes[parts[0]]
                if parts[1] in cls.nested_classes:
                    comp_ctx.parent_class = parts[0]
                    comp_ctx.nested_class = parts[1]
                    comp_ctx.parent_class_file = comp_ctx.target_file

    def _handle_new_file_context(self, comp_ctx: ComponentContext, request: str):
        parent_dir = str(Path(comp_ctx.target_file).parent)
        parent_path = self.repo_dir / parent_dir

        if parent_path.exists():
            for f in sorted(parent_path.glob("*.py")):
                rel = str(f.relative_to(self.repo_dir))
                if rel != comp_ctx.target_file:
                    module = self.analyzer.parse_file(self.repo_dir, rel)
                    if module and module.classes:
                        first_cls = next(iter(module.classes.values()))
                        comp_ctx.relevant_snippets.append({
                            "file": rel,
                            "snippet": f"# Reference: {rel} with class {first_cls.name}\n{module.source[:3000]}",
                            "purpose": "Sibling file showing pattern for new file",
                        })
                        break

        comp_ctx.init_py_exports = self._check_init_py(comp_ctx.target_file)

    def _find_similar_functions(self, comp_ctx: ComponentContext,
                                module: ModuleInfo) -> List[Dict[str, Any]]:
        wanted = identifier_terms(comp_ctx.name.split(".")[-1])
        similar = []
        functions = [("", name, fn) for name, fn in module.functions.items()]
        for class_name, cls in module.classes.items():
            functions.extend((class_name, name, fn) for name, fn in cls.methods.items())
        for class_name, name, method in functions:
            if name == comp_ctx.name.split(".")[-1]:
                continue
            words = identifier_terms(name)
            score = len(wanted & words) / max(1, len(wanted | words))
            if score >= 0.2:
                similar.append({"class": class_name, "method": name,
                                "signature": "def %s(%s)" % (name, ", ".join(method.args)),
                                "body": method.body_source, "similarity": score})
        return sorted(similar, key=lambda item: (-item["similarity"], item["class"], item["method"]))[:5]

    def _check_init_py(self, file_path: str) -> Optional[InitPyInfo]:
        """Check __init__.py with sibling export pattern analysis."""
        parent = str(Path(file_path).parent)
        init_file = f"{parent}/__init__.py"

        content = get_file_content(self.repo_dir, init_file)
        if content is None:
            return None

        module = self.analyzer.parse_file(self.repo_dir, init_file)
        if module is None:
            return None

        comp_name = Path(file_path).stem
        imports_us = any(
            imp["type"] == "from" and comp_name in [n["name"] for n in imp.get("names", [])]
            for imp in module.imports
        )

        # Analyze sibling export pattern: how do OTHER modules in this package get exported?
        sibling_pattern = ""
        if module.dunder_all is not None:
            # Check if functions from submodules appear in __all__
            for name in module.dunder_all[:20]:
                if name != comp_name:
                    sibling_pattern = f"__all__ contains names from submodules (e.g. {name})"
                    break

        # Check if there are from .submodule import func patterns
        for imp in module.imports:
            if imp["type"] == "from" and imp.get("level", 0) > 0:
                for n in imp.get("names", []):
                    n_name = n if isinstance(n, str) else n.get("name", "")
                    if n_name[0].islower() and n_name != comp_name:
                        sibling_pattern = f"Pattern: `from .{imp['module']} import {n_name}`"
                        break

        return InitPyInfo(
            file=init_file,
            has_all=module.dunder_all is not None,
            all_names=module.dunder_all or [],
            imports_our_module=imports_us,
            source_preview=content[:2000],
            sibling_export_pattern=sibling_pattern,
        )

    def _resolve_inter_component_deps(self, ctx: InstanceContext):
        """Detect inter-component dependencies.

        Key insight from user: unix_partition calls unix_device.
        Both are new components in the same file.
        We detect this by:
        1. Same-file components: check if one's name appears in request/doc of another
        2. Cross-file components: check if signatures reference each other
        """
        if len(ctx.components) < 2:
            return

        # Build lookup: short_name -> ComponentContext
        by_short_name = {}
        for comp in ctx.components:
            short = comp.name.split(".")[-1]
            by_short_name[short] = comp

        for comp in ctx.components:
            short = comp.name.split(".")[-1]
            # Check if the request mentions this component being called by another
            for other_short, other_comp in by_short_name.items():
                if other_short == short:
                    continue
                # Check in request text
                request_lower = ctx.request.lower()
                # Check if other component's doc/request suggests calling us
                if other_comp.doc and short in other_comp.doc:
                    other_comp.depends_on_components.append(short)
                # Check same-file: if both target same file, likely interdependent
                if (other_comp.target_file == comp.target_file and
                    other_comp.comp_type == "function" and comp.comp_type == "function"):
                    # Heuristic: if the function names share domain words
                    other_words = set(re.findall(r'[a-z]+', other_short))
                    my_words = set(re.findall(r'[a-z]+', short))
                    shared = other_words & my_words
                    # Even without shared words, functions in same class are likely related
                    # Check if one name is a "broader" version of the other
                    # e.g. unix_device vs unix_partition → unix is shared
                    if shared and len(shared) >= 1:
                        # The more specific one likely depends on the more general one
                        if len(other_short) > len(short):
                            other_comp.depends_on_components.append(short)

    def _raw_components_to_defs(self, raw: List[Dict]) -> List[Dict]:
        defs = []
        for item in raw:
            components = []
            for comp in item.get("components", []):
                components.append({
                    "name": comp["name"],
                    "signature": comp.get("signature", ""),
                    "type": comp.get("type", "function"),
                    "doc": comp.get("doc", ""),
                })
            defs.append({"file": item["file"], "components": components})
        return defs

    def _estimate_context_size(self, ctx: InstanceContext) -> int:
        total = len(ctx.request)
        for comp in ctx.components:
            total += len(comp.target_file_source)
            total += len(comp.doc)
            for m in comp.sibling_methods.values():
                total += len(m.body_source)
            for node in comp.reference_call_chain:
                total += len(node.body_source)
            total += sum(len(s["snippet"]) for s in comp.relevant_snippets)
        return total

    def format_context_for_llm(self, ctx: InstanceContext) -> str:
        """Format structured context into a prompt-friendly string for LLM."""
        parts = []

        # Header
        parts.append(f"# Feature Request\n{ctx.request}\n")
        if ctx.history_patterns:
            from explorer.history_patterns import render_commit_patterns
            parts.append(render_commit_patterns(ctx.history_patterns))

        # Components overview
        parts.append(f"# Components to Implement ({len(ctx.components)} total)")
        for i, comp in enumerate(ctx.components):
            parts.append(f"\n## Component {i+1}: {comp.name}")
            parts.append(f"- Type: {comp.comp_type}")
            parts.append(f"- File: {comp.target_file}")
            parts.append(f"- Signature: {comp.signature}")
            if comp.doc:
                parts.append(f"- Docstring:\n{comp.doc}")
            if comp.parent_class:
                parts.append(f"- Parent class: {comp.parent_class}" +
                            (f".{comp.nested_class}" if comp.nested_class else ""))
            if comp.depends_on_components:
                parts.append(f"- DEPENDS ON other new components: {comp.depends_on_components}")
        parts.append("")

        # Per-component context
        for i, comp in enumerate(ctx.components):
            parts.append(f"\n{'='*40}")
            parts.append(f"# Detailed Context for {comp.name}")
            parts.append(f"{'='*40}")

            # Class-level variables (TYPE_MAPPING, etc.)
            if comp.class_vars:
                parts.append(f"\n## Class-level variables in {comp.parent_class}")
                for var_name, var_type in comp.class_vars.items():
                    parts.append(f"  {var_name}: {var_type}")

            # Instance variables from __init__
            if comp.class_init_vars:
                parts.append(f"\n## Instance variables (from __init__)")
                for var, val in comp.class_init_vars.items():
                    parts.append(f"  self.{var} = {val}")

            # Sibling methods with full bodies (for pattern learning)
            if comp.sibling_methods:
                parts.append(f"\n## Existing methods in {comp.parent_class or 'module'}")
                comp_short = comp.name.split(".")[-1]
                for mname, method in comp.sibling_methods.items():
                    if mname == comp_short:
                        continue
                    parts.append(f"\n### {mname}{self._format_method_signature(method)}")
                    if method.docstring:
                        parts.append(f'"""{method.docstring}"""')
                    body = method.body_source
                    if len(body) > 2000:
                        body = body[:2000] + "\n    # ... (truncated)"
                    parts.append(body)

            # Call graph of all siblings
            if comp.calling_patterns:
                parts.append(f"\n## Method call graph (self.xxx() calls)")
                for caller, callees in comp.calling_patterns.items():
                    parts.append(f"  {caller} → [{', '.join(callees)}]")

            # REFERENCE CALL CHAIN - the key insight for "follow funcB's pattern"
            if comp.reference_call_chain:
                parts.append(f"\n## REFERENCE CALL CHAIN (follow this pattern)")
                parts.append(f"The most similar existing method and its full call chain:")
                for node in comp.reference_call_chain:
                    parts.append(f"\n### {node.class_name}.{node.method_name}()")
                    if node.callees:
                        parts.append(f"(calls: {', '.join(node.callees)})")
                    body = node.body_source
                    if len(body) > 2500:
                        body = body[:2500] + "\n    # ... (truncated)"
                    parts.append(body)

            # Inter-component dependencies
            if comp.depends_on_components:
                parts.append(f"\n## INTER-COMPONENT DEPENDENCIES")
                parts.append(f"This component DEPENDS ON other new components: {comp.depends_on_components}")
                parts.append(f"You MUST ensure {comp.name} can call/reference these components.")
                # Add the depended component's context
                for dep_name in comp.depends_on_components:
                    for other_comp in ctx.components:
                        if other_comp.name.split(".")[-1] == dep_name:
                            parts.append(f"\n### Depended component: {other_comp.name}")
                            parts.append(f"File: {other_comp.target_file}")
                            parts.append(f"Signature: {other_comp.signature}")
                            if other_comp.doc:
                                parts.append(f"Doc: {other_comp.doc}")
                            break

            # Base classes
            if comp.base_classes:
                parts.append(f"\n## Base class definitions")
                for base_name, (base_file, base_module) in comp.base_classes.items():
                    base_cls_name = base_name.split(".")[-1]
                    if base_cls_name in base_module.classes:
                        base_cls = base_module.classes[base_cls_name]
                        parts.append(f"\n### {base_name} (from {base_file})")
                        parts.append(f"Bases: {', '.join(base_cls.bases)}")
                        parts.append(f"Methods: {', '.join(base_cls.methods.keys())}")
                        # Include relevant base methods
                        for mname, method in base_cls.methods.items():
                            comp_words = set(re.findall(r'[a-z]+', comp.name.split(".")[-1]))
                            m_words = set(re.findall(r'[a-z]+', mname))
                            if comp_words & m_words and len(comp_words) > 0:
                                parts.append(f"\n  Relevant base method '{mname}':")
                                body = method.body_source[:2000]
                                parts.append(f"  {body}")

            # Similar functions
            if comp.similar_functions:
                parts.append(f"\n## Similar existing functions (by name pattern)")
                for sim in comp.similar_functions[:3]:
                    parts.append(f"\n### {sim['class']}.{sim['method']} (similarity: {sim['similarity']:.1%})")
                    parts.append(sim["body"][:2000])

            # __init__.py context
            if comp.init_py_exports:
                init = comp.init_py_exports
                parts.append(f"\n## __init__.py context ({init.file})")
                parts.append(f"```python")
                parts.append(init.source_preview[:2000])
                parts.append(f"```")
                if init.has_all:
                    parts.append(f"Has __all__ = {init.all_names[:20]}")
                if init.imports_our_module:
                    parts.append(f"STATUS: Already imports our module ✓")
                else:
                    parts.append(f"STATUS: Does NOT import our module ✗")
                    parts.append(f"ACTION: You likely need to add an import for the new component in this __init__.py")
                if init.sibling_export_pattern:
                    parts.append(f"Sibling export pattern: {init.sibling_export_pattern}")

            # Relevant snippets
            if comp.relevant_snippets:
                parts.append(f"\n## Additional context")
                for snippet in comp.relevant_snippets:
                    parts.append(f"\n### {snippet['purpose']} (from {snippet['file']})")
                    parts.append(snippet["snippet"][:2000])

            # Target file info
            if comp.target_file_source:
                parts.append(f"\n## Target file: {comp.target_file} ({len(comp.target_file_source)} chars)")

        # Extra changes warning
        if ctx.has_extra_changes:
            parts.append(f"\n# IMPORTANT: Extra changes required")
            parts.append("In addition to the new components, other code changes are needed:")
            parts.append("- __init__.py modifications (adding imports/exports)")
            parts.append("- Import additions in existing files")
            parts.append("- Configuration/registration changes")
            parts.append("- Any file mentioned in the feature request that isn't a component")

        result = "\n".join(parts)
        if len(result) > self.max_context_chars:
            result = result[:self.max_context_chars] + "\n\n# [Context truncated due to size limit]"
        if ctx.recalled_files:
            parts.append("\n## Retrieved integration locations\n")
            for candidate in ctx.recalled_files:
                parts.append("File: %s (score %.2f)" % (candidate.file, candidate.score))
                parts.extend("- " + reason for reason in candidate.reasons)
                for snippet in candidate.snippets:
                    parts.append(snippet)

        return result

    def _format_method_signature(self, method: FunctionInfo) -> str:
        args = ["self"]
        if not method.is_staticmethod:
            prefix = "cls" if method.is_classmethod else "self"
            args = [prefix]
        args.extend(method.args)
        ret = f" -> {method.return_annotation}" if method.return_annotation else ""
        return f"({', '.join(args)}){ret}"
