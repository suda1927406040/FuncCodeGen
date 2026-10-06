"""AST-based code analysis for understanding repo structure.

This module implements deterministic (zero LLM calls) code analysis:
- Parse Python files into AST trees
- Extract class structures, method signatures, decorators
- Find class hierarchy (bases, MRO)
- Identify sibling methods and their calling patterns
- Detect __init__.py exports and __all__
"""
import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any


@dataclass
class FunctionInfo:
    name: str
    args: List[str]                # argument names (excluding self)
    decorators: List[str]
    docstring: str = ""
    body_source: str = ""          # raw source of function body
    return_annotation: str = ""
    is_staticmethod: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    line_start: int = 0
    line_end: int = 0


@dataclass
class ClassInfo:
    name: str
    bases: List[str]               # base class names
    methods: Dict[str, FunctionInfo] = field(default_factory=dict)
    class_vars: Dict[str, str] = field(default_factory=dict)  # name -> type annotation or "unknown"
    docstring: str = ""
    nested_classes: Dict[str, "ClassInfo"] = field(default_factory=dict)
    line_start: int = 0
    line_end: int = 0

    def all_methods(self) -> Dict[str, FunctionInfo]:
        """Get all methods including from nested classes (flattened with class prefix)."""
        result = dict(self.methods)
        for nc_name, nc in self.nested_classes.items():
            for m_name, m in nc.methods.items():
                result[f"{nc_name}.{m_name}"] = m
        return result


@dataclass
class ModuleInfo:
    file_path: str
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    dunder_all: Optional[List[str]] = None
    source: str = ""


class ASTAnalyzer:
    """Deterministic Python AST analyzer."""

    def __init__(self, max_file_chars: int = 20000):
        self.max_file_chars = max_file_chars
        self._cache: Dict[str, Optional[ModuleInfo]] = {}

    def parse_file(self, repo_dir: Path, file_path: str) -> Optional[ModuleInfo]:
        """Parse a Python file and extract structured information."""
        cache_key = f"{repo_dir}:{file_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        full_path = repo_dir / file_path
        if not full_path.exists():
            self._cache[cache_key] = None
            return None

        source = full_path.read_text(errors="replace")
        # Parse complete files; truncating before AST parsing loses symbols or
        # creates a SyntaxError in the middle of a function. Rendering owns budgets.

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            self._cache[cache_key] = None
            return None

        info = ModuleInfo(
            file_path=file_path,
            source=source,
        )

        # Extract imports first
        info.imports = self._extract_imports(tree)

        # Extract __all__
        info.dunder_all = self._extract_dunder_all(tree)

        # Extract module-level functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_info = self._extract_class(node, source)
                info.classes[cls_info.name] = cls_info
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function(node, source)
                info.functions[func_info.name] = func_info

        self._cache[cache_key] = info
        return info

    def _extract_imports(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract all import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = []
                for alias in node.names:
                    names.append({
                        "name": alias.name,
                        "alias": alias.asname,
                    })
                imports.append({
                    "type": "from",
                    "module": module,
                    "names": names,
                    "level": node.level,  # for relative imports
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                    })
        return imports

    def _extract_dunder_all(self, tree: ast.Module) -> Optional[List[str]]:
        """Extract __all__ = [...] if present."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            names = []
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    names.append(str(elt.value))
                            return names
        return None

    def _extract_class(self, node: ast.ClassDef, source: str) -> ClassInfo:
        """Extract class definition with all methods and nested classes."""
        cls = ClassInfo(
            name=node.name,
            bases=[self._node_to_str(b) for b in node.bases],
            docstring=ast.get_docstring(node) or "",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
        )

        for item in ast.iter_child_nodes(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_function(item, source)
                cls.methods[func.name] = func
            elif isinstance(item, ast.ClassDef):
                nested = self._extract_class(item, source)
                cls.nested_classes[nested.name] = nested
            elif isinstance(item, ast.Assign):
                # Class-level variables
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        cls.class_vars[target.id] = "unknown"
            elif isinstance(item, ast.AnnAssign):
                # Typed class variables: var: Type = value
                if isinstance(item.target, ast.Name):
                    cls.class_vars[item.target.id] = self._node_to_str(item.annotation) if item.annotation else "unknown"

        return cls

    def _extract_function(self, node, source: str) -> FunctionInfo:
        """Extract function/method definition."""
        args = []
        decorators = []
        is_static = is_class = is_prop = False

        # Extract decorators
        for dec in node.decorator_list:
            dec_str = self._node_to_str(dec)
            decorators.append(dec_str)
            if dec_str == "staticmethod":
                is_static = True
            elif dec_str == "classmethod":
                is_class = True
            elif dec_str == "property":
                is_prop = True

        # Extract args (skip 'self' and 'cls')
        all_args = node.args
        skip_first = (not is_static and bool(all_args.args)
                      and all_args.args[0].arg in {"self", "cls"})
        arg_list = all_args.args[1:] if skip_first and all_args.args else all_args.args
        for arg in arg_list:
            args.append(arg.arg)

        # Return annotation
        return_ann = ""
        if node.returns:
            return_ann = self._node_to_str(node.returns)

        # Extract body source
        body_lines = source.split("\n")
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        end = node.end_lineno or node.lineno
        body_source = "\n".join(body_lines[start:end])

        return FunctionInfo(
            name=node.name,
            args=args,
            decorators=decorators,
            docstring=ast.get_docstring(node) or "",
            body_source=body_source,
            return_annotation=return_ann,
            is_staticmethod=is_static,
            is_classmethod=is_class,
            is_property=is_prop,
            line_start=node.lineno,
            line_end=end,
        )

    def _node_to_str(self, node) -> str:
        """Convert AST node to string representation."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._node_to_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Subscript):
            return f"{self._node_to_str(node.value)}[{self._node_to_str(node.slice)}]"
        if isinstance(node, ast.Index):  # Python 3.8 compat
            return self._node_to_str(node.value)
        if isinstance(node, ast.Tuple):
            return ", ".join(self._node_to_str(e) for e in node.elts)
        if isinstance(node, ast.List):
            return "[" + ", ".join(self._node_to_str(e) for e in node.elts) + "]"
        if isinstance(node, ast.BinOp):
            return f"{self._node_to_str(node.left)} | {self._node_to_str(node.right)}"
        if isinstance(node, ast.Starred):
            return f"*{self._node_to_str(node.value)}"
        if isinstance(node, ast.Call):
            return f"{self._node_to_str(node.func)}(...)"
        if isinstance(node, ast.Getattr):
            return f"{self._node_to_str(node.value)}.{node.attr}"
        try:
            return ast.dump(node)
        except Exception:
            return "?"

    def find_method_callers(self, module_info: ModuleInfo, method_name: str,
                            class_name: Optional[str] = None) -> List[Tuple[str, str]]:
        """Find which methods call a given method.

        Returns list of (caller_class, caller_method) tuples.
        """
        callers = []
        target_patterns = [f"self.{method_name}(", f"self.{method_name} ("]

        def _check_body(body_source: str, caller_name: str, cname: str):
            for pat in target_patterns:
                if pat in body_source:
                    callers.append((cname, caller_name))
                    return

        if class_name and class_name in module_info.classes:
            cls = module_info.classes[class_name]
            for mname, method in cls.methods.items():
                if mname != method_name:
                    _check_body(method.body_source, mname, class_name)
            for nc_name, nc in cls.nested_classes.items():
                for mname, method in nc.methods.items():
                    if mname != method_name:
                        _check_body(method.body_source, f"{nc_name}.{mname}", class_name)
        else:
            # Search all classes
            for cname, cls in module_info.classes.items():
                for mname, method in cls.methods.items():
                    if mname != method_name:
                        _check_body(method.body_source, mname, cname)

        return callers

    def find_method_callees(self, func_info: FunctionInfo) -> List[str]:
        """Find which self.xxx() calls a method makes.

        Returns list of method names called via self.
        """
        import re
        pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return sorted(set(re.findall(pattern, func_info.body_source)))

    def get_sibling_methods(self, module_info: ModuleInfo,
                            class_name: str, nested_class: Optional[str] = None) -> Dict[str, FunctionInfo]:
        """Get all sibling methods in the same class."""
        if class_name not in module_info.classes:
            return {}

        cls = module_info.classes[class_name]
        if nested_class:
            if nested_class in cls.nested_classes:
                return cls.nested_classes[nested_class].methods
            return {}
        return cls.methods

    def get_class_init_vars(self, module_info: ModuleInfo, class_name: str,
                            nested_class: Optional[str] = None) -> Dict[str, str]:
        """Extract instance variables set in __init__.

        Returns dict of var_name -> assignment pattern.
        """
        target_cls = None
        if class_name in module_info.classes:
            cls = module_info.classes[class_name]
            if nested_class:
                target_cls = cls.nested_classes.get(nested_class)
            else:
                target_cls = cls

        if not target_cls or "__init__" not in target_cls.methods:
            return {}

        init_func = target_cls.methods["__init__"]
        import re
        # Match self.xxx = ... patterns
        pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        vars_found = {}
        for match in re.finditer(pattern, init_func.body_source):
            var_name = match.group(1)
            # Try to get the RHS
            line = init_func.body_source.split("\n")
            for l in line:
                if f"self.{var_name} =" in l:
                    rhs = l.split(f"self.{var_name} =")[1].strip().rstrip(",")
                    vars_found[var_name] = rhs[:100]
                    break

        return vars_found

    def clear_cache(self):
        self._cache.clear()
