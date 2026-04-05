"""Import chain tracer - resolve imports to file paths using BFS.

Deterministic (zero LLM calls). Resolves:
- from X import Y → find X's file path
- from . import Y → resolve relative imports
- import X → find X's file path
- Follow base class imports to find parent class definitions
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from explorer.ast_analyzer import ASTAnalyzer, ModuleInfo


@dataclass
class ImportNode:
    """A resolved import with its file path."""
    module_name: str           # e.g. "sqlglot.dialects.dialect"
    file_path: str             # e.g. "sqlglot/dialects/dialect.py"
    import_names: List[str]    # e.g. ["Dialect", "Dialects"]
    is_relative: bool = False


class ImportTracer:
    """Trace import chains to resolve dependencies."""

    def __init__(self, repo_dir: Path, analyzer: ASTAnalyzer, max_depth: int = 5):
        self.repo_dir = repo_dir
        self.analyzer = analyzer
        self.max_depth = max_depth

    def resolve_import(self, import_info: Dict, current_file: str) -> Optional[ImportNode]:
        """Resolve a single import to its file path.

        Args:
            import_info: Parsed import dict from ASTAnalyzer
            current_file: The file containing this import

        Returns:
            ImportNode if resolved, None otherwise
        """
        if import_info["type"] == "from":
            return self._resolve_from_import(import_info, current_file)
        else:
            return self._resolve_absolute_import(import_info)

    def _resolve_from_import(self, info: Dict, current_file: str) -> Optional[ImportNode]:
        """Resolve 'from X import Y' statements."""
        module = info["module"]
        level = info.get("level", 0)
        names = [n["name"] for n in info["names"]]

        if level > 0:
            # Relative import: from ..dialects import X
            base_dir = Path(current_file).parent
            for _ in range(level - 1):
                base_dir = base_dir.parent
            module_parts = module.split(".") if module else []
            full_parts = list(base_dir.parts) + module_parts
            module = ".".join(full_parts)

        # Try as a Python module (file or package)
        candidates = self._module_to_paths(module)
        for path in candidates:
            if (self.repo_dir / path).exists():
                return ImportNode(
                    module_name=module,
                    file_path=path,
                    import_names=names,
                    is_relative=level > 0,
                )

        # Try the import name itself as a file within the module
        # e.g. from sqlglot.dialects import druid → sqlglot/dialects/druid.py
        for name in names:
            candidates = self._module_to_paths(f"{module}.{name}" if module else name)
            for path in candidates:
                if (self.repo_dir / path).exists():
                    return ImportNode(
                        module_name=f"{module}.{name}" if module else name,
                        file_path=path,
                        import_names=[name],
                        is_relative=level > 0,
                    )

        return None

    def _resolve_absolute_import(self, info: Dict) -> Optional[ImportNode]:
        """Resolve 'import X' statements."""
        module = info["module"]
        candidates = self._module_to_paths(module)
        for path in candidates:
            if (self.repo_dir / path).exists():
                return ImportNode(
                    module_name=module,
                    file_path=path,
                    import_names=[],
                )
        return None

    def trace_imports_bfs(self, target_file: str, target_names: Optional[Set[str]] = None,
                          max_files: int = 20) -> Dict[str, ModuleInfo]:
        """BFS from target file, following imports.

        Only follows imports that reference target_names (if provided)
        or appear in the same module.

        Returns dict of file_path -> ModuleInfo for all discovered files.
        """
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(target_file, 0)]
        result: Dict[str, ModuleInfo] = {}

        while queue and len(result) < max_files:
            current_file, depth = queue.pop(0)
            if current_file in visited or depth > self.max_depth:
                continue
            visited.add(current_file)

            module_info = self.analyzer.parse_file(self.repo_dir, current_file)
            if module_info is None:
                continue
            result[current_file] = module_info

            # Follow imports
            for imp in module_info.imports:
                resolved = self.resolve_import(imp, current_file)
                if resolved and resolved.file_path not in visited:
                    # Filter: only follow relevant imports
                    if target_names is None or self._is_relevant_import(imp, target_names):
                        queue.append((resolved.file_path, depth + 1))

        return result

    def trace_base_class(self, class_name: str, module_info: ModuleInfo,
                         current_file: str) -> Dict[str, Tuple[str, ModuleInfo]]:
        """Trace base class definitions through import chain.

        Returns dict of class_name -> (file_path, ModuleInfo).
        """
        result = {}
        if class_name not in module_info.classes:
            return result

        cls = module_info.classes[class_name]
        for base in cls.bases:
            # base might be "Dialect" or "generator.Generator" or "exp.Ceil"
            base_file = self._find_name_definition(base, module_info, current_file)
            if base_file:
                base_module = self.analyzer.parse_file(self.repo_dir, base_file)
                if base_module:
                    result[base] = (base_file, base_module)
        return result

    def find_name_definition_file(self, name: str, module_info: ModuleInfo,
                                  current_file: str) -> Optional[str]:
        """Find which file defines a given name.

        Handles:
        - Simple names: "Dialect" → search imports, then current file
        - Dotted names: "generator.Generator" → resolve "generator" module, find "Generator"
        """
        return self._find_name_definition(name, module_info, current_file)

    def _find_name_definition(self, name: str, module_info: ModuleInfo,
                              current_file: str) -> Optional[str]:
        """Internal: resolve a name to its defining file."""
        # Handle dotted names: "generator.Generator" → module=generator, name=Generator
        if "." in name:
            parts = name.split(".", 1)
            module_part, rest = parts[0], parts[1]
            # Find the module
            for imp in module_info.imports:
                resolved = self.resolve_import(imp, current_file)
                if resolved:
                    for n in imp.get("names", []):
                        n_name = n if isinstance(n, str) else n.get("name", "")
                        if n_name == module_part or imp.get("module", "").endswith(module_part):
                            # Recurse into the resolved file
                            sub_module = self.analyzer.parse_file(self.repo_dir, resolved.file_path)
                            if sub_module:
                                return self._find_name_definition(rest, sub_module, resolved.file_path)
                            return resolved.file_path
            return None

        # Simple name: search imports
        for imp in module_info.imports:
            if imp["type"] == "from":
                for n in imp.get("names", []):
                    n_name = n if isinstance(n, str) else n.get("name", "")
                    if n_name == name:
                        resolved = self.resolve_import(imp, current_file)
                        if resolved:
                            return resolved.file_path
            elif imp["type"] == "import":
                if imp["module"].endswith(name):
                    resolved = self.resolve_import(imp, current_file)
                    if resolved:
                        return resolved.file_path

        # Not found in imports, might be defined in current file
        if name in module_info.classes or name in module_info.functions:
            return current_file

        return None

    def _module_to_paths(self, module: str) -> List[str]:
        """Convert module name to possible file paths.

        e.g. "sqlglot.dialects.dialect" → [
            "sqlglot/dialects/dialect.py",
            "sqlglot/dialects/dialect/__init__.py",
        ]
        """
        parts = module.split(".")
        base = "/".join(parts)
        return [
            f"{base}.py",
            f"{base}/__init__.py",
        ]

    def _is_relevant_import(self, imp: Dict, target_names: Set[str]) -> bool:
        """Check if an import references any of our target names."""
        if imp["type"] == "from":
            for n in imp.get("names", []):
                n_name = n if isinstance(n, str) else n.get("name", "")
                if n_name in target_names:
                    return True
        elif imp["type"] == "import":
            if any(t in imp["module"] for t in target_names):
                return True
        return False
