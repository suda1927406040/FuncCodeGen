"""Plan, validate, expand base context, then generate one final patch response."""
import ast
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import re

from config import MAX_CONTEXT_CHARS
from explorer.context_renderer import render_code_files
from explorer.feature_retriever import RecallCandidate, RepositoryIndex, identifier_terms, is_source_file
from explorer.history_patterns import render_commit_patterns


class PlanError(ValueError):
    def __init__(self, message, report=None):
        self.report = report or {"valid": False, "errors": [message]}
        super().__init__(message)


PLAN_FORMAT = """Return one JSON object with this schema. Do not generate code or edits.
{
  "contracts": ["Required behavior and existing behavior that must remain valid"],
  "steps": [
    {"id": "S1", "file": "pkg/api.py", "action": "modify", "symbol": "Engine.select",
     "purpose": "Brief description of this responsibility and the necessary change",
     "constraints": ["Preserve distinct state/guard meanings"], "depends_on": [],
     "evidence": [{"source": "current", "file": "pkg/api.py", "symbol": "Engine.select"}]}
  ],
  "missing_context": [{"file": "", "symbol": "helper_name", "reason": "Need its callers and state writers"}]
}
Allowed actions: modify, create (new file), create_symbol (new method/function/field in an existing file), inspect.
Use the exact current qualified symbol for modify; use an empty symbol for imports/exports/configuration.
For create_symbol, explicitly name the new symbol; don't describe it as an existing method.
Evidence sources: current (base file/symbol), history (commit + file), requirement (declared file/symbol).
Use these evidence object formats:
- current: {"source": "current", "file": "actual/base/path.py", "symbol": "CurrentClass.method"}
- history: {"source": "history", "commit": "<exact visible COMMIT id>", "file": "path/changed/in/that/commit.py"}
- requirement: {"source": "requirement", "file": "<declared file>", "symbol": "<exact declared symbol>"}
History evidence MUST include the commit key and an ID from the supplied COMMIT blocks. A file must be
listed as changed in that particular commit. Do not use history to cite the new symbol being implemented.
For new definitions, use requirement evidence and copy an exact pair from Declared requirement symbols below.
For imports, exports, and module constants, use an empty symbol with current file evidence.
History is a reference: map applicable responsibilities to current base code. Package exports are necessary only
for public entry points required by the task; internal helpers need not become package APIs.
Every edit step needs at least one evidence reference. Dependencies must reference step IDs without cycles.
Use missing_context for unseen definitions; do not invent an existing file or assume an unresolved symbol exists.
Keep purposes and contracts brief. Output JSON only, no hidden reasoning or additional prose.
"""


def _task(instance, ctx):
    statement = instance["problem_statement"]
    match = re.search(r"<definitions>(.*?)</definitions>", statement, re.DOTALL)
    definitions = match.group(1).strip() if match else json.dumps(
        instance.get("new_components") or [], ensure_ascii=False)
    return "<request>\n%s\n</request>\n<definitions>\n%s\n</definitions>\n" % (ctx.request, definitions)


def _visible_files(prompt):
    section = prompt.split("\n<code>\n", 1)[1].split("\n</code>", 1)[0]
    return re.findall(r"^\[start of ([^\n\]]+)\]$", section, re.MULTILINE)


def _visible_symbols(prompt):
    found = set()
    section = prompt.split("\n<code>\n", 1)[1].split("\n</code>", 1)[0]
    for match in re.finditer(r"\[start of ([^\n\]]+)\]\n(.*?)\n\[end of \1\]", section, re.DOTALL):
        file, source = match.groups()
        if Path(file).suffix not in {".py", ".pyi"}:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        def collect(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    collect(node.body, prefix + node.name + ".")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.add(file + ':' + prefix + node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            found.add(file + ':' + prefix + target.id)
        collect(tree.body)
    return found


class PlanContext:
    def __init__(self, repo_dir, ctx, index=None):
        self.repo_dir = Path(repo_dir)
        self.ctx = ctx
        self.index = index or RepositoryIndex(self.repo_dir, ctx.base_commit)
        self.known_files = set(self.index.all_files)
        self.declared = {(comp.target_file, comp.name) for comp in ctx.components}
        self.targets = {comp.target_file for comp in ctx.components}
        self.history = {pattern.commit: pattern for pattern in ctx.history_patterns}
        self._class_fields_cache = {}
        self._module_bindings_cache = {}

    def _module_bindings(self, file):
        if file not in self._module_bindings_cache:
            bindings = {}
            try:
                source = (self.repo_dir / file).read_text(errors='replace')
                lines = source.splitlines(keepends=True)
                for node in ast.parse(source).body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        for target in targets:
                            for name in ast.walk(target):
                                if isinstance(name, ast.Name):
                                    bindings[name.id] = ''.join(lines[node.lineno - 1:node.end_lineno])
            except (OSError, SyntaxError):
                pass
            self._module_bindings_cache[file] = bindings
        return self._module_bindings_cache[file]

    def _class_fields(self, file):
        if file in self._class_fields_cache:
            return self._class_fields_cache[file]
        fields = defaultdict(list)
        try:
            tree = ast.parse((self.repo_dir / file).read_text(errors='replace'))
        except (SyntaxError, OSError):
            return fields
        def collect(nodes, prefix=''):
            for node in nodes:
                if not isinstance(node, ast.ClassDef):
                    continue
                owner = prefix + node.name
                for child in node.body:
                    if isinstance(child, (ast.Assign, ast.AnnAssign)):
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        for target in targets:
                            if isinstance(target, ast.Name):
                                fields[owner + '.' + target.id].append(owner + '.' + target.id)
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for value in ast.walk(child):
                            if (isinstance(value, ast.Attribute) and isinstance(value.ctx, ast.Store)
                                    and isinstance(value.value, ast.Name) and value.value.id in {'self', 'cls'}):
                                fields[owner + '.' + value.attr].append(owner + '.' + child.name)
                collect(node.body, owner + '.')
        collect(tree.body)
        self._class_fields_cache[file] = fields
        return fields

    def _file(self, file):
        if not isinstance(file, str) or not file or '\\' in file:
            raise PlanError("Plan file must be a relative POSIX path")
        path = PurePosixPath(file)
        if path.is_absolute() or '..' in path.parts or str(path) != file or '.git' in path.parts:
            raise PlanError("Invalid plan path: " + file)
        try:
            (self.repo_dir / file).resolve().relative_to(self.repo_dir.resolve())
        except ValueError:
            raise PlanError("Plan path leaves the repository: " + file)
        if set(path.parts) & {"tests", "test", "docs", "doc", ".venv", "venv", "__pycache__"} or path.name.startswith('test_'):
            raise PlanError("Plan path is outside production context: " + file)
        return file

    def resolve(self, file, symbol, seen=None):
        """Resolve current methods/classes/fields; return actual source owners."""
        index = self.index
        if not isinstance(symbol, str):
            raise PlanError('Context symbol must be a string')
        seen = set() if seen is None else seen
        marker = (file, symbol)
        if marker in seen or len(seen) >= 8:
            raise PlanError('Cyclic or unresolved symbol inheritance: ' + symbol)
        seen.add(marker)
        if not file:
            if symbol in index.modules:
                return self._file(index.modules[symbol]), []
            keys = index.by_name.get(symbol.split('.')[-1], [])
            exact = [key for key in keys if index.symbols[key].name == symbol or
                     ('.' in index.symbols[key].name and symbol.endswith('.' + index.symbols[key].name))]
            matches = exact or (keys if len(keys) == 1 else [])
            if len(matches) != 1:
                raise PlanError("Unresolved or ambiguous context symbol: " + symbol)
            resolved = index.symbols[matches[0]]
            return self._file(resolved.file), [resolved.name]
        file = self._file(file)
        if file not in self.known_files or not (self.repo_dir / file).is_file():
            raise PlanError("File is not present in the base tree: " + file)
        if not symbol or Path(file).suffix not in {'.py', '.pyi'}:
            return file, []
        for module, path in self.index.modules.items():
            if path == file and symbol.startswith(module + '.'):
                symbol = symbol[len(module) + 1:]
                break
        if symbol in self._module_bindings(file):
            return file, [symbol]
        field_sources = self._class_fields(file).get(symbol)
        if field_sources:
            return file, list(dict.fromkeys(field_sources))[:3]
        key = index._resolve_symbol(file, symbol)
        if key:
            value = index.symbols[key]
            return value.file, [value.name]
        class_key = file + ':' + symbol
        if class_key in index.class_bases:
            methods = [value.name for value in index.by_file[file] if value.class_name == symbol]
            methods.sort(key=lambda name: (not name.endswith('.__init__'), name))
            return file, methods[:3]
        owner, _, leaf = symbol.rpartition('.')
        if file + ':' + owner in index.class_bases:
            method = index._resolve_method(file + ':' + owner, leaf)
            if method:
                value = index.symbols[method]
                return value.file, [value.name]
            writers = [value.name for value in index.by_file[file]
                       if value.class_name == owner and leaf.lstrip('_') in value.writes]
            if writers:
                return file, sorted(writers, key=lambda name: (not name.endswith('.__init__'), name))[:3]
            for base in index.class_bases[file + ':' + owner]:
                parent = index._resolve_class(file, base)
                if parent:
                    parent_file, parent_name = parent.split(':', 1)
                    try:
                        return self.resolve(parent_file, parent_name + '.' + leaf, set(seen))
                    except PlanError:
                        continue
        raise PlanError("Symbol is not present in the base tree: %s:%s" % (file, symbol))

    def planning_prompt(self, instance):
        # Compact factual evidence, not legacy instructions to clone a sibling
        # or export every new helper.
        facts = []
        for candidate in self.ctx.recalled_files:
            facts.append({"file": candidate.file, "reasons": candidate.reasons[:2],
                          "symbols": candidate.symbols, "locations": candidate.locations[:2]})
        evidence = json.dumps(facts, ensure_ascii=False, separators=(',', ':'))
        if len(evidence) > 9000:
            evidence = json.dumps([{k: v for k, v in fact.items() if k != 'locations'} for fact in facts], ensure_ascii=False)
        inventory = '\n'.join(sorted(self.targets | {candidate.file for candidate in self.ctx.recalled_files}))
        history = render_commit_patterns(self.ctx.history_patterns, 8000, include_plan_instruction=False)
        prefix = ("Plan the current repository feature using current code and ancestral coordinated changes.\n"
                  + _task(instance, self.ctx) + "\nKnown candidate paths:\n" + inventory
                  + "\nDeclared requirement symbols (exact file/symbol pairs):\n"
                  + json.dumps([{'file': file, 'symbol': symbol} for file, symbol in sorted(self.declared)], ensure_ascii=False)
                  + "\nStatic evidence (observations, not mandatory edits):\n" + evidence + history + '\n' + PLAN_FORMAT)
        if len(prefix) > MAX_CONTEXT_CHARS - 2000:
            raise PlanError("Planning instructions exceed context budget")
        files = list(dict.fromkeys(sorted(self.targets) + [candidate.file for candidate in self.ctx.recalled_files]))
        code = render_code_files(self.repo_dir, files, self.ctx, MAX_CONTEXT_CHARS - len(prefix) - 30)
        prompt = prefix + '\n<code>\n' + code + '\n</code>\n'
        return prompt

    def parse_validate(self, response):
        raw = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1)
        try:
            plan = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise PlanError("Planner did not return complete valid JSON") from error
        if not isinstance(plan, dict) or not isinstance(plan.get('steps'), list) or not 1 <= len(plan['steps']) <= 24:
            raise PlanError("Plan needs 1–24 steps")
        contracts = plan.get('contracts')
        if not isinstance(contracts, list) or not contracts or not all(isinstance(value, str) and value.strip() for value in contracts):
            raise PlanError("Plan needs nonempty behavior contracts")
        normalized = []
        report = {"valid": True, "steps": [], "context_requests": [], "warnings": [], "ignored_steps": []}
        excluded = set()
        production = []
        for step in plan['steps']:
            file = step.get('file', '') if isinstance(step, dict) else ''
            parts = PurePosixPath(file).parts if isinstance(file, str) else ()
            if set(parts) & {'tests', 'test', 'docs', 'doc'} or (parts and parts[-1].startswith('test_')):
                excluded.add(step.get('id'))
                report['ignored_steps'].append(step)
                report['warnings'].append('Ignored nonproduction step: ' + str(file))
            else:
                production.append(step)
        if not production:
            raise PlanError('Plan contains no production changes')
        raw_steps = {step.get('id'): step for step in production if isinstance(step, dict)}
        def ancestors(sid, seen=None):
            seen = set() if seen is None else seen
            if sid in seen:
                return seen
            seen.add(sid)
            for dep in raw_steps.get(sid, {}).get('depends_on', []):
                if isinstance(dep, str) and dep not in excluded:
                    ancestors(dep, seen)
            return seen
        creates = {step['file']: step.get('id') for step in production
                   if isinstance(step, dict) and step.get('action') == 'create' and isinstance(step.get('file'), str)}
        for step in production:
            if not isinstance(step, dict):
                raise PlanError("Every plan step must be an object")
            sid = step.get('id', '')
            if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}', sid):
                raise PlanError("Invalid plan step ID")
            file = self._file(step.get('file'))
            action = step.get('action')
            symbol = step.get('symbol', '')
            if action not in {'modify', 'create', 'create_symbol', 'inspect'} or not isinstance(symbol, str):
                raise PlanError("Invalid plan action or symbol")
            if not isinstance(step.get('purpose'), str) or not step['purpose'].strip():
                raise PlanError("Plan step needs a brief purpose")
            constraints = step.get('constraints', [])
            dependencies = step.get('depends_on', [])
            if isinstance(dependencies, list):
                dependencies = [dep for dep in dependencies if dep not in excluded]
            if not isinstance(constraints, list) or not all(isinstance(value, str) for value in constraints):
                raise PlanError("Invalid constraints")
            if not isinstance(dependencies, list) or not all(isinstance(value, str) for value in dependencies):
                raise PlanError("Invalid step dependencies")
            if action == 'create':
                if file in self.known_files or (self.repo_dir / file).exists():
                    raise PlanError("create is only for an absent file: " + file)
                current_file, current_symbols = file, []
            elif action == 'create_symbol':
                if not symbol:
                    raise PlanError("create_symbol must name the new symbol")
                if file not in self.known_files and (file, symbol) in self.declared:
                    creator = creates.get(file)
                    if creator and creator not in ancestors(sid):
                        raise PlanError('New-file symbol must depend on its create step: ' + file)
                    current_file, current_symbols = file, []
                    if not creator:
                        action = 'create'
                        report['warnings'].append(sid + ': declared absent file normalized to create')
                else:
                    current_file, current_symbols = self.resolve(file, '')
                if file + ':' + symbol in self.index.symbols:
                    raise PlanError("create_symbol names an existing method: " + symbol)
                owner = symbol.rpartition('.')[0]
                if owner and file + ':' + owner in self.index.class_bases:
                    _, current_symbols = self.resolve(file, owner)
            else:
                if action == 'modify' and (file, symbol) in self.declared and file + ':' + symbol not in self.index.symbols:
                    report['warnings'].append(sid + ': declared new symbol normalized to create_symbol')
                    action = 'create_symbol' if file in self.known_files else 'create'
                    current_file, current_symbols = file, []
                else:
                    current_file, current_symbols = self.resolve(file, symbol)
            evidence = step.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                raise PlanError("Every step needs current/history/requirement evidence")
            checked = []
            for ref in evidence:
                if not isinstance(ref, dict):
                    raise PlanError("Invalid evidence object")
                source = ref.get('source')
                ref_file = self._file(ref.get('file'))
                if not isinstance(ref.get('symbol', ''), str):
                    raise PlanError('Evidence symbol must be a string')
                if source == 'current':
                    if (ref_file, ref.get('symbol', '')) in self.declared and ref_file not in self.known_files:
                        raise PlanError('Use requirement evidence for a new file: ' + ref_file)
                    resolved_file, resolved_symbols = self.resolve(ref_file, ref.get('symbol', ''))
                    checked.append({'source': source, 'file': resolved_file, 'symbols': resolved_symbols})
                elif source == 'requirement':
                    if (ref_file, ref.get('symbol', '')) not in self.declared:
                        raise PlanError("Unknown declared requirement evidence")
                    checked.append({'source': source, 'file': ref_file, 'symbol': ref['symbol']})
                elif source == 'history':
                    sha = ref.get('commit', '')
                    matches = [key for key in self.history if isinstance(sha, str) and len(sha) >= 7 and key.startswith(sha)]
                    if len(matches) != 1 or ref_file not in {change.file for change in self.history[matches[0]].changes}:
                        raise PlanError("Evidence is not from a supplied ancestral commit")
                    checked.append({'source': source, 'file': ref_file, 'commit': matches[0]})
                else:
                    raise PlanError("Unknown evidence source")
            item = {'id': sid, 'file': file, 'action': action, 'symbol': symbol, 'purpose': step['purpose'],
                    'constraints': constraints, 'depends_on': dependencies, 'evidence': checked,
                    'current_file': current_file, 'current_symbols': current_symbols}
            normalized.append(item)
            report['steps'].append({'id': sid, 'file': current_file, 'symbols': current_symbols, 'action': action})
        ids = {step['id'] for step in normalized}
        if len(ids) != len(normalized) or any(dep not in ids for step in normalized for dep in step['depends_on']):
            raise PlanError("Duplicate step IDs or unknown dependencies")
        pending = list(normalized)
        ordered, done = [], set()
        while pending:
            ready = [step for step in pending if set(step['depends_on']) <= done]
            if not ready:
                raise PlanError("Plan dependencies contain a cycle")
            for step in ready:
                ordered.append(step)
                done.add(step['id'])
                pending.remove(step)
        requests = plan.get('missing_context', [])
        if not isinstance(requests, list) or len(requests) > 16:
            raise PlanError("missing_context must be a list with at most 16 entries")
        supplements = []
        for request in requests:
            try:
                if not isinstance(request, dict) or not isinstance(request.get('symbol', ''), str):
                    raise PlanError("Invalid context request")
                file, symbols = self.resolve(request.get('file', ''), request.get('symbol', ''))
                supplements.append({'file': file, 'symbols': symbols, 'reason': str(request.get('reason', ''))})
                report['context_requests'].append({'request': request, 'resolved_file': file, 'symbols': symbols})
            except PlanError as error:
                report['warnings'].append(str(error))
        result = {'contracts': contracts, 'steps': ordered, 'context': supplements}
        if len(json.dumps(result, ensure_ascii=False)) > 16000:
            raise PlanError("Plan is too large for bounded generation context")
        return result, report

    def generation_prompt(self, instance, plan, output_format):
        required = defaultdict(list)
        planned_files = []
        for step in plan['steps']:
            planned_files.append(step['file'])
            planned_files.append(step['current_file'])
            required[step['current_file']].extend(step['current_symbols'])
        for item in plan['context']:
            planned_files.append(item['file'])
            required[item['file']].extend(item['symbols'])
        dependencies = []
        frequency = Counter()
        for value in self.index.symbols.values():
            frequency.update(value.reads | value.writes)
        seeds = [self.index.symbols[file + ':' + name] for file, names in required.items() for name in names
                 if file + ':' + name in self.index.symbols]
        neighbors = []
        for seed in seeds:
            keys = self.index.reverse[seed.key] | self.index.forward[seed.key]
            for key in sorted(keys):
                neighbors.append(self.index.symbols[key])
                dependencies.append({'kind': 'static call relation', 'from': seed.key, 'to': key})
        for value in self.index.symbols.values():
            for seed in seeds:
                fields = (value.writes & seed.reads) | (value.reads & seed.writes)
                rare = sorted(name for name in fields if frequency[name] <= 10)
                if rare and value.file != seed.file:
                    neighbors.append(value)
                    dependencies.append({'kind': 'shared state name; verify receiver', 'from': seed.key,
                                         'to': value.key, 'fields': rare})
        priority = list(dict.fromkeys(planned_files + sorted(self.targets)))
        supplemented = []
        for value in neighbors:
            if value.file not in priority and len(supplemented) >= 6:
                continue
            if value.file not in priority:
                priority.append(value.file)
                supplemented.append(value.file)
        for file in list(priority):
            for companion in [str(Path(file).parent / '__init__.py'), file + 'i']:
                if companion in self.known_files and companion not in priority:
                    priority.append(companion)
        priority += [candidate.file for candidate in self.ctx.recalled_files if candidate.file not in priority]
        render_ctx = copy(self.ctx)
        render_ctx.recall_diagnostics = dict(self.ctx.recall_diagnostics)
        render_ctx.recall_diagnostics['target_context_symbols'] = dict(self.ctx.recall_diagnostics.get('target_context_symbols', {}))
        render_ctx.recall_diagnostics['target_context_symbols'].update(required)
        render_ctx.recalled_files = []
        minimum, caps = {}, {}
        for file in priority:
            names = list(dict.fromkeys(required.get(file, []) + render_ctx.recall_diagnostics['target_context_symbols'].get(file, [])))
            names += [value.name for value in neighbors if value.file == file and value.name not in names][:3]
            sources = [self.index.symbols[file + ':' + name].source for name in names if file + ':' + name in self.index.symbols]
            render_ctx.recalled_files.append(RecallCandidate(file, 1, symbols=names, snippets=sources))
            if file in required:
                caps[file] = 45000
                minimum[file] = sum(len(self.index.symbols[file + ':' + name].source)
                                    for name in set(required[file]) if file + ':' + name in self.index.symbols) + sum(len(self._module_bindings(file).get(name, ''))
                                    for name in set(required[file])) + 700
        factual_plan = json.dumps(plan, ensure_ascii=False, indent=2)
        relation_text = json.dumps(dependencies[:30], ensure_ascii=False)
        prefix = ("Implement the feature once, using the checked plan and current code below.\n"
                  + _task(instance, self.ctx) + '\n<checked_plan>\n' + factual_plan + '\n</checked_plan>\n'
                  + "The script checked path/symbol existence and dependency order, not semantic correctness. "
                    "Treat contracts as proposals to verify against the request and current code. "
                    "Preserve existing API behavior and distinct state/guard meanings. Internal helpers do not require "
                    "public exports unless the task requires them. Dependency evidence is for inspection, not mandatory edits.\n"
                  + relation_text + render_commit_patterns(self.ctx.history_patterns, 5000, include_plan_instruction=False)
                  + "Implement every declared component and requested behavior even if the checked plan overlooked it. "
                    "Output only the final edit blocks; do not generate alternative solutions or another plan.\n")
        tail = '\n' + output_format
        budget = MAX_CONTEXT_CHARS - len(prefix) - len(tail) - 30
        if budget < 1000:
            raise PlanError("Validated plan leaves insufficient code context")
        code = render_code_files(self.repo_dir, priority, render_ctx, budget, file_caps=caps, minimum_budgets=minimum)
        prompt = prefix + '\n<code>\n' + code + '\n</code>\n' + tail
        shown_files, shown_symbols = set(_visible_files(prompt)), _visible_symbols(prompt)
        missing_files = set(planned_files) - shown_files
        needed = {file + ':' + name for file, names in required.items() for name in names}
        missing_symbols = needed - shown_symbols
        if missing_files or missing_symbols:
            raise PlanError("Required planned code does not fit the generation prompt",
                            {'valid': False, 'missing_files': sorted(missing_files), 'missing_symbols': sorted(missing_symbols)})
        return prompt, {'planned_files': list(dict.fromkeys(planned_files)), 'supplemented_files': supplemented,
                        'visible_code_files': sorted(shown_files), 'visible_symbols': sorted(shown_symbols),
                        'dependencies': dependencies[:30], 'prompt_chars': len(prompt)}
