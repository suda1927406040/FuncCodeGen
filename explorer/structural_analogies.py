"""Find repository-local feature lifecycles that may guide a new component.

Name similarity only proposes a dispatcher.  An analogy is exposed to the
planner only when current code also shows input forms, owned state, a policy
query over that state, and a dispatcher call.  The result remains evidence to
verify, rather than a template that must be copied.
"""
import ast
import re
import textwrap


def _words(value):
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', value)
    result = set()
    for word in re.findall(r'[a-z][a-z0-9]*', value.lower()):
        if len(word) < 3:
            continue
        if word.endswith('ies'):
            word = word[:-3] + 'y'
        elif word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]
        result.add(word)
    return result


def _tree(source):
    try:
        return ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, TypeError):
        return None


def _self_fields(source, context):
    tree = _tree(source)
    if tree is None:
        return set()
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == 'self'
            and isinstance(node.ctx, context)}


def _input_discriminators(source):
    tree = _tree(source)
    if tree is None:
        return []
    forms = []
    for node in ast.walk(tree):
        values = []
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'startswith'):
            values = node.args[:1]
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                forms.append(value.value)
    return list(dict.fromkeys(forms))


def discover_structural_analogies(ctx, limit=6):
    """Return closed-loop analogy candidates from the current base tree.

    A candidate needs stronger evidence than a similar method name: the
    dispatcher must call a sibling worker, that worker must read at least two
    concept-specific fields written by ``__init__``, and ``__init__`` must
    parse an input form for the same concept.
    """
    result = []
    for component in ctx.components:
        chain = component.reference_call_chain
        init_method = component.sibling_methods.get('__init__')
        if not component.parent_class or not init_method or len(chain) < 2:
            continue
        dispatcher = chain[0]
        target_words = _words(component.name.split('.')[-1])
        dispatcher_words = _words(dispatcher.method_name)
        target_concepts = sorted(target_words - dispatcher_words,
                                 key=lambda word: (-len(word), word))
        if len(target_concepts) != 1:
            continue
        target_concept = target_concepts[0]
        init_writes = _self_fields(init_method.body_source, ast.Store)
        discriminator_forms = _input_discriminators(init_method.body_source)
        discriminator_words = set().union(*(_words(value) for value in discriminator_forms))
        dispatcher_calls = set(component.calling_patterns.get(dispatcher.method_name, []))
        for worker in chain[1:]:
            if worker.method_name not in dispatcher_calls:
                continue
            reads = _self_fields(worker.body_source, ast.Load) & init_writes
            if len(reads) < 2:
                continue
            shared_field_words = None
            for field in reads:
                field_words = _words(field)
                shared_field_words = (field_words if shared_field_words is None
                                      else shared_field_words & field_words)
            worker_words = _words(worker.method_name)
            source_concepts = sorted((shared_field_words or set()) & worker_words
                                     & discriminator_words,
                                     key=lambda word: (-len(word), word))
            if len(source_concepts) != 1:
                continue
            source_concept = source_concepts[0]
            forms = [value for value in discriminator_forms
                     if source_concept in _words(value)]
            if not forms:
                continue
            state_fields = sorted(field for field in reads if source_concept in _words(field))
            if len(state_fields) < 2:
                continue
            result.append({
                'status': 'candidate_to_verify',
                'component': component.name,
                'file': component.target_file,
                'source_concept': source_concept,
                'target_concept': target_concept,
                'observed_chain': [
                    {'role': 'input_state_owner',
                     'symbol': component.parent_class + '.__init__',
                     'input_forms': forms, 'state_fields_written': state_fields},
                    {'role': 'policy_query',
                     'symbol': component.parent_class + '.' + worker.method_name,
                     'state_fields_read': state_fields},
                    {'role': 'dispatcher',
                     'symbol': component.parent_class + '.' + dispatcher.method_name,
                     'calls': [worker.method_name]},
                ],
                'verification': [
                    'Check whether the request gives the target concept independent policy state.',
                    'Check every source input form, including exact and qualified forms, for a target counterpart.',
                    'Map lifecycle roles and repository conventions; do not copy source-specific conditions verbatim.',
                ],
            })
            break
        if len(result) >= limit:
            break
    return result
