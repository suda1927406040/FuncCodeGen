"""Prompt templates for the 3-phase LLM generation pipeline.

Phase 1: Understanding (small model) - What does each component do?
Phase 2: Pattern Extraction (medium model) - What patterns to follow?
Phase 3: Code Generation (strong model) - Generate the actual code
"""


def prompt_understand(request: str, formatted_context: str,
                      component_names: list) -> str:
    """Phase 1 prompt: Understand what each component should do.

    Uses small/cheap model. Input: structured context. Output: understanding JSON.
    """
    components_str = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(component_names))

    return f"""You are analyzing a feature request for a Python codebase.

# Feature Request
{request}

# Codebase Context
{formatted_context}

# Components to Implement
{components_str}

# Task
For each component, explain:
1. What it should do (functional requirements)
2. What existing code it should reference/follow
3. What dependencies it has (other components, base class methods, imports)
4. Whether __init__.py or other registration files need modification

Output a JSON object with this exact structure:
{{
  "components": [
    {{
      "name": "component_name",
      "purpose": "what this component does",
      "references": ["list of existing methods/functions to follow"],
      "dependencies": ["other components or imports needed"],
      "init_py_action": "add_import" | "no_change" | "unknown",
      "init_py_detail": "what to add/change in __init__.py if needed",
      "implementation_strategy": "brief description of how to implement"
    }}
  ],
  "extra_changes": [
    {{
      "file": "path/to/file",
      "change": "description of change needed",
      "reason": "why this change is needed"
    }}
  ]
}}

Be precise and specific. Reference actual method names and class names from the context."""


def prompt_extract_patterns(request: str, formatted_context: str,
                             understanding: dict) -> str:
    """Phase 2 prompt: Extract reusable patterns.

    Uses medium model. Input: context + understanding. Output: pattern guide.
    """
    import json
    understanding_str = json.dumps(understanding, indent=2)

    return f"""You are extracting reusable code patterns from a Python codebase for implementing a new feature.

# Feature Request
{request}

# Understanding of Components
{understanding_str}

# Codebase Context
{formatted_context}

# Task
Extract concrete patterns that should be followed when implementing the new components.
For each pattern, provide:
1. The pattern name (e.g., "method registration pattern", "error handling pattern")
2. A code template showing the pattern
3. Which components should use this pattern
4. Specific modifications needed (e.g., adding to __all__, registering in a dict)

Output a JSON object:
{{
  "patterns": [
    {{
      "name": "pattern name",
      "template": "code template as string",
      "applicable_to": ["component names"],
      "notes": "important details"
    }}
  ],
  "style_guide": {{
    "naming": "naming conventions observed",
    "imports": "import style to follow",
    "docstrings": "docstring format to use",
    "type_annotations": "annotation style"
  }},
  "init_py_changes": [
    {{
      "file": "__init__.py path",
      "action": "exact line to add or change",
      "position": "after which existing line/entry"
    }}
  ],
  "other_file_changes": [
    {{
      "file": "file path",
      "action": "what to change",
      "code": "exact code to add"
    }}
  ]
}}"""


def prompt_generate(request: str, formatted_context: str,
                     understanding: dict, patterns: dict,
                     component_names: list,
                     target_files: list) -> str:
    """Phase 3 prompt: Generate actual code.

    Uses strong model. Input: all context. Output: complete implementation.
    """
    import json
    understanding_str = json.dumps(understanding, indent=2)
    patterns_str = json.dumps(patterns, indent=2)

    components_section = "\n".join(f"  - {name}" for name in component_names)
    files_section = "\n".join(f"  - {f}" for f in target_files)

    return f"""You are implementing a new feature in a Python codebase. Generate the complete code changes needed.

# Feature Request
{request}

# Components to Implement
{components_section}

# Target Files
{files_section}

# Understanding
{understanding_str}

# Patterns to Follow
{patterns_str}

# Codebase Context
{formatted_context}

# Task
Generate ALL code changes needed to implement this feature. This includes:
1. New functions/classes to add
2. Modifications to existing files (imports, registrations, __all__, etc.)
3. Any other changes identified in the understanding phase

Output a JSON object with this exact structure:
{{
  "files": [
    {{
      "path": "relative/file/path.py",
      "action": "create" | "modify",
      "description": "what this change does",
      "content": "complete new file content (for 'create' action)",
      "search": "exact string to find (for 'modify' action)",
      "replace": "replacement string (for 'modify' action)"
    }}
  ]
}}

IMPORTANT RULES:
- For 'create' action: provide the complete file content
- For 'modify' action: provide exact search/replace strings that will be applied
  - The 'search' string must match EXACTLY (including whitespace) in the current file
  - Include enough context in 'search' to be unique (3-5 lines)
  - Multiple modify actions for the same file are allowed
- Follow the patterns and style guide exactly
- Include all necessary imports
- Include proper docstrings matching the project style
- Handle __init__.py changes if identified
- Ensure all inter-component dependencies are satisfied
- Do NOT include test code - only implementation code"""
