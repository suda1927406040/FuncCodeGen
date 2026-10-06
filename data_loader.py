"""Load FEA-bench dataset from arrow format."""
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import FEABENCH_DATA_PATH


def load_dataset(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all instances from FEA-bench arrow file."""
    import pyarrow as pa
    path = path or FEABENCH_DATA_PATH
    reader = pa.ipc.open_stream(str(path))
    table = reader.read_all()
    instances = []
    for i in range(table.num_rows):
        row = table.take([i]).to_pydict()
        instance = {k: row[k][0] for k in row}
        instances.append(instance)
    return instances


def load_instance(instance_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load a single instance by instance_id."""
    for inst in load_dataset(path):
        if inst["instance_id"] == instance_id:
            return inst
    return None


def parse_problem_statement(problem_statement: str) -> Dict[str, Any]:
    """Parse problem_statement into structured components.

    Returns:
        {
            "request": str,              # The feature request text
            "definitions": [             # List of new component definitions
                {
                    "file": str,
                    "components": [
                        {
                            "name": str,
                            "signature": str,
                            "type": "class"|"function",
                            "doc": str,
                        }
                    ]
                }
            ],
            "has_extra_changes": bool,   # Whether non-component changes needed
        }
    """
    result = {
        "request": "",
        "definitions": [],
        "has_extra_changes": False,
    }

    # Extract request
    if "<request>" in problem_statement and "</request>" in problem_statement:
        result["request"] = problem_statement.split("<request>")[1].split("</request>")[0].strip()
    elif "<<NEW FEATURE REQUEST>>" in problem_statement:
        # Fallback: grab text between headers
        text = problem_statement.split("<<NEW FEATURE REQUEST>>")[1]
        if "<request>" in text:
            text = text.split("<request>")[1].split("</request>")[0].strip()
        else:
            text = text.split("\n")[0].strip()
        result["request"] = text

    # Extract definitions
    if "<definitions>" in problem_statement and "</definitions>" in problem_statement:
        defs_text = problem_statement.split("<definitions>")[1].split("</definitions>")[0]
        result["definitions"] = _parse_definitions(defs_text)

    # Check for extra changes hint
    result["has_extra_changes"] = "also need to make other code changes" in problem_statement

    return result


def _parse_definitions(text: str) -> List[Dict[str, Any]]:
    """Parse <definitions> block into structured list."""
    definitions = []
    current_file = None
    current_components = []

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("[start of new definitions in"):
            # e.g. [start of new definitions in sqlglot/dialects/druid.py]
            current_file = line.split(" in ")[1].rstrip("]")
            current_components = []
        elif line.startswith("[end of new definitions in"):
            if current_file and current_components:
                definitions.append({
                    "file": current_file,
                    "components": current_components,
                })
            current_file = None
            current_components = []
        elif line.startswith("(definition of"):
            # e.g. (definition of Druid:) or (definition of Generator.ceil_floor:)
            paren_idx = line.index(")")
            def_header = line[1:paren_idx]  # "definition of Druid:"
            name_part = def_header.replace("definition of ", "").rstrip(":")
            # The rest of the line (after the closing paren) is the signature
            signature = line[paren_idx + 1:].strip()
            comp_type = "class" if signature.startswith("class ") else "function"
            current_components.append({
                "name": name_part,
                "signature": signature,
                "type": comp_type,
                "doc": "",
            })

    return definitions


def get_instance_repo_dir(instance_id: str, repo: str) -> str:
    """Get the local directory name for an instance's repo.

    Args:
        instance_id: e.g. 'tobymao__sqlglot-4579'
        repo: e.g. 'tobymao/sqlglot'

    Returns:
        Directory name: 'tobymao__sqlglot-4579'
    """
    return instance_id


def get_github_url(repo: str) -> str:
    """Convert repo slug to GitHub URL."""
    return f"https://github.com/{repo}"
