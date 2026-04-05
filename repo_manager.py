"""Manage repo checkout and preparation for FEA-bench instances."""
import subprocess
import shutil
import os
from pathlib import Path
from typing import Optional
from config import REPOS_DIR, GIT_CHECKOUT_TIMEOUT


def prepare_repo(instance_id: str, base_commit: str, repo: str,
                 repos_dir: Optional[Path] = None) -> Path:
    """Prepare a repo at the correct base_commit.

    Strategy:
    1. Check if repo already exists in repos_dir
    2. If not, clone from GitHub
    3. Checkout base_commit (force if needed)

    Returns:
        Path to the prepared repo directory.
    """
    repos_dir = repos_dir or REPOS_DIR
    repo_dir = repos_dir / instance_id

    if not repo_dir.exists():
        # Need to clone
        github_url = f"https://github.com/{repo}"
        _clone_repo(github_url, repo_dir)

    # Checkout base_commit
    _force_checkout(repo_dir, base_commit)
    return repo_dir


def _clone_repo(url: str, dest: Path):
    """Clone a repo from GitHub."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", url, str(dest)],
            capture_output=True, text=True, timeout=300,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone {url}: {e.stderr}")


def _force_checkout(repo_dir: Path, commit: str):
    """Force checkout a specific commit, handling dirty state."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def _run(cmd, **kwargs):
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=GIT_CHECKOUT_TIMEOUT,
            cwd=str(repo_dir), env=env, **kwargs,
        )

    # Step 1: Clean any untracked files and reset
    _run(["git", "clean", "-fdq"])
    _run(["git", "reset", "--hard", "HEAD"])

    # Step 2: Try normal checkout first
    result = _run(["git", "checkout", commit])
    if result.returncode == 0:
        return

    # Step 3: If that fails, try fetching the commit
    _run(["git", "fetch", "origin"])
    result = _run(["git", "checkout", commit])
    if result.returncode == 0:
        return

    # Step 4: Force checkout (detached HEAD)
    _run(["git", "clean", "-fdq"])
    _run(["git", "reset", "--hard"])
    result = _run(["git", "checkout", "-f", commit])
    if result.returncode == 0:
        return

    # Step 5: Nuclear option - remove and re-clone
    error_msg = result.stderr
    raise RuntimeError(
        f"Could not checkout {commit} in {repo_dir}: {error_msg}. "
        f"You may need to re-clone the repo."
    )


def apply_patch(repo_dir: Path, patch_text: str) -> bool:
    """Apply a git patch to the repo directory.

    Returns True if successful.
    """
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=patch_text, capture_output=True, text=True,
        cwd=str(repo_dir),
    )
    if result.returncode == 0:
        return True

    # Try with --3way for better conflict resolution
    result = subprocess.run(
        ["git", "apply", "--3way", "--whitespace=nowarn"],
        input=patch_text, capture_output=True, text=True,
        cwd=str(repo_dir),
    )
    return result.returncode == 0


def get_file_content(repo_dir: Path, file_path: str) -> Optional[str]:
    """Read a file from the repo, return None if doesn't exist."""
    full_path = repo_dir / file_path
    if full_path.exists():
        return full_path.read_text(errors="replace")
    return None


def list_files(repo_dir: Path, pattern: str = "**/*.py") -> list:
    """List files matching a glob pattern in the repo."""
    return [str(p.relative_to(repo_dir)) for p in repo_dir.glob(pattern)]
