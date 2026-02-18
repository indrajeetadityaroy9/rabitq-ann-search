"""Central project path definitions for deterministic path resolution."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def resolve_project_path(path: Path | str) -> Path:
    return (PROJECT_ROOT / Path(path)).resolve()
