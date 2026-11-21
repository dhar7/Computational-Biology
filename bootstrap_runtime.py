"""Re-execute a project script with this repository's virtualenv when present."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prefer_project_venv() -> None:
    project_root = Path(__file__).resolve().parent
    candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    interpreter = next((path for path in candidates if path.is_file()), None)
    if interpreter is None:
        return
    expected_prefix = interpreter.parent.parent if interpreter.parent.name == "bin" else interpreter.parent.parent
    if Path(sys.prefix).resolve() == expected_prefix.resolve():
        return
    script = Path(sys.argv[0]).resolve()
    os.execv(str(interpreter), [str(interpreter), str(script), *sys.argv[1:]])
