"""Bootstrap the project-local environment without Claude Code or global packages."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()
    if args.rebuild_index and args.skip_index:
        parser.error("--rebuild-index and --skip-index are mutually exclusive.")
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")
    python = PROJECT_DIR / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not python.is_file():
        print("Creating the project-local virtual environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(PROJECT_DIR / ".venv")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ANONYMIZED_TELEMETRY": "False"}
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements.txt")],
        cwd=PROJECT_DIR, env=env, check=True,
    )
    subprocess.run(
        [str(python), str(PROJECT_DIR / "local_agent.py"), "--check"],
        cwd=PROJECT_DIR, env=env, check=True,
    )
    if not args.skip_index:
        # Inspect the collection itself: an existing folder alone is not a valid index.
        check = subprocess.run(
            [str(python), "-c", (
                "import rag_tools; c, _ = rag_tools._get_rag(); "
                "assert c.count() > 0, 'The document collection is empty'"
            )],
            cwd=PROJECT_DIR, env=env, check=False,
        )
        if args.rebuild_index or check.returncode:
            command = [str(python), str(PROJECT_DIR / "build_docs_db.py")]
            if args.rebuild_index:
                command.append("--rebuild")
            subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)
    print("Setup complete. Start the assistant with the virtual environment's Python and local_agent.py.")


if __name__ == "__main__":
    main()
