from pathlib import Path

root = Path.cwd()

files = [
    "audit_trace.py",
    "render_audit.py",
    "historian_mcp_server.py",
    ".claude/commands/audit-trail.md",
    "how_it_works.md",
]

terms = (
    "audit_start_session",
    "audit_end_session",
    "def start_session",
    "def end_session",
    "def log_tool",
    "def log_",
    "def finalize",
    "def render",
    "active_session",
    "current_session",
    "trace.jsonl",
    "audit_end_session",
    "render_audit.py",
)

for relative in files:
    path = root / relative
    if not path.exists():
        print(f"\n### NO EXISTE: {relative}")
        continue

    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    print(f"\n{'=' * 75}\n### {relative} ({len(lines)} lineas)")

    selected = set()
    for index, line in enumerate(lines):
        if any(term.lower() in line.lower() for term in terms):
            selected.update(range(max(0, index - 4), min(len(lines), index + 9)))

    if not selected:
        print("\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[:100])))
        continue

    previous = -2
    for index in sorted(selected):
        if index > previous + 1:
            print("...")
        print(f"{index + 1}: {lines[index]}")
        previous = index
