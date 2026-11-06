from pathlib import Path

path = Path("local_agent.py")
lines = path.read_text(encoding="utf-8-sig").splitlines()

print("=== local_agent.py: imports ===")
for i, line in enumerate(lines[:35], 1):
    print(f"{i}: {line}")

print("\n=== local_agent.py: bucle del agente y finalizacion ===")
start = max(0, next(
    (i for i, line in enumerate(lines) if "async def run_agent" in line),
    0
) - 3)

for i in range(start, len(lines)):
    print(f"{i + 1}: {lines[i]}")
