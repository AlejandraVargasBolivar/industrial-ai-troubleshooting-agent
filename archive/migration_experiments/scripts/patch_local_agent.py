from pathlib import Path
import shutil

path = Path("local_agent.py")
source = path.read_text(encoding="utf-8-sig")

start_marker = "async def run_agent(question: str) -> None:"
end_marker = "\n\nasync def main() -> None:"

if start_marker not in source or end_marker not in source:
    raise RuntimeError(
        "No se encontraron los límites esperados de run_agent(). "
        "No se ha modificado local_agent.py."
    )

backup = Path("local_agent.py.backup")
if not backup.exists():
    shutil.copy2(path, backup)
    print(f"Copia de seguridad creada: {backup}")
else:
    print(f"Se conserva la copia de seguridad existente: {backup}")

new_function = r'''
async def run_agent(question: str) -> None:
    server_params = StdioServerParameters(
        command=PYTHON_EXE,
        args=[MCP_SERVER],
        cwd=PROJECT_DIR,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            mcp_tools_result = await session.list_tools()
            tools = convert_mcp_tools(mcp_tools_result.tools)

            # The local agent manages the audit lifecycle itself.
            # Keep the audit tools out of the model's tool list to avoid
            # duplicate starts or premature session closure.
            tools = [
                tool for tool in tools
                if tool["function"]["name"] not in {
                    "audit_start_session",
                    "audit_end_session",
                }
            ]

            print(f"\nModelo: {OLLAMA_MODEL}")
            print(f"Herramientas MCP disponibles: {len(mcp_tools_result.tools)}")
            print("\nIniciando auditoría...\n")

            audit_started = False

            try:
                start_result = await session.call_tool(
                    "audit_start_session",
                    arguments={"question": question},
                )

                start_text = "\n".join(
                    item.text
                    for item in start_result.content
                    if hasattr(item, "text")
                )

                if start_result.isError:
                    raise RuntimeError(
                        f"No se pudo iniciar la auditoría: {start_text}"
                    )

                audit_started = True
                print(f"Auditoría iniciada:\n{start_text}\n")

                claude_md_path = os.path.join(PROJECT_DIR, "CLAUDE.md")
                with open(claude_md_path, "r", encoding="utf-8") as file:
                    project_instructions = file.read()

                system_prompt = (
                    project_instructions
                    + "\n\n## Local Ollama Agent\n"
                    + "You are running locally through Ollama instead of Claude Code. "
                    + "Follow the project instructions above. Use the available MCP "
                    + "tools and existing project scripts. Never invent measurements, "
                    + "timestamps, tool results, evidence steps, or document citations. "
                    + "For diagnostic investigations, follow the audited root-cause "
                    + "methodology. Before diagnostic tool calls, log hypotheses using "
                    + "audit_log_reasoning; after calls, log observations and conclusions "
                    + "with evidence_steps. Record rejected hypotheses, distinguish cause "
                    + "from symptom, and provide evidence-based corrective actions. "
                    + "Do not call audit_start_session or audit_end_session; the local "
                    + "agent manages those automatically. For ML Fault Detection and "
                    + "MSPC requests, follow the corresponding project workflows. "
                    + "If a required step cannot be completed, report the limitation."
                )

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]

                async with httpx.AsyncClient() as client:
                    for round_number in range(MAX_TOOL_ROUNDS):
                        message = await ollama_chat(client, messages, tools)
                        messages.append(message)

                        tool_calls = message.get("tool_calls", [])

                        if not tool_calls:
                            print(
                                message.get(
                                    "content",
                                    "The model returned no text.",
                                )
                            )
                            break

                        for tool_call in tool_calls:
                            function = tool_call["function"]
                            tool_name = function["name"]
                            arguments = function.get("arguments", {})

                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)

                            print(
                                f"[Herramienta] {tool_name}"
                                f"({json.dumps(arguments, ensure_ascii=False)})"
                            )

                            try:
                                result = await session.call_tool(
                                    tool_name,
                                    arguments=arguments,
                                )

                                tool_output = "\n".join(
                                    item.text
                                    for item in result.content
                                    if hasattr(item, "text")
                                )

                                if result.isError:
                                    tool_output = (
                                        f"Tool execution error: {tool_output}"
                                    )

                            except Exception as exc:
                                tool_output = f"Tool execution failed: {exc}"

                            messages.append({
                                "role": "tool",
                                "tool_name": tool_name,
                                "content": tool_output,
                            })
                    else:
                        print(
                            "Se alcanzó el límite de llamadas a herramientas. "
                            "La auditoría se cerrará con los resultados disponibles."
                        )

            finally:
                if audit_started:
                    print("\nFinalizando auditoría...")

                    try:
                        end_result = await session.call_tool(
                            "audit_end_session",
                            arguments={},
                        )

                        end_text = "\n".join(
                            item.text
                            for item in end_result.content
                            if hasattr(item, "text")
                        )

                        print("\n=== RESULTADO DEL CIERRE DE AUDITORÍA ===")
                        print(end_text)

                        if end_result.isError:
                            print(
                                "ADVERTENCIA: el servidor MCP informó un error "
                                "al cerrar la auditoría."
                            )

                    except Exception as exc:
                        print(
                            "ERROR al finalizar la auditoría: "
                            f"{exc}"
                        )
                        print(
                            "La sesión puede haber quedado sin finalizar. "
                            "Consulta la carpeta traces antes de repetir "
                            "la investigación."
                        )
'''

updated = (
    source[:source.index(start_marker)]
    + new_function.strip("\n")
    + source[source.index(end_marker):]
)

compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")
print("local_agent.py actualizado correctamente.")
print("La sintaxis de Python es válida.")
print("La copia original se conserva en local_agent.py.backup.")
