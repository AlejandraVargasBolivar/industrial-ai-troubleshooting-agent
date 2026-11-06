# Historical migration experiments

These files are retained for provenance, not used by the local runtime.

- `scripts/`: the original inspection, manual MCP checks and prototype patch scripts.
- `backups/`: the original pre-migration Python backups, preserved without content edits.

Do not run the old patch or audit-closing scripts against the current application.
They assume the former file layout and, in some cases, modify the prototype or attempt
to close an audit from another server process. Use the supported CLI and regression
tests instead.

See [the migration guide](../../docs/MIGRATION_GUIDE.md) for the active file inventory.
