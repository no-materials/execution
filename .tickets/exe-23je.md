---
id: exe-23je
status: open
deps: []
links: []
created: 2026-02-04T17:01:51Z
type: task
priority: 2
assignee: nomaterials
tags: [execution_tape, inputs, design]
---
# Inputs: extern consts + snapshot binding

## Goal
Provide program-level inputs via const pool externs backed by an input_table, with deterministic snapshot/replay and no explicit observe opcode.

## Checklist
- [x] Phase 0: encoding decisions + spec deltas
- [x] Phase 1: program model + serialization (input_table + extern const tag)
- [x] Phase 2: inputs module (resolver, snapshot, canonical encoding)
- [ ] Phase 3: VM integration (bind snapshot; const pool extern resolution)
- [ ] Phase 4: verifier updates (extern const type checks)
- [ ] Phase 5: tests + conformance coverage

## Notes
- Prefer const pool externs over entry-arg-only inputs.
- Extern input consts reference input_table only (no ValueType in const entry).
- No version bump for now; optional section semantics.
- Phase 0 decisions: `input_table` section tag = 9; const pool tag `ExternInput` = 8; input_table entry = `symbol_id + value_type`; snapshot canonical encoding uses const tags and excludes `ExternInput`.
