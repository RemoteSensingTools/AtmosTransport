# CLAUDE_additions.md — Merged into CLAUDE.md (2026-04-19)

**Status:** retired. Content merged into the top-level
[CLAUDE.md](../../CLAUDE.md) during plan 17 Session 2.

The "Planning discipline", "Testing discipline", "Julia/language
gotchas", "Operator design patterns", "Plan execution rhythm",
"Branch hygiene", and "What NOT to do" sections now live under
corresponding top-level headings in CLAUDE.md:

- **Plan execution rhythm** — commit 0 pattern, rollback discipline,
  measurement as decision gate, survey before greenfield
- **Branch hygiene** — stack plan branches, merge when stable
- **Julia / language gotchas** — FT coercion, kwarg defaults, Adapt
  parametric types, `Ref{Int}` kernel-safety, `Val(:unchecked)`
  Adapt path
- **Testing** (§"Testing discipline") — accessor-API contract,
  default-kwarg bit-exact rule, baseline 77-failure invariant
- **Workflow: Adding a new physics operator** — extended with the
  `No<Operator>` dead-branch pattern, adjoint-preserving solver
  structure, array-level palindrome entry point

## Where new lessons should go

Future plan retrospectives that surface generalizable patterns
should be merged directly into the relevant CLAUDE.md section in
the plan's final commit, alongside the per-plan NOTES.md. Do NOT
re-create a "CLAUDE_additions.md" accumulator — that splits the
source of truth and requires a second merge step.

Plan NOTES.md files remain the per-plan narrative of discovery.
CLAUDE.md carries the distilled guidance that survives to the
next plan.
