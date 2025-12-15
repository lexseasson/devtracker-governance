# Governance Model

DevTracker is a governance layer for human–LLM collaboration. Its primary goal is to prevent a common failure mode in agentic workflows: semantic drift caused by fragmented sources of truth and uncontrolled automation edits.

This document defines the governance contract and its safety properties.

## Problem: semantic drift in agentic systems

In real projects, project state is distributed across:
- Git (what changed),
- docs (what was intended),
- chats (why it changed),
- trackers (what is owned and prioritized).

LLMs and agents can operate with partial visibility, generating incorrect state assumptions. The worst-case outcome is not “wrong answers”, but unauthorized edits to human-owned semantics (priority, roadmap, intent), which corrupts the system’s decision boundary.

## Core thesis

DevTracker separates **meaning** from **evidence**:

- **Meaning** is human-owned and should remain stable unless explicitly changed by humans.
- **Evidence** is machine-owned and can be derived from observable signals (Git activity, tests, linting, type checks).

This separation turns the tracker into a governance boundary object: humans can coordinate and approve meaning, while automation keeps evidence fresh and auditable.

## Field ownership and authority model

DevTracker defines three categories of fields:

### 1) Human-owned fields (immutable by automation)

These include purpose, commercial potential, inputs/outputs definitions, roadmap semantics, and any domain reasoning. Automation MUST NOT modify these fields.

Rationale:
- These fields encode intent, not observation.
- Allowing automation to edit intent introduces silent, compounding errors.

### 2) Core mutable fields (governance evidence)

These fields may be updated by DevTracker under explicit flags:
- estado_dev
- ultima_actualizacion
- notas
- update_author

Rationale:
- They represent lifecycle evidence and audit annotations.
- They help keep the tracker operationally current without overwriting meaning.

### 3) Metrics mutable fields (computed signals, opt-in)

These include:
- quality_score
- confidence_score
- velocity_7d / velocity_30d
- churn_30d
- stability_days
- last_git_touch_utc
- agent_type

Rationale:
- Metrics are useful for dashboards and prioritization, but can be debated.
- Therefore they are strictly opt-in and must be attributable.

## Safety invariants (what must always be true)

1) **No silent semantic edits**
   - Automation cannot write into human-owned fields.
2) **Explicit application**
   - Updates are proposed first, then applied only with explicit flags.
3) **Reversibility**
   - Every application creates a backup and an append-only journal entry.
4) **Attribution**
   - update_author indicates who/what applied changes.
5) **Determinism of computed outputs**
   - Given the same Git state + tracker input, DevTracker should produce consistent proposals.

## Review workflow (recommended)

1) Run DevTracker in propose mode:
   - generates proposed_updates_core.csv and proposed_updates_metrics.csv
2) Human reviews proposals (Excel-friendly).
3) Apply core updates when accepted.
4) Apply metrics updates if desired (opt-in).
5) Commit tracker changes if the tracker is kept in version control.

## Operational stance

DevTracker is intentionally conservative:
- It prefers not to update rather than risk corrupting meaning.
- It uses narrow, auditable heuristics instead of opaque inference for governance-critical fields.
