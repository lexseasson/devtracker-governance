# Architecture

DevTracker is a repo auditor + tracker governance engine. It is designed to produce both:
- human-readable governance reports, and
- machine-readable artifacts suitable for dashboards or API exposure.

## High-level flow

Input:
- a Git repository (current working directory repo),
- a canonical tracker CSV (Excel-friendly source of truth for inventory/governance).

Pipeline:

1) Tracker ingestion and sanitation
2) Git audit (diff/status/log)
3) Quality suite (pytest/ruff/mypy)
4) Change-to-entity mapping (agents/tools/apps by path convention)
5) Update proposal (core vs metrics)
6) Optional controlled apply (hard field ownership rules)
7) Artifact/report emission + append-only journal

## Data model: Tracker as a governance boundary object

The tracker CSV is the central contract. It contains:
- identity and ownership (office_id, agent_name, repo_id, code_path),
- human meaning (purpose, priority, business intent),
- governance evidence (estado_dev, ultima_actualizacion, notas),
- computed operational metrics (quality/confidence/velocity/churn/stability).

DevTracker treats the tracker as an interface between:
- humans (who define intent),
- automation (which maintains evidence),
- and downstream consumers (dashboards, APIs, LLM tools).

## Components

### A) Tracker IO layer
Responsibilities:
- robust CSV read (delimiter detection, encoding fallback),
- schema enforcement,
- canonical write (utf-8-sig + ';' delimiter for Excel).

Guarantees:
- a single header row,
- stable column ordering (standard columns first, extras preserved).

### B) Git audit layer
Signals:
- branch, HEAD, describe,
- merge-base resolution against a base ref,
- changed files from diff and dirty working tree,
- recent log lines.

Output:
- a structured snapshot of “what changed” relative to base.

### C) Entity mapping layer
Heuristic conventions:
- agents/<name>/... -> agent entity
- tools/<name>/... -> tool entity
- apps/<name>/...  -> app entity

This creates a minimal index of impacted entities.

### D) Quality suite layer
Runs:
- pytest
- ruff
- mypy

Produces:
- a boolean all_ok
- a continuous quality_score in [0, 1] as a baseline signal.

### E) Governance policy layer
Generates proposals under explicit rules:
- CORE updates are minimal governance evidence fields.
- METRICS updates are optional computed signals.

A safety gate prevents touching human-owned semantics regardless of proposal content.

### F) Output layer
Artifacts:
- JSON run snapshot (machine-readable)
- Markdown report (human-readable)
- CSV proposed updates (reviewable)
- append-only journal event (audit trail)

## Scaling paths

### 1) Multi-repo governance
Extend repo_id and allow running DevTracker across multiple repos:
- one tracker governs many repos,
- each run updates only entities mapped to that repo_id.

### 2) API mode
Expose the latest snapshot for dashboards and LLM tool calling:
- GET /latest.json
- GET /agents
- GET /agents/{name}
- GET /status

The critical property is read-heavy access: models can read evidence without rewriting meaning.

### 3) CI / scheduled execution
DevTracker can run:
- in GitHub Actions (CI checks + artifact upload),
- as a scheduled job (cron),
- inside Docker for reproducibility.

Recommended stance:
- propose mode by default in CI,
- apply mode only with explicit approvals.
