# DevTracker Governance

DevTracker is a lightweight governance and “external memory” layer for human–LLM collaboration.

It audits a Git repository, runs a quality suite, and updates a human-owned tracker (CSV) using strict, auditable rules—so automation can write **evidence** without overwriting human **meaning**.

## Why this exists

Agentic systems fail in practice when project truth fragments across Git, issues, chat logs, docs, and spreadsheets. The model then:
- operates on stale/partial state (hallucinated project status), or
- overwrites human semantics (priority, intent, roadmap) because no boundary is enforced.

DevTracker enforces that boundary as a contract.

## Core idea: separate semantics from evidence

- **Human-owned semantics (never edited by DevTracker):**
  purpose, priority, roadmap semantics, business intent, ownership decisions.

- **Automation-owned evidence (auditable fields only):**
  timestamps, audit notes, lifecycle signals, quality/velocity/stability metrics.

This makes the tracker usable as a shared contract between humans and automation.

## What it does

Given:
- a Git repository (this repo or another repo),
- a tracker CSV (source of truth for inventory / roadmap / governance),

DevTracker can:

1) **Sanitize** the tracker  
   - single header (removes duplicated Excel headers),
   - canonical delimiter/encoding (Excel-friendly),
   - schema enforcement (missing columns added safely).

2) **Audit** changes  
   - git diff + status + log,
   - maps changed entities (agents/tools/apps) via path conventions,
   - runs a quality suite (pytest + ruff + mypy).

3) **Propose updates (review-first by design)**  
   - `proposed_updates_core_*.csv` (governance evidence only),
   - `proposed_updates_metrics_*.csv` (computed metrics only).

4) **Apply updates under explicit policy flags**  
   - `--apply` applies only core evidence fields,
   - `--apply-metrics` applies computed metrics fields,
   - backups + append-only journal for reversibility and auditability.

## Outputs

- `artifacts/dev_tracker/*.json` — machine-readable snapshots (API-ready)
- `reports/dev_tracker/dev_tracker_status.md` — human-readable audit report
- `artifacts/dev_tracker/proposed_updates_*.csv` — reviewable proposals (Excel-friendly)

## Quickstart

### Install
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
