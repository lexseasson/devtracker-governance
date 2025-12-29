# DevTracker Governance

DevTracker is a lightweight governance and “external memory” layer for human–LLM collaboration.
“This repo documents a case where governance, not prompting, was the main scaling bottleneck.”
It audits a Git repository, runs a quality suite, and updates a human-owned tracker (CSV) using strict, auditable rules—so automation can write **evidence** without overwriting human **meaning**.

## Why this exists

Agentic systems fail in practice when project truth fragments across Git, issues, chat logs, docs, and spreadsheets. 
The model then:
- operates on stale/partial state (hallucinated project status), or
- overwrites human semantics (priority, intent, roadmap) because no boundary is enforced.
The cost is not hallucinations.
The cost is senior time lost to re-explaining, re-validating, and re-aligning work that already happened.

## Who this is for

DevTracker is designed for teams and leaders who are already building agentic systems and need to answer, consistently:

- What actually progressed this week?
- Which decisions were made, and why?
- What evidence supports our current state?
- Where are we losing senior engineering time?

Typical readers:
- CTOs / Heads of Engineering scaling AI systems
- AI Directors responsible for agentic delivery
- Engineering Managers dealing with invisible rework


DevTracker enforces that boundary as a contract.

## Core idea: separate semantics from evidence

- **Human-owned semantics (never edited by DevTracker):**
  purpose, priority, roadmap semantics, business intent, ownership decisions.

- **Automation-owned evidence (auditable fields only):**
  timestamps, audit notes, lifecycle signals, quality/velocity/stability metrics.

This makes the tracker usable as a shared contract between humans and automation.

“This repository documents an approach to governed human–LLM collaboration.
It is intentionally opinionated and not intended as a plug-and-play framework.”

## The failure mode it addresses

Without explicit governance, human–LLM collaboration fails in a predictable way:
activity increases, but explainable progress collapses.

## Operating pattern (not just tooling)

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

Why this repository is public

This repository is not meant to be forked and customized casually.

It documents a real operating pattern for governed human–LLM collaboration, built to expose where agentic systems actually fail in production.

The value is not in the code alone, but in the architectural boundaries and governance decisions it makes explicit.

## Quickstart

### Install
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt



