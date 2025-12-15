from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# =========================
# Tracker contract (v0.5+)
# =========================
# Nota: agregamos columnas "computadas" y de "gobernanza" que NO rompen el uso humano:
# - humano sigue editando propósito/backlog/semántica
# - dev_tracker escribe evidencia/métricas bajo flags explícitos

TRACKER_COLUMNS: List[str] = [
    # Identidad / ownership
    "office_id",
    "office_nombre",
    "agent_name",
    "agent_type",  # executor | development | observer
    "estado",
    "prioridad",
    "layer",
    "rol_play",
    "repo_id",
    # Intención / negocio
    "proposito_tecnico",
    "potencial_comercial",
    # Contratos de IO / data
    "inputs",
    "outputs",
    "pipelines_uso",
    "tablas_db",
    "data_sources",
    "core_metrics",
    # Dev lifecycle
    "estado_dev",
    "ultima_actualizacion",
    "last_git_touch_utc",
    "update_author",  # human | dev_tracker | agent:<name>
    # Evidencia / notas
    "notas",
    "code_path",
    "dev_stream",
    # Métricas (computadas)
    "quality_score",  # 0..1 derivado de pytest/ruff/mypy
    "confidence_score",  # 0..1 (calidad + estabilidad + señales)
    "velocity_7d",  # commits/diff touches últimos 7d (aprox)
    "velocity_30d",
    "churn_30d",  # cambios repetidos sin promoción de estado
    "stability_days",  # días desde último cambio detectado
]

DEFAULT_ARTIFACTS_DIR = Path("artifacts/dev_tracker")
DEFAULT_REPORTS_DIR = Path("reports/dev_tracker")
DEFAULT_JOURNAL = Path("artifacts/system/dev_journal.jsonl")

DEV_STATE_ORDER: List[str] = [
    "planned_not_implemented",
    "idea_stage",
    "stub_prototype",
    "beta_active",
    "needs_refactor",
    "prod_stable",
]

MAX_MD_UPDATES_TABLE = 500  # no escatimamos aquí
MAX_MD_ENTITIES_TABLE = 500

# =========================
# Governance / Authority (v0.7)
# =========================
CORE_MUTABLE_FIELDS = {
    "estado_dev",
    "ultima_actualizacion",
    "notas",
    "update_author",
}

METRICS_MUTABLE_FIELDS = {
    "agent_type",
    "quality_score",
    "confidence_score",
    "velocity_7d",
    "velocity_30d",
    "churn_30d",
    "stability_days",
    "last_git_touch_utc",
}

HUMAN_ONLY_FIELDS = set(TRACKER_COLUMNS) - CORE_MUTABLE_FIELDS - METRICS_MUTABLE_FIELDS

DASHBOARD_KPIS = [
    "confidence_score",
    "quality_score",
    "velocity_7d",
    "velocity_30d",
    "stability_days",
    "estado_dev",
    "agent_type",
]

# =========================
# Modelos
# =========================
@dataclass(frozen=True)
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class TrackerRow:
    raw: dict[str, str]

    @property
    def agent_name(self) -> str:
        return (self.raw.get("agent_name") or "").strip()

    def get(self, key: str) -> str:
        return (self.raw.get(key) or "").strip()

    def set(self, key: str, value: str) -> None:
        self.raw[key] = value


# =========================
# Tiempo / IO
# =========================
def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _now_utc_compact() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _parse_utc_iso(s: str) -> Optional[datetime]:
    s2 = (s or "").strip()
    if not s2:
        return None
    try:
        if s2.endswith("Z"):
            return datetime.fromisoformat(s2.replace("Z", "+00:00"))
        return datetime.fromisoformat(s2)
    except ValueError:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> CmdResult:
    logger.debug("Run: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    return CmdResult(
        cmd=cmd,
        returncode=proc.returncode,
        stdout=(proc.stdout or "").strip(),
        stderr=(proc.stderr or "").strip(),
    )


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


# =========================
# Lectura tracker robusta
# =========================
def _decode_with_fallback(data: bytes) -> Tuple[str, str]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_exc: Optional[UnicodeError] = None
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except UnicodeError as exc:
            last_exc = exc
    raise UnicodeError(f"No se pudo decodificar con {encodings}: {last_exc}")


def _first_non_empty_line(lines: List[str]) -> str:
    for line in lines:
        if line.strip():
            return line
    return ""


def _detect_delimiter(header_line: str) -> str:
    semis = header_line.count(";")
    commas = header_line.count(",")
    return ";" if semis > commas else ","


def _is_header_row(row: dict[str, str], fieldnames: List[str]) -> bool:
    if not fieldnames:
        return False

    office_id_val = (row.get("office_id") or "").strip().lower()
    if office_id_val == "office_id":
        return True

    hits = 0
    for fn in fieldnames:
        v = (row.get(fn) or "").strip()
        if v == fn:
            hits += 1
    return hits >= max(2, int(len(fieldnames) * 0.6))


def _read_tracker_rows(csv_path: Path) -> Tuple[List[TrackerRow], str, str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe tracker CSV: {csv_path}")

    raw_bytes = csv_path.read_bytes()
    text, enc_used = _decode_with_fallback(raw_bytes)

    lines = text.splitlines()
    header_line = _first_non_empty_line(lines)
    delimiter = _detect_delimiter(header_line)

    fh = io.StringIO(text)
    reader = csv.DictReader(fh, delimiter=delimiter)

    if reader.fieldnames is None:
        raise ValueError("Tracker CSV sin header detectable.")

    fieldnames = [fn.strip() for fn in reader.fieldnames]

    required = {"office_id", "agent_name"}
    missing_req = [c for c in required if c not in set(fieldnames)]
    if missing_req:
        msg = (
            "Tracker inválido: faltan columnas mínimas "
            f"{missing_req}. Header={fieldnames}"
        )
        raise ValueError(msg)

    rows: List[TrackerRow] = []
    for raw in reader:
        clean: dict[str, str] = {}
        for k, v in raw.items():
            if k is None:
                continue
            clean[k.strip()] = (v or "").strip()

        if _is_header_row(clean, fieldnames):
            continue

        rows.append(TrackerRow(raw=clean))

    return rows, enc_used, delimiter, fieldnames


def _ensure_tracker_schema(
    rows: Sequence[TrackerRow],
    existing_fieldnames: List[str],
) -> List[str]:
    extras = [c for c in existing_fieldnames if c not in TRACKER_COLUMNS]
    out_fieldnames = TRACKER_COLUMNS + extras

    for r in rows:
        for c in TRACKER_COLUMNS:
            if c not in r.raw:
                r.raw[c] = ""
    return out_fieldnames


def _write_tracker_csv(
    csv_path: Path,
    rows: Sequence[TrackerRow],
    encoding: str,
) -> None:
    """
    Escritura canónica del tracker BASE:

    - Delimitador fijo ';' (Excel-friendly).
    - Encoding fijo 'utf-8-sig' para Excel.
    - Columnas estándar primero; extras preservadas al final.
    - No reintroduce header duplicado: siempre escribe un único header.
    """
    seen_keys: List[str] = []
    for r in rows:
        for k in r.raw.keys():
            if k not in seen_keys:
                seen_keys.append(k)

    extras = [k for k in seen_keys if k not in TRACKER_COLUMNS]
    fieldnames = TRACKER_COLUMNS + extras

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    out_enc = "utf-8-sig"

    with tmp.open("w", encoding=out_enc, newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter=";",
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for r in rows:
            out = {k: r.raw.get(k, "") for k in fieldnames}
            w.writerow(out)

    tmp.replace(csv_path)
    _ = encoding  # reservado para reporting/journal (no usado para escritura canónica)


# =========================
# Git: helpers
# =========================
def _git_ref_exists(root: Path, ref: str) -> bool:
    res = _run_cmd(["git", "rev-parse", "--verify", "--quiet", ref], cwd=root)
    return res.returncode == 0


def _git_head(root: Path) -> str:
    res = _run_cmd(["git", "rev-parse", "HEAD"], cwd=root)
    return res.stdout if res.returncode == 0 else "unknown"


def _git_branch(root: Path) -> str:
    res = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return res.stdout if res.returncode == 0 else "unknown"


def _git_describe(root: Path) -> str:
    res = _run_cmd(["git", "describe", "--tags", "--always", "--dirty"], cwd=root)
    return res.stdout if res.returncode == 0 else "unknown"


def _git_status_porcelain(root: Path) -> List[str]:
    res = _run_cmd(["git", "status", "--porcelain"], cwd=root)
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def _paths_from_porcelain(status_lines: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for line in status_lines:
        if not line:
            continue
        if line.startswith("?? "):
            out.append(Path(line[3:].strip()))
            continue
        if len(line) >= 4:
            out.append(Path(line[3:].strip()))
    return out


def _resolve_base_ref(root: Path, requested: str, fallback_depth: int) -> str:
    if requested and _git_ref_exists(root, requested):
        return requested

    candidates = ["origin/main", "origin/master", "origin/HEAD", "main", "master"]
    for ref in candidates:
        if _git_ref_exists(root, ref):
            return ref

    return f"HEAD~{fallback_depth}"


def _git_merge_base(root: Path, base_ref: str) -> str:
    res = _run_cmd(["git", "merge-base", "HEAD", base_ref], cwd=root)
    return res.stdout if res.returncode == 0 else ""


def _git_changed_files(root: Path, base_ref: str) -> List[Path]:
    spec = f"{base_ref}...HEAD"
    res = _run_cmd(["git", "diff", "--name-only", spec], cwd=root)
    if res.returncode != 0:
        return []
    return [Path(ln.strip()) for ln in res.stdout.splitlines() if ln.strip()]


def _git_diffstat(root: Path, base_ref: str) -> str:
    spec = f"{base_ref}...HEAD"
    res = _run_cmd(["git", "diff", "--stat", spec], cwd=root)
    return res.stdout if res.returncode == 0 else ""


def _git_log_oneline(root: Path, base_ref: str, limit: int) -> List[str]:
    spec = f"{base_ref}..HEAD"
    res = _run_cmd(["git", "log", "--oneline", spec, f"-n{limit}"], cwd=root)
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def _git_last_touch_iso(root: Path, path_prefix: str) -> str:
    """
    Último commit que tocó un path (best-effort).
    Devuelve ISO-Z o "" si no hay.
    """
    pfx = (path_prefix or "").strip()
    if not pfx:
        return ""
    res = _run_cmd(
        ["git", "log", "-1", "--format=%cI", "--", pfx],
        cwd=root,
    )
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def _git_commit_count_since(root: Path, since_days: int, path_prefix: str) -> int:
    pfx = (path_prefix or "").strip()
    if not pfx:
        return 0
    since = f"--since={since_days}.days"
    res = _run_cmd(["git", "rev-list", "--count", since, "HEAD", "--", pfx], cwd=root)
    if res.returncode != 0:
        return 0
    try:
        return int(res.stdout.strip())
    except ValueError:
        return 0


# =========================
# Mapeo cambios a entidades
# =========================
def _infer_agent_from_path(p: Path) -> Optional[str]:
    parts = p.as_posix().split("/")
    if len(parts) >= 2 and parts[0] == "agents":
        return parts[1]
    return None


def _infer_tool_from_path(p: Path) -> Optional[str]:
    parts = p.as_posix().split("/")
    if len(parts) >= 2 and parts[0] == "tools":
        return parts[1]
    return None


def _infer_app_from_path(p: Path) -> Optional[str]:
    parts = p.as_posix().split("/")
    if len(parts) >= 2 and parts[0] == "apps":
        return parts[1]
    return None


def _changed_entities(changed_files: Sequence[Path]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}

    def _add(kind: str, name: str, file_path: Path) -> None:
        key = f"{kind}:{name}"
        if key not in entities:
            entities[key] = {"kind": kind, "name": name, "files": []}
        entities[key]["files"].append(file_path.as_posix())

    for p in changed_files:
        ag = _infer_agent_from_path(p)
        if ag:
            _add("agent", ag, p)
            continue
        tl = _infer_tool_from_path(p)
        if tl:
            _add("tool", tl, p)
            continue
        app = _infer_app_from_path(p)
        if app:
            _add("app", app, p)
            continue

    return entities


# =========================
# Quality suite
# =========================
def _run_quality_suite(root: Path) -> dict[str, Any]:
    pytest_res = _run_cmd([sys.executable, "-m", "pytest", "-q"], cwd=root)
    ruff_res = _run_cmd([sys.executable, "-m", "ruff", "check", "."], cwd=root)
    mypy_res = _run_cmd(
        [sys.executable, "-m", "mypy", "--config-file", "mypy.ini", "."],
        cwd=root,
    )

    all_ok = (
        pytest_res.returncode == 0
        and ruff_res.returncode == 0
        and mypy_res.returncode == 0
    )

    # Score continuo (0..1) para producto: penaliza fallos, no solo booleano
    # (si algún comando no corre -> returncode != 0 -> baja score)
    parts_ok = [
        1.0 if pytest_res.returncode == 0 else 0.0,
        1.0 if ruff_res.returncode == 0 else 0.0,
        1.0 if mypy_res.returncode == 0 else 0.0,
    ]
    quality_score = sum(parts_ok) / 3.0

    return {
        "all_ok": all_ok,
        "quality_score": quality_score,
        "pytest": {
            "cmd": pytest_res.cmd,
            "returncode": pytest_res.returncode,
            "ok": pytest_res.returncode == 0,
        },
        "ruff": {
            "cmd": ruff_res.cmd,
            "returncode": ruff_res.returncode,
            "ok": ruff_res.returncode == 0,
        },
        "mypy": {
            "cmd": mypy_res.cmd,
            "returncode": mypy_res.returncode,
            "ok": mypy_res.returncode == 0,
        },
        "stdout": {
            "pytest": pytest_res.stdout,
            "ruff": ruff_res.stdout,
            "mypy": mypy_res.stdout,
        },
        "stderr": {
            "pytest": pytest_res.stderr,
            "ruff": ruff_res.stderr,
            "mypy": mypy_res.stderr,
        },
    }


# =========================
# Journal: lectura y métricas históricas
# =========================
def _read_jsonl(path: Path, max_lines: int = 50_000) -> List[dict[str, Any]]:
    if not path.exists():
        return []
    out: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out


def _filter_events_by_days(events: Sequence[dict[str, Any]], days: int) -> List[dict[str, Any]]:
    if days <= 0:
        return list(events)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kept: List[dict[str, Any]] = []
    for ev in events:
        ts = _parse_utc_iso(str(ev.get("ts_utc", "")))
        if ts is None:
            continue
        if ts >= cutoff:
            kept.append(ev)
    return kept


def _extract_agent_updates_from_events(
    events: Sequence[dict[str, Any]],
) -> Dict[str, List[dict[str, Any]]]:
    """
    Best-effort: intenta reconstruir actividad por agente desde dev_tracker runs.
    Si en el futuro guardás "proposed" con agent_name, esto queda perfecto.
    Hoy usamos:
      - counts agregados + (si payload guarda proposed) lo usa.
    """
    by_agent: Dict[str, List[dict[str, Any]]] = {}
    for ev in events:
        if str(ev.get("kind", "")) not in {"dev_tracker_run"}:
            continue
        proposed = ev.get("tracker_updates", {}).get("proposed")
        if isinstance(proposed, list):
            for u in proposed:
                name = str(u.get("agent_name", "")).strip()
                if not name:
                    continue
                by_agent.setdefault(name, []).append(u)
    return by_agent


# =========================
# Política de estado / clasificación
# =========================
def _dev_state_rank(state: str) -> int:
    s = (state or "").strip()
    if not s:
        return -1
    try:
        return DEV_STATE_ORDER.index(s)
    except ValueError:
        return 2


def _infer_agent_type(row: TrackerRow) -> str:
    # Heurística simple y explícita: producto debe ser predecible
    estado_dev = row.get("estado_dev")
    rol = (row.get("rol_play") or "").lower()
    name = row.agent_name.lower()

    if "govern" in rol or "audit" in rol or "tracker" in name or "monitor" in name:
        return "observer"
    if estado_dev == "prod_stable":
        return "executor"
    return "development"


def _compute_stability_days(ts_iso: str, last_git_touch_utc: str) -> str:
    now = _parse_utc_iso(ts_iso)
    last = _parse_utc_iso(last_git_touch_utc)
    if now is None or last is None:
        return ""
    delta = now - last
    days = max(0, int(delta.total_seconds() // 86400))
    return str(days)


def _compute_confidence_score(
    quality_score: float,
    estado_dev: str,
    stability_days: Optional[int],
    velocity_30d: Optional[int],
    churn_30d: Optional[float],
) -> float:
    """
    Score 0..1: combina señal de calidad + madurez + estabilidad + costo de churn.
    Diseñado para ser comercial: simple, monotónico, auditable.
    """
    maturity_boost = 0.0
    rank = _dev_state_rank(estado_dev)
    if rank >= _dev_state_rank("prod_stable"):
        maturity_boost = 0.20
    elif rank >= _dev_state_rank("beta_active"):
        maturity_boost = 0.10

    stab = float(stability_days or 0)
    stab_term = min(0.20, stab / 60.0 * 0.20)  # 0..0.20 en 60 días

    vel = float(velocity_30d or 0)
    vel_term = min(0.10, vel / 30.0 * 0.10)  # actividad moderada suma, hiperactividad no

    churn = float(churn_30d or 0.0)
    churn_penalty = min(0.25, churn * 0.25)  # churn 1.0 => -0.25

    base = 0.50 * quality_score + maturity_boost + stab_term + vel_term - churn_penalty
    return max(0.0, min(1.0, base))


# =========================
# Propuestas de update (core + métricas)
# =========================
def _propose_updates(
    tracker_rows: Sequence[TrackerRow],
    entity_changes: dict[str, dict[str, Any]],
    tests_all_ok: bool,
    ts_iso: str,
    author: str,
) -> List[dict[str, Any]]:
    changed_agents = {
        str(e.get("name", "")).strip()
        for e in entity_changes.values()
        if e.get("kind") == "agent"
    }

    updates: List[dict[str, Any]] = []

    for row in tracker_rows:
        name = row.agent_name
        if not name or name not in changed_agents:
            continue

        old_last = row.get("ultima_actualizacion")
        if old_last != ts_iso:
            updates.append(
                {
                    "agent_name": name,
                    "field": "ultima_actualizacion",
                    "old": old_last,
                    "new": ts_iso,
                    "reason": "Agent touched in git diff/status; refresh timestamp.",
                }
            )

        old_notes = row.get("notas")
        touched = entity_changes.get(f"agent:{name}", {}).get("files", [])
        note_append = (
            f"[dev_tracker] touched={len(touched)} files; "
            f"tests_ok={tests_all_ok}; ts={ts_iso}"
        )
        if note_append not in old_notes:
            new_notes = note_append if not old_notes else f"{old_notes} | {note_append}"
            updates.append(
                {
                    "agent_name": name,
                    "field": "notas",
                    "old": old_notes,
                    "new": new_notes,
                    "reason": "Append dev_tracker evidence line.",
                }
            )

        # Estado dev: reglas conservadoras (no downgrade)
        old_dev = row.get("estado_dev")
        new_dev = old_dev

        if not tests_all_ok:
            if _dev_state_rank(old_dev) < _dev_state_rank("needs_refactor"):
                new_dev = "needs_refactor"
        else:
            if _dev_state_rank(old_dev) <= _dev_state_rank("stub_prototype"):
                new_dev = "beta_active"

        if new_dev != old_dev:
            updates.append(
                {
                    "agent_name": name,
                    "field": "estado_dev",
                    "old": old_dev,
                    "new": new_dev,
                    "reason": (
                        "Policy: if tests pass and agent changed -> beta_active; "
                        "if tests fail -> needs_refactor (no downgrade)."
                    ),
                }
            )

        # Autor de update (siempre evidencia)
        old_author = row.get("update_author")
        if old_author != author:
            updates.append(
                {
                    "agent_name": name,
                    "field": "update_author",
                    "old": old_author,
                    "new": author,
                    "reason": "Audit trail: the actor applying core dev_tracker updates.",
                }
            )

    return updates


def _propose_metric_updates(
    root: Path,
    tracker_rows: Sequence[TrackerRow],
    ts_iso: str,
    quality_score: float,
    journal_events: Sequence[dict[str, Any]],
    metrics_days_short: int,
    metrics_days_long: int,
) -> List[dict[str, Any]]:
    """
    Métricas por agente. NO toca backlog/semántica humana.
    Se aplican sólo si el operador pasa --apply-metrics.
    """
    updates: List[dict[str, Any]] = []

    # Best-effort: si en el futuro guardás updates por agente en journal, esto sube precisión.
    recent_events = _filter_events_by_days(journal_events, metrics_days_long)
    by_agent_updates = _extract_agent_updates_from_events(recent_events)

    for row in tracker_rows:
        name = row.agent_name
        if not name:
            continue

        # agent_type
        inferred_type = _infer_agent_type(row)
        if row.get("agent_type") != inferred_type:
            updates.append(
                {
                    "agent_name": name,
                    "field": "agent_type",
                    "old": row.get("agent_type"),
                    "new": inferred_type,
                    "reason": "Heuristic classification: observer/executor/development.",
                }
            )

        # quality_score (global suite; puede derivarse a agente como base)
        qs_str = f"{quality_score:.3f}"
        if row.get("quality_score") != qs_str:
            updates.append(
                {
                    "agent_name": name,
                    "field": "quality_score",
                    "old": row.get("quality_score"),
                    "new": qs_str,
                    "reason": "Quality suite score (repo-level) written as baseline evidence.",
                }
            )

        # last git touch + stability
        code_path = row.get("code_path")
        last_touch = _git_last_touch_iso(root, code_path) if code_path else ""
        if last_touch and row.get("last_git_touch_utc") != last_touch:
            updates.append(
                {
                    "agent_name": name,
                    "field": "last_git_touch_utc",
                    "old": row.get("last_git_touch_utc"),
                    "new": last_touch,
                    "reason": "Derived from git log -1 for code_path.",
                }
            )

        stability_str = _compute_stability_days(ts_iso, last_touch) if last_touch else ""
        if stability_str and row.get("stability_days") != stability_str:
            updates.append(
                {
                    "agent_name": name,
                    "field": "stability_days",
                    "old": row.get("stability_days"),
                    "new": stability_str,
                    "reason": "Days since last git touch for code_path.",
                }
            )

        # velocity (commit count) últimos N días, por code_path
        v7 = _git_commit_count_since(root, metrics_days_short, code_path) if code_path else 0
        v30 = _git_commit_count_since(root, metrics_days_long, code_path) if code_path else 0

        if row.get("velocity_7d") != str(v7):
            updates.append(
                {
                    "agent_name": name,
                    "field": "velocity_7d",
                    "old": row.get("velocity_7d"),
                    "new": str(v7),
                    "reason": "Commits count since window for code_path.",
                }
            )
        if row.get("velocity_30d") != str(v30):
            updates.append(
                {
                    "agent_name": name,
                    "field": "velocity_30d",
                    "old": row.get("velocity_30d"),
                    "new": str(v30),
                    "reason": "Commits count since long window for code_path.",
                }
            )

        # churn_30d: proxy usando journal (si existe per-agent). Si no, fallback 0.
        # Definición proxy: proporción de updates de estado_dev respecto a total updates.
        churn = 0.0
        ulist = by_agent_updates.get(name, [])
        if ulist:
            total = len(ulist)
            state_updates = sum(1 for u in ulist if str(u.get("field")) == "estado_dev")
            # churn alto si hay muchas anotaciones sin progresar, 
            # pero hoy no tenemos "progreso" explícito.
            churn = max(0.0, 1.0 - (state_updates / max(1, total)))

        churn_str = f"{churn:.3f}"
        if row.get("churn_30d") != churn_str:
            updates.append(
                {
                    "agent_name": name,
                    "field": "churn_30d",
                    "old": row.get("churn_30d"),
                    "new": churn_str,
                    "reason": 
                    "Proxy churn from journal per-agent updates (best-effort).",
                }
            )

        # confidence_score
        try:
            stab_int = int(stability_str) if stability_str else None
        except ValueError:
            stab_int = None
        estado_dev = row.get("estado_dev")
        conf = _compute_confidence_score(
            quality_score=quality_score,
            estado_dev=estado_dev,
            stability_days=stab_int,
            velocity_30d=v30,
            churn_30d=churn,
        )
        conf_str = f"{conf:.3f}"
        if row.get("confidence_score") != conf_str:
            updates.append(
                {
                    "agent_name": name,
                    "field": "confidence_score",
                    "old": row.get("confidence_score"),
                    "new": conf_str,
                    "reason": 
                    "Composite confidence from quality+maturity+stability+velocity-churn.",
                }
            )

    return updates


def _apply_updates_in_memory(
    rows: Sequence[TrackerRow],
    updates: Sequence[dict[str, Any]],
) -> int:
    by_agent: dict[str, List[dict[str, Any]]] = {}
    for u in updates:
        field = str(u.get("field"))
        if field not in CORE_MUTABLE_FIELDS and field not in METRICS_MUTABLE_FIELDS:
            continue  # hard safety: never touch human-owned fields
        by_agent.setdefault(str(u["agent_name"]), []).append(u)

    applied = 0
    for r in rows:
        name = r.agent_name
        if not name or name not in by_agent:
            continue
        for u in by_agent[name]:
            r.set(str(u["field"]), str(u["new"]))
            applied += 1
    return applied


# =========================
# Reportes
# =========================
def _write_proposed_updates_csv(path: Path, updates: Sequence[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames = ["agent_name", "field", "old", "new", "reason"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for u in updates:
            w.writerow({k: u.get(k, "") for k in fieldnames})


def _write_md_report(
    path: Path,
    ts_iso: str,
    git_info: dict[str, Any],
    suite: dict[str, Any],
    entity_changes: dict[str, dict[str, Any]],
    updates_core: Sequence[dict[str, Any]],
    updates_metrics: Sequence[dict[str, Any]],
) -> None:
    _ensure_dir(path.parent)

    agents_changed = [e for e in entity_changes.values() if e.get("kind") == "agent"]
    tools_changed = [e for e in entity_changes.values() if e.get("kind") == "tool"]
    apps_changed = [e for e in entity_changes.values() if e.get("kind") == "app"]

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Dev tracker — project follower — {ts_iso}\n\n")

        fh.write("## Git state\n\n")
        fh.write(f"- Branch: `{git_info.get('branch', '')}`\n")
        fh.write(f"- HEAD: `{git_info.get('head', '')}`\n")
        fh.write(f"- Describe: `{git_info.get('describe', '')}`\n")
        fh.write(f"- Base ref: `{git_info.get('base_ref', '')}`\n")
        fh.write(f"- Merge base: `{git_info.get('merge_base', '')}`\n")
        fh.write(f"- Dirty: **{git_info.get('dirty', False)}**\n")
        fh.write(
            f"- Status lines: **{len(git_info.get('status_porcelain', []))}**\n"
        )
        fh.write(
            f"- Diff changed files: **{len(git_info.get('diff_files', []))}**\n"
        )
        fh.write("\n")

        fh.write("## Quality suite\n\n")
        fh.write(f"- All OK: **{suite.get('all_ok', False)}**\n")
        fh.write(f"- Quality score: **{suite.get('quality_score', 0.0):.3f}**\n")
        fh.write(
            "- Pytest rc={p} | Ruff rc={r} | Mypy rc={m}\n\n".format(
                p=suite["pytest"]["returncode"],
                r=suite["ruff"]["returncode"],
                m=suite["mypy"]["returncode"],
            )
        )

        fh.write("## Change map\n\n")
        fh.write(f"- Agents changed: **{len(agents_changed)}**\n")
        fh.write(f"- Tools changed: **{len(tools_changed)}**\n")
        fh.write(f"- Apps changed: **{len(apps_changed)}**\n\n")

        if agents_changed:
            fh.write("### Agents with changes\n\n")
            fh.write("| agent | files |\n")
            fh.write("|---|---:|\n")
            shown = 0
            for e in sorted(agents_changed, key=lambda x: str(x.get("name", ""))):
                fh.write(f"| {e.get('name', '')} | {len(e.get('files', []))} |\n")
                shown += 1
                if shown >= MAX_MD_ENTITIES_TABLE:
                    break
            if len(agents_changed) > MAX_MD_ENTITIES_TABLE:
                fh.write(
                    f"\n_Se truncó la tabla: {len(agents_changed)} agents totales._\n"
                )
            fh.write("\n")

        fh.write("## Proposed updates — core (semántica mínima)\n\n")
        fh.write(f"- Updates proposed (core): **{len(updates_core)}**\n\n")
        if updates_core:
            fh.write("| agent_name | field | old | new |\n")
            fh.write("|---|---|---|---|\n")
            for u in updates_core[:MAX_MD_UPDATES_TABLE]:
                old = str(u.get("old", "")).replace("\n", " ")
                new = str(u.get("new", "")).replace("\n", " ")
                fh.write(
                    f"| {u.get('agent_name', '')} | {u.get('field', '')} | "
                    f"{old} | {new} |\n"
                )
            if len(updates_core) > MAX_MD_UPDATES_TABLE:
                fh.write(
                    f"\n_Se truncó la tabla: {len(updates_core)} updates core._\n"
                )
            fh.write("\n")

        fh.write("## Proposed updates — metrics (producto)\n\n")
        fh.write(f"- Updates proposed (metrics): **{len(updates_metrics)}**\n\n")
        if updates_metrics:
            fh.write("| agent_name | field | old | new |\n")
            fh.write("|---|---|---|---|\n")
            for u in updates_metrics[:MAX_MD_UPDATES_TABLE]:
                old = str(u.get("old", "")).replace("\n", " ")
                new = str(u.get("new", "")).replace("\n", " ")
                fh.write(
                    f"| {u.get('agent_name', '')} | {u.get('field', '')} | "
                    f"{old} | {new} |\n"
                )
            if len(updates_metrics) > MAX_MD_UPDATES_TABLE:
                fh.write(
                    f"\n_Se truncó la tabla: {len(updates_metrics)} updates métricos._\n"
                )


# =========================
# Runner
# =========================
def run(
    tracker_csv: Path,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    journal_path: Path = DEFAULT_JOURNAL,
    base_ref: str = "origin/main",
    log_limit: int = 100,
    fallback_depth: int = 50,
    apply: bool = False,
    apply_metrics: bool = False,
    sanitize_only: bool = False,
    journal_lookback_days: int = 60,
    metrics_days_short: int = 7,
    metrics_days_long: int = 30,
    author: str = "dev_tracker",
) -> dict[str, Path]:
    """
    Dev tracker v0.5+ (musculoso)

    Modos:
    - --sanitize-only: sanea tracker (header único + schema + encoding/delim canónicos)
    - normal: audita git + corre quality suite + propone updates
    - --apply: aplica sólo updates core (estado_dev, ultima_actualizacion, notas, update_author)
    - --apply-metrics: además escribe métricas/agent_type/confidence/etc

    Diseño:
    - Core updates son "mínimos" para no tocar semántica humana.
    - Métricas son opt-in porque pueden ser discutibles en proyectos externos.
    """
    root = _repo_root()
    ts_compact = _now_utc_compact()
    ts_iso = _now_utc_iso()

    tracker_abs = root / tracker_csv
    artifacts_abs = root / artifacts_dir
    reports_abs = root / reports_dir
    journal_abs = root / journal_path

    _ensure_dir(artifacts_abs)
    _ensure_dir(reports_abs)

    rows, enc_used, delim_used, fieldnames_in = _read_tracker_rows(tracker_abs)
    fieldnames_out = _ensure_tracker_schema(rows, fieldnames_in)

    if sanitize_only:
        backup = tracker_abs.with_suffix(tracker_abs.suffix + f".bak_{ts_compact}")
        backup.write_bytes(tracker_abs.read_bytes())

        _write_tracker_csv(tracker_abs, rows, encoding=enc_used)

        _append_jsonl(
            journal_abs,
            {
                "ts_utc": ts_iso,
                "kind": "dev_tracker_sanitize_only",
                "schema": "dev_tracker.v0.5",
                "tracker": str(tracker_csv.as_posix()),
                "backup": str(backup.relative_to(root).as_posix()),
                "rows": len(rows),
                "encoding_in": enc_used,
                "delimiter_in": delim_used,
                "encoding_out": "utf-8-sig",
                "delimiter_out": ";",
            },
        )
        logger.info(
            "OK: sanitized tracker and wrote %s (backup %s)",
            tracker_abs,
            backup,
        )
        return {"tracker": tracker_abs}

    # Git
    branch = _git_branch(root)
    head = _git_head(root)
    describe = _git_describe(root)
    resolved_base = _resolve_base_ref(root, base_ref, fallback_depth=fallback_depth)
    merge_base = _git_merge_base(root, resolved_base)

    status_lines = _git_status_porcelain(root)
    dirty = bool(status_lines)

    diff_files = _git_changed_files(root, resolved_base)
    diffstat = _git_diffstat(root, resolved_base)
    log_lines = _git_log_oneline(root, resolved_base, limit=log_limit)

    status_files = _paths_from_porcelain(status_lines)
    all_changed = list({p.as_posix(): p for p in (diff_files + status_files)}.values())
    entity_changes = _changed_entities(all_changed)

    # Quality
    suite = _run_quality_suite(root)
    tests_all_ok = bool(suite.get("all_ok", False))
    quality_score = float(suite.get("quality_score", 0.0))

    # Journal events (para métricas históricas)
    journal_events = _read_jsonl(journal_abs)
    journal_events = _filter_events_by_days(journal_events, journal_lookback_days)

    # Core proposals
    updates_core = _propose_updates(
        tracker_rows=rows,
        entity_changes=entity_changes,
        tests_all_ok=tests_all_ok,
        ts_iso=ts_iso,
        author=author,
    )

    # Metrics proposals (opt-in)
    updates_metrics = _propose_metric_updates(
        root=root,
        tracker_rows=rows,
        ts_iso=ts_iso,
        quality_score=quality_score,
        journal_events=journal_events,
        metrics_days_short=metrics_days_short,
        metrics_days_long=metrics_days_long,
    )

    # Apply policy:
    # - --apply aplica core
    # - --apply-metrics aplica métricas (además o solo si también hay core)
    applied_count_core = 0
    applied_count_metrics = 0
    backup_path: Optional[Path] = None

    if (apply and updates_core) or (apply_metrics and updates_metrics):
        backup_path = tracker_abs.with_suffix(tracker_abs.suffix + f".bak_{ts_compact}")
        backup_path.write_bytes(tracker_abs.read_bytes())

        if apply and updates_core:
            applied_count_core = _apply_updates_in_memory(rows, updates_core)

        if apply_metrics and updates_metrics:
            applied_count_metrics = _apply_updates_in_memory(rows, updates_metrics)

        _write_tracker_csv(tracker_abs, rows, encoding=enc_used)

    # Outputs
    json_path = artifacts_abs / f"dev_tracker_{ts_compact}.json"
    proposed_core_csv = artifacts_abs / f"proposed_updates_core_{ts_compact}.csv"
    proposed_metrics_csv = artifacts_abs / f"proposed_updates_metrics_{ts_compact}.csv"
    md_path = reports_abs / "dev_tracker_status.md"

    agents_changed = sum(1 for e in entity_changes.values() if e.get("kind") == "agent")
    tools_changed = sum(1 for e in entity_changes.values() if e.get("kind") == "tool")
    apps_changed = sum(1 for e in entity_changes.values() if e.get("kind") == "app")

    payload: dict[str, Any] = {
        "generated_utc": ts_iso,
        "schema": "dev_tracker.v0.5",
        "tracker": {
            "path": str(tracker_csv.as_posix()),
            "encoding_in": enc_used,
            "delimiter_in": delim_used,
            "rows": len(rows),
            "columns_out": fieldnames_out,
            "encoding_out": "utf-8-sig",
            "delimiter_out": ";",
            "repo_id": root.name,
        },
        "git": {
            "branch": branch,
            "head": head,
            "describe": describe,
            "requested_base_ref": base_ref,
            "base_ref": resolved_base,
            "merge_base": merge_base,
            "dirty": dirty,
            "status_porcelain": status_lines,
            "diff_files": [p.as_posix() for p in diff_files],
            "status_files": [p.as_posix() for p in status_files],
            "all_changed_files": [p.as_posix() for p in all_changed],
            "diffstat": diffstat,
            "log_oneline": log_lines,
        },
        "quality_suite": suite,
        "changes": {
            "entities": entity_changes,
            "counts": {
                "agents_changed": agents_changed,
                "tools_changed": tools_changed,
                "apps_changed": apps_changed,
            },
        },
        "policy": {
            "core_updates_apply": apply,
            "metrics_updates_apply": apply_metrics,
            "author": author,
            "windows": {
                "journal_lookback_days": journal_lookback_days,
                "metrics_days_short": metrics_days_short,
                "metrics_days_long": metrics_days_long,
            },
        },
        "tracker_updates": {
            "proposed_core": updates_core,
            "proposed_metrics": updates_metrics,
            "apply_requested": apply,
            "apply_metrics_requested": apply_metrics,
            "applied_core_count": applied_count_core,
            "applied_metrics_count": applied_count_metrics,
            "backup_path": (
                str(backup_path.relative_to(root).as_posix()) if backup_path else ""
            ),
        },
        "outputs": {
            "json": str(json_path.relative_to(root).as_posix()),
            "md": str(md_path.relative_to(root).as_posix()),
            "proposed_updates_core_csv": str(proposed_core_csv.relative_to(root).as_posix()),
            "proposed_updates_metrics_csv": str(
                proposed_metrics_csv.relative_to(root).as_posix()
            ),
        },
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # =========================
    # v0.8 API-ready snapshot
    # =========================
    api_snapshot = artifacts_abs / "latest.json"
    api_snapshot.write_text(
        json.dumps(
            {
                "generated_utc": ts_iso,
                "schema": "dev_tracker.v0.8",
                "repo_id": root.name,
                "agents": [
                    r.raw for r in rows
                ],
                "dashboard_kpis": DASHBOARD_KPIS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_proposed_updates_csv(proposed_core_csv, updates_core)
    _write_proposed_updates_csv(proposed_metrics_csv, updates_metrics)
    _write_md_report(
        md_path,
        ts_iso=ts_iso,
        git_info={
            "branch": branch,
            "head": head,
            "describe": describe,
            "base_ref": resolved_base,
            "merge_base": merge_base,
            "dirty": dirty,
            "status_porcelain": status_lines,
            "diff_files": [p.as_posix() for p in diff_files],
        },
        suite=suite,
        entity_changes=entity_changes,
        updates_core=updates_core,
        updates_metrics=updates_metrics,
    )

    # Journal (append-only)
    journal_event: dict[str, Any] = {
        "ts_utc": ts_iso,
        "kind": "dev_tracker_run",
        "schema": "dev_tracker.v0.5",
        "git": {
            "branch": branch,
            "head": head,
            "describe": describe,
            "base_ref": resolved_base,
            "merge_base": merge_base,
            "dirty": dirty,
        },
        "counts": {
            "tracker_rows": len(rows),
            "status_lines": len(status_lines),
            "diff_files": len(diff_files),
            "agents_changed": agents_changed,
            "tools_changed": tools_changed,
            "apps_changed": apps_changed,
            "proposed_core_updates": len(updates_core),
            "proposed_metrics_updates": len(updates_metrics),
            "applied_core_updates": applied_count_core,
            "applied_metrics_updates": applied_count_metrics,
        },
        "quality": {
            "all_ok": tests_all_ok,
            "quality_score": quality_score,
        },
        "out": {
            "json": str(json_path.relative_to(root).as_posix()),
            "md": str(md_path.relative_to(root).as_posix()),
            "proposed_core": str(proposed_core_csv.relative_to(root).as_posix()),
            "proposed_metrics": str(proposed_metrics_csv.relative_to(root).as_posix()),
            "backup": str(backup_path.relative_to(root).as_posix()) if backup_path else "",
        },
    }
    _append_jsonl(journal_abs, journal_event)

    logger.info("OK: wrote %s", json_path)
    logger.info("OK: wrote %s", proposed_core_csv)
    logger.info("OK: wrote %s", proposed_metrics_csv)
    logger.info("OK: wrote %s", md_path)

    if backup_path:
        logger.info("OK: backup %s", backup_path)
    if apply and updates_core:
        logger.info("OK: applied core=%d to %s", applied_count_core, tracker_abs)
    if apply_metrics and updates_metrics:
        logger.info("OK: applied metrics=%d to %s", applied_count_metrics, tracker_abs)

    return {
        "json": json_path,
        "md": md_path,
        "proposed_updates_core_csv": proposed_core_csv,
        "proposed_updates_metrics_csv": proposed_metrics_csv,
    }


# =========================
# CLI
# =========================
def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "dev_tracker — project follower (git auditor + tracker sanitizer/updater + metrics)"
        )
    )
    ap.add_argument(
        "--tracker",
        default="docs/AGENTES_TRACKER_BASE.csv",
        help="Tracker BASE (maestro)",
    )
    ap.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    ap.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    ap.add_argument("--journal", default=str(DEFAULT_JOURNAL))

    ap.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref base for diff/log (auto-fallback if missing)",
    )
    ap.add_argument(
        "--log-limit",
        type=int,
        default=100,
        help="git log --oneline limit from base_ref..HEAD",
    )
    ap.add_argument(
        "--fallback-depth",
        type=int,
        default=50,
        help="If no base ref exists, use HEAD~N",
    )

    ap.add_argument(
        "--apply",
        action="store_true",
        help="Apply CORE updates (estado_dev/ultima_actualizacion/notas/update_author).",
    )
    ap.add_argument(
        "--apply-metrics",
        action="store_true",
        help="Apply METRICS updates (agent_type, confidence, velocity, churn, etc).",
    )
    ap.add_argument(
        "--sanitize-only",
        action="store_true",
        help="Only sanitize tracker: remove dup headers + enforce schema (creates backup).",
    )

    ap.add_argument(
        "--journal-lookback-days",
        type=int,
        default=60,
        help="Days of journal history to consider for metrics (best-effort).",
    )
    ap.add_argument(
        "--metrics-days-short",
        type=int,
        default=7,
        help="Short window for velocity metrics.",
    )
    ap.add_argument(
        "--metrics-days-long",
        type=int,
        default=30,
        help="Long window for velocity/churn metrics.",
    )
    ap.add_argument(
        "--author",
        default="dev_tracker",
        help="Actor name to stamp in update_author when applying core updates.",
    )

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    run(
        tracker_csv=Path(args.tracker),
        artifacts_dir=Path(args.artifacts_dir),
        reports_dir=Path(args.reports_dir),
        journal_path=Path(args.journal),
        base_ref=args.base_ref,
        log_limit=args.log_limit,
        fallback_depth=args.fallback_depth,
        apply=bool(args.apply),
        apply_metrics=bool(args.apply_metrics),
        sanitize_only=bool(args.sanitize_only),
        journal_lookback_days=args.journal_lookback_days,
        metrics_days_short=args.metrics_days_short,
        metrics_days_long=args.metrics_days_long,
        author=str(args.author),

       
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
