#!/usr/bin/env -S uv run -s
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27",
#   "typer>=0.12",
#   "structlog>=24.1",
# ]
# ///
"""
Cron-friendly GitHub→Todoist sync using gh(1) + Todoist REST v2.

Env:
  TODOIST_API_TOKEN  (required)
  GH_PATH            (optional, default: 'gh')

Example cron (runs at :15 past the hour):
  15 * * * * /path/to/gh_todoist_sync.py --project "GitHub"
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import httpx
import logging
import structlog
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = structlog.get_logger("gh_todoist_sync")

# ---------- config / io ----------

def _default_db_path() -> Path:
    xdg_state = os.environ.get("XDG_STATE_HOME") or os.path.join(Path.home(), ".local", "state")
    return Path(xdg_state) / "gh_todoist_sync.db"

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

# ---------- sqlite ----------

DDL = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS topics (
  gh_id TEXT PRIMARY KEY,
  gh_url TEXT NOT NULL UNIQUE,
  repo  TEXT NOT NULL,
  number INTEGER NOT NULL,
  is_pr INTEGER NOT NULL,
  todoist_id TEXT,
  gh_updated_at TEXT NOT NULL,
  last_synced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

def db_open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON;")
    con.executescript(DDL)
    return con

def db_get(con: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    cur = con.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default

def db_set(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    con.commit()

def db_upsert_topic(
    con: sqlite3.Connection, *,
    gh_id: str, gh_url: str, repo: str, number: int, is_pr: bool,
    todoist_id: Optional[str], gh_updated_at: str
) -> None:
    con.execute(
        """
        INSERT INTO topics(gh_id, gh_url, repo, number, is_pr, todoist_id, gh_updated_at, last_synced_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(gh_id) DO UPDATE SET
          gh_url=excluded.gh_url,
          repo=excluded.repo,
          number=excluded.number,
          is_pr=excluded.is_pr,
          todoist_id=COALESCE(excluded.todoist_id, topics.todoist_id),
          gh_updated_at=excluded.gh_updated_at,
          last_synced_at=excluded.last_synced_at
        """,
        (gh_id, gh_url, repo, number, int(is_pr), todoist_id, gh_updated_at, _now_iso()),
    )
    con.commit()

def db_get_topic_by_gh(con: sqlite3.Connection, gh_id: str) -> Optional[tuple[str, str]]:
    cur = con.execute("SELECT todoist_id, gh_updated_at FROM topics WHERE gh_id=?", (gh_id,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None

def db_set_todoist_id(con: sqlite3.Connection, gh_id: str, todoist_id: str) -> None:
    con.execute("UPDATE topics SET todoist_id=?, last_synced_at=? WHERE gh_id=?", (todoist_id, _now_iso(), gh_id))
    con.commit()

# ---------- models ----------

@dc.dataclass
class Topic:
    gh_id: str
    url: str
    repo: str
    number: int
    title: str
    body: str
    labels: list[str]
    assignees: list[str]
    state: Literal["OPEN", "CLOSED"]
    state_reason: Optional[str]  # issues only; "completed" | "not_planned" | None
    updated_at: datetime
    closed_at: Optional[datetime]
    is_pr: bool
    merged: Optional[bool] = None  # PRs only, lazy-populated

# ---------- GitHub via gh(1) ----------

def gh_cmd(*args: str, gh_path: Optional[str] = None, timeout: int = 120) -> str:
    exe = gh_path or os.environ.get("GH_PATH") or "gh"
    cmd = [exe, *args]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"gh failed: {shlex.join(cmd)} :: {p.stderr.strip()}")
    return p.stdout

def gh_search_assigned(since: Optional[datetime], gh_path: Optional[str]) -> list[Topic]:
    # One call for open assigned; one call for recently-updated (incl. closed). Keeps runtime low yet robust.
    fields = ",".join([
        "id","number","title","body","labels","assignees","state","updatedAt","closedAt","isPullRequest","url","repository"
    ])
    items: list[dict[str, Any]] = []

    # Always fetch open assigned across repos.
    out_open = gh_cmd(
        "search","issues",
        "--assignee","@me",
        "--state","open",
        "--json", fields,
        "--limit","1000",
        gh_path=gh_path
    )
    items.extend(json.loads(out_open))

    # Also fetch recently updated (captures closures/renames/label changes).
    if since is None:
        since = datetime.now(UTC) - timedelta(days=30)
    since_q = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_recent = gh_cmd(
        "search","issues",
        f"updated:>={since_q} assignee:@me",
        "--assignee","@me",
        "--json", fields,
        "--limit","1000",
        gh_path=gh_path
    )
    items.extend(json.loads(out_recent))

    topics: dict[str, Topic] = {}
    for it in items:
        labels = [lbl["name"] if isinstance(lbl, dict) and "name" in lbl else str(lbl) for lbl in (it.get("labels") or [])]
        assignees = [a.get("login","") for a in (it.get("assignees") or [])]
        repo = (it.get("repository") or {}).get("nameWithOwner") or (it.get("repository") or {}).get("fullName") or "?"
        t = Topic(
            gh_id=str(it["id"]),
            url=it["url"],
            repo=repo,
            number=int(it["number"]),
            title=it.get("title",""),
            body=(it.get("body") or "")[:2000],
            labels=labels,
            assignees=assignees,
            state=(it.get("state","OPEN") or "OPEN").upper(),  # OPEN|CLOSED
            state_reason=(it.get("stateReason") or None),
            updated_at=_parse_dt(it.get("updatedAt")),
            closed_at=_parse_dt(it.get("closedAt")),
            is_pr=bool(it.get("isPullRequest")),
        )
        topics[t.gh_id] = t  # de-dup by gh_id
    return list(topics.values())

def _parse_dt(x: Optional[str]) -> Optional[datetime]:
    if not x:
        return None
    try:
        return datetime.fromisoformat(x.replace("Z","+00:00")).astimezone(UTC)
    except Exception:
        return None

def gh_pr_merged(url: str, gh_path: Optional[str]) -> Optional[bool]:
    try:
        out = gh_cmd("pr","view", url, "--json","merged", gh_path=gh_path, timeout=60)
        data = json.loads(out)
        return bool(data.get("merged"))
    except Exception as e:
        log.warning("gh_pr_merged_failed", url=url, error=str(e))
        return None

# ---------- Todoist REST v2 ----------

class Todoist:
    def __init__(self, token: str, base_url: str = "https://api.todoist.com/rest/v2"):
        self.base = base_url.rstrip("/")
        self.h = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
        self.cli = httpx.Client(timeout=30)

    def _req(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[dict[str, str]] = None,
        **kw,
    ) -> httpx.Response:
        url = f"{self.base}{path}"
        hdr = {**self.h}
        if headers:
            hdr.update(headers)
        return self.cli.request(method, url, headers=hdr, **kw)

    def get_labels(self) -> dict[str,str]:
        r = self._req("GET","/labels")
        r.raise_for_status()
        return {lbl["name"]: lbl["id"] for lbl in r.json()}

    def ensure_labels(self, names: Iterable[str]) -> list[str]:
        # Todoist accepts label names directly on tasks; still pre-create to avoid 404s.
        existing = self.get_labels()
        created = []
        for name in sorted(set(n for n in names if n)):
            if name not in existing:
                rr = self._req("POST","/labels", json={"name": name})
                if rr.status_code in (200, 201):
                    lid = rr.json()["id"]
                    existing[name] = lid
                    created.append(name)
                else:
                    log.warning("label_create_failed", label=name, status=rr.status_code, body=rr.text)
        if created:
            log.info("labels_created", count=len(created), labels=created)
        return list(set(names))

    def get_projects(self) -> dict[str,str]:
        r = self._req("GET","/projects")
        r.raise_for_status()
        return {p["name"]: p["id"] for p in r.json()}

    def resolve_project_id(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return self.get_projects().get(name)

    def create_task(self, *, content: str, description: str, labels: list[str], project_id: Optional[str], xreq: str) -> Optional[str]:
        body = {"content": content, "description": description}
        if labels:
            body["labels"] = labels
        if project_id:
            body["project_id"] = project_id
        r = self._req("POST","/tasks", json=body, headers={"X-Request-Id": xreq})
        if r.status_code in (200, 201):
            return r.json()["id"]
        log.error("task_create_failed", status=r.status_code, body=r.text)
        return None

    def update_task(self, task_id: str, *, content: Optional[str]=None, description: Optional[str]=None, labels: Optional[list[str]]=None) -> bool:
        body = {}
        if content is not None:
            body["content"] = content
        if description is not None:
            body["description"] = description
        if labels is not None:
            body["labels"] = labels
        if not body:
            return True
        r = self._req("POST", f"/tasks/{task_id}", json=body)
        if r.status_code in (200, 204):
            return True
        log.error("task_update_failed", task_id=task_id, status=r.status_code, body=r.text)
        return False

    def close_task(self, task_id: str) -> bool:
        r = self._req("POST", f"/tasks/{task_id}/close")
        if r.status_code in (200, 204):
            return True
        log.error("task_close_failed", task_id=task_id, status=r.status_code, body=r.text)
        return False

    def reopen_task(self, task_id: str) -> bool:
        r = self._req("POST", f"/tasks/{task_id}/reopen")
        if r.status_code in (200, 204):
            return True
        log.error("task_reopen_failed", task_id=task_id, status=r.status_code, body=r.text)
        return False

    def delete_task(self, task_id: str) -> bool:
        r = self._req("DELETE", f"/tasks/{task_id}")
        if r.status_code in (200, 204):
            return True
        log.error("task_delete_failed", task_id=task_id, status=r.status_code, body=r.text)
        return False

# ---------- sync logic ----------

def make_content(t: Topic) -> str:
    kind = "PR" if t.is_pr else "Issue"
    return f"[{kind}] {t.repo}#{t.number}: {t.title}".strip()

def make_description(t: Topic) -> str:
    header = f"{t.url}\n"
    body = (t.body or "").strip()
    if body:
        # Keep desc succinct for Todoist UI
        body = body if len(body) <= 1000 else body[:1000] + "…"
        return header + "\n" + body
    return header

WONTFIX_LABELS = {"wontfix", "won't fix", "not planned", "invalid"}

def disposition(t: Topic, gh_path: Optional[str]) -> Literal["open","complete","delete"]:
    if t.state.upper() != "CLOSED":
        return "open"
    # Issues: use state_reason if present.
    if not t.is_pr and t.state_reason:
        sr = t.state_reason.lower()
        if sr in ("completed","fixed","done"):
            return "complete"
        if sr in ("not_planned","not planned","wontfix"):
            return "delete"
    # PRs: treat merged as "complete", closed-unmerged as "complete" unless labeled wontfix-ish.
    if t.is_pr:
        if t.merged is None:
            t.merged = gh_pr_merged(t.url, gh_path)
        if t.merged:
            return "complete"
        if {lbl.lower() for lbl in t.labels} & WONTFIX_LABELS:
            return "delete"
        return "complete"
    # Fallback: complete.
    return "complete"

def idempotency_key(t: Topic) -> str:
    return hashlib.sha1(t.url.encode()).hexdigest()

def should_update(local_updated: Optional[str], remote_updated: datetime) -> bool:
    if local_updated is None:
        return True
    try:
        lu = datetime.fromisoformat(local_updated)
    except Exception:
        return True
    return remote_updated > lu

# ---------- logging setup ----------

def configure_logging(verbosity: int) -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            {0: logging.INFO, 1: logging.DEBUG, 2: getattr(structlog, "TRACE", logging.DEBUG)}.get(verbosity, logging.INFO)
        ),
    )

# ---------- CLI ----------

@app.command()
def sync(
    project: Optional[str] = typer.Option(None, "--project", help="Todoist project name (default: Inbox)"),
    db_path: Path = typer.Option(_default_db_path(), "--db", help="SQLite state path"),
    gh: Optional[str] = typer.Option(None, "--gh", help="Path to gh(1); defaults to $GH_PATH or 'gh'"),
    dry_run: bool = typer.Option(False, "--dry-run", help="No writes to Todoist / DB"),
    verbose: int = typer.Option(0, "-v", count=True, help="-v or -vv for more logs"),
):
    """Sync assigned GitHub topics to Todoist tasks."""
    configure_logging(verbose)

    token = os.environ.get("TODOIST_API_TOKEN")
    if not token:
        typer.echo("ERROR: TODOIST_API_TOKEN is not set", err=True)
        raise typer.Exit(2)

    con = db_open(db_path)
    last_sync_s = db_get(con, "last_sync_iso")
    last_sync = datetime.fromisoformat(last_sync_s).astimezone(UTC) if last_sync_s else None
    log.info("starting", last_sync_iso=last_sync_s or None, dry_run=dry_run)

    topics = gh_search_assigned(last_sync, gh_path=gh)
    log.info("github_items_fetched", count=len(topics))

    td = Todoist(token)
    project_id = td.resolve_project_id(project) if project else None

    # Ensure labels exist globally (union of all GH labels we may use)
    all_labels = sorted({lbl for t in topics for lbl in t.labels})
    if not dry_run:
        td.ensure_labels(all_labels)

    created = updated = completed = deleted = reopened = 0

    for t in topics:
        disp = disposition(t, gh_path=gh)
        # Ensure mapping record exists; update last seen GH state.
        existing = db_get_topic_by_gh(con, t.gh_id)
        existing_tid = existing[0] if existing else None
        existing_updated = existing[1] if existing else None

        # Upsert DB row early so we keep GH metadata even on failures.
        db_upsert_topic(
            con,
            gh_id=t.gh_id, gh_url=t.url, repo=t.repo, number=t.number, is_pr=t.is_pr,
            todoist_id=existing_tid, gh_updated_at=t.updated_at.isoformat() if t.updated_at else _now_iso()
        )

        content = make_content(t)
        description = make_description(t)
        labels = list(t.labels)

        if not existing_tid:
            if disp == "delete":
                log.info("skip_create_deleted", gh_id=t.gh_id, url=t.url)
                continue
            if dry_run:
                log.info("create_task_dryrun", gh_id=t.gh_id, content=content)
                created += 1
                continue
            tid = td.create_task(content=content, description=description, labels=labels, project_id=project_id, xreq=idempotency_key(t))
            if tid:
                db_set_todoist_id(con, t.gh_id, tid)
                existing_tid = tid
                created += 1
                log.info("task_created", gh_id=t.gh_id, todoist_id=tid, url=t.url)
            else:
                continue  # failed create; don't try to update/close
        else:
            # Update if GH changed since last time.
            if should_update(existing_updated, t.updated_at or datetime.now(UTC)):
                if dry_run:
                    log.info("update_task_dryrun", gh_id=t.gh_id, todoist_id=existing_tid, content=content)
                    updated += 1
                else:
                    if td.update_task(existing_tid, content=content, description=description, labels=labels):
                        updated += 1
                        log.info("task_updated", gh_id=t.gh_id, todoist_id=existing_tid)
            else:
                log.debug("no_update_needed", gh_id=t.gh_id)

        # Handle closure/disposal after ensuring task exists/mapped.
        if existing_tid:
            # Re-open task if the GitHub topic became open again
            if disp == "open":
                if dry_run:
                    log.info("reopen_task_dryrun", gh_id=t.gh_id, todoist_id=existing_tid)
                    reopened += 1
                else:
                    if td.reopen_task(existing_tid):
                        reopened += 1
                        log.info("task_reopened", gh_id=t.gh_id, todoist_id=existing_tid)

            match disp:
                case "complete":
                    if dry_run:
                        log.info("complete_task_dryrun", gh_id=t.gh_id, todoist_id=existing_tid)
                        completed += 1
                    else:
                        if td.close_task(existing_tid):
                            completed += 1
                            log.info("task_completed", gh_id=t.gh_id, todoist_id=existing_tid)
                case "delete":
                    if dry_run:
                        log.info("delete_task_dryrun", gh_id=t.gh_id, todoist_id=existing_tid)
                        deleted += 1
                    else:
                        if td.delete_task(existing_tid):
                            deleted += 1
                            log.info("task_deleted", gh_id=t.gh_id, todoist_id=existing_tid)

        # Avoid hammering APIs if assigned list is large.
        time.sleep(0.05)

    if not dry_run:
        db_set(con, "last_sync_iso", _now_iso())

    log.info("done", created=created, updated=updated, completed=completed, deleted=deleted, reopened=reopened, dry_run=dry_run)

# ---------- entry ----------

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        # last-ditch log (cron visibility)
        print(json.dumps({
            "ts": datetime.now(UTC).isoformat(),
            "level": "error",
            "event": "fatal",
            "error": str(e),
        }), file=sys.stderr)
        sys.exit(1)
