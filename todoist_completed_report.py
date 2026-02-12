#!/usr/bin/env python3
"""
todoist_completed_report.py

Single-file CLI to fetch completed Todoist tasks for a given timeframe and optional project,
and output them in a format suitable for piping into an LLM for a weekly situation report.

Features:
- Uses only Python standard library (no third-party deps).
- Auth via --token or environment variables TODOIST_TOKEN / TODOIST_API_TOKEN.
- Timeframe options:
  - --since / --until (accepts YYYY-MM-DD or ISO 8601; naïve times treated as local time)
  - --last-week (previous ISO week, Mon 00:00:00 to Sun 23:59:59)
  - --week YYYY-Www (e.g., 2025-W33), or --week this / --week last
  - Default is the last 7 days up to now
- Project filter by name or ID via --project. If name is provided, it is resolved via REST API.
- Output formats: markdown (default) or jsonl.
- Handles pagination and basic rate limiting (HTTP 429) with retries.

Examples:
  export TODOIST_TOKEN="YOUR_TOKEN_HERE"
  python3 todoist_completed_report.py --last-week --project "Work" --format markdown
  python3 todoist_completed_report.py --since 2025-08-01 --until 2025-08-07 --format jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, OrderedDict
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_SYNC = "https://api.todoist.com/sync/v10"
API_BASE_REST = "https://api.todoist.com/api/v1"


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_date(s: str) -> date:
    # s like YYYY-MM-DD
    return date.fromisoformat(s)


def _parse_datetime_guess(s: str, *, end_of_day: bool = False) -> datetime:
    """
    Parse a date/time string:
    - YYYY-MM-DD => local midnight (or 23:59:59 if end_of_day)
    - ISO 8601 with or without Z; naïve => interpreted as local time
    Returns timezone-aware datetime in local tz (convert upstream as needed).
    """
    s = s.strip()
    local_tz = _local_tz()
    # Date only
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = _parse_date(s)
        t = dtime(23, 59, 59) if end_of_day else dtime(0, 0, 0)
        return datetime.combine(d, t, tzinfo=local_tz)
    # ISO with Z
    if s.endswith("Z"):
        # Remove Z, parse, attach UTC
        base = s[:-1]
        try:
            dt = datetime.fromisoformat(base)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            pass
    # ISO without Z
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
        return dt
    except ValueError:
        raise SystemExit(f"Invalid date/time: {s!r}")


def _iso_week_range(year: int, week: int, tz) -> Tuple[datetime, datetime]:
    """
    Return (start, end) datetimes for ISO week (Mon..Sun), localized to tz.
    start = Mon 00:00:00, end = Sun 23:59:59 (inclusive).
    """
    monday = date.fromisocalendar(year, week, 1)
    start = datetime.combine(monday, dtime(0, 0, 0), tzinfo=tz)
    end = datetime.combine(monday + timedelta(days=6), dtime(23, 59, 59), tzinfo=tz)
    return start, end


def _compute_timeframe(
    since: Optional[str],
    until: Optional[str],
    week: Optional[str],
    last_week: bool,
) -> Tuple[str, str]:
    """
    Compute (since_iso_z, until_iso_z) strings in UTC Z format for Todoist.
    Priority:
      1) --since/--until (if provided)
      2) --week (YYYY-Www | this | last)
      3) --last-week
      4) default: last 7 days until now
    """
    local_tz = _local_tz()

    if since or until:
        if since:
            since_dt = _parse_datetime_guess(since, end_of_day=False)
        else:
            # default: 7 days before until (or now)
            u = _parse_datetime_guess(until, end_of_day=True) if until else datetime.now(local_tz)
            since_dt = u - timedelta(days=7)
        if until:
            until_dt = _parse_datetime_guess(until, end_of_day=True)
        else:
            until_dt = datetime.now(local_tz)
        return _to_utc_z(since_dt), _to_utc_z(until_dt)

    if week:
        wk = week.strip().lower()
        if wk in ("this", "current"):
            today = datetime.now(local_tz).date()
            year, wknum, _ = today.isocalendar()
            s, e = _iso_week_range(year, wknum, local_tz)
        elif wk in ("last", "previous", "prev"):
            today = datetime.now(local_tz).date()
            year, wknum, _ = today.isocalendar()
            # Move to previous week
            d = today - timedelta(days=7)
            year, wknum, _ = d.isocalendar()
            s, e = _iso_week_range(year, wknum, local_tz)
        else:
            # Expect YYYY-Www (e.g., 2025-W33) or YYYYWww
            wk_str = wk.upper().replace("W", "-W")
            try:
                parts = wk_str.split("-W")
                if len(parts) != 2:
                    raise ValueError
                y = int(parts[0])
                w = int(parts[1])
                s, e = _iso_week_range(y, w, local_tz)
            except Exception:
                raise SystemExit(f"Invalid --week value: {week!r}. Use 'this', 'last', or 'YYYY-Www'.")
        return _to_utc_z(s), _to_utc_z(e)

    if last_week:
        today = datetime.now(local_tz).date()
        d = today - timedelta(days=7)
        y, w, _ = d.isocalendar()
        s, e = _iso_week_range(y, w, local_tz)
        return _to_utc_z(s), _to_utc_z(e)

    # Default: last 7 days ending now
    now = datetime.now(local_tz)
    start = now - timedelta(days=7)
    return _to_utc_z(start), _to_utc_z(now)


def _http_get_json(url: str, *, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None, retries: int = 5) -> Any:
    if params:
        query = urlencode(params, doseq=True)
        if "?" in url:
            url = f"{url}&{query}"
        else:
            url = f"{url}?{query}"
    backoff = 1.0
    for attempt in range(retries):
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype or data.strip().startswith(b"{") or data.strip().startswith(b"["):
                    return json.loads(data.decode("utf-8"))
                return data.decode("utf-8")
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                if attempt == retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            # For 4xx other than 429 or non-retriable
            raise
        except URLError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)
    raise RuntimeError("Unreachable")


def fetch_projects(token: str) -> Dict[str, str]:
    """
    Returns a mapping project_id(str) -> project_name(str) for active projects.
    """
    url = f"{API_BASE_REST}/projects"
    headers = {"Authorization": f"Bearer {token}"}
    data = _http_get_json(url, headers=headers)
    out: Dict[str, str] = {}
    if isinstance(data, list):
        for p in data:
            pid = str(p.get("id"))
            name = p.get("name") or ""
            out[pid] = name
    return out


def resolve_project_id_by_name(projects: Dict[str, str], name: str) -> Optional[str]:
    """
    Case-insensitive exact match; falls back to case-insensitive unique prefix match.
    """
    name_ci = name.strip().casefold()
    # exact
    for pid, pname in projects.items():
        if pname.casefold() == name_ci:
            return pid
    # unique prefix
    matches = [pid for pid, pname in projects.items() if pname.casefold().startswith(name_ci)]
    if len(matches) == 1:
        return matches[0]
    return None


def fetch_projects_full(token: str) -> List[Dict[str, Any]]:
    """
    Return raw project objects from Todoist API v1, including parent_id.
    """
    url = f"{API_BASE_REST}/projects"
    headers = {"Authorization": f"Bearer {token}"}
    data = _http_get_json(url, headers=headers)
    return data if isinstance(data, list) else []


def collect_descendant_project_ids(projects: List[Dict[str, Any]], root_id: str) -> set[str]:
    """
    Given raw project list and a root project id (string), return {root_id} plus all descendant ids.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    for p in projects:
        pid = p.get("id")
        if pid is None:
            continue
        pid_str = str(pid)
        parent = p.get("parent_id")
        if parent is not None:
            parent_str = str(parent)
            children[parent_str].append(pid_str)
    result: set[str] = set()
    stack: List[str] = [str(root_id)]
    while stack:
        cur = stack.pop()
        if cur in result:
            continue
        result.add(cur)
        for child in children.get(cur, []):
            if child not in result:
                stack.append(child)
    return result


def fetch_completed(
    token: str,
    *,
    since_iso_z: str,
    until_iso_z: str,
    project_id: Optional[str] = None,
    limit: int = 200,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Returns (items, projects_map, sections_map)
    items: list of completed items as returned by Todoist.
    projects_map: project_id(str)->name
    sections_map: section_id(str)->name
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{API_BASE_SYNC}/completed/get_all"

    items: List[Dict[str, Any]] = []
    projects_map: Dict[str, str] = {}
    sections_map: Dict[str, str] = {}

    offset = 0
    while True:
        params: Dict[str, Any] = {
            "since": since_iso_z,
            "until": until_iso_z,
            "limit": limit,
            "offset": offset,
            "include_task": "true",
            # "annotate_notes": "true",  # Uncomment if you want notes (if supported)
        }
        if project_id:
            params["project_id"] = project_id
        data = _http_get_json(url, headers=headers, params=params)
        page_items = data.get("items", []) if isinstance(data, dict) else []
        page_projects = data.get("projects", []) if isinstance(data, dict) else []
        page_sections = data.get("sections", []) if isinstance(data, dict) else []
        for it in page_items:
            items.append(it)

        # Normalize/merge projects (can be list[dict] or dict[id]->(name|dict))
        if isinstance(page_projects, dict):
            for pid_raw, p in page_projects.items():
                pid = str(pid_raw)
                if isinstance(p, dict):
                    name = p.get("name") or p.get("project_name") or ""
                else:
                    name = str(p) if p is not None else ""
                if pid:
                    projects_map[pid] = name
        elif isinstance(page_projects, list):
            for p in page_projects:
                if isinstance(p, dict):
                    pid_val = p.get("id") or p.get("project_id") or p.get("projectid") or p.get("projectId") or p.get("project")
                    pid = str(pid_val) if pid_val is not None else ""
                    name = p.get("name") or p.get("project_name") or ""
                    if pid:
                        projects_map[pid] = name
                else:
                    pid = str(p)
                    if pid:
                        projects_map.setdefault(pid, f"Project {pid}")

        # Normalize/merge sections (can be list[dict] or dict[id]->(name|dict))
        if isinstance(page_sections, dict):
            for sid_raw, s in page_sections.items():
                sid = str(sid_raw)
                if isinstance(s, dict):
                    name = s.get("name") or ""
                else:
                    name = str(s) if s is not None else ""
                if sid:
                    sections_map[sid] = name
        elif isinstance(page_sections, list):
            for s in page_sections:
                if isinstance(s, dict):
                    sid_val = s.get("id") or s.get("section_id")
                    sid = str(sid_val) if sid_val is not None else ""
                    name = s.get("name") or ""
                    if sid:
                        sections_map[sid] = name
                else:
                    sid = str(s)
                    if sid:
                        sections_map.setdefault(sid, f"Section {sid}")

        if not page_items or len(page_items) < limit:
            break
        offset += len(page_items)

    return items, projects_map, sections_map


def fetch_task_detail(token: str, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single task (including completed/archived) via API v1.
    Returns the task dict or None if not found.
    """
    url = f"{API_BASE_REST}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        data = _http_get_json(url, headers=headers)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except URLError:
        return None
    return data if isinstance(data, dict) else None


def fetch_task_details_bulk(token: str, ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch multiple tasks by id (one-by-one; REST has no batch for this).
    Returns mapping id -> task dict for those found.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tid in ids:
        tid_str = str(tid or "")
        if not tid_str:
            continue
        d = fetch_task_detail(token, tid_str)
        if isinstance(d, dict):
            out[str(d.get("id") or tid_str)] = d
    return out


def enrich_items_with_task_details(token: str, items: List[Dict[str, Any]], *, verbose: int = 0) -> Dict[str, Dict[str, Any]]:
    """
    Ensure each completed item has a 'task' dict with at least description/parent_id when possible.
    Also fetch parent task details so we can resolve parent titles even if the parent isn't in the timeframe.
    Returns a dict of all fetched task details {task_id: task_dict}.
    """
    # Determine which items are missing useful task metadata
    need_ids: set[str] = set()
    for it in items:
        tid = str(it.get("task_id") or it.get("id") or "")
        if not tid:
            continue
        t = it.get("task")
        if not isinstance(t, dict) or (t.get("description") in (None, "") and t.get("parent_id") is None):
            need_ids.add(tid)

    details: Dict[str, Dict[str, Any]] = {}
    if need_ids:
        details = fetch_task_details_bulk(token, need_ids)
        # Attach details into items
        for it in items:
            tid = str(it.get("task_id") or it.get("id") or "")
            if not tid:
                continue
            det = details.get(tid)
            if det:
                t = it.get("task")
                if isinstance(t, dict):
                    # Merge with preference for existing 'task' fields
                    merged = {**det, **t}
                else:
                    merged = det
                it["task"] = merged

    # Collect parent_ids and fetch details for those parents not already known
    parent_ids: set[str] = set()
    for it in items:
        pid = (it.get("task") or {}).get("parent_id")
        if pid:
            parent_ids.add(str(pid))

    known_ids: set[str] = set(details.keys())
    for it in items:
        known_ids.add(str(it.get("task_id") or it.get("id") or ""))

    unresolved_parents = {pid for pid in parent_ids if pid and pid not in known_ids}
    if unresolved_parents:
        parent_details = fetch_task_details_bulk(token, unresolved_parents)
        details.update(parent_details)

    if verbose:
        print(f"Info: enriched {len(details)} task detail(s) via REST.", file=sys.stderr)

    return details


def _clean_content(s: str) -> str:
    s = s.replace("\r\n", " ").replace("\n", " ").strip()
    return " ".join(s.split())


def _weekday_name(iso_date: str) -> str:
    y, m, d = map(int, iso_date.split("-"))
    return date(y, m, d).strftime("%A")


def _group_by_day(items: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        completed_at = it.get("completed_at")
        if not completed_at:
            continue
        day = completed_at[:10]  # YYYY-MM-DD (UTC from API)
        buckets[day].append(it)
    # Sort by day then by completion time
    ordered = OrderedDict()
    for day in sorted(buckets.keys()):
        ordered[day] = sorted(buckets[day], key=lambda x: x.get("completed_at", ""))
    return ordered


def output_markdown(
    items: List[Dict[str, Any]],
    projects: Dict[str, str],
    sections: Dict[str, str],
    *,
    since_iso_z: str,
    until_iso_z: str,
    project_label: Optional[str],
    task_details: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    by_day = _group_by_day(items)
    total = sum(len(v) for v in by_day.values())

    # Build lookup of task_id -> content for parent resolution
    id_to_content: Dict[str, str] = {}
    for t in items:
        tid = str(t.get("task_id") or t.get("id") or "")
        if tid:
            content_src = (t.get("task") or {}).get("content") or t.get("content") or ""
            id_to_content[tid] = _clean_content(str(content_src))
    if task_details:
        for tid, td in task_details.items():
            if not tid:
                continue
            c = td.get("content") or ""
            if c:
                id_to_content[str(tid)] = _clean_content(str(c))

    print("# Todoist Completed Tasks — Weekly Situation Report")
    print()
    print(f"- Timeframe (UTC): {since_iso_z} — {until_iso_z}")
    print(f"- Project: {project_label or 'All Projects'}")
    print(f"- Total completed tasks: {total}")
    print()
    print("---")
    print()

    for day, tasks in by_day.items():
        wk = _weekday_name(day)
        print(f"## {day} ({wk}) — {len(tasks)} task{'s' if len(tasks)!=1 else ''}")
        for it in tasks:
            content = _clean_content(str(it.get("content", "")))
            desc_val = (it.get("task") or {}).get("description") or it.get("description") or it.get("desc") or ""
            description = _clean_content(str(desc_val)) if desc_val else ""
            pid = str(it.get("project_id") or "")
            sid = str(it.get("section_id") or "")
            pname = projects.get(pid, f"Project {pid}" if pid else "Unknown Project")
            sname = sections.get(sid, "")
            # Indicate subtasks lightly if parent_id present (prefer nested task.parent_id when available)
            parent_id_val = (it.get("task") or {}).get("parent_id")
            if parent_id_val is None:
                parent_id_val = it.get("parent_id")
            is_subtask = bool(parent_id_val)
            parent_note = ""
            if is_subtask:
                parent_id_str = str(parent_id_val or "")
                parent_title = id_to_content.get(parent_id_str)
                parent_note = f" (subtask of: {parent_title})" if parent_title else " (subtask)"
            prefix = "  - " if is_subtask else "- "
            if sname:
                print(f"{prefix}[{pname} / {sname}] {content}{parent_note}")
            else:
                print(f"{prefix}[{pname}] {content}{parent_note}")
            if description:
                desc_indent = "    " if is_subtask else "  "
                print(f"{desc_indent}- Description: {description}")
        print()


def output_jsonl(
    items: List[Dict[str, Any]],
    projects: Dict[str, str],
    sections: Dict[str, str],
    *,
    since_iso_z: str,
    until_iso_z: str,
    project_label: Optional[str],
    task_details: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    meta = {
        "type": "todoist_completed_tasks",
        "since_utc": since_iso_z,
        "until_utc": until_iso_z,
        "project": project_label,
        "total": len(items),
        "generated_at": _to_utc_z(datetime.now(timezone.utc)),
    }
    print(json.dumps({"meta": meta}, separators=(",", ":")))
    # Build lookup of task_id -> content for parent resolution
    id_to_content: Dict[str, str] = {}
    for t in items:
        tid = str(t.get("task_id") or t.get("id") or "")
        if tid:
            content_src = (t.get("task") or {}).get("content") or t.get("content") or ""
            id_to_content[tid] = _clean_content(str(content_src))
    if task_details:
        for tid, td in task_details.items():
            if not tid:
                continue
            c = td.get("content") or ""
            if c:
                id_to_content[str(tid)] = _clean_content(str(c))
    for it in sorted(items, key=lambda x: x.get("completed_at", "")):
        pid = str(it.get("project_id") or "")
        sid = str(it.get("section_id") or "")
        desc_val = (it.get("task") or {}).get("description") or it.get("description") or it.get("desc") or ""
        description = _clean_content(str(desc_val)) if desc_val else ""
        parent_id_val = (it.get("task") or {}).get("parent_id")
        if parent_id_val is None:
            parent_id_val = it.get("parent_id")
        parent_id_str = str(parent_id_val or "")
        is_subtask = bool(parent_id_str)
        parent_content = id_to_content.get(parent_id_str) if is_subtask else None
        record = {
            "completed_at": it.get("completed_at"),
            "project_id": pid or None,
            "project_name": projects.get(pid),
            "section_id": sid or None,
            "section_name": sections.get(sid),
            "task_id": str(it.get("task_id") or it.get("id") or ""),
            "content": _clean_content(str(it.get("content", ""))),
            "description": description or None,
            "parent_id": parent_id_str or None,
            "parent_content": parent_content,
            "is_subtask": is_subtask,
        }
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch completed Todoist tasks and output in LLM-friendly format (markdown/jsonl)."
    )
    ap.add_argument("--token", help="Todoist API token. Defaults to $TODOIST_TOKEN or $TODOIST_API_TOKEN.")
    ap.add_argument("--project", "-p", help="Project name or ID to filter. If omitted, includes all projects.")
    ap.add_argument("--since", help="Start datetime (YYYY-MM-DD or ISO). Naïve times treated as local time.")
    ap.add_argument("--until", help="End datetime (YYYY-MM-DD or ISO). Naïve times treated as local time.")
    ap.add_argument("--week", help="ISO week: 'this', 'last', or 'YYYY-Www' (e.g., 2025-W33).")
    ap.add_argument("--last-week", action="store_true", help="Use the previous ISO week (Mon..Sun).")
    ap.add_argument("--format", "-f", choices=("markdown", "jsonl"), default="markdown", help="Output format.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (can use -vv).")

    args = ap.parse_args(argv)

    token = args.token or os.getenv("TODOIST_TOKEN") or os.getenv("TODOIST_API_TOKEN")
    if not token:
        print("Error: Provide a Todoist API token via --token or TODOIST_TOKEN / TODOIST_API_TOKEN env var.", file=sys.stderr)
        return 2

    since_iso_z, until_iso_z = _compute_timeframe(args.since, args.until, args.week, args.last_week)

    project_label: Optional[str] = None
    project_id: Optional[str] = None
    filter_ids: Optional[set[str]] = None

    # If user provided a project string, try to detect ID or resolve name -> id.
    if args.project:
        p = args.project.strip()
        project_label = p
        # If it's a numeric-looking ID, accept as-is
        if p.isdigit():
            project_id = p
        else:
            # Resolve via REST projects
            projects_active = fetch_projects(token)
            resolved = resolve_project_id_by_name(projects_active, p)
            if resolved:
                project_id = resolved

    # If we resolved a project_id, collect descendant project IDs (include sub-projects)
    if project_id is not None:
        try:
            projects_full = fetch_projects_full(token)
            ids_set = collect_descendant_project_ids(projects_full, project_id)
            # Only use client-side filtering when there are descendants; otherwise let server filter
            if len(ids_set) > 1:
                filter_ids = ids_set
        except Exception as e:
            # On any failure, fall back gracefully to server-side filtering
            if args.verbose:
                print(f"Warning: could not compute descendant project IDs: {e}", file=sys.stderr)

    if args.verbose:
        if filter_ids:
            print(f"Fetching completed tasks: since={since_iso_z}, until={until_iso_z}, project_id=ALL (client-side filter {len(filter_ids)} ids)", file=sys.stderr)
        else:
            print(f"Fetching completed tasks: since={since_iso_z}, until={until_iso_z}, project_id={project_id or 'ALL'}", file=sys.stderr)

    items, projects_map, sections_map = fetch_completed(
        token, since_iso_z=since_iso_z, until_iso_z=until_iso_z, project_id=(None if filter_ids else project_id)
    )
    if filter_ids:
        items = [it for it in items if str(it.get("project_id") or "") in filter_ids]

    # If project was a NAME but could not be resolved, try filtering client-side by name
    if args.project and project_id is None:
        # Try to find project(s) with this name in the returned payload
        target_name_ci = args.project.strip().casefold()
        # Build reverse map: name_ci -> list of ids
        name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for pid, pname in projects_map.items():
            name_to_ids[pname.casefold()].append(pid)
        ids = name_to_ids.get(target_name_ci, [])
        if len(ids) == 1:
            pid = ids[0]
            items = [it for it in items if str(it.get("project_id") or "") == pid]
            project_label = projects_map.get(pid, args.project)
            if args.verbose:
                print(f"Filtered client-side to project '{project_label}' (id={pid}).", file=sys.stderr)
        elif len(ids) > 1:
            if args.verbose:
                print(
                    f"Multiple projects named '{args.project}' found in results; including all matching.",
                    file=sys.stderr,
                )
            items = [it for it in items if str(it.get("project_id") or "") in set(ids)]
        else:
            print(
                f"Warning: Project '{args.project}' not found via REST or in results; no server-side filtering applied.",
                file=sys.stderr,
            )

    # Enrich items with task details (description, parent_id) when not provided by completed API
    task_details = enrich_items_with_task_details(token, items, verbose=args.verbose)

    # Sort items by completion time
    items.sort(key=lambda x: x.get("completed_at", ""))

    if args.format == "markdown":
        output_markdown(items, projects_map, sections_map, since_iso_z=since_iso_z, until_iso_z=until_iso_z, project_label=project_label, task_details=task_details)
    else:
        output_jsonl(items, projects_map, sections_map, since_iso_z=since_iso_z, until_iso_z=until_iso_z, project_label=project_label, task_details=task_details)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
