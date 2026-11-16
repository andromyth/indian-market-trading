#!/usr/bin/env python3
"""Publish one session health record to the session-health branch.

Uses the GitHub Contents API with the workflow token so the write is atomic
and its failure mode is a plain HTTP status instead of a git rebase dance.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
BRANCH = "session-health"
KEEP_HISTORY_LINES = 2000


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    token = os.environ["GH_TOKEN"]
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
        return json.loads(body) if body else {}


def _get_file(path: str) -> tuple[bytes | None, str | None]:
    try:
        payload = _request("GET", f"/repos/{os.environ['REPOSITORY']}/contents/{path}?ref={BRANCH}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, None
        raise
    return base64.b64decode(payload["content"]), payload.get("sha")


def _put_file(path: str, content: bytes, sha: str | None, message: str) -> None:
    body = {
        "message": message,
        "content": base64.b64encode(content).decode(),
        "branch": BRANCH,
    }
    if sha:
        body["sha"] = sha
    _request("PUT", f"/repos/{os.environ['REPOSITORY']}/contents/{path}", body)


def main() -> int:
    source = sys.argv[1]
    with open(source, "r", encoding="utf-8") as handle:
        health = json.load(handle)
    stamp = str(health.get("checked_at_ist") or "")
    pretty = (json.dumps(health, indent=2, sort_keys=True) + "\n").encode()

    _, sha = _get_file("session_health/latest.json")
    _put_file("session_health/latest.json", pretty, sha, f"Session health {stamp}")

    raw, sha = _get_file("session_health/history.jsonl")
    lines = (raw or b"").decode().splitlines()
    lines.append(json.dumps(health, sort_keys=True, default=str))
    lines = lines[-KEEP_HISTORY_LINES:]
    _put_file(
        "session_health/history.jsonl",
        ("\n".join(lines) + "\n").encode(),
        sha,
        f"Session health {stamp}",
    )
    print("PUBLISHED", stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
