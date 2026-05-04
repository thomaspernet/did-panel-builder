---
description: "Start or stop the monitored app's dev stack. Usage: /app-control open|close"
---

Start or stop the target repo's dev stack (backend + frontend) cleanly, without leaving orphan processes.

## Parse arguments

`$ARGUMENTS` is exactly one token: `open` or `close`.

- `$ARGUMENTS` = `"open"` → ACTION=open
- `$ARGUMENTS` = `"close"` → ACTION=close

If `$ARGUMENTS` is empty or anything else, **stop** and tell the user: *"Usage: /app-control open|close"*.

## Detect repo

Run this once and reuse for every `devwatch` call:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
REPO_ROOT=$(git rev-parse --show-toplevel)
PID_FILE="$REPO_ROOT/.devwatch/app-control.pids"
LOG_DIR="$REPO_ROOT/.devwatch/logs"
mkdir -p "$REPO_ROOT/.devwatch" "$LOG_DIR"
```

The PID file and logs live under `.devwatch/` in the target repo. Add `.devwatch/` to `.gitignore` if it isn't already — do **not** commit PID files or logs.

## Context loading

Read `CLAUDE.md` (or `README.md`) in the target repo to learn how its dev stack is launched. Typical patterns:

- **Node / Next.js**: `npm run dev`, `npm run start:dev`, `pnpm dev`, `yarn dev`.
- **Python / FastAPI**: `uv run uvicorn <module>:app --port <N>`, `python -m <module>`, `./manage.py runserver`.
- **Multi-service**: a repo-root `start.sh`, `docker compose up`, or a `Procfile`. Respect it — if it exists, run it instead of reinventing the startup sequence.

If the repo documents a specific launch command (e.g. `./start.sh`, `make dev`), use it. Otherwise read `package.json` scripts and `pyproject.toml` to identify the correct dev commands.

## Pick free ports (do this BEFORE launching)

**Do not assume default ports are free.** Ports like `8000`, `8001`, `3000`, `3001` are commonly taken on developer machines. Retrying on `address already in use` wastes minutes; asking the kernel for a free port costs milliseconds.

Pick one free port per service up front, then pass them into the launch commands:

```bash
pick_port() {
    python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}
BACKEND_PORT=$(pick_port)
FRONTEND_PORT=$(pick_port)
echo "Backend port: $BACKEND_PORT"
echo "Frontend port: $FRONTEND_PORT"
```

Wire them into the launch commands (examples):

- **FastAPI / uvicorn**: `uv run uvicorn server.main:app --host 127.0.0.1 --port "$BACKEND_PORT"`
- **Next.js**: `npm run dev -- --port "$FRONTEND_PORT"` (or `-p "$FRONTEND_PORT"`)
- **If the frontend needs to reach the backend**, pass the chosen backend port in via env (`NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:$BACKEND_PORT"`, etc.) instead of hardcoding.

If the target repo has `.claude/rules/dev-ports.md`, treat it as authoritative — follow whatever recipe it prescribes. Otherwise use the block above.

## Intelligence (what you decide)

### If ACTION=open

1. **Pre-flight** — if `$PID_FILE` already exists and the PIDs are alive, the stack is already running. Report it and stop without double-starting.
    ```bash
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                echo "App already running (pid $pid). Run /app-control close first."
                exit 0
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    ```

2. **Identify the launch commands.** Usually one backend + one frontend. Write down the exact shell command for each, wiring in the `$BACKEND_PORT` / `$FRONTEND_PORT` you picked above (e.g. `uv run uvicorn server.main:app --host 127.0.0.1 --port "$BACKEND_PORT"` and `cd dashboard && npm run dev -- --port "$FRONTEND_PORT"`).

3. **Launch each command detached** with stdout/stderr redirected to a log file. Record the PID:
    ```bash
    : > "$PID_FILE"
    launch() {
        local name="$1"; shift
        local log="$LOG_DIR/${name}.log"
        nohup bash -c "cd '$REPO_ROOT' && $*" > "$log" 2>&1 &
        echo $! >> "$PID_FILE"
        echo "  $name pid $! → $log"
    }
    launch backend "uv run uvicorn server.main:app --host 127.0.0.1 --port $BACKEND_PORT"
    launch frontend "cd dashboard && npm run dev -- --port $FRONTEND_PORT"
    ```
    Always invoke commands from `$REPO_ROOT` so Python and Node resolve their packages correctly (don't `cd server` before `uvicorn` — the `server` module lives at the repo root). On close we kill each recorded PID and its child tree via `pkill -P` — macOS doesn't ship `setsid`, so we rely on process-tree walking instead of process groups.

4. **Verify the processes are alive** after a short delay (at least 2 seconds) so we don't report `running` for a stack that crashed on boot:
    ```bash
    sleep 3
    ALL_ALIVE=1
    PIDS_CSV=""
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            PIDS_CSV="${PIDS_CSV:+$PIDS_CSV,}$pid"
        else
            ALL_ALIVE=0
        fi
    done < "$PID_FILE"
    ```
    If any PID is already dead, tail the matching log file, report what went wrong, and stop (do **not** report `running`).

5. **Report running state to Lingtai:**
    ```bash
    devwatch --repo "$REPO" app-state \
      --status running \
      --pids "$PIDS_CSV" \
      --pid-file "$PID_FILE" \
      --message "Dev stack up (backend + frontend)"
    ```

6. **Print the localhost URLs** the user should open in their browser. Use the `$BACKEND_PORT` / `$FRONTEND_PORT` you picked in the "Pick free ports" step — those are the ports the servers are bound to. Cross-check the dev-server logs (Next.js prints `- Local: http://localhost:<PORT>`, uvicorn prints `Uvicorn running on http://127.0.0.1:<PORT>`) and use the log-reported port if it differs (some dev servers reassign). Report:
    - Dashboard (frontend): `http://localhost:$FRONTEND_PORT`
    - API (backend): `http://localhost:$BACKEND_PORT`

    Always include the log paths so the user can tail output.

### If ACTION=close

1. **If `$PID_FILE` is missing**, the app is probably already stopped. Still report `stopped` to Lingtai so the dashboard reflects reality, then exit:
    ```bash
    if [ ! -f "$PID_FILE" ]; then
        devwatch --repo "$REPO" app-state --status stopped --message "No PID file found"
        echo "No running app detected."
        exit 0
    fi
    ```

2. **Kill each recorded PID and its children.** `pkill -P <pid>` drops the whole child tree — the dev server wrapper plus any workers it spawned. On macOS we can't rely on process groups (no `setsid`), so we walk the tree explicitly:
    ```bash
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    sleep 2
    # Escalate to SIGKILL on anything that survived SIGTERM.
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    ```

3. **Verify nothing is left.** Scan for known dev-server process names (e.g. `uvicorn`, `next dev`, `node .*dev`) and warn the user if any survived — they likely started from a different launch path and need manual cleanup.

4. **Report stopped state to Lingtai:**
    ```bash
    devwatch --repo "$REPO" app-state \
      --status stopped \
      --message "Dev stack stopped"
    ```

## Boundary

This skill only starts or stops the monitored app's dev stack. It does **not**:
- Install dependencies (`npm install`, `uv sync`).
- Run migrations, seed data, or build artefacts.
- Touch the Lingtai server itself — the user runs that separately.

If the stack fails to start, report the failure and stop. Do not attempt to fix the underlying repo.
