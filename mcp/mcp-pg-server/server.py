import atexit
import os
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool

# Official MCP FastMCP (package name: mcp)
from mcp.server.fastmcp import FastMCP


# ----------------------------
# Env config (match TablePlus)
# ----------------------------
SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", ""))
SSH_USER = os.environ.get("SSH_USER", "")
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "")

# Where Postgres is reachable FROM the SSH server
PG_REMOTE_HOST = os.environ.get("PG_REMOTE_HOST", "127.0.0.1")
PG_REMOTE_PORT = int(os.environ.get("PG_REMOTE_PORT", "5432"))

# Local bind (TablePlus uses 127.0.0.1:5432, but 5432 may be busy locally)
LOCAL_PG_HOST = "127.0.0.1"
LOCAL_PG_PORT = int(os.environ.get("LOCAL_PG_PORT", "0"))  # 0 = auto pick free port

# DB credentials (NOT SSH)
PG_DB = os.environ.get("PG_DB", "merchant-invoice")
PG_USER = os.environ.get("PG_USER", "merchant-invoice")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

# Safety
DEFAULT_MAX_ROWS = int(os.environ.get("PG_MAX_ROWS", "200"))
STATEMENT_TIMEOUT_MS = int(os.environ.get("PG_STATEMENT_TIMEOUT_MS", "5000"))


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOCAL_PG_HOST, 0))
        return s.getsockname()[1]


def _wait_port_open(host: str, port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for tunnel on {host}:{port}. Last error: {last_err}")


if not PG_PASSWORD:
    raise RuntimeError("PG_PASSWORD is required (DB password, not SSH password).")

if LOCAL_PG_PORT == 0:
    LOCAL_PG_PORT = _pick_free_port()

# ----------------------------
# Start SSH local port forward
# ----------------------------
# Equivalent to TablePlus "Over SSH":
# ssh -N -L <local_port>:<pg_remote_host>:<pg_remote_port> <ssh_user>@<ssh_host>
ssh_cmd = [
    "ssh",
    "-N",  # no remote command, just forwarding
    "-L", f"{LOCAL_PG_HOST}:{LOCAL_PG_PORT}:{PG_REMOTE_HOST}:{PG_REMOTE_PORT}",
    "-p", str(SSH_PORT),
    "-i", SSH_KEY_PATH,
    # recommended keepalive so it doesn't die silently
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    f"{SSH_USER}@{SSH_HOST}",
]

ssh_proc = subprocess.Popen(
    ssh_cmd,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

def _cleanup():
    try:
        if ssh_proc.poll() is None:
            ssh_proc.terminate()
            try:
                ssh_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                ssh_proc.kill()
    except Exception:
        pass

atexit.register(_cleanup)

# Wait until tunnel is ready (or ssh died)
try:
    _wait_port_open(LOCAL_PG_HOST, LOCAL_PG_PORT, timeout_s=10.0)
except Exception:
    # If tunnel failed, show ssh stderr for debugging
    _cleanup()
    stderr = ""
    try:
        stderr = (ssh_proc.stderr.read() or "").strip()
    except Exception:
        pass
    raise RuntimeError(f"SSH tunnel failed. ssh stderr:\n{stderr}")  # surfaces in logs

# ----------------------------
# Postgres pool via tunnel
# ----------------------------
conninfo = (
    f"host={LOCAL_PG_HOST} port={LOCAL_PG_PORT} "
    f"dbname={PG_DB} user={PG_USER} password={PG_PASSWORD} "
    f"options='-c statement_timeout={STATEMENT_TIMEOUT_MS}'"
)

pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=5)

# ----------------------------
# MCP tools
# ----------------------------
mcp = FastMCP("postgres-tools-over-ssh")


class QueryArgs(BaseModel):
    sql: str = Field(..., description="SQL query. Only SELECT/CTE allowed. Use %s placeholders.")
    params: List[Any] = Field(default_factory=list, description="Parameters for %s placeholders.")
    max_rows: int = Field(DEFAULT_MAX_ROWS, ge=1, le=5000, description="Max rows to return.")


def _is_safe_readonly_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate ", "grant ", "revoke "]
    return not any(tok in s for tok in forbidden)


@mcp.tool()
def pg_query(sql: str, params: List[Any] = None, max_rows: int = DEFAULT_MAX_ROWS) -> Dict[str, Any]:
    """Run a READ-ONLY SQL query over the SSH tunnel. Returns columns + rows."""
    params = params or []

    if not _is_safe_readonly_sql(sql):
        return {"ok": False, "error": "Only read-only queries are allowed (SELECT/CTE)."}

    q = sql.strip().rstrip(";")
    if "limit" not in q.lower():
        q = f"{q}\nLIMIT {max_rows}"

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            cols = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            data = [list(r) for r in rows]

    return {"ok": True, "columns": cols, "rows": data, "row_count": len(data)}


@mcp.tool()
def pg_list_tables(schema: str = "public") -> Dict[str, Any]:
    """List tables in a schema."""
    sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = %s
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [schema])
            tables = [r[0] for r in cur.fetchall()]
    return {"ok": True, "schema": schema, "tables": tables}


@mcp.tool()
def pg_describe_table(table: str, schema: str = "public") -> Dict[str, Any]:
    """Describe columns of a table."""
    sql = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [schema, table])
            cols = [
                {"column_name": r[0], "data_type": r[1], "is_nullable": r[2], "column_default": r[3]}
                for r in cur.fetchall()
            ]
    return {"ok": True, "schema": schema, "table": table, "columns": cols}


if __name__ == "__main__":
    # IMPORTANT: For stdio MCP servers, avoid noisy prints to stdout.
    # If you need debugging, write to stderr.
    mcp.run()
