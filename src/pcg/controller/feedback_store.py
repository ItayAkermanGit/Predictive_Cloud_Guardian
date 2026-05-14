# SQLite-backed feedback store for the active learning loop.
#
# Schema overview (3 tables):
#
# alerts
#   Stores every fired alert with its full context. This is the ground truth
#   record of what the system decided and when.
#
# feedback
#   Human (or automated) label attached to a fired alert:
#     - "true_positive"  → real incident, model was correct.
#     - "false_positive" → alert fired but no incident occurred.
#     - "false_negative" → incident occurred but no alert fired (manually logged).
#   false_positive rows are the primary input to the retraining queue.
#
# retraining_queue
#   Each false-positive feedback record places a row here. The retraining
#   process reads pending rows, uses them to adjust thresholds or collect
#   training windows, and marks them processed.
#
# Thread safety: each call opens its own connection (check_same_thread=False).
# For production use a connection pool; for this system SQLite is sufficient.

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterator, Literal, Optional

FeedbackLabel = Literal["true_positive", "false_positive", "false_negative"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id       TEXT    NOT NULL,
    fired_at        TEXT    NOT NULL,
    severity        TEXT    NOT NULL,
    combined_risk   REAL    NOT NULL,
    forecast_risk   REAL    NOT NULL,
    anomaly_risk    REAL    NOT NULL,
    anomaly_index   REAL    NOT NULL,
    root_cause      TEXT,
    breaches_json   TEXT    NOT NULL,  -- JSON dict of metric->margin
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        INTEGER NOT NULL REFERENCES alerts(id),
    label           TEXT    NOT NULL CHECK(label IN ('true_positive','false_positive','false_negative')),
    notes           TEXT,
    submitted_by    TEXT    NOT NULL DEFAULT 'system',
    submitted_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS retraining_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id     INTEGER NOT NULL REFERENCES feedback(id),
    alert_id        INTEGER NOT NULL REFERENCES alerts(id),
    server_id       TEXT    NOT NULL,
    fired_at        TEXT    NOT NULL,
    combined_risk   REAL    NOT NULL,
    reason          TEXT    NOT NULL,   -- human-readable why this went to queue
    status          TEXT    NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending','processing','done','skipped')),
    enqueued_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    processed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_server ON alerts(server_id);
CREATE INDEX IF NOT EXISTS idx_alerts_fired  ON alerts(fired_at);
CREATE INDEX IF NOT EXISTS idx_feedback_alert ON feedback(alert_id);
CREATE INDEX IF NOT EXISTS idx_queue_status  ON retraining_queue(status);
"""


@dataclass
class StoredAlert:
    id: int
    server_id: str
    fired_at: str
    severity: str
    combined_risk: float
    forecast_risk: float
    anomaly_risk: float
    anomaly_index: float
    root_cause: Optional[str]
    breaches: dict[str, float]
    created_at: str


@dataclass
class QueueItem:
    id: int
    feedback_id: int
    alert_id: int
    server_id: str
    fired_at: str
    combined_risk: float
    reason: str
    status: str
    enqueued_at: str
    processed_at: Optional[str]


class FeedbackStore:
    # Persistent store for alerts, feedback labels, and the retraining queue.
    #
    # For ":memory:" databases a single persistent connection is kept open
    # because each new sqlite3.connect(":memory:") creates a separate,
    # empty database — sharing the same connection avoids that pitfall.
    # For file-based databases a new connection is opened per operation,
    # which is the correct pattern for multi-process file access.

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._in_memory = self._db_path == ":memory:"
        self._shared_conn: Optional[sqlite3.Connection] = None
        if self._in_memory:
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        self._init_schema()

    # ─── Schema ──────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        if self._in_memory:
            assert self._shared_conn is not None
            self._shared_conn.executescript(_SCHEMA_SQL)
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            try:
                conn.executescript(_SCHEMA_SQL)
            finally:
                conn.close()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._in_memory:
            # Re-use the single shared connection; don't close it.
            assert self._shared_conn is not None
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ─── Alerts ──────────────────────────────────────────────────────────────

    def insert_alert(
        self,
        server_id: str,
        fired_at: datetime,
        severity: str,
        combined_risk: float,
        forecast_risk: float,
        anomaly_risk: float,
        anomaly_index: float,
        root_cause: Optional[str],
        breaches: dict[str, float],
    ) -> int:
        # Insert a fired alert and return its auto-assigned row id.
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts
                    (server_id, fired_at, severity, combined_risk, forecast_risk,
                     anomaly_risk, anomaly_index, root_cause, breaches_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    server_id,
                    fired_at.isoformat(),
                    severity,
                    combined_risk,
                    forecast_risk,
                    anomaly_risk,
                    anomaly_index,
                    root_cause,
                    json.dumps(breaches),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_alert(self, alert_id: int) -> Optional[StoredAlert]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_alert(row)

    def list_alerts(
        self,
        server_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[StoredAlert]:
        with self._connect() as conn:
            if server_id:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE server_id = ? ORDER BY fired_at DESC LIMIT ?",
                    (server_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY fired_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> StoredAlert:
        return StoredAlert(
            id=row["id"],
            server_id=row["server_id"],
            fired_at=row["fired_at"],
            severity=row["severity"],
            combined_risk=row["combined_risk"],
            forecast_risk=row["forecast_risk"],
            anomaly_risk=row["anomaly_risk"],
            anomaly_index=row["anomaly_index"],
            root_cause=row["root_cause"],
            breaches=json.loads(row["breaches_json"]),
            created_at=row["created_at"],
        )

    # ─── Feedback ─────────────────────────────────────────────────────────────

    def submit_feedback(
        self,
        alert_id: int,
        label: FeedbackLabel,
        notes: Optional[str] = None,
        submitted_by: str = "system",
    ) -> int:
        # Record a feedback label for an alert.
        # If the label is "false_positive", automatically enqueue for retraining.
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback (alert_id, label, notes, submitted_by)
                VALUES (?, ?, ?, ?)
                """,
                (alert_id, label, notes, submitted_by),
            )
            feedback_id = cur.lastrowid

            if label == "false_positive":
                # Fetch alert details to populate the queue row.
                row = conn.execute(
                    "SELECT * FROM alerts WHERE id = ?", (alert_id,)
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        INSERT INTO retraining_queue
                            (feedback_id, alert_id, server_id, fired_at,
                             combined_risk, reason)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            feedback_id,
                            alert_id,
                            row["server_id"],
                            row["fired_at"],
                            row["combined_risk"],
                            f"false_positive reported by {submitted_by}",
                        ),
                    )

        return feedback_id  # type: ignore[return-value]

    def get_feedback_for_alert(self, alert_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE alert_id = ? ORDER BY submitted_at",
                (alert_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Retraining queue ─────────────────────────────────────────────────────

    def pending_queue_items(self, limit: int = 50) -> list[QueueItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM retraining_queue
                WHERE status = 'pending'
                ORDER BY enqueued_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_queue(r) for r in rows]

    def mark_queue_item(
        self,
        queue_id: int,
        status: Literal["processing", "done", "skipped"],
    ) -> None:
        processed_at = datetime.utcnow().isoformat() if status in ("done", "skipped") else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE retraining_queue
                SET status = ?, processed_at = ?
                WHERE id = ?
                """,
                (status, processed_at, queue_id),
            )

    def queue_length(self, status: str = "pending") -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM retraining_queue WHERE status = ?",
                (status,),
            ).fetchone()
        return row["n"] if row else 0

    @staticmethod
    def _row_to_queue(row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            id=row["id"],
            feedback_id=row["feedback_id"],
            alert_id=row["alert_id"],
            server_id=row["server_id"],
            fired_at=row["fired_at"],
            combined_risk=row["combined_risk"],
            reason=row["reason"],
            status=row["status"],
            enqueued_at=row["enqueued_at"],
            processed_at=row["processed_at"],
        )
