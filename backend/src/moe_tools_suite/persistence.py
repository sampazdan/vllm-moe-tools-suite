from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .domain import (
    BenchmarkRun,
    ExpertProfile,
    JobRecord,
    JobStatus,
    ModelSession,
    ModelState,
    RoutingSummary,
)


class Base(DeclarativeBase):
    pass


class ModelSessionRow(Base):
    __tablename__ = "model_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    profile_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class BenchmarkRunRow(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    completed_items: Mapped[int] = mapped_column(Integer, nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, nullable=False)
    items_json: Mapped[str] = mapped_column(Text, nullable=False)
    routing_artifact: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    progress_current: Mapped[int] = mapped_column(Integer, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_id: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SqliteStore:
    """Durable metadata and routing artifacts for one application writer."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.artifact_dir = self.data_dir / "artifacts" / "routing"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.data_dir / "metadata.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        event.listen(self.engine, "connect", _configure_sqlite)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def save_model_session(self, model_session: ModelSession) -> None:
        now = datetime.now(UTC)
        with self.sessions.begin() as database:
            row = database.get(ModelSessionRow, model_session.id)
            profile_json = (
                model_session.profile.model_dump_json()
                if model_session.profile is not None
                else None
            )
            if row is None:
                database.add(
                    ModelSessionRow(
                        id=model_session.id,
                        model_id=model_session.model_id,
                        state=model_session.state.value,
                        mode=model_session.mode,
                        profile_json=profile_json,
                        created_at=model_session.created_at,
                        updated_at=now,
                    )
                )
            else:
                row.state = model_session.state.value
                row.profile_json = profile_json
                row.updated_at = now

    def reconcile_interrupted_sessions(self) -> int:
        active_states = {
            ModelState.STARTING.value,
            ModelState.READY.value,
            ModelState.STOPPING.value,
        }
        reconciled = 0
        with self.sessions.begin() as database:
            rows = database.query(ModelSessionRow).filter(
                ModelSessionRow.state.in_(active_states)
            )
            for row in rows:
                row.state = ModelState.FAILED.value
                row.updated_at = datetime.now(UTC)
                reconciled += 1
        return reconciled

    def load_model_sessions(self) -> dict[str, ModelSession]:
        loaded: dict[str, ModelSession] = {}
        with self.sessions() as database:
            rows = database.query(ModelSessionRow).order_by(
                ModelSessionRow.created_at
            )
            for row in rows:
                profile = (
                    ExpertProfile.model_validate_json(row.profile_json)
                    if row.profile_json is not None
                    else None
                )
                loaded[row.id] = ModelSession(
                    id=row.id,
                    model_id=row.model_id,
                    state=row.state,
                    mode=row.mode,
                    profile=profile,
                    created_at=_as_utc(row.created_at),
                )
        return loaded

    def save_job(self, job: JobRecord, payload: dict[str, object]) -> None:
        with self.sessions.begin() as database:
            database.merge(
                JobRow(
                    id=job.id,
                    kind=job.kind.value,
                    status=job.status.value,
                    progress_current=job.progress_current,
                    progress_total=job.progress_total,
                    payload_json=json.dumps(payload, separators=(",", ":")),
                    result_id=job.result_id,
                    error=job.error,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
            )

    def reconcile_interrupted_jobs(self) -> int:
        active_states = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
        reconciled = 0
        with self.sessions.begin() as database:
            rows = database.query(JobRow).filter(JobRow.status.in_(active_states))
            for row in rows:
                row.status = JobStatus.FAILED.value
                row.error = "application restarted before the job completed"
                row.completed_at = datetime.now(UTC)
                reconciled += 1
        return reconciled

    def load_jobs(self) -> dict[str, tuple[JobRecord, dict[str, object]]]:
        loaded: dict[str, tuple[JobRecord, dict[str, object]]] = {}
        with self.sessions() as database:
            rows = database.query(JobRow).order_by(JobRow.created_at)
            for row in rows:
                job = JobRecord(
                    id=row.id,
                    kind=row.kind,
                    status=row.status,
                    progress_current=row.progress_current,
                    progress_total=row.progress_total,
                    result_id=row.result_id,
                    error=row.error,
                    created_at=_as_utc(row.created_at),
                    started_at=_as_utc(row.started_at) if row.started_at else None,
                    completed_at=(
                        _as_utc(row.completed_at) if row.completed_at else None
                    ),
                )
                loaded[row.id] = (job, json.loads(row.payload_json))
        return loaded

    def save_run(self, run: BenchmarkRun, routing: RoutingSummary) -> None:
        relative_artifact = Path("artifacts") / "routing" / f"{run.id}.npz"
        artifact_path = self.data_dir / relative_artifact
        temporary_path = artifact_path.with_name(
            f".{artifact_path.name}.{uuid4().hex}.tmp"
        )
        with temporary_path.open("wb") as artifact:
            np.savez_compressed(
                artifact,
                layer_ids=np.asarray(routing.layer_ids, dtype=np.int32),
                selection_counts=np.asarray(
                    routing.selection_counts, dtype=np.int64
                ),
                routing_mass=np.asarray(routing.routing_mass, dtype=np.float64),
                total_routed_slots=np.asarray(
                    routing.total_routed_slots, dtype=np.int64
                ),
            )
        os.replace(temporary_path, artifact_path)

        with self.sessions.begin() as database:
            database.merge(
                BenchmarkRunRow(
                    id=run.id,
                    benchmark_id=run.benchmark_id,
                    model_session_id=run.model_session_id,
                    status=run.status,
                    score=run.score,
                    completed_items=run.completed_items,
                    total_items=run.total_items,
                    items_json=json.dumps(
                        [item.model_dump(mode="json") for item in run.items],
                        separators=(",", ":"),
                    ),
                    routing_artifact=relative_artifact.as_posix(),
                    created_at=run.created_at,
                )
            )

    def load_runs(self) -> dict[str, tuple[BenchmarkRun, RoutingSummary]]:
        loaded: dict[str, tuple[BenchmarkRun, RoutingSummary]] = {}
        with self.sessions() as database:
            rows = database.query(BenchmarkRunRow).order_by(
                BenchmarkRunRow.created_at
            )
            for row in rows:
                artifact_path = self._resolve_artifact(row.routing_artifact)
                if not artifact_path.is_file():
                    continue
                with np.load(artifact_path, allow_pickle=False) as artifact:
                    routing = RoutingSummary(
                        run_id=row.id,
                        layer_ids=artifact["layer_ids"].astype(int).tolist(),
                        selection_counts=artifact["selection_counts"].tolist(),
                        routing_mass=artifact["routing_mass"].tolist(),
                        total_routed_slots=int(artifact["total_routed_slots"]),
                    )
                run = BenchmarkRun(
                    id=row.id,
                    benchmark_id=row.benchmark_id,
                    model_session_id=row.model_session_id,
                    status=row.status,
                    score=row.score,
                    completed_items=row.completed_items,
                    total_items=row.total_items,
                    items=json.loads(row.items_json),
                    created_at=_as_utc(row.created_at),
                )
                loaded[row.id] = (run, routing)
        return loaded

    def count_rows(self, model) -> int:
        with self.sessions() as database:
            return database.query(model).count()

    def _resolve_artifact(self, relative_path: str) -> Path:
        resolved = (self.data_dir / relative_path).resolve()
        if not resolved.is_relative_to(self.artifact_dir):
            raise ValueError("routing artifact escaped the configured data directory")
        return resolved


def _configure_sqlite(dbapi_connection, connection_record) -> None:
    del connection_record
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=DELETE")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
