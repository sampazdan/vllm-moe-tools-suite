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
    BenchmarkCohort,
    BenchmarkDatasetRecord,
    BenchmarkRun,
    ComparisonRecord,
    ExpertProfile,
    JobRecord,
    JobStatus,
    ModelSession,
    ModelState,
    ProfileValidation,
    RoutingSummary,
    RunProvenance,
    SavedExpertProfile,
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


class ModelSessionProfileRow(Base):
    __tablename__ = "model_session_profiles"

    model_session_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False, index=True)


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


class BenchmarkDatasetRow(Base):
    __tablename__ = "benchmark_datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    info_json: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class BenchmarkCohortRow(Base):
    __tablename__ = "benchmark_cohorts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    benchmark_revision: Mapped[str] = mapped_column(String, nullable=False)
    dataset_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_template_version: Mapped[str] = mapped_column(String, nullable=False)
    scoring_version: Mapped[str] = mapped_column(String, nullable=False)
    item_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    generation_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RunMetadataRow(Base):
    __tablename__ = "benchmark_run_metadata"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    cohort_id: Mapped[str | None] = mapped_column(String, index=True)
    scored_items: Mapped[int] = mapped_column(Integer, nullable=False)
    provenance_json: Mapped[str | None] = mapped_column(Text)


class ExpertProfileRow(Base):
    __tablename__ = "expert_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    profile_json: Mapped[str] = mapped_column(Text, nullable=False)
    profile_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(String, index=True)
    parent_profile_id: Mapped[str | None] = mapped_column(String, index=True)
    metric: Mapped[str | None] = mapped_column(String)
    validation_json: Mapped[str] = mapped_column(Text, nullable=False)
    observed_mass_retained: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ComparisonRow(Base):
    __tablename__ = "comparisons"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    baseline_run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    candidate_run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    profile_id: Mapped[str | None] = mapped_column(String, index=True)
    profile_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    benchmark_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    cohort_item_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    baseline_score: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_delta: Mapped[float] = mapped_column(Float, nullable=False)
    regressions: Mapped[int] = mapped_column(Integer, nullable=False)
    recoveries: Mapped[int] = mapped_column(Integer, nullable=False)
    retained_passes: Mapped[int] = mapped_column(Integer, nullable=False)
    retained_failures: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ComparisonMetadataRow(Base):
    __tablename__ = "comparison_metadata"

    comparison_id: Mapped[str] = mapped_column(String, primary_key=True)
    unscored_items: Mapped[int] = mapped_column(Integer, nullable=False)


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
            profile_link = database.get(ModelSessionProfileRow, model_session.id)
            if model_session.profile_id is not None:
                if profile_link is None:
                    database.add(
                        ModelSessionProfileRow(
                            model_session_id=model_session.id,
                            profile_id=model_session.profile_id,
                        )
                    )
                else:
                    profile_link.profile_id = model_session.profile_id
            elif profile_link is not None:
                database.delete(profile_link)

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
            profile_ids = {
                row.model_session_id: row.profile_id
                for row in database.query(ModelSessionProfileRow)
            }
            rows = database.query(ModelSessionRow).order_by(ModelSessionRow.created_at)
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
                    profile_id=profile_ids.get(row.id),
                    created_at=_as_utc(row.created_at),
                )
        return loaded

    def save_benchmark_dataset(self, dataset: BenchmarkDatasetRecord) -> None:
        with self.sessions.begin() as database:
            database.merge(
                BenchmarkDatasetRow(
                    id=dataset.id,
                    info_json=dataset.info.model_dump_json(),
                    content_hash=dataset.content_hash,
                    created_at=dataset.created_at,
                )
            )

    def load_benchmark_datasets(self) -> dict[str, BenchmarkDatasetRecord]:
        loaded: dict[str, BenchmarkDatasetRecord] = {}
        with self.sessions() as database:
            rows = database.query(BenchmarkDatasetRow).order_by(
                BenchmarkDatasetRow.created_at
            )
            for row in rows:
                loaded[row.id] = BenchmarkDatasetRecord(
                    id=row.id,
                    info=json.loads(row.info_json),
                    content_hash=row.content_hash,
                    created_at=_as_utc(row.created_at),
                )
        return loaded

    def save_cohort(self, cohort: BenchmarkCohort) -> None:
        with self.sessions.begin() as database:
            database.merge(
                BenchmarkCohortRow(
                    id=cohort.id,
                    benchmark_id=cohort.benchmark_id,
                    benchmark_revision=cohort.benchmark_revision,
                    dataset_content_hash=cohort.dataset_content_hash,
                    prompt_template_version=cohort.prompt_template_version,
                    scoring_version=cohort.scoring_version,
                    item_ids_json=json.dumps(cohort.item_ids, separators=(",", ":")),
                    fingerprint=cohort.fingerprint,
                    generation_json=cohort.generation.model_dump_json(),
                    created_at=cohort.created_at,
                )
            )

    def load_cohorts(self) -> dict[str, BenchmarkCohort]:
        loaded: dict[str, BenchmarkCohort] = {}
        with self.sessions() as database:
            rows = database.query(BenchmarkCohortRow).order_by(
                BenchmarkCohortRow.created_at
            )
            for row in rows:
                loaded[row.id] = BenchmarkCohort(
                    id=row.id,
                    benchmark_id=row.benchmark_id,
                    benchmark_revision=row.benchmark_revision,
                    dataset_content_hash=row.dataset_content_hash,
                    prompt_template_version=row.prompt_template_version,
                    scoring_version=row.scoring_version,
                    item_ids=json.loads(row.item_ids_json),
                    fingerprint=row.fingerprint,
                    generation=json.loads(row.generation_json),
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

    def save_run(
        self,
        run: BenchmarkRun,
        routing: RoutingSummary,
        provenance: RunProvenance | None = None,
    ) -> None:
        relative_artifact = Path("artifacts") / "routing" / f"{run.id}.npz"
        artifact_path = self.data_dir / relative_artifact
        temporary_path = artifact_path.with_name(
            f".{artifact_path.name}.{uuid4().hex}.tmp"
        )
        with temporary_path.open("wb") as artifact:
            np.savez_compressed(
                artifact,
                layer_ids=np.asarray(routing.layer_ids, dtype=np.int32),
                selection_counts=np.asarray(routing.selection_counts, dtype=np.int64),
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
                    score=run.score if run.score is not None else -1.0,
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
            database.merge(
                RunMetadataRow(
                    run_id=run.id,
                    cohort_id=run.cohort_id,
                    scored_items=run.scored_items,
                    provenance_json=(
                        provenance.model_dump_json() if provenance is not None else None
                    ),
                )
            )

    def load_runs(
        self,
    ) -> dict[str, tuple[BenchmarkRun, RoutingSummary, RunProvenance | None]]:
        loaded: dict[
            str, tuple[BenchmarkRun, RoutingSummary, RunProvenance | None]
        ] = {}
        with self.sessions() as database:
            metadata = {row.run_id: row for row in database.query(RunMetadataRow)}
            rows = database.query(BenchmarkRunRow).order_by(BenchmarkRunRow.created_at)
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
                run_metadata = metadata.get(row.id)
                items = json.loads(row.items_json)
                run = BenchmarkRun(
                    id=row.id,
                    benchmark_id=row.benchmark_id,
                    model_session_id=row.model_session_id,
                    status=row.status,
                    score=row.score if row.score >= 0 else None,
                    scored_items=(
                        run_metadata.scored_items
                        if run_metadata is not None
                        else sum(item.get("passed") is not None for item in items)
                    ),
                    completed_items=row.completed_items,
                    total_items=row.total_items,
                    items=items,
                    cohort_id=(
                        run_metadata.cohort_id if run_metadata is not None else None
                    ),
                    created_at=_as_utc(row.created_at),
                )
                provenance = (
                    RunProvenance.model_validate_json(run_metadata.provenance_json)
                    if run_metadata is not None
                    and run_metadata.provenance_json is not None
                    else None
                )
                loaded[row.id] = (run, routing, provenance)
        return loaded

    def save_expert_profile(self, profile: SavedExpertProfile) -> None:
        with self.sessions.begin() as database:
            database.merge(
                ExpertProfileRow(
                    id=profile.id,
                    name=profile.name,
                    description=profile.description,
                    model_id=profile.model_id,
                    profile_json=profile.profile.model_dump_json(),
                    profile_fingerprint=profile.profile_fingerprint,
                    source=profile.source.value,
                    source_run_id=profile.source_run_id,
                    parent_profile_id=profile.parent_profile_id,
                    metric=profile.metric,
                    validation_json=profile.validation.model_dump_json(),
                    observed_mass_retained=profile.observed_mass_retained,
                    created_at=profile.created_at,
                )
            )

    def load_expert_profiles(self) -> dict[str, SavedExpertProfile]:
        loaded: dict[str, SavedExpertProfile] = {}
        with self.sessions() as database:
            rows = database.query(ExpertProfileRow).order_by(
                ExpertProfileRow.created_at
            )
            for row in rows:
                loaded[row.id] = SavedExpertProfile(
                    id=row.id,
                    name=row.name,
                    description=row.description,
                    model_id=row.model_id,
                    profile=ExpertProfile.model_validate_json(row.profile_json),
                    profile_fingerprint=row.profile_fingerprint,
                    source=row.source,
                    source_run_id=row.source_run_id,
                    parent_profile_id=row.parent_profile_id,
                    metric=row.metric,
                    validation=ProfileValidation.model_validate_json(
                        row.validation_json
                    ),
                    observed_mass_retained=row.observed_mass_retained,
                    created_at=_as_utc(row.created_at),
                )
        return loaded

    def save_comparison(self, comparison: ComparisonRecord) -> None:
        with self.sessions.begin() as database:
            database.merge(
                ComparisonRow(
                    id=comparison.id,
                    name=comparison.name,
                    baseline_run_id=comparison.baseline_run_id,
                    candidate_run_id=comparison.candidate_run_id,
                    profile_id=comparison.profile_id,
                    profile_fingerprint=comparison.profile_fingerprint,
                    benchmark_id=comparison.benchmark_id,
                    cohort_item_ids_json=json.dumps(
                        comparison.cohort_item_ids, separators=(",", ":")
                    ),
                    baseline_score=(
                        comparison.baseline_score
                        if comparison.baseline_score is not None
                        else -1.0
                    ),
                    candidate_score=(
                        comparison.candidate_score
                        if comparison.candidate_score is not None
                        else -1.0
                    ),
                    score_delta=(
                        comparison.score_delta
                        if comparison.score_delta is not None
                        else -2.0
                    ),
                    regressions=comparison.regressions,
                    recoveries=comparison.recoveries,
                    retained_passes=comparison.retained_passes,
                    retained_failures=comparison.retained_failures,
                    created_at=comparison.created_at,
                )
            )
            database.merge(
                ComparisonMetadataRow(
                    comparison_id=comparison.id,
                    unscored_items=comparison.unscored_items,
                )
            )

    def load_comparisons(self) -> dict[str, ComparisonRecord]:
        loaded: dict[str, ComparisonRecord] = {}
        with self.sessions() as database:
            metadata = {
                row.comparison_id: row for row in database.query(ComparisonMetadataRow)
            }
            rows = database.query(ComparisonRow).order_by(ComparisonRow.created_at)
            for row in rows:
                loaded[row.id] = ComparisonRecord(
                    id=row.id,
                    name=row.name,
                    baseline_run_id=row.baseline_run_id,
                    candidate_run_id=row.candidate_run_id,
                    profile_id=row.profile_id,
                    profile_fingerprint=row.profile_fingerprint,
                    benchmark_id=row.benchmark_id,
                    cohort_item_ids=json.loads(row.cohort_item_ids_json),
                    baseline_score=(
                        row.baseline_score if row.baseline_score >= 0 else None
                    ),
                    candidate_score=(
                        row.candidate_score if row.candidate_score >= 0 else None
                    ),
                    score_delta=(row.score_delta if row.score_delta >= -1 else None),
                    regressions=row.regressions,
                    recoveries=row.recoveries,
                    retained_passes=row.retained_passes,
                    retained_failures=row.retained_failures,
                    unscored_items=(
                        metadata[row.id].unscored_items if row.id in metadata else 0
                    ),
                    created_at=_as_utc(row.created_at),
                )
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
