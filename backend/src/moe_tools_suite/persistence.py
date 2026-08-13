from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .agentic.domain import AgentRun, AgentTrial, InferenceCall, SandboxSession
from .domain import (
    BenchmarkCohort,
    BenchmarkDatasetRecord,
    BenchmarkRun,
    ComparisonRecord,
    EvaluationContract,
    ExecutionPolicy,
    ExpertProfile,
    JobKind,
    JobPhaseRecord,
    JobRecord,
    JobStatus,
    LLMJudgeResult,
    ModelLoadPhase,
    ModelRuntimeRecipe,
    ModelSession,
    ModelState,
    ModelTopology,
    ProfileValidation,
    RoutingSummary,
    RunProvenance,
    SavedExpertProfile,
)
from .v2_domain import (
    EvaluationResultRecord,
    Experiment,
    ExperimentLane,
    ExperimentStatus,
    InterventionContextRef,
    PerformanceSnapshot,
    RunEvent,
    RunEventKind,
    RunUnit,
    RunUnitStatus,
    WorkloadDescriptor,
    WorkloadRun,
    WorkloadRunStatus,
    utc_now,
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
    model_revision: Mapped[str | None] = mapped_column(String(40))
    topology_json: Mapped[str | None] = mapped_column(Text)
    runtime_recipe_json: Mapped[str | None] = mapped_column(Text)
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
    evaluation_contract_json: Mapped[str | None] = mapped_column(Text)
    execution_policy_json: Mapped[str | None] = mapped_column(Text)
    evaluation_contract_fingerprint: Mapped[str | None] = mapped_column(String(64))
    execution_policy_fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RunMetadataRow(Base):
    __tablename__ = "benchmark_run_metadata"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    cohort_id: Mapped[str | None] = mapped_column(String, index=True)
    scored_items: Mapped[int] = mapped_column(Integer, nullable=False)
    provenance_json: Mapped[str | None] = mapped_column(Text)
    record_json: Mapped[str | None] = mapped_column(Text)


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
    metadata_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ExpertProfileAgentTrialRow(Base):
    __tablename__ = "expert_profile_agent_trials"

    profile_id: Mapped[str] = mapped_column(String, primary_key=True)
    trial_id: Mapped[str] = mapped_column(String, nullable=False, index=True)


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
    retry_of_job_id: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
    phase_history_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task_pack_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AgentTrialRow(Base):
    __tablename__ = "agent_trials"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AgentInferenceRow(Base):
    __tablename__ = "agent_inferences"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trial_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class SandboxSessionRow(Base):
    __tablename__ = "agent_sandbox_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trial_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    provider_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String, nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class JudgeEvaluationRow(Base):
    __tablename__ = "judge_evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    request_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    provider: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    rubric_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class InterventionContextRow(Base):
    __tablename__ = "intervention_contexts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    profile_id: Mapped[str | None] = mapped_column(String, index=True)
    profile_fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    context_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    topology_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class WorkloadDescriptorRow(Base):
    __tablename__ = "workload_descriptors"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    content_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    record_json: Mapped[str] = mapped_column(Text, nullable=False)


class ExperimentRow(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    workload_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    latest_event_sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ExperimentLaneRow(Base):
    __tablename__ = "experiment_lanes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)


class WorkloadRunRow(Base):
    __tablename__ = "workload_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lane_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    workload_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RunUnitRow(Base):
    __tablename__ = "run_units"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lane_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    workload_run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RunEventRow(Base):
    __tablename__ = "run_events"

    experiment_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lane_id: Mapped[str | None] = mapped_column(String, index=True)
    workload_run_id: Mapped[str | None] = mapped_column(String, index=True)
    run_unit_id: Mapped[str | None] = mapped_column(String, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ExperimentEvaluationResultRow(Base):
    __tablename__ = "experiment_evaluation_results"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workload_run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    run_unit_id: Mapped[str | None] = mapped_column(String, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PerformanceSnapshotRow(Base):
    __tablename__ = "performance_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workload_run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    run_unit_id: Mapped[str | None] = mapped_column(String, index=True)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


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
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply additive SQLite migrations required by persisted API contracts."""
        additions = {
            "benchmark_cohorts": {
                "evaluation_contract_json": "TEXT",
                "execution_policy_json": "TEXT",
                "evaluation_contract_fingerprint": "VARCHAR(64)",
                "execution_policy_fingerprint": "VARCHAR(64)",
            },
            "benchmark_run_metadata": {"record_json": "TEXT"},
            "expert_profiles": {"metadata_json": "TEXT"},
            "model_sessions": {
                "model_revision": "VARCHAR(40)",
                "topology_json": "TEXT",
                "runtime_recipe_json": "TEXT",
            },
            "jobs": {
                "phase_history_json": "TEXT",
                "retry_of_job_id": "VARCHAR",
            },
            "experiments": {"latest_event_sequence": "INTEGER NOT NULL DEFAULT 0"},
        }
        with self.engine.begin() as connection:
            inspector = inspect(connection)
            for table_name, columns in additions.items():
                existing = {
                    column["name"] for column in inspector.get_columns(table_name)
                }
                for column_name, column_type in columns.items():
                    if column_name in existing:
                        continue
                    connection.execute(
                        text(
                            f'ALTER TABLE "{table_name}" '
                            f'ADD COLUMN "{column_name}" {column_type}'
                        )
                    )
            if inspector.has_table("experiments") and inspector.has_table("run_events"):
                connection.execute(
                    text(
                        "UPDATE experiments SET latest_event_sequence = "
                        "MAX(latest_event_sequence, COALESCE((SELECT MAX(sequence) "
                        "FROM run_events WHERE run_events.experiment_id = "
                        "experiments.id), 0))"
                    )
                )

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
                        model_revision=model_session.model_revision,
                        topology_json=(
                            model_session.topology.model_dump_json()
                            if model_session.topology is not None
                            else None
                        ),
                        runtime_recipe_json=(
                            model_session.runtime_recipe.model_dump_json()
                            if model_session.runtime_recipe is not None
                            else None
                        ),
                        created_at=model_session.created_at,
                        updated_at=now,
                    )
                )
            else:
                row.state = model_session.state.value
                row.profile_json = profile_json
                row.model_revision = model_session.model_revision
                row.topology_json = (
                    model_session.topology.model_dump_json()
                    if model_session.topology is not None
                    else None
                )
                row.runtime_recipe_json = (
                    model_session.runtime_recipe.model_dump_json()
                    if model_session.runtime_recipe is not None
                    else None
                )
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

    def save_model_load_submission(
        self,
        model_session: ModelSession,
        job: JobRecord,
        payload: dict[str, object],
    ) -> None:
        """Persist a planned model session and its job atomically."""
        now = datetime.now(UTC)
        with self.sessions.begin() as database:
            database.add(
                ModelSessionRow(
                    id=model_session.id,
                    model_id=model_session.model_id,
                    state=model_session.state.value,
                    mode=model_session.mode,
                    profile_json=(
                        model_session.profile.model_dump_json()
                        if model_session.profile is not None
                        else None
                    ),
                    model_revision=model_session.model_revision,
                    topology_json=(
                        model_session.topology.model_dump_json()
                        if model_session.topology is not None
                        else None
                    ),
                    runtime_recipe_json=(
                        model_session.runtime_recipe.model_dump_json()
                        if model_session.runtime_recipe is not None
                        else None
                    ),
                    created_at=model_session.created_at,
                    updated_at=now,
                )
            )
            if model_session.profile_id is not None:
                database.add(
                    ModelSessionProfileRow(
                        model_session_id=model_session.id,
                        profile_id=model_session.profile_id,
                    )
                )
            database.add(
                JobRow(
                    id=job.id,
                    kind=job.kind.value,
                    status=job.status.value,
                    progress_current=job.progress_current,
                    progress_total=job.progress_total,
                    payload_json=json.dumps(payload, separators=(",", ":")),
                    result_id=job.result_id,
                    retry_of_job_id=job.retry_of_job_id,
                    error=job.error,
                    phase_history_json=json.dumps(
                        [phase.model_dump(mode="json") for phase in job.phase_history],
                        separators=(",", ":"),
                    ),
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
            )

    def reconcile_interrupted_state(self) -> tuple[int, int]:
        """Fail active sessions and jobs in one restart transaction."""
        session_states = {
            ModelState.STARTING.value,
            ModelState.READY.value,
            ModelState.STOPPING.value,
        }
        job_states = {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.CANCELLING.value,
        }
        now = datetime.now(UTC)
        reconciled_sessions = 0
        reconciled_jobs = 0
        with self.sessions.begin() as database:
            session_rows = database.query(ModelSessionRow).filter(
                ModelSessionRow.state.in_(session_states)
            )
            for row in session_rows:
                row.state = ModelState.FAILED.value
                row.updated_at = now
                reconciled_sessions += 1
            job_rows = database.query(JobRow).filter(JobRow.status.in_(job_states))
            for row in job_rows:
                row.status = JobStatus.FAILED.value
                if row.kind == JobKind.MODEL_LOAD.value and row.result_id is not None:
                    row.error = (
                        "application restarted during model loading; the linked "
                        "model session was marked failed and the request can be retried"
                    )
                    phase_history = (
                        [
                            JobPhaseRecord.model_validate(value)
                            for value in json.loads(row.phase_history_json)
                        ]
                        if row.phase_history_json
                        else []
                    )
                    for phase in reversed(phase_history):
                        if phase.status == "active":
                            phase.status = "failed"
                            phase.completed_at = now
                            break
                    phase_history.append(
                        JobPhaseRecord(
                            phase=ModelLoadPhase.FAILED,
                            status="failed",
                            source="application",
                            started_at=now,
                            completed_at=now,
                            detail=row.error,
                            failure_code="application_restarted",
                            recovery_action=(
                                "Confirm no model load is active, then retry the "
                                "same pinned model."
                            ),
                            diagnostics=row.error,
                        )
                    )
                    row.phase_history_json = json.dumps(
                        [phase.model_dump(mode="json") for phase in phase_history],
                        separators=(",", ":"),
                    )
                else:
                    row.error = "application restarted before the job completed"
                row.completed_at = now
                reconciled_jobs += 1
        return reconciled_sessions, reconciled_jobs

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
                    model_revision=row.model_revision,
                    topology=(
                        ModelTopology.model_validate_json(row.topology_json)
                        if row.topology_json is not None
                        else None
                    ),
                    runtime_recipe=(
                        ModelRuntimeRecipe.model_validate_json(row.runtime_recipe_json)
                        if row.runtime_recipe_json is not None
                        else None
                    ),
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
                    evaluation_contract_json=(
                        cohort.evaluation_contract.model_dump_json()
                        if cohort.evaluation_contract is not None
                        else None
                    ),
                    execution_policy_json=(
                        cohort.execution_policy.model_dump_json()
                        if cohort.execution_policy is not None
                        else None
                    ),
                    evaluation_contract_fingerprint=(
                        cohort.evaluation_contract_fingerprint
                    ),
                    execution_policy_fingerprint=(cohort.execution_policy_fingerprint),
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
                    evaluation_contract=(
                        EvaluationContract.model_validate_json(
                            row.evaluation_contract_json
                        )
                        if row.evaluation_contract_json is not None
                        else None
                    ),
                    execution_policy=(
                        ExecutionPolicy.model_validate_json(row.execution_policy_json)
                        if row.execution_policy_json is not None
                        else None
                    ),
                    evaluation_contract_fingerprint=(
                        row.evaluation_contract_fingerprint
                    ),
                    execution_policy_fingerprint=(row.execution_policy_fingerprint),
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
                    retry_of_job_id=job.retry_of_job_id,
                    error=job.error,
                    phase_history_json=json.dumps(
                        [phase.model_dump(mode="json") for phase in job.phase_history],
                        separators=(",", ":"),
                    ),
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
            )

    def reconcile_interrupted_jobs(self) -> int:
        active_states = {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.CANCELLING.value,
        }
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
                    retry_of_job_id=row.retry_of_job_id,
                    error=row.error,
                    phase_history=(
                        [
                            JobPhaseRecord.model_validate(value)
                            for value in json.loads(row.phase_history_json)
                        ]
                        if row.phase_history_json
                        else []
                    ),
                    created_at=_as_utc(row.created_at),
                    started_at=_as_utc(row.started_at) if row.started_at else None,
                    completed_at=(
                        _as_utc(row.completed_at) if row.completed_at else None
                    ),
                )
                loaded[row.id] = (job, json.loads(row.payload_json))
        return loaded

    def save_agent_run(self, run: AgentRun) -> None:
        with self.sessions.begin() as database:
            database.merge(
                AgentRunRow(
                    id=run.id,
                    status=run.status.value,
                    model_session_id=run.model_session_id,
                    task_pack_id=run.task_pack_id,
                    record_json=run.model_dump_json(),
                    created_at=run.created_at,
                    updated_at=datetime.now(UTC),
                )
            )

    def load_agent_runs(self) -> dict[str, AgentRun]:
        with self.sessions() as database:
            rows = database.query(AgentRunRow).order_by(AgentRunRow.created_at)
            return {
                row.id: AgentRun.model_validate_json(row.record_json) for row in rows
            }

    def save_agent_trial(self, trial: AgentTrial) -> None:
        with self.sessions.begin() as database:
            database.merge(
                AgentTrialRow(
                    id=trial.id,
                    run_id=trial.run_id,
                    task_id=trial.task_id,
                    status=trial.status.value,
                    record_json=trial.model_dump_json(),
                    created_at=trial.created_at,
                    updated_at=datetime.now(UTC),
                )
            )

    def load_agent_trials(self) -> dict[str, AgentTrial]:
        with self.sessions() as database:
            rows = database.query(AgentTrialRow).order_by(AgentTrialRow.created_at)
            return {
                row.id: AgentTrial.model_validate_json(row.record_json) for row in rows
            }

    def save_agent_inference(self, inference: InferenceCall) -> None:
        with self.sessions.begin() as database:
            database.merge(
                AgentInferenceRow(
                    id=inference.id,
                    trial_id=inference.trial_id,
                    record_json=inference.model_dump_json(),
                    created_at=inference.created_at,
                )
            )

    def load_agent_inferences(self) -> dict[str, InferenceCall]:
        with self.sessions() as database:
            rows = database.query(AgentInferenceRow).order_by(
                AgentInferenceRow.created_at
            )
            return {
                row.id: InferenceCall.model_validate_json(row.record_json)
                for row in rows
            }

    def save_sandbox_session(self, sandbox: SandboxSession) -> None:
        with self.sessions.begin() as database:
            database.merge(
                SandboxSessionRow(
                    id=sandbox.id,
                    trial_id=sandbox.trial_id,
                    provider_id=sandbox.provider_id,
                    state=sandbox.state.value,
                    external_id=sandbox.external_id,
                    record_json=sandbox.model_dump_json(),
                    created_at=sandbox.created_at,
                    updated_at=sandbox.updated_at,
                )
            )

    def load_sandbox_sessions(self) -> dict[str, SandboxSession]:
        with self.sessions() as database:
            rows = database.query(SandboxSessionRow).order_by(
                SandboxSessionRow.created_at
            )
            return {
                row.id: SandboxSession.model_validate_json(row.record_json)
                for row in rows
            }

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
                    record_json=run.model_dump_json(exclude={"items"}),
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
                if run_metadata is not None and run_metadata.record_json is not None:
                    run_record = json.loads(run_metadata.record_json)
                    run_record["items"] = items
                    run = BenchmarkRun.model_validate(run_record)
                else:
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
                    metadata_json=json.dumps(
                        {
                            "source_refs": [
                                ref.model_dump(mode="json")
                                for ref in profile.source_refs
                            ],
                            "source_fingerprint": profile.source_fingerprint,
                            "profile_fingerprint_version": (
                                profile.profile_fingerprint_version
                            ),
                            "selection_strategy": profile.selection_strategy,
                            "selection_config": profile.selection_config,
                        },
                        separators=(",", ":"),
                    ),
                    created_at=profile.created_at,
                )
            )
            trial_link = database.get(ExpertProfileAgentTrialRow, profile.id)
            if profile.source_trial_id is not None:
                if trial_link is None:
                    database.add(
                        ExpertProfileAgentTrialRow(
                            profile_id=profile.id,
                            trial_id=profile.source_trial_id,
                        )
                    )
                else:
                    trial_link.trial_id = profile.source_trial_id
            elif trial_link is not None:
                database.delete(trial_link)

    def load_expert_profiles(self) -> dict[str, SavedExpertProfile]:
        loaded: dict[str, SavedExpertProfile] = {}
        with self.sessions() as database:
            trial_ids = {
                row.profile_id: row.trial_id
                for row in database.query(ExpertProfileAgentTrialRow)
            }
            rows = database.query(ExpertProfileRow).order_by(
                ExpertProfileRow.created_at
            )
            for row in rows:
                profile_metadata = (
                    json.loads(row.metadata_json)
                    if row.metadata_json is not None
                    else {}
                )
                loaded[row.id] = SavedExpertProfile(
                    id=row.id,
                    name=row.name,
                    description=row.description,
                    model_id=row.model_id,
                    profile=ExpertProfile.model_validate_json(row.profile_json),
                    profile_fingerprint=row.profile_fingerprint,
                    profile_fingerprint_version=profile_metadata.get(
                        "profile_fingerprint_version"
                    ),
                    source=row.source,
                    source_run_id=row.source_run_id,
                    source_trial_id=trial_ids.get(row.id),
                    source_refs=profile_metadata.get("source_refs", []),
                    source_fingerprint=profile_metadata.get("source_fingerprint"),
                    parent_profile_id=row.parent_profile_id,
                    metric=row.metric,
                    selection_strategy=profile_metadata.get("selection_strategy"),
                    selection_config=profile_metadata.get("selection_config", {}),
                    validation=ProfileValidation.model_validate_json(
                        row.validation_json
                    ),
                    observed_mass_retained=row.observed_mass_retained,
                    created_at=_as_utc(row.created_at),
                )
        return loaded

    def get_judge_result(self, request_hash: str) -> LLMJudgeResult | None:
        with self.sessions() as database:
            row = (
                database.query(JudgeEvaluationRow)
                .filter(JudgeEvaluationRow.request_hash == request_hash)
                .one_or_none()
            )
            return (
                LLMJudgeResult.model_validate_json(row.result_json)
                if row is not None
                else None
            )

    def save_judge_result(self, result: LLMJudgeResult) -> None:
        with self.sessions.begin() as database:
            database.merge(
                JudgeEvaluationRow(
                    id=result.id,
                    request_hash=result.request_hash,
                    provider=result.provider.value,
                    model=result.model,
                    config_fingerprint=result.config_fingerprint,
                    rubric_hash=result.rubric_hash,
                    result_json=result.model_dump_json(),
                    created_at=result.created_at,
                )
            )

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

    def save_intervention_context(self, context: InterventionContextRef) -> None:
        with self.sessions.begin() as database:
            database.merge(
                InterventionContextRow(
                    id=context.context_id,
                    model_id=context.model_id,
                    kind=context.kind.value,
                    profile_id=context.profile_id,
                    profile_fingerprint=context.profile_fingerprint,
                    context_fingerprint=context.context_fingerprint,
                    topology_fingerprint=context.topology_fingerprint,
                    record_json=context.model_dump_json(),
                    created_at=utc_now(),
                )
            )

    def load_intervention_contexts(self) -> dict[str, InterventionContextRef]:
        with self.sessions() as database:
            rows = database.query(InterventionContextRow).order_by(
                InterventionContextRow.created_at
            )
            return {
                row.id: InterventionContextRef.model_validate_json(row.record_json)
                for row in rows
            }

    def save_workload_descriptor(self, descriptor: WorkloadDescriptor) -> None:
        with self.sessions.begin() as database:
            database.merge(
                WorkloadDescriptorRow(
                    id=descriptor.id,
                    kind=descriptor.kind.value,
                    content_fingerprint=descriptor.content_fingerprint,
                    record_json=descriptor.model_dump_json(),
                )
            )

    def save_experiment(self, experiment: Experiment) -> None:
        with self.sessions.begin() as database:
            persisted_sequence = database.scalar(
                select(ExperimentRow.latest_event_sequence).where(
                    ExperimentRow.id == experiment.id
                )
            )
            if persisted_sequence is not None:
                experiment.latest_event_sequence = max(
                    experiment.latest_event_sequence, persisted_sequence
                )
            database.merge(self._experiment_row(experiment))
            for lane in experiment.lanes:
                database.merge(self._experiment_lane_row(lane))

    def save_experiment_bundle(
        self,
        experiment: Experiment,
        workload_runs: list[WorkloadRun],
        units: list[RunUnit],
        initial_event: RunEvent,
    ) -> None:
        """Persist an executable experiment graph before it becomes observable."""

        if initial_event.experiment_id != experiment.id or initial_event.sequence != 1:
            raise ValueError("initial experiment event must have sequence one")
        with self.sessions.begin() as database:
            database.merge(self._experiment_row(experiment))
            database.merge(
                WorkloadDescriptorRow(
                    id=experiment.workload.id,
                    kind=experiment.workload.kind.value,
                    content_fingerprint=experiment.workload.content_fingerprint,
                    record_json=experiment.workload.model_dump_json(),
                )
            )
            for lane in experiment.lanes:
                database.merge(self._experiment_lane_row(lane))
                database.merge(
                    InterventionContextRow(
                        id=lane.context.context_id,
                        model_id=lane.context.model_id,
                        kind=lane.context.kind.value,
                        profile_id=lane.context.profile_id,
                        profile_fingerprint=lane.context.profile_fingerprint,
                        context_fingerprint=lane.context.context_fingerprint,
                        topology_fingerprint=lane.context.topology_fingerprint,
                        record_json=lane.context.model_dump_json(),
                        created_at=experiment.created_at,
                    )
                )
            for workload_run in workload_runs:
                database.merge(self._workload_run_row(workload_run))
            for unit in units:
                database.merge(self._run_unit_row(unit))
            database.add(self._run_event_row(initial_event))
            database.execute(
                update(ExperimentRow)
                .where(ExperimentRow.id == experiment.id)
                .values(latest_event_sequence=initial_event.sequence)
            )

    def save_workload_run(self, workload_run: WorkloadRun) -> None:
        with self.sessions.begin() as database:
            database.merge(self._workload_run_row(workload_run))

    def save_run_unit(self, unit: RunUnit) -> None:
        with self.sessions.begin() as database:
            database.merge(self._run_unit_row(unit))
            if unit.evaluation is not None:
                database.merge(self._evaluation_result_row(unit.evaluation))
            if unit.performance is not None:
                database.merge(self._performance_snapshot_row(unit.performance))

    def append_run_event(
        self,
        *,
        experiment_id: str,
        kind: RunEventKind,
        phase: str,
        message: str = "",
        lane_id: str | None = None,
        workload_run_id: str | None = None,
        run_unit_id: str | None = None,
        checkpoint: int | None = None,
        data: dict[str, object] | None = None,
    ) -> RunEvent:
        """Append one event using the next durable per-experiment sequence."""

        with self.sessions.begin() as database:
            next_sequence = database.execute(
                update(ExperimentRow)
                .where(ExperimentRow.id == experiment_id)
                .values(latest_event_sequence=(ExperimentRow.latest_event_sequence + 1))
                .returning(ExperimentRow.latest_event_sequence)
            ).scalar_one_or_none()
            if next_sequence is None:
                raise KeyError(experiment_id)
            event_record = RunEvent(
                experiment_id=experiment_id,
                sequence=int(next_sequence),
                kind=kind,
                phase=phase,
                message=message,
                lane_id=lane_id,
                workload_run_id=workload_run_id,
                run_unit_id=run_unit_id,
                checkpoint=checkpoint,
                data=data or {},
            )
            database.add(self._run_event_row(event_record))
        return event_record

    def save_experiment_event(
        self,
        experiment: Experiment,
        *,
        kind: RunEventKind,
        phase: str,
        message: str = "",
        lane_id: str | None = None,
        workload_runs: list[WorkloadRun] | None = None,
        unit: RunUnit | None = None,
        checkpoint: int | None = None,
        data: dict[str, object] | None = None,
    ) -> RunEvent:
        """Commit experiment state and its next event as one transition."""

        with self.sessions.begin() as database:
            next_sequence = database.execute(
                update(ExperimentRow)
                .where(ExperimentRow.id == experiment.id)
                .values(latest_event_sequence=(ExperimentRow.latest_event_sequence + 1))
                .returning(ExperimentRow.latest_event_sequence)
            ).scalar_one_or_none()
            if next_sequence is None:
                raise KeyError(experiment.id)
            committed_sequence = int(next_sequence)
            persisted_experiment = experiment.model_copy(
                update={"latest_event_sequence": committed_sequence}
            )
            event = RunEvent(
                experiment_id=experiment.id,
                sequence=committed_sequence,
                kind=kind,
                phase=phase,
                message=message,
                lane_id=lane_id,
                workload_run_id=(
                    unit.workload_run_id
                    if unit is not None
                    else (
                        workload_runs[0].id
                        if workload_runs is not None and len(workload_runs) == 1
                        else None
                    )
                ),
                run_unit_id=(unit.id if unit else None),
                checkpoint=checkpoint,
                data=data or {},
            )
            database.merge(self._experiment_row(persisted_experiment))
            for lane in experiment.lanes:
                database.merge(self._experiment_lane_row(lane))
            for workload_run in workload_runs or []:
                database.merge(self._workload_run_row(workload_run))
            if unit is not None:
                database.merge(self._run_unit_row(unit))
                if unit.evaluation is not None:
                    database.merge(self._evaluation_result_row(unit.evaluation))
                if unit.performance is not None:
                    database.merge(self._performance_snapshot_row(unit.performance))
            database.add(self._run_event_row(event))
        experiment.latest_event_sequence = committed_sequence
        return event

    def load_experiments(self) -> dict[str, Experiment]:
        with self.sessions() as database:
            rows = database.query(ExperimentRow).order_by(ExperimentRow.created_at)
            loaded = {
                row.id: Experiment.model_validate_json(row.record_json) for row in rows
            }
            latest_sequences = dict(
                database.query(
                    RunEventRow.experiment_id,
                    func.max(RunEventRow.sequence),
                ).group_by(RunEventRow.experiment_id)
            )
        for experiment_id, experiment in loaded.items():
            experiment.latest_event_sequence = int(
                latest_sequences.get(experiment_id, 0) or 0
            )
        return loaded

    def load_workload_runs(self) -> dict[str, WorkloadRun]:
        with self.sessions() as database:
            rows = database.query(WorkloadRunRow).order_by(WorkloadRunRow.created_at)
            return {
                row.id: WorkloadRun.model_validate_json(row.record_json) for row in rows
            }

    def load_run_units(self) -> dict[str, RunUnit]:
        with self.sessions() as database:
            rows = database.query(RunUnitRow).order_by(
                RunUnitRow.experiment_id, RunUnitRow.ordinal
            )
            return {
                row.id: RunUnit.model_validate_json(row.record_json) for row in rows
            }

    def load_run_events(
        self,
        experiment_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> tuple[list[RunEvent], int, bool]:
        if after < 0:
            raise ValueError("event cursor cannot be negative")
        if not 1 <= limit <= 1000:
            raise ValueError("event page limit must be between 1 and 1000")
        with self.sessions() as database:
            rows = (
                database.query(RunEventRow)
                .filter(
                    RunEventRow.experiment_id == experiment_id,
                    RunEventRow.sequence > after,
                )
                .order_by(RunEventRow.sequence)
                .limit(limit + 1)
                .all()
            )
            latest = (
                database.query(func.max(RunEventRow.sequence))
                .filter(RunEventRow.experiment_id == experiment_id)
                .scalar()
            )
        has_more = len(rows) > limit
        return (
            [RunEvent.model_validate_json(row.record_json) for row in rows[:limit]],
            int(latest or 0),
            has_more,
        )

    def reconcile_interrupted_experiments(self) -> int:
        """Fail active V2 records without inventing completed work after restart."""

        active_experiments = {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.CANCELLING,
        }
        active_runs = {WorkloadRunStatus.QUEUED, WorkloadRunStatus.RUNNING}
        active_units = {RunUnitStatus.QUEUED, RunUnitStatus.RUNNING}
        now = utc_now()
        reconciled = 0
        with self.sessions.begin() as database:
            for row in database.query(ExperimentRow):
                experiment = Experiment.model_validate_json(row.record_json)
                if experiment.status not in active_experiments:
                    continue
                experiment.status = ExperimentStatus.FAILED
                experiment.error = (
                    "application restarted before the experiment completed"
                )
                experiment.completed_at = now
                for lane in experiment.lanes:
                    if lane.status in active_runs:
                        lane.status = WorkloadRunStatus.FAILED
                    lane_row = database.get(ExperimentLaneRow, lane.id)
                    if lane_row is not None:
                        lane_row.status = lane.status.value
                        lane_row.record_json = lane.model_dump_json()
                latest_sequence = database.scalar(
                    select(func.max(RunEventRow.sequence)).where(
                        RunEventRow.experiment_id == experiment.id
                    )
                )
                next_sequence = int(latest_sequence or 0) + 1
                experiment.latest_event_sequence = next_sequence
                database.add(
                    self._run_event_row(
                        RunEvent(
                            experiment_id=experiment.id,
                            sequence=next_sequence,
                            kind=RunEventKind.FAILED,
                            phase="failed",
                            message=experiment.error,
                            data={"reconciled_after_restart": True},
                            created_at=now,
                        )
                    )
                )
                row.status = experiment.status.value
                row.latest_event_sequence = next_sequence
                row.record_json = experiment.model_dump_json()
                row.updated_at = now
                reconciled += 1
            for row in database.query(WorkloadRunRow):
                run = WorkloadRun.model_validate_json(row.record_json)
                if run.status not in active_runs:
                    continue
                run.status = WorkloadRunStatus.FAILED
                run.error = "application restarted before the workload run completed"
                run.completed_at = now
                row.status = run.status.value
                row.record_json = run.model_dump_json()
                row.updated_at = now
            for row in database.query(RunUnitRow):
                unit = RunUnit.model_validate_json(row.record_json)
                if unit.status not in active_units:
                    continue
                unit.status = RunUnitStatus.ERROR
                unit.completed_at = now
                row.status = unit.status.value
                row.record_json = unit.model_dump_json()
                row.updated_at = now
        return reconciled

    @staticmethod
    def _experiment_row(experiment: Experiment) -> ExperimentRow:
        return ExperimentRow(
            id=experiment.id,
            job_id=experiment.job_id,
            model_id=experiment.model_id,
            workload_id=experiment.workload.id,
            status=experiment.status.value,
            latest_event_sequence=experiment.latest_event_sequence,
            record_json=experiment.model_dump_json(),
            created_at=experiment.created_at,
            updated_at=utc_now(),
        )

    @staticmethod
    def _experiment_lane_row(lane: ExperimentLane) -> ExperimentLaneRow:
        return ExperimentLaneRow(
            id=lane.id,
            experiment_id=lane.experiment_id,
            role=lane.role.value,
            status=lane.status.value,
            record_json=lane.model_dump_json(),
        )

    @staticmethod
    def _workload_run_row(workload_run: WorkloadRun) -> WorkloadRunRow:
        return WorkloadRunRow(
            id=workload_run.id,
            experiment_id=workload_run.experiment_id,
            lane_id=workload_run.lane_id,
            workload_id=workload_run.workload.id,
            status=workload_run.status.value,
            record_json=workload_run.model_dump_json(),
            created_at=workload_run.created_at,
            updated_at=utc_now(),
        )

    @staticmethod
    def _run_unit_row(unit: RunUnit) -> RunUnitRow:
        return RunUnitRow(
            id=unit.id,
            experiment_id=unit.experiment_id,
            lane_id=unit.lane_id,
            workload_run_id=unit.workload_run_id,
            ordinal=unit.ordinal,
            unit_key=unit.unit_key,
            status=unit.status.value,
            record_json=unit.model_dump_json(),
            created_at=unit.created_at,
            updated_at=utc_now(),
        )

    @staticmethod
    def _run_event_row(event_record: RunEvent) -> RunEventRow:
        return RunEventRow(
            experiment_id=event_record.experiment_id,
            sequence=event_record.sequence,
            kind=event_record.kind.value,
            lane_id=event_record.lane_id,
            workload_run_id=event_record.workload_run_id,
            run_unit_id=event_record.run_unit_id,
            record_json=event_record.model_dump_json(),
            created_at=event_record.created_at,
        )

    @staticmethod
    def _evaluation_result_row(
        result: EvaluationResultRecord,
    ) -> ExperimentEvaluationResultRow:
        return ExperimentEvaluationResultRow(
            id=result.id,
            workload_run_id=result.workload_run_id,
            run_unit_id=result.run_unit_id,
            record_json=result.model_dump_json(),
            created_at=result.created_at,
        )

    @staticmethod
    def _performance_snapshot_row(
        snapshot: PerformanceSnapshot,
    ) -> PerformanceSnapshotRow:
        return PerformanceSnapshotRow(
            id=snapshot.id,
            workload_run_id=snapshot.workload_run_id,
            run_unit_id=snapshot.run_unit_id,
            record_json=snapshot.model_dump_json(),
            created_at=snapshot.captured_at,
        )

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
