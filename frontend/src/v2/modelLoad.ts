import type { JobPhaseRecord, JobRecord } from "../types";

export function currentModelLoadPhase(job: JobRecord): JobPhaseRecord | null {
  const history = job.phase_history ?? [];
  return [...history].reverse().find((record) => record.status === "active")
    ?? history.at(-1)
    ?? null;
}

export function failedModelLoadPhase(job: JobRecord): JobPhaseRecord | null {
  return [...(job.phase_history ?? [])].reverse().find(
    (record) => record.status === "failed" || record.phase === "failed",
  ) ?? null;
}

export function failedModelLoadStage(job: JobRecord): JobPhaseRecord | null {
  return [...(job.phase_history ?? [])].reverse().find(
    (record) => record.status === "failed" && record.phase !== "failed",
  ) ?? null;
}

export function modelLoadElapsedMs(job: JobRecord, now = Date.now()): number {
  const start = Date.parse(job.started_at ?? job.created_at);
  const end = job.completed_at ? Date.parse(job.completed_at) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  return Math.max(0, end - start);
}

export function modelLoadPhaseElapsedMs(
  record: JobPhaseRecord,
  now = Date.now(),
): number | null {
  if (record.observability === "unavailable") return null;
  const start = Date.parse(record.started_at);
  const end = record.completed_at ? Date.parse(record.completed_at) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return Math.max(0, end - start);
}

export function canRetryModelLoad(job: JobRecord): boolean {
  if (job.kind !== "model_load") return false;
  if (job.status !== "failed" && job.status !== "cancelled") return false;
  return failedModelLoadPhase(job)?.failure_code !== "cleanup_failed";
}

export function phaseCounters(record: JobPhaseRecord): string | null {
  const counters: string[] = [];
  if (record.bytes_current != null || record.bytes_total != null) {
    counters.push(
      `${formatBytes(record.bytes_current)} / ${formatBytes(record.bytes_total)}`,
    );
  }
  if (record.files_current != null || record.files_total != null) {
    counters.push(`${record.files_current ?? "—"} / ${record.files_total ?? "—"} files`);
  }
  return counters.length > 0 ? counters.join(" · ") : null;
}

export function formatDuration(milliseconds: number | null): string {
  if (milliseconds == null) return "Unavailable";
  const seconds = Math.max(0, Math.floor(milliseconds / 1_000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainder}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatBytes(value: number | null): string {
  if (value == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
