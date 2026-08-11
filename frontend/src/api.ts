import type { JobRecord } from "./types";

let csrfToken: string | null = null;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class JobTerminalError extends Error {
  constructor(readonly job: JobRecord) {
    super(job.error ?? (job.status === "cancelled" ? "Job cancelled" : `Job ${job.status}`));
    this.name = "JobTerminalError";
  }
}

export interface WaitForJobOptions {
  onProgress?: (job: JobRecord) => void;
  onTransientError?: (error: Error | null, attempt: number) => void;
  signal?: AbortSignal;
  pollIntervalMs?: number;
  maxRetryIntervalMs?: number;
  getJob?: (jobId: string, signal?: AbortSignal) => Promise<JobRecord>;
  wait?: (milliseconds: number, signal?: AbortSignal) => Promise<void>;
}

export function setCsrfToken(token: string | null) {
  csrfToken = token;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method?.toUpperCase() ?? "GET";
  const mutation = ["POST", "PUT", "PATCH", "DELETE"].includes(method);
  const response = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(mutation && csrfToken ? { "X-CSRF-Token": csrfToken } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: unknown;
      message?: string;
    } | null;
    const detail = payload?.detail;
    throw new ApiError(
      errorMessage(detail, payload?.message, response.status),
      response.status,
      detail,
    );
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function waitForJob(
  submitted: JobRecord,
  optionsOrProgress: WaitForJobOptions | ((job: JobRecord) => void) = {},
): Promise<JobRecord> {
  const options = typeof optionsOrProgress === "function"
    ? { onProgress: optionsOrProgress }
    : optionsOrProgress;
  const pollInterval = options.pollIntervalMs ?? 500;
  const maxRetryInterval = options.maxRetryIntervalMs ?? 5_000;
  const getJob = options.getJob ?? ((jobId: string, signal?: AbortSignal) =>
    api<JobRecord>(`/api/jobs/${jobId}`, { signal }));
  const wait = options.wait ?? waitForPoll;
  let job = submitted;
  let transientAttempts = 0;
  options.onProgress?.(job);
  while (isActiveJob(job)) {
    const delay = transientAttempts === 0
      ? pollInterval
      : Math.min(pollInterval * 2 ** transientAttempts, maxRetryInterval);
    await wait(delay, options.signal);
    try {
      job = await getJob(job.id, options.signal);
      if (transientAttempts > 0) options.onTransientError?.(null, 0);
      transientAttempts = 0;
      options.onProgress?.(job);
    } catch (error) {
      if (options.signal?.aborted) throw abortError();
      if (!isTransientApiError(error)) throw error;
      transientAttempts += 1;
      options.onTransientError?.(asError(error), transientAttempts);
    }
  }
  if (job.status === "completed" || (job.status === "cancelled" && job.result_id)) {
    return job;
  }
  throw new JobTerminalError(job);
}

export function isActiveJob(job: JobRecord | null | undefined) {
  return (
    job?.status === "queued" ||
    job?.status === "running" ||
    job?.status === "cancelling"
  );
}

export function isTransientApiError(error: unknown) {
  if (error instanceof ApiError) {
    return (
      error.status === 408 ||
      error.status === 425 ||
      error.status === 429 ||
      error.status >= 500
    );
  }
  return error instanceof TypeError;
}

export function activeJobFromConflict(error: unknown): JobRecord | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  if (!error.detail || typeof error.detail !== "object") return null;
  const detail = error.detail as {
    code?: unknown;
    active_job?: unknown;
  };
  if (detail.code !== "active_job_conflict" || !isJobRecord(detail.active_job)) {
    return null;
  }
  return detail.active_job;
}

function errorMessage(detail: unknown, fallback: string | undefined, status: number) {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object") {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) return message;
  }
  return fallback ?? `Request failed (${status})`;
}

function asError(error: unknown) {
  return error instanceof Error ? error : new Error("Connection interrupted");
}

function isJobRecord(value: unknown): value is JobRecord {
  if (!value || typeof value !== "object") return false;
  const job = value as Partial<JobRecord>;
  return (
    typeof job.id === "string" &&
    typeof job.kind === "string" &&
    typeof job.status === "string" &&
    typeof job.progress_current === "number" &&
    typeof job.progress_total === "number"
  );
}

function abortError() {
  return new DOMException("Job polling stopped", "AbortError");
}

function waitForPoll(milliseconds: number, signal?: AbortSignal) {
  if (signal?.aborted) return Promise.reject(abortError());
  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => finish(resolve), milliseconds);
    const onOnline = () => finish(resolve);
    const onVisible = () => {
      if (document.visibilityState === "visible") finish(resolve);
    };
    const onAbort = () => finish(() => reject(abortError()));
    const finish = (complete: () => void) => {
      window.clearTimeout(timeout);
      window.removeEventListener("online", onOnline);
      document.removeEventListener("visibilitychange", onVisible);
      signal?.removeEventListener("abort", onAbort);
      complete();
    };
    window.addEventListener("online", onOnline, { once: true });
    document.addEventListener("visibilitychange", onVisible);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
