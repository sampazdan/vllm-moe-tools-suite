import type { JobRecord } from "./types";

let csrfToken: string | null = null;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
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
      detail?: string;
    } | null;
    throw new ApiError(
      payload?.detail ?? `Request failed (${response.status})`,
      response.status,
    );
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function waitForJob(
  submitted: JobRecord,
  onProgress?: (job: JobRecord) => void,
): Promise<JobRecord> {
  let job = submitted;
  onProgress?.(job);
  while (
    job.status === "queued" ||
    job.status === "running" ||
    job.status === "cancelling"
  ) {
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    job = await api<JobRecord>(`/api/jobs/${job.id}`);
    onProgress?.(job);
  }
  if (job.status === "completed" || (job.status === "cancelled" && job.result_id)) {
    return job;
  }
  throw new Error(job.error ?? `Job ${job.status}`);
}
