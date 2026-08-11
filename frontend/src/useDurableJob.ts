import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
  JobTerminalError,
  api,
  isActiveJob,
  isTransientApiError,
  waitForJob,
} from "./api";
import type { JobRecord } from "./types";

interface DurableJobOptions {
  enabled: boolean;
  onRecovered: (job: JobRecord) => void | Promise<void>;
}

export function useDurableJob({ enabled, onRecovered }: DurableJobOptions) {
  const queryClient = useQueryClient();
  const [activeJob, setActiveJob] = useState<JobRecord | null>(null);
  const [connectionIssue, setConnectionIssue] = useState<string | null>(null);
  const [recoveryError, setRecoveryError] = useState<string | null>(null);
  const [recoveryAttempt, setRecoveryAttempt] = useState(0);
  const locallyWatchedJob = useRef<string | null>(null);
  const recoveringJob = useRef<string | null>(null);
  const recoveredCallback = useRef(onRecovered);

  useEffect(() => {
    recoveredCallback.current = onRecovered;
  }, [onRecovered]);

  const activeJobQuery = useQuery({
    queryKey: ["active-job"],
    queryFn: () => api<JobRecord | null>("/api/jobs/active"),
    enabled,
    retry: (failureCount, error) =>
      isTransientApiError(error) && failureCount < 4,
    retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 5_000),
    refetchInterval: (query) =>
      isActiveJob(query.state.data) && query.state.data?.kind !== "agent_run"
        ? false
        : 5_000,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });

  const discoveredJob = activeJobQuery.data;
  const discoveredJobId = discoveredJob && isActiveJob(discoveredJob)
    ? discoveredJob.id
    : null;

  useEffect(() => {
    if (!enabled || !discoveredJobId || !discoveredJob) return;
    const job = discoveredJob;
    if (job.kind === "agent_run") {
      setActiveJob((current) => current?.kind === "agent_run" ? null : current);
      void recoveredCallback.current(job);
      return;
    }
    setActiveJob(job);
    if (
      locallyWatchedJob.current === discoveredJobId ||
      recoveringJob.current === discoveredJobId
    ) {
      return;
    }

    const controller = new AbortController();
    recoveringJob.current = discoveredJobId;
    setRecoveryError(null);
    void waitForJob(job, {
      signal: controller.signal,
      onProgress: updateJob,
      onTransientError: updateConnectionIssue,
    })
      .then(async (terminalJob) => {
        clearJob(terminalJob.id);
        await recoveredCallback.current(terminalJob);
      })
      .catch(async (error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof JobTerminalError) {
          clearJob(error.job.id);
          await recoveredCallback.current(error.job);
          if (error.job.status === "failed") setRecoveryError(error.message);
          return;
        }
        setRecoveryError(
          error instanceof Error ? error.message : "Could not resume background work",
        );
      })
      .finally(() => {
        if (recoveringJob.current === discoveredJobId) {
          recoveringJob.current = null;
        }
      });

    return () => controller.abort();
  }, [discoveredJobId, enabled, recoveryAttempt]);

  async function watchJob(submitted: JobRecord) {
    locallyWatchedJob.current = submitted.id;
    setRecoveryError(null);
    updateJob(submitted);
    try {
      const terminalJob = await waitForJob(submitted, {
        onProgress: updateJob,
        onTransientError: updateConnectionIssue,
      });
      clearJob(terminalJob.id);
      return terminalJob;
    } catch (error) {
      if (error instanceof JobTerminalError) {
        clearJob(error.job.id);
        await recoveredCallback.current(error.job);
      } else {
        setRecoveryAttempt((current) => current + 1);
      }
      throw error;
    } finally {
      if (locallyWatchedJob.current === submitted.id) {
        locallyWatchedJob.current = null;
      }
    }
  }

  function adoptJob(job: JobRecord) {
    setRecoveryError(null);
    setConnectionIssue(null);
    queryClient.setQueryData(["active-job"], job);
    if (job.kind === "agent_run") {
      setActiveJob(null);
      void recoveredCallback.current(job);
      return;
    }
    setActiveJob(job);
    setRecoveryAttempt((current) => current + 1);
  }

  function retryRecovery() {
    setRecoveryError(null);
    setRecoveryAttempt((current) => current + 1);
    void activeJobQuery.refetch().then(({ data }) => {
      if (isActiveJob(data)) return;
      setActiveJob(null);
      setConnectionIssue(null);
    });
  }

  function updateJob(job: JobRecord) {
    setActiveJob(job);
    queryClient.setQueryData(["active-job"], isActiveJob(job) ? job : null);
  }

  function updateConnectionIssue(error: Error | null, attempt: number) {
    setConnectionIssue(
      error
        ? `Connection interrupted · reconnecting (attempt ${attempt})`
        : null,
    );
  }

  function clearJob(jobId: string) {
    setActiveJob((current) => current?.id === jobId ? null : current);
    setConnectionIssue(null);
    queryClient.setQueryData(["active-job"], null);
    void queryClient.invalidateQueries({ queryKey: ["active-job"] });
  }

  const discoveryError = activeJobQuery.error instanceof Error
    ? activeJobQuery.error.message
    : null;

  return {
    activeJob,
    connectionIssue,
    recoveryError,
    discoveryError,
    watchJob,
    adoptJob,
    retryRecovery,
  };
}
