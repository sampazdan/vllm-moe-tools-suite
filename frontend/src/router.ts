import { useEffect, useState } from "react";

export type AppRoute =
  | { name: "dashboard"; path: "/" }
  | { name: "benchmarks"; path: "/benchmarks" }
  | { name: "benchmark"; path: string; benchmarkId: string }
  | { name: "runs"; path: "/runs" }
  | { name: "run"; path: string; runId: string }
  | { name: "agentRuns"; path: "/agent-runs" }
  | { name: "agentRun"; path: string; runId: string }
  | {
      name: "agentComparison";
      path: string;
      baselineRunId: string;
      candidateRunId: string;
    }
  | {
      name: "profileStudio";
      path: "/profiles/new";
      runId: string | null;
      trialIds: string[];
      profileId: string | null;
    }
  | { name: "profiles"; path: "/profiles" }
  | { name: "profile"; path: string; profileId: string }
  | { name: "comparisons"; path: "/comparisons" }
  | { name: "comparison"; path: string; comparisonId: string }
  | { name: "notFound"; path: string };

export function parseRoute(pathname: string, search = ""): AppRoute {
  const path = normalizePath(pathname);
  const segments = path.split("/").filter(Boolean).map(decodeSegment);
  if (segments.length === 0) return { name: "dashboard", path: "/" };
  if (segments.length === 1 && segments[0] === "benchmarks") {
    return { name: "benchmarks", path: "/benchmarks" };
  }
  if (segments.length === 2 && segments[0] === "benchmarks") {
    return { name: "benchmark", path, benchmarkId: segments[1] };
  }
  if (segments.length === 1 && segments[0] === "runs") {
    return { name: "runs", path: "/runs" };
  }
  if (segments.length === 2 && segments[0] === "runs") {
    return { name: "run", path, runId: segments[1] };
  }
  if (segments.length === 1 && segments[0] === "agent-runs") {
    return { name: "agentRuns", path: "/agent-runs" };
  }
  if (segments.length === 2 && segments[0] === "agent-runs") {
    return { name: "agentRun", path, runId: segments[1] };
  }
  if (segments.length === 3 && segments[0] === "agent-comparisons") {
    return {
      name: "agentComparison",
      path,
      baselineRunId: segments[1],
      candidateRunId: segments[2],
    };
  }
  if (segments.length === 1 && segments[0] === "profiles") {
    return { name: "profiles", path: "/profiles" };
  }
  if (segments.length === 2 && segments[0] === "profiles" && segments[1] === "new") {
    const parameters = new URLSearchParams(search);
    return {
      name: "profileStudio",
      path: "/profiles/new",
      runId: parameters.get("run"),
      trialIds: Array.from(new Set(parameters.getAll("trial").filter(Boolean))),
      profileId: parameters.get("profile"),
    };
  }
  if (segments.length === 2 && segments[0] === "profiles") {
    return { name: "profile", path, profileId: segments[1] };
  }
  if (segments.length === 1 && segments[0] === "comparisons") {
    return { name: "comparisons", path: "/comparisons" };
  }
  if (segments.length === 2 && segments[0] === "comparisons") {
    return { name: "comparison", path, comparisonId: segments[1] };
  }
  return { name: "notFound", path };
}

export function useAppRoute() {
  const [route, setRoute] = useState(() =>
    parseRoute(window.location.pathname, window.location.search),
  );

  useEffect(() => {
    const update = () =>
      setRoute(parseRoute(window.location.pathname, window.location.search));
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);

  return route;
}

export function navigate(href: string, options?: { replace?: boolean }) {
  const next = new URL(href, window.location.origin);
  if (
    next.pathname === window.location.pathname &&
    next.search === window.location.search &&
    next.hash === window.location.hash
  ) {
    return;
  }
  const method = options?.replace ? "replaceState" : "pushState";
  window.history[method]({}, "", `${next.pathname}${next.search}${next.hash}`);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo({ top: 0, behavior: "instant" });
}

export function routeHref(
  route: "benchmark" | "run" | "agentRun" | "profile" | "comparison",
  id: string,
) {
  const roots = {
    benchmark: "benchmarks",
    run: "runs",
    agentRun: "agent-runs",
    profile: "profiles",
    comparison: "comparisons",
  } as const;
  return `/${roots[route]}/${encodeURIComponent(id)}`;
}

function normalizePath(pathname: string) {
  if (!pathname || pathname === "/") return "/";
  return `/${pathname.split("/").filter(Boolean).join("/")}`;
}

function decodeSegment(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
