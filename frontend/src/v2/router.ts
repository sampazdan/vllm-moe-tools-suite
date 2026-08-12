import { useEffect, useState } from "react";

export type V2Route =
  | { name: "experiments"; path: string }
  | { name: "experimentNew"; path: string; workloadId: string | null }
  | { name: "experiment"; path: string; experimentId: string }
  | { name: "workloads"; path: string }
  | { name: "profiles"; path: string }
  | {
      name: "profileStudio";
      path: "/profiles/new";
      runId: string | null;
      trialIds: string[];
      profileId: string | null;
    }
  | { name: "profile"; path: string; profileId: string }
  | { name: "models"; path: string }
  | { name: "settings"; path: string }
  | { name: "legacy"; path: string }
  | { name: "notFound"; path: string };

const legacyRoots = new Set([
  "benchmarks",
  "runs",
  "agent-runs",
  "agent-comparisons",
  "comparisons",
]);

export function parseV2Route(pathname: string, search = ""): V2Route {
  const path = normalizePath(pathname);
  const segments = path.split("/").filter(Boolean).map(decodeSegment);
  if (segments.length === 0 || (segments.length === 1 && segments[0] === "experiments")) {
    return { name: "experiments", path: segments.length ? "/experiments" : "/" };
  }
  if (segments[0] === "experiments" && segments.length === 2) {
    if (segments[1] === "new") {
      return {
        name: "experimentNew",
        path,
        workloadId: new URLSearchParams(search).get("workload"),
      };
    }
    return { name: "experiment", path, experimentId: segments[1] };
  }
  if (
    segments.length === 1 &&
    (segments[0] === "workloads" || segments[0] === "workload-library")
  ) {
    return { name: "workloads", path };
  }
  if (segments[0] === "profiles") {
    if (segments.length === 1) return { name: "profiles", path };
    if (segments.length === 2 && segments[1] === "new") {
      const parameters = new URLSearchParams(search);
      return {
        name: "profileStudio",
        path: "/profiles/new",
        runId: parameters.get("run"),
        trialIds: Array.from(new Set(parameters.getAll("trial").filter(Boolean))),
        profileId: parameters.get("profile"),
      };
    }
    if (segments.length === 2 && segments[1] !== "new") {
      return { name: "profile", path, profileId: segments[1] };
    }
    return { name: "legacy", path };
  }
  if (segments.length === 1 && segments[0] === "models") {
    return { name: "models", path };
  }
  if (segments.length === 1 && segments[0] === "settings") {
    return { name: "settings", path };
  }
  if (segments.length === 1 && segments[0] === "legacy") {
    return { name: "legacy", path };
  }
  if (legacyRoots.has(segments[0] ?? "")) return { name: "legacy", path };
  return { name: "notFound", path };
}

export function useV2Route() {
  const [route, setRoute] = useState(() =>
    parseV2Route(window.location.pathname, window.location.search),
  );
  useEffect(() => {
    const update = () =>
      setRoute(parseV2Route(window.location.pathname, window.location.search));
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  return route;
}

export function v2Href(
  kind: "experiment" | "profile" | "newExperiment",
  id?: string,
) {
  if (kind === "newExperiment") {
    return id ? `/experiments/new?workload=${encodeURIComponent(id)}` : "/experiments/new";
  }
  const root = kind === "experiment" ? "experiments" : "profiles";
  return `/${root}/${encodeURIComponent(id ?? "")}`;
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
