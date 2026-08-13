import type { WorkloadKind } from "./types";

export const MAX_SELECTED_WORKLOAD_UNITS = 10_000;
export const DEFAULT_SELECTED_WORKLOAD_UNITS = 100;
export const MAX_RENDERED_WORKLOAD_UNITS = 200;

export function uniqueWorkloadUnitIds(unitIds: string[]) {
  return [...new Set(unitIds.filter(Boolean))];
}

export function defaultWorkloadUnitSelection(unitIds: string[]) {
  return uniqueWorkloadUnitIds(unitIds).slice(
    0,
    DEFAULT_SELECTED_WORKLOAD_UNITS,
  );
}

export function selectAllWorkloadUnits(unitIds: string[]) {
  return uniqueWorkloadUnitIds(unitIds).slice(0, MAX_SELECTED_WORKLOAD_UNITS);
}

export function orderedWorkloadUnitSelection(
  unitIds: string[],
  selected: ReadonlySet<string>,
) {
  return uniqueWorkloadUnitIds(unitIds)
    .filter((unitId) => selected.has(unitId))
    .slice(0, MAX_SELECTED_WORKLOAD_UNITS);
}

export function plannedExperimentUnits({
  conditionCount,
  kind,
  paired,
  selectedUnitCount,
}: {
  conditionCount: number;
  kind: WorkloadKind;
  paired: boolean;
  selectedUnitCount: number;
}) {
  const conditions = kind === "state_drift" ? conditionCount : 1;
  return selectedUnitCount * conditions * (paired ? 2 : 1);
}
