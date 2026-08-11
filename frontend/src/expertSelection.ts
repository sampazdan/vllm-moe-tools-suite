import type { ExpertProfile } from "./types";

function fullExpertIds(numExperts: number) {
  return Array.from({ length: numExperts }, (_, expertId) => expertId);
}

function expertsForLayer(
  profile: ExpertProfile,
  layerId: number,
  numExperts: number,
) {
  return profile.layers[String(layerId)]?.keep ?? fullExpertIds(numExperts);
}

export function normalizeEngagementMatrix(values: number[][]): number[][] {
  const total = values.reduce(
    (matrixTotal, row) =>
      matrixTotal + row.reduce((rowTotal, value) => rowTotal + value, 0),
    0,
  );
  return total > 0
    ? values.map((row) => row.map((value) => value / total))
    : values.map((row) => row.map(() => 0));
}

export function createFullProfile(layerIds: number[], numExperts: number): ExpertProfile {
  return {
    version: 1,
    layers: Object.fromEntries(
      layerIds.map((layerId) => [
        String(layerId),
        { keep: fullExpertIds(numExperts) },
      ]),
    ),
  };
}

export function toggleProfileExpert(
  profile: ExpertProfile,
  layerId: number,
  expertId: number,
  numExperts: number,
  minimumExperts: number,
) {
  const current = expertsForLayer(profile, layerId, numExperts);
  const exists = current.includes(expertId);
  if (exists && current.length <= minimumExperts) return profile;
  const keep = exists
    ? current.filter((value) => value !== expertId)
    : [...current, expertId].sort((left, right) => left - right);
  return replaceProfileLayer(profile, layerId, keep);
}

export function selectTopExperts(
  profile: ExpertProfile,
  layerIds: number[],
  values: number[][],
  keepPerLayer: number,
) {
  const layers = { ...profile.layers };
  layerIds.forEach((layerId, layerIndex) => {
    const row = values[layerIndex] ?? [];
    layers[String(layerId)] = {
      keep: row
        .map((value, expertId) => ({ value, expertId }))
        .sort((left, right) => right.value - left.value || left.expertId - right.expertId)
        .slice(0, Math.min(Math.max(keepPerLayer, 0), row.length))
        .map(({ expertId }) => expertId)
        .sort((left, right) => left - right),
    };
  });
  return { ...profile, layers };
}

export function selectGlobalTopExperts(
  profile: ExpertProfile,
  layerIds: number[],
  values: number[][],
  globalBudget: number,
  minimumPerLayer: number,
) {
  const minimumBudget = minimumPerLayer * layerIds.length;
  const safeBudget = Math.max(minimumBudget, Math.floor(globalBudget));
  const rankedByLayer = layerIds.map((layerId, layerIndex) => ({
    layerId,
    layerIndex,
    ranked: rankRow(values[layerIndex] ?? []),
  }));
  const keep = new Map<number, Set<number>>();
  rankedByLayer.forEach(({ layerId, ranked }) => {
    keep.set(
      layerId,
      new Set(ranked.slice(0, minimumPerLayer).map(({ expertId }) => expertId)),
    );
  });
  const candidates = rankedByLayer
    .flatMap(({ layerId, ranked }) =>
      ranked.slice(minimumPerLayer).map((entry) => ({ ...entry, layerId })),
    )
    .sort((left, right) =>
      right.value - left.value ||
      left.layerId - right.layerId ||
      left.expertId - right.expertId,
    );
  let selected = minimumBudget;
  for (const candidate of candidates) {
    if (selected >= safeBudget) break;
    keep.get(candidate.layerId)?.add(candidate.expertId);
    selected += 1;
  }
  return profileWithSelection(profile, keep);
}

export function selectByCumulativeMass(
  profile: ExpertProfile,
  layerIds: number[],
  values: number[][],
  threshold: number,
  minimumPerLayer: number,
) {
  const boundedThreshold = Math.min(Math.max(threshold, 0), 1);
  const rankedByLayer = layerIds.map((layerId, layerIndex) => ({
    layerId,
    ranked: rankRow(values[layerIndex] ?? []),
  }));
  const keep = new Map<number, Set<number>>();
  let retained = 0;
  let total = 0;
  rankedByLayer.forEach(({ layerId, ranked }) => {
    total += ranked.reduce((sum, entry) => sum + Math.max(entry.value, 0), 0);
    const minimum = ranked.slice(0, minimumPerLayer);
    keep.set(layerId, new Set(minimum.map(({ expertId }) => expertId)));
    retained += minimum.reduce(
      (sum, entry) => sum + Math.max(entry.value, 0),
      0,
    );
  });
  const candidates = rankedByLayer
    .flatMap(({ layerId, ranked }) =>
      ranked.slice(minimumPerLayer).map((entry) => ({ ...entry, layerId })),
    )
    .sort((left, right) =>
      right.value - left.value ||
      left.layerId - right.layerId ||
      left.expertId - right.expertId,
    );
  for (const candidate of candidates) {
    if (total === 0 || retained / total >= boundedThreshold) break;
    keep.get(candidate.layerId)?.add(candidate.expertId);
    retained += Math.max(candidate.value, 0);
  }
  return profileWithSelection(profile, keep);
}

export function replaceProfileLayer(
  profile: ExpertProfile,
  layerId: number,
  keep: number[],
) {
  return {
    ...profile,
    layers: {
      ...profile.layers,
      [String(layerId)]: { keep: [...keep].sort((left, right) => left - right) },
    },
  };
}

export function profileSelectionCount(
  profile: ExpertProfile,
  layerIds: number[],
  numExperts: number,
) {
  return layerIds.reduce(
    (total, layerId) => total + expertsForLayer(profile, layerId, numExperts).length,
    0,
  );
}

export function observedMassRetained(
  profile: ExpertProfile,
  layerIds: number[],
  numExperts: number,
  routingMass: number[][],
) {
  let retained = 0;
  let total = 0;
  layerIds.forEach((layerId, layerIndex) => {
    const selected = new Set(expertsForLayer(profile, layerId, numExperts));
    for (let expertId = 0; expertId < (routingMass[layerIndex]?.length ?? 0); expertId += 1) {
      const value = routingMass[layerIndex][expertId] ?? 0;
      total += value;
      if (selected.has(expertId)) retained += value;
    }
  });
  return total > 0 ? retained / total : 0;
}

function rankRow(row: number[]) {
  return row
    .map((value, expertId) => ({ value, expertId }))
    .sort(
      (left, right) =>
        right.value - left.value || left.expertId - right.expertId,
    );
}

function profileWithSelection(
  profile: ExpertProfile,
  keep: Map<number, Set<number>>,
) {
  const layers = { ...profile.layers };
  keep.forEach((expertIds, layerId) => {
    layers[String(layerId)] = {
      keep: [...expertIds].sort((left, right) => left - right),
    };
  });
  return { ...profile, layers };
}
