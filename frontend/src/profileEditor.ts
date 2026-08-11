import type { ExpertProfile } from "./types";

export function fullExpertIds(numExperts: number) {
  return Array.from({ length: numExperts }, (_, expertId) => expertId);
}

export function expertsForLayer(
  profile: ExpertProfile,
  layerId: number,
  numExperts: number,
) {
  const configured = profile.layers[String(layerId)];
  return configured ? [...configured.keep] : fullExpertIds(numExperts);
}

export function materializeProfileForEditor(
  profile: ExpertProfile,
  routedLayerIds: number[],
  numExperts: number,
): ExpertProfile {
  const layers = Object.fromEntries(
    Object.entries(profile.layers).map(([layerId, layer]) => [
      layerId,
      { keep: [...layer.keep] },
    ]),
  );
  for (const layerId of routedLayerIds) {
    layers[String(layerId)] = {
      keep: expertsForLayer(profile, layerId, numExperts),
    };
  }
  return { version: profile.version, layers };
}

export function profileMeetsTopKFloor(
  profile: ExpertProfile,
  routedLayerIds: number[],
  numExperts: number,
  topK: number,
) {
  return routedLayerIds.every(
    (layerId) => expertsForLayer(profile, layerId, numExperts).length >= topK,
  );
}
