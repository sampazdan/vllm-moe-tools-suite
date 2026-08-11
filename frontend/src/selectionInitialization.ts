interface BenchmarkSelectionItem {
  id: string;
}

export interface BenchmarkSelectionInitialization {
  benchmarkId: string;
  selectedIds: string[];
  focusedItemId: string | null;
}

export function benchmarkSelectionInitialization(
  initializedBenchmarkId: string | null,
  benchmarkId: string,
  items: BenchmarkSelectionItem[] | undefined,
): BenchmarkSelectionInitialization | null {
  if (!items || initializedBenchmarkId === benchmarkId) return null;
  return {
    benchmarkId,
    selectedIds: items.map((item) => item.id),
    focusedItemId: items[0]?.id ?? null,
  };
}
