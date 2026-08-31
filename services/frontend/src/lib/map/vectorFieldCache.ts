import type { VectorFieldData } from "@/lib/api/types";

/** Bounded in-memory decoded field cache (max 6 leads to constrain memory to < 10MB). */
const MAX_CACHE_ENTRIES = 6;
const cache = new Map<string, VectorFieldData>();

export function getCachedVectorField(url: string): VectorFieldData | undefined {
  const item = cache.get(url);
  if (item !== undefined) {
    // Refresh LRU position
    cache.delete(url);
    cache.set(url, item);
  }
  return item;
}

export function setCachedVectorField(url: string, data: VectorFieldData): void {
  if (cache.has(url)) {
    cache.delete(url);
  } else if (cache.size >= MAX_CACHE_ENTRIES) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }
  cache.set(url, data);
}

export function clearVectorFieldCache(): void {
  cache.clear();
}
