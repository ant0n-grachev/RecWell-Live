import type {FacilityPayload} from "../types/facility";

export const CACHE_KEY = "recwell:facilityCache";
export const CACHE_VERSION = 1;
export const CACHE_REFRESH_GUARD_MS = 60 * 1000;

type CacheEntry = {
    version: number;
    payload: FacilityPayload;
    fetchedAt: number;
    lastUpdated: string | null;
};

type CacheMap = Record<string, CacheEntry>;

const hasWindow = () => typeof window !== "undefined";

const readCache = (): CacheMap => {
    if (!hasWindow()) return {};

    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return {};

    try {
        const parsed = JSON.parse(raw) as CacheMap;
        return parsed ?? {};
    } catch {
        return {};
    }
};

const writeCache = (map: CacheMap) => {
    if (!hasWindow()) return;
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(map));
};

export const getFacilityCache = (facilityId: 1186 | 1656): CacheEntry | null => {
    const map = readCache();
    const entry = map[String(facilityId)];
    if (!entry || entry.version !== CACHE_VERSION) return null;
    return entry;
};

export const setFacilityCache = (
    facilityId: 1186 | 1656,
    payload: FacilityPayload,
    lastUpdated: string | null
): void => {
    if (!hasWindow()) return;
    const map = readCache();
    map[String(facilityId)] = {
        version: CACHE_VERSION,
        payload,
        fetchedAt: Date.now(),
        lastUpdated,
    };
    writeCache(map);
};
