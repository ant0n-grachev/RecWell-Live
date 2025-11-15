import axios from "axios";
import type {Location, FacilityPayload} from "../types/facility";
import {nick} from "../data/nick";
import {bakke} from "../data/bakke";

const URL =
    "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount?AccountAPIKey=7938fc89-a15c-492d-9566-12c961bc1f27";

const safePercent = (current: number | null, max: number | null) => {
    if (!current || !max) return null;
    return Math.round((current / max) * 100);
};

const flatten = (floors: Record<number, Location[]>) => {
    return Object.values(floors)
        .flat()
        .sort(
            (a, b) =>
                a.floor - b.floor || a.locationName.localeCompare(b.locationName)
        );
};

export async function fetchFacility(
    facilityId: 1186 | 1656,
    signal?: AbortSignal
): Promise<FacilityPayload> {
    const layout: Record<number, Location[]> =
        facilityId === 1186
            ? JSON.parse(JSON.stringify(nick))
            : JSON.parse(JSON.stringify(bakke));

    const resp = await axios.get(URL, {signal});
    const live = resp.data;

    const index: Record<number, Location> = {};

    for (const floor of Object.values(layout)) {
        for (const loc of floor) {
            index[loc.locationId] = loc;

            loc.isClosed = null;
            loc.currentCapacity = null;
            loc.percentageCapacity = null;
            loc.lastUpdated = null;
        }
    }

    for (const f of live) {
        const loc = index[f.LocationId];
        if (!loc) continue;

        loc.isClosed = f.IsClosed;
        loc.currentCapacity = f.LastCount;
        loc.percentageCapacity = safePercent(f.LastCount, loc.maxCapacity);
        loc.lastUpdated = f.LastUpdatedDateAndTime ?? null;
    }

    return {
        facilityId,
        facilityName:
            facilityId === 1186
                ? "Nicholas Recreation Center"
                : "Bakke Recreation & Wellbeing Center",
        floors: layout,
        locations: flatten(layout),
    };
}
