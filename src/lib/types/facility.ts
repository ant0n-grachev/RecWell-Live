export interface Location {
    facilityId: number;
    locationId: number;
    locationName: string;
    floor: number;
    isClosed: boolean | null;
    currentCapacity: number | null;
    maxCapacity: number | null;
    percentageCapacity: number | null;
    lastUpdated: string | null;
}

export interface FacilityPayload {
    facilityId: number;
    facilityName: string;
    floors: Record<number, Location[]>;
    locations: Location[];
}
