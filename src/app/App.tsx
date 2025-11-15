import {useCallback, useEffect, useState} from "react";
import {
    Alert,
    Container,
    Box,
    CircularProgress,
    Typography,
    Link,
    IconButton,
    Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import FacilitySelector from "../features/facilities/components/FacilitySelector";
import OccupancyCard from "../features/facilities/components/OccupancyCard";
import SectionSummary from "../features/facilities/components/SectionSummary";
import SectionSummaryOther from "../features/facilities/components/SectionSummaryOther";
import {fetchFacility} from "../lib/api/recwellParser";
import type {FacilityPayload} from "../lib/types/facility";
import {
    BAKKE_ALL,
    BAKKE_COURTS,
    BAKKE_FITNESS,
    BAKKE_ICE,
    BAKKE_MENDOTA,
    BAKKE_POOL,
    BAKKE_TRACK,
    NICK_COURTS,
    NICK_FITNESS,
    NICK_POOL,
    NICK_RACQUETBALL,
    NICK_TRACK,
} from "../features/facilities/constants";
import {getFacilityCache, setFacilityCache} from "../lib/storage/facilityCache";
const FACILITY_STORAGE_KEY = "recwell:selectedFacility";

const getStoredFacility = (): 1186 | 1656 => {
    if (typeof window === "undefined") return 1186;
    const stored = Number(window.localStorage.getItem(FACILITY_STORAGE_KEY));
    return stored === 1656 ? 1656 : 1186;
};

export default function App() {
    const [facility, setFacility] = useState<1186 | 1656>(getStoredFacility);
    const [refreshKey, setRefreshKey] = useState(0);
    const [data, setData] = useState<FacilityPayload | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [freshness, setFreshness] = useState<"live" | "cached" | null>(null);
    const [lastAutoRefresh, setLastAutoRefresh] = useState(() => Date.now());
    const [lastManualRefresh, setLastManualRefresh] = useState(0);

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(FACILITY_STORAGE_KEY, String(facility));
    }, [facility]);

    const getLatestTimestamp = (payload: FacilityPayload | null | undefined) => {
        if (!payload) return null;
        return (
            payload.locations
                .map((l) => l.lastUpdated)
                .filter(Boolean)
                .sort()
                .slice(-1)[0] ?? null
        );
    };

    useEffect(() => {
        if (!facility) return;

        let isCancelled = false;
        let activeController: AbortController | null = null;

        const cached = getFacilityCache(facility);
        if (cached && !isCancelled) {
            setData(cached.payload);
            setFreshness(null);
        } else if (!cached && !isCancelled) {
            setData(null);
            setFreshness(null);
        }

        const load = async () => {
            activeController?.abort();
            const controller = new AbortController();
            activeController = controller;
            setIsLoading(true);
            setError(null);
            setFreshness(null);

            try {
                const payload = await fetchFacility(facility, controller.signal);
                if (!isCancelled) {
                    setData(payload);
                    setError(null);
                    setFreshness("live");
                    setFacilityCache(facility, payload, getLatestTimestamp(payload));
                }
            } catch (error) {
                if (!controller.signal.aborted) {
                    console.error("Failed to fetch facility data", error);
                    const fallback = getFacilityCache(facility);
                    if (fallback) {
                        setData(fallback.payload);
                        setFreshness("cached");
                        setError(null);
                    } else {
                        setData(null);
                        setFreshness(null);
                        setError("Data unavailable right now. Please try again shortly.");
                    }
                }
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsLoading(false);
                }
            }
        };

        load();

        return () => {
            isCancelled = true;
            activeController?.abort();
        };
    }, [facility, refreshKey]);

    const triggerRefresh = useCallback((action: () => void) => {
        setIsLoading(true);
        setError(null);
        setFreshness(null);
        setLastAutoRefresh(Date.now());
        action();
    }, []);

    useEffect(() => {
        const interval = window.setInterval(() => {
            if (!isLoading && Date.now() - lastAutoRefresh >= 15 * 60 * 1000) {
                triggerRefresh(() => setRefreshKey((key) => key + 1));
            }
        }, 60 * 1000);

        return () => {
            window.clearInterval(interval);
        };
    }, [isLoading, lastAutoRefresh, triggerRefresh]);

    const handleFacilitySelect = (next: 1186 | 1656) => {
        if (next === facility) return;
        triggerRefresh(() => setFacility(next));
    };

    const total = data
        ? data.locations.reduce((sum, l) => sum + (l.currentCapacity ?? 0), 0)
        : 0;

    const max = data
        ? data.locations.reduce((sum, l) => sum + (l.maxCapacity ?? 0), 0)
        : 0;

    const lastUpdated = data
        ? data.locations
            .map((l) => l.lastUpdated)
            .filter(Boolean)
            .sort()
            .slice(-1)[0]
        : null;

    const manualRefresh = () => {
        if (isLoading) return;
        const now = Date.now();
        if (now - lastManualRefresh < 3000) return;
        setLastManualRefresh(now);
        triggerRefresh(() => setRefreshKey((key) => key + 1));
    };

    const formattedCachedTime = lastUpdated
        ? new Date(lastUpdated).toLocaleString([], {hour: "numeric", minute: "2-digit", month: "short", day: "numeric"})
        : null;

    return (
        <Box sx={{py: {xs: 2, sm: 3}, bgcolor: "background.default", minHeight: "100vh"}}>
            <Container
                maxWidth="sm"
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: {xs: 2, sm: 3},
                    pt: {xs: 1.5, sm: 2},
                    px: {xs: 2, sm: 0},
                }}
            >
                <Box sx={{display: "flex", alignItems: "center", gap: 1}}>
                    <Box sx={{flexGrow: 1}}>
                        <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>
                    </Box>
                    <Tooltip title="Refresh">
                        <span>
                            <IconButton
                                onClick={manualRefresh}
                                disabled={isLoading}
                                aria-label="Refresh occupancy data"
                            >
                                {isLoading ? (
                                    <CircularProgress size={18} thickness={5}/>
                                ) : (
                                    <RefreshIcon fontSize="small"/>
                                )}
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>

                {isLoading && !data && (
                    <Box
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 2,
                            py: 5,
                        }}
                    >
                        <CircularProgress size={28} thickness={5}/>
                        <Typography color="text.secondary" fontWeight={600}>
                            Loading...
                        </Typography>
                    </Box>
                )}

                {error && (
                    <Alert severity="warning" sx={{borderRadius: 2}}>
                        {error}
                    </Alert>
                )}

                {data && (
                    <>
                        <OccupancyCard
                            title="Total Occupancy"
                            total={total}
                            max={max}
                            lastUpdated={lastUpdated}
                        />

                        {freshness === "cached" && (
                            <Alert severity="info" variant="outlined" sx={{borderRadius: 2}}>
                                Showing cached data
                                {formattedCachedTime ? ` from ${formattedCachedTime}` : ""}. We'll
                                refresh again when the service is back.
                            </Alert>
                        )}

                        {facility === 1186 && (
                            <>
                                <SectionSummary
                                    title="🏋️ Fitness Floors"
                                    ids={NICK_FITNESS}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🏀 Basketball Courts"
                                    ids={NICK_COURTS}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="👟 Running Track"
                                    ids={NICK_TRACK}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🏊‍♀️ Swimming Pool"
                                    ids={NICK_POOL}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🎾 Racquetball Courts"
                                    ids={NICK_RACQUETBALL}
                                    locations={data.locations}
                                />
                            </>
                        )}

                        {facility === 1656 && (
                            <>
                                <SectionSummary
                                    title="🏋️ Fitness Floors"
                                    ids={BAKKE_FITNESS}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🏀 Basketball Courts"
                                    ids={BAKKE_COURTS}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="👟 Running Track"
                                    ids={BAKKE_TRACK}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🏊‍♂️ Swimming Pool"
                                    ids={BAKKE_POOL}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🧗 Rock Climbing"
                                    ids={BAKKE_MENDOTA}
                                    locations={data.locations}
                                />

                                <SectionSummary
                                    title="🧊 Ice Skating"
                                    ids={BAKKE_ICE}
                                    locations={data.locations}
                                />

                                <SectionSummaryOther
                                    title="🧩 Other Facilities"
                                    exclude={BAKKE_ALL}
                                    locations={data.locations}
                                />
                            </>
                        )}
                    </>
                )}

                <Box component="footer" sx={{textAlign: "center", color: "text.secondary", fontSize: "0.85rem", py: 2}}>
                    Built by {" "}
                    <Link href="https://anton.grachev.us" target="_blank" rel="noopener noreferrer" underline="hover">
                        Anton
                    </Link>
                </Box>
            </Container>
        </Box>
    );
}
