import {Typography, Stack, Box} from "@mui/material";
import ModernCard from "../../../shared/components/ModernCard";
import type {Location} from "../../../lib/types/facility";
import {clampPercent, getOccupancyColor} from "../../../shared/utils/styles";

interface Props {
    title: string;
    ids: number[];
    locations: Location[];
}

export default function SectionSummary({title, ids, locations}: Props) {
    const list = locations
        .filter((l) => ids.includes(l.locationId))
        .sort((a, b) => ids.indexOf(a.locationId) - ids.indexOf(b.locationId));

    const single = list.length === 1;
    const only = single ? list[0] : null;

    // Single-location CLOSED handling
    const singleClosed = single && only && only.isClosed === true;

    // Totals (irrelevant when singleClosed is true)
    const total = list.reduce((s, l) => s + (l.currentCapacity ?? 0), 0);
    const max = list.reduce((s, l) => s + (l.maxCapacity ?? 0), 0);
    const percent = clampPercent(max ? (total / max) * 100 : 0);
    const percentColor = getOccupancyColor(percent);

    return (
        <ModernCard>
            <Typography variant="h6">
                {title}
            </Typography>

            {/* ----- Single facility CLOSED ----- */}
            {singleClosed && (
                <Typography
                    variant="h4"
                    fontWeight={700}
                    sx={{color: "error.main"}}
                >
                    CLOSED
                </Typography>
            )}

            {/* ----- Single facility OPEN ----- */}
            {single && !singleClosed && (
                <>
                    <Typography variant="h5">
                        {only!.currentCapacity ?? 0} / {only!.maxCapacity ?? 0}
                    </Typography>

                    <Typography sx={{color: percentColor, fontWeight: 600}}>
                        {percent}% full
                    </Typography>
                </>
            )}

            {/* ----- Multi-facility category ----- */}
            {!single && (
                <>
                    <Typography variant="h5">
                        {total} / {max}
                    </Typography>

                    <Typography sx={{color: percentColor, fontWeight: 600}}>
                        {percent}% full
                    </Typography>

                    <Stack spacing={1} sx={{mt: 1}}>
                        {list.map((loc) => {
                            const isClosed = loc.isClosed === true;

                            if (isClosed) {
                                return (
                                    <Box
                                        key={loc.locationId}
                                        sx={{
                                            display: "flex",
                                            flexDirection: {xs: "column", sm: "row"},
                                            justifyContent: "space-between",
                                            alignItems: {xs: "flex-start", sm: "center"},
                                            gap: {xs: 0.5, sm: 1},
                                            width: "100%",
                                        }}
                                    >
                                        <Typography variant="body2" fontWeight={600}>
                                            {loc.locationName}
                                        </Typography>

                                        <Typography
                                            variant="body2"
                                            color="error.main"
                                            fontWeight={700}
                                        >
                                            CLOSED
                                        </Typography>
                                    </Box>
                                );
                            }

                            const p = loc.maxCapacity
                                ? clampPercent(
                                    ((loc.currentCapacity ?? 0) / loc.maxCapacity) * 100
                                )
                                : 0;
                            const color2 = getOccupancyColor(p);

                            return (
                                <Box
                                    key={loc.locationId}
                                    sx={{
                                        display: "flex",
                                        flexDirection: {xs: "column", sm: "row"},
                                        justifyContent: "space-between",
                                        alignItems: {xs: "flex-start", sm: "center"},
                                        gap: {xs: 0.5, sm: 2},
                                        width: "100%",
                                    }}
                                >
                                    <Typography variant="body2" fontWeight={600}>
                                        {loc.locationName}
                                    </Typography>

                                    <Stack
                                        direction={{xs: "column", sm: "row"}}
                                        spacing={{xs: 0.5, sm: 1}}
                                        alignItems={{xs: "flex-start", sm: "center"}}
                                        sx={{textAlign: {xs: "left", sm: "inherit"}}}
                                    >
                                        <Typography variant="body2" fontWeight={600}>
                                            {loc.currentCapacity ?? 0} / {loc.maxCapacity ?? 0}
                                        </Typography>
                                        <Typography variant="body2" sx={{color: color2}}>
                                            ({p}%)
                                        </Typography>
                                    </Stack>
                                </Box>
                            );
                        })}
                    </Stack>
                </>
            )}
        </ModernCard>
    );
}
