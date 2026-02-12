import {Alert, Box, CircularProgress, IconButton, Stack, Typography} from "@mui/material";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ModernCard from "../shared/components/ModernCard";
import type {ForecastDay, ForecastWindow} from "../lib/types/forecast";

interface Props {
    day: ForecastDay | null;
    dayOffset: number;
    canPrev: boolean;
    canNext: boolean;
    onPrev: () => void;
    onNext: () => void;
    isLoading: boolean;
    error: string | null;
}

const CHICAGO_TIMEZONE = "America/Chicago";

const formatTime = (value?: string): string => {
    if (!value) return "N/A";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "N/A";
    return parsed.toLocaleTimeString([], {
        timeZone: CHICAGO_TIMEZONE,
        hour: "numeric",
        minute: "2-digit",
    });
};

const formatWindow = (window: ForecastWindow): string => {
    const start = formatTime(window.start);
    const end = formatTime(window.end);
    return `${start}-${end}`;
};

const renderWindows = (title: string, color: string, windows?: ForecastWindow[]) => {
    const sorted = (windows ?? [])
        .slice()
        .sort((a, b) => {
            const left = Date.parse(a.start || "");
            const right = Date.parse(b.start || "");
            if (Number.isNaN(left) && Number.isNaN(right)) return 0;
            if (Number.isNaN(left)) return 1;
            if (Number.isNaN(right)) return -1;
            return left - right;
        });
    return (
        <Box>
            <Typography
                variant="body2"
                sx={{fontWeight: 700, textTransform: "uppercase", color}}
            >
                {title}
            </Typography>
            <Stack spacing={0.5} sx={{mt: 0.5}}>
                {sorted.length === 0 && (
                    <Typography variant="body2" sx={{fontWeight: 600}} color="text.secondary">
                        None
                    </Typography>
                )}
                {sorted.map((window, index) => (
                    <Typography key={`${title}-${index}`} variant="body2" sx={{fontWeight: 600}}>
                        {formatWindow(window)}
                    </Typography>
                ))}
            </Stack>
        </Box>
    );
};

export default function ForecastWindowsCard({
    day,
    dayOffset,
    canPrev,
    canNext,
    onPrev,
    onNext,
    isLoading,
    error,
}: Props) {
    const dateLabel = day?.date ? ` (${day.date})` : "";
    const title =
        dayOffset === 0
            ? `Forecast Today${dateLabel}`
            : dayOffset === 1
              ? `Forecast Tomorrow${dateLabel}`
              : dayOffset === 2
                ? `Forecast In 2 Days${dateLabel}`
                : `Forecast In 3 Days${dateLabel}`;

    return (
        <ModernCard>
            <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
                <Typography variant="subtitle2" color="text.secondary">
                    {title}
                </Typography>
                <Stack direction="row" spacing={0.25}>
                    <IconButton
                        size="small"
                        onClick={onPrev}
                        disabled={!canPrev || isLoading}
                        aria-label="Previous forecast day"
                    >
                        <ChevronLeftIcon fontSize="small"/>
                    </IconButton>
                    <IconButton
                        size="small"
                        onClick={onNext}
                        disabled={!canNext || isLoading}
                        aria-label="Next forecast day"
                    >
                        <ChevronRightIcon fontSize="small"/>
                    </IconButton>
                </Stack>
            </Box>

            {isLoading && (
                <Box sx={{display: "flex", alignItems: "center", gap: 1, mt: 1}}>
                    <CircularProgress size={16} thickness={5}/>
                    <Typography variant="body2" color="text.secondary">
                        Loading forecast...
                    </Typography>
                </Box>
            )}

            {!isLoading && error && (
                <Alert severity="info" variant="outlined" sx={{mt: 1}}>
                    {error}
                </Alert>
            )}

            {!isLoading && !error && day && (
                <Stack spacing={1.5} sx={{mt: 1}}>
                    {renderWindows("Low crowd", "success.main", day.bestWindows)}
                    {renderWindows("Peak crowd", "error.main", day.avoidWindows)}
                </Stack>
            )}
        </ModernCard>
    );
}
