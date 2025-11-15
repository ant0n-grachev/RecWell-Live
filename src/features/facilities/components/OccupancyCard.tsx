import {Typography} from "@mui/material";
import ModernCard from "../../../shared/components/ModernCard";
import {clampPercent, getOccupancyColor} from "../../../shared/utils/styles";

interface Props {
    title: string;
    total: number;
    max: number;
    lastUpdated?: string | null;
}

export default function OccupancyCard({title, total, max, lastUpdated}: Props) {
    const percent = clampPercent(max ? (total / max) * 100 : 0);
    const color = getOccupancyColor(percent);

    const formatted =
        lastUpdated
            ? new Date(lastUpdated).toLocaleTimeString([], {hour: "numeric", minute: "2-digit"})
            : null;

    return (
        <ModernCard>
            <Typography variant="subtitle2" color="text.secondary">
                {title}
            </Typography>

            <Typography variant="h4" fontWeight={700}>
                {total} / {max}
            </Typography>

            <Typography sx={{color, fontWeight: 600}}>{percent}% full</Typography>

            {formatted && (
                <Typography variant="body2" color="text.secondary">
                    Last updated: {formatted}
                </Typography>
            )}
        </ModernCard>
    );
}
