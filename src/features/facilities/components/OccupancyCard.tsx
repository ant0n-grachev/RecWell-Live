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

    let formatted: string | null = null;

    if (lastUpdated) {
        const parsed = new Date(lastUpdated);

        if (!isNaN(parsed.getTime())) {
            const time = parsed.toLocaleTimeString([], {hour: "numeric", minute: "2-digit"});
            const now = new Date();
            const dayStart = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate());
            const diffMs = dayStart(now).getTime() - dayStart(parsed).getTime();
            const daysAgo = Math.max(0, Math.floor(diffMs / (1000 * 60 * 60 * 24)));

            let descriptor: string;

            if (daysAgo === 0) {
                descriptor = "Updated today at";
            } else if (daysAgo === 1) {
                descriptor = "Updated yesterday at";
            } else {
                descriptor = `Updated ${daysAgo} days ago at`;
            }

            formatted = `${descriptor} ${time}`;
        }
    }

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
                    {formatted}
                </Typography>
            )}
        </ModernCard>
    );
}
