import {Card, CardContent, Typography} from "@mui/material";
import type {Location} from "../../../lib/types/facility";

interface Props {
    locations: Location[];
}

export default function LocationList({locations}: Props) {
    return (
        <Card sx={{mb: 3}}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    All Locations
                </Typography>

                {locations.map((loc) => (
                    <Typography key={loc.locationId} sx={{mb: 1}}>
                        {loc.locationName} — {loc.currentCapacity ?? 0} /{" "}
                        {loc.maxCapacity ?? "?"}
                    </Typography>
                ))}
            </CardContent>
        </Card>
    );
}
