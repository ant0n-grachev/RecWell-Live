import {ToggleButton, ToggleButtonGroup} from "@mui/material";

interface Props {
    facility: 1186 | 1656 | null;
    onSelect: (f: 1186 | 1656) => void;
}

export default function FacilitySelector({facility, onSelect}: Props) {
    return (
        <ToggleButtonGroup
            value={facility}
            exclusive
            onChange={(_e, val) => {
                if (val !== null) onSelect(val);
            }}
            fullWidth
            size="small"
            color="primary"
            aria-label="Facility selector"
            sx={{
                '& .MuiToggleButton-root': {
                    flex: 1,
                    fontWeight: 600,
                    textTransform: "none",
                },
            }}
        >
            <ToggleButton value={1186}>Nick</ToggleButton>
            <ToggleButton value={1656}>Bakke</ToggleButton>
        </ToggleButtonGroup>
    );
}
