# Map Overlays and Tooltips

Last Updated: 2026-02-19  
Applies To: `frontend/app/components/MapView.tsx`

## Overlay Types

- Stops overlay
- Incident overlay
- Segment overlay

The map renders overlays from existing payload fields (`geometry`, `segment_breakdown`, `incident_events`) and user state.

## Segment Tooltip Content

Segment tooltips can surface:

- segment distance and duration
- grade / energy fields (when present)
- component costs (`time_cost`, `fuel_cost`, `toll_cost`, `carbon_cost`)

## Performance Notes

- Segment overlays should stay bounded (sampling/bucketing in UI).
- Keep tooltip rendering lightweight to avoid map interaction lag.

## Related Docs

- [Documentation Index](README.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)
