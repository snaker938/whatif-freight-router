# Map Overlays and Tooltips

Last Updated: 2026-03-31  
Applies To: `frontend/app/components/MapView.tsx` and `frontend/app/lib/mapOverlays.ts`

This page summarizes the current map overlay layers, tooltip behavior, and performance limits.

## Overlay Types

Current map layers include:

- selected smart route
- smart alternative routes
- OSRM baseline route
- ORS baseline route
- academic/reference route overlay
- route failure overlay
- preview connector and preview route segments
- origin and destination markers
- managed stop marker popup actions
- duty-stop overlays
- incident overlays
- tutorial guide and confirm-pin tooltips
- time-lapse position marker
- route legend badges for selected, alternative, and failure states

Overlay sources are derived from route geometry, `segment_breakdown`, `incident_events`, managed pin state, duty-chain stop state, baseline/reference routes, preview-node helpers, and tutorial state.

## Tooltip and Popup Content

### Route Badges

The map shows compact badge markers for:

- selected route
- alternative routes
- failure overlays

### Segment Tooltips

Current segment tooltips show:

- distance
- ETA
- cost in GBP
- CO2 in kg
- incident delay
- bucketed geometry so tooltips stay readable without one popup per polyline point

### Incident Popups

Current incident popups show:

- incident type
- delay
- start offset
- segment index
- source

### Managed Stop Popups

Managed stop popups allow:

- rename
- coordinate copy
- close
- delete where applicable
- tutorial-aware copy and close behavior during guided steps

## Performance Notes

- overlay bucketing helpers live in `frontend/app/lib/mapOverlays.ts`
- route segment tooltip layers are bucketed to avoid per-segment overload
- `MAX_POLYLINE_POINTS = 1000` is enforced in `frontend/app/components/MapView.tsx`
- preview-route helpers use generated preview nodes and dot segments rather than backend routes
- guide targets can stay permanently labeled while a tutorial step is active

## Accessibility Caveats

- some overlays are still hover-driven
- popups and tooltip flows are not uniformly keyboard-equivalent
- tutorial mode can intentionally lock or dim map interaction to guide users through a sequence
- map-native interactions are still the source of truth for several overlay actions, so the docs should not imply a fully mirrored list-based fallback

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)
