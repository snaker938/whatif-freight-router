# Map Overlays and Tooltips

Last Updated: 2026-04-09
Applies To: `frontend/app/components/MapView.tsx` and `frontend/app/lib/mapOverlays.ts`

## Overlay Types

- stop overlays
- preview connector overlays
- incident overlays
- segment overlays
- selected-route badges
- OSRM baseline badges
- OpenRouteService baseline badges
- academic reference badges
- failure overlays

The map renders overlays from current payload fields and UI state, primarily:

- `geometry`
- `segment_breakdown`
- `incident_events`
- baseline route state
- tutorial map state
- overlay visibility toggles

Current visibility toggles include:

- `showStopOverlay`
- `showIncidentOverlay`
- `showSegmentTooltips`
- `showPreviewConnector`
- `showBaselineRoute`
- `showGoogleBaselineRoute`
- `showReferenceRoute`

## Stop And Preview Overlays

- Stop overlays are built from the origin, destination, and duty stops.
- Duplicate coordinates are deduplicated so the map does not stack multiple markers on the same location.
- The stop overlay uses the current label set and still distinguishes start, end, and duty-stop roles.
- Preview connectors are used before a computed route is present so the map still communicates the planned path between points.
- Preview nodes are color-graded from the start color to the end color, which makes multi-leg tutorial routes easier to read at a glance.

## Segment Overlay And Tooltip Content

Segment tooltips currently surface:

- segment label
- distance
- ETA / duration
- monetary cost
- CO2
- incident delay

The data comes from the route segment breakdown, which is bucketed before rendering so the map can stay responsive on long routes. The segment breakdown panel is also capped at a preview size before the user explicitly expands it.

Current segment-related behavior:

- `buildSegmentBuckets` groups rows into bounded buckets and uses those buckets for map-level hover tooltips
- the breakdown panel defaults to a 40-row preview before the user asks for the rest
- the tooltip stays sticky and lightweight so it does not fight with pan and zoom interactions

## Incident Overlay Content

Incident overlays currently display:

- delay in seconds
- start offset in seconds
- segment index
- source

The map places incident markers by projecting the simulated event onto the route geometry. Supported incident types are still `dwell`, `accident`, and `closure`.

## Failure Overlay Content

When route compute fails, the map can draw a failure path and show a diagnostic tooltip with:

- `reason_code`
- stage and stage detail when present
- the backend-facing message

That makes failure states explainable without forcing the user to open raw logs first.

## Baseline And Reference Labels

The map currently distinguishes between these route overlays:

- the selected smart route
- the OSRM baseline route
- the OpenRouteService baseline route or proxy baseline route, depending on the method returned by the backend
- the academic reference route used in the selection comparison panel

Each of those routes can have its own badge and color treatment, which is useful when the user is comparing trade-offs rather than only viewing the single selected route.

## Performance Notes

- Segment overlays should stay bounded through sampling and bucketing in the UI.
- Tooltip rendering should stay lightweight to avoid map interaction lag.
- Route sampling is intentionally simplified for the preview and badge layers so the map can carry longer routes without becoming visually noisy.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)

