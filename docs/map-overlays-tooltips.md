# Map Overlays and Per-Segment Tooltips

This document describes the map-level overlays added for incidents, stops, and per-segment route insights.

## Scope

The map now supports:

- stop overlays (origin, destination, and parsed duty-chain stops)
- incident overlays (synthetic dwell/accident/closure markers)
- per-segment tooltip overlays (bucketed for performance)

These features are frontend-driven and use existing route payload fields (`geometry`, `segment_breakdown`, `incident_events`).

## Overlay Controls

The map HUD includes three toggles:

- `Stops`
- `Incidents`
- `Segments`

Defaults:

- all three overlays are enabled by default
- toggle state is local UI state for the active page session

Each toggle displays a count badge:

- stops count from merged overlay stop points
- incidents count from selected route incident events
- segments count capped by bucket limit

## Stops Overlay Behavior

Stop overlays combine:

1. origin pin
2. destination pin
3. duty-chain stops parsed from the duty planner text area

Implementation notes:

- malformed duty stop lines are ignored in overlay parsing (non-blocking while typing)
- duplicate coordinates are deduplicated for overlay rendering
- duty stops are shown with numbered badges and popup details (index, label, coordinates)
- a dashed polyline connects ordered overlay stops

## Incident Overlay Behavior

Incidents are rendered as map markers with popups:

- `dwell` (amber)
- `accident` (orange-red)
- `closure` (red)

Popup details include:

- incident type
- delay seconds
- start offset seconds
- segment index
- source

Coordinate mapping:

- primary: interpolate along route by `start_offset_s / route_duration_s`
- fallback: segment-index ratio if duration-based progress is unavailable

## Per-Segment Tooltip Behavior

Per-segment tooltips are rendered as interactive route overlays:

- segments are grouped into adaptive buckets
- bucket cap is `120` to protect map performance
- each bucket polyline shows a hover tooltip with:
  - segment label/range
  - distance (km)
  - duration (s)
  - monetary cost
  - CO2
  - incident delay (s)

The hovered bucket is highlighted to make the active segment clear.

## Segment Bucketing Rationale

The backend can return many segment rows. Rendering every segment as an individual interactive path can degrade performance on dense routes.

Current approach:

- cap rendered interactive overlays to `<= 120` buckets
- aggregate contiguous segment rows per bucket
- map bucket ranges back to route geometry using ratio-based coordinate slicing

This keeps overlays responsive while preserving meaningful per-segment insights.

## Known Limitations

- segment tooltips are frontend-derived and may be approximate for heavily downsampled geometry
- incident coordinates are inferred from route progress and are not independent geocoded feed points
- duty stop overlay parsing is permissive for UX; strict validation still occurs for duty-chain execution requests

## Backward Compatibility

- no backend endpoint changes were required for this overlay/tooltips wave
- existing routing, comparison, and duty-chain compute flows continue to work unchanged
