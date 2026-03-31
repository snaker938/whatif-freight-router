# Frontend Accessibility and i18n

Last Updated: 2026-03-31  
Applies To: `frontend/app/page.tsx`, `frontend/app/components/TutorialOverlay.tsx`, `frontend/app/components/MapView.tsx`, `frontend/app/global.css`, `frontend/app/lib/i18n.ts`

This page summarizes the current accessibility and localization surface of the frontend.

## i18n Scope

Current localization files and helpers:

- `frontend/app/lib/i18n.ts`
- `frontend/app/lib/format.ts`

Current locales:

- `en`
- `es`

Current coverage is partial. The app has localized strings for key route-compute and map-overlay labels, but it does not yet implement full locale routing, exhaustive translation coverage, or advanced pluralization across the entire UI.

## Accessibility Coverage

Current implemented accessibility features include:

- a skip link from the page shell into the controls panel
- live regions for loading, compute progress, and status updates
- readiness/status cards with `role="status"`
- route-graph warmup progress surfaced as a real progress bar with ARIA attributes
- tutorial overlay dialog semantics with label/description wiring
- tutorial overlay focus management
- tutorial overlay Escape handling
- tutorial checklist with current-step and status semantics
- multiple screen-reader-only helper regions
- reduced-motion-aware behavior in UI styling and tutorial map interactions
- sidebar toggle state exposed with `aria-pressed`
- route and map failure cards surfaced outside the canvas so they remain discoverable when the map itself is not enough

## Tutorial Overlay Notes

`frontend/app/components/TutorialOverlay.tsx` currently provides:

- dialog semantics
- modal and non-modal variants depending on running scope
- `aria-modal` true for normal guided mode and false for map-only running scope
- desktop-only blocked mode for the full guided tutorial
- current-task and checklist status output
- resume/restart flows when saved tutorial progress exists

## Map Accessibility Notes

`frontend/app/components/MapView.tsx` currently includes:

- marker popups with explicit controls
- tutorial confirmation tooltips with group labels
- route legend and failure-overlay status text outside the canvas
- permanent guide tooltips while a tutorial target is active
- screen-reader-relevant route failure status outside the map canvas

Current limitations remain:

- many map interactions are still pointer-driven
- hover-oriented tooltips are inherently less accessible than explicit list views
- tutorial locking and map dimming intentionally constrain interaction during guided steps
- some interactions intentionally remain map-native rather than duplicating every action in a parallel list UI

## Validation

From repo root:

```powershell
pnpm -C frontend build
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Map Overlays and Tooltips](map-overlays-tooltips.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
