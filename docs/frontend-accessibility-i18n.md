# Frontend Accessibility and i18n

Last Updated: 2026-04-09
Applies To: `frontend/app/lib/i18n.ts`, `frontend/app/lib/format.ts`, and the sidebar/workflow UI

## i18n Scope

- Locale dictionary: `frontend/app/lib/i18n.ts`
- Formatting helpers: `frontend/app/lib/format.ts`
- Current locales: `en`, `es`

The current translation layer covers:

- panel title and subtitle copy
- compute mode labels
- route hints while selecting start/end points
- live progress and success/failure messages
- overlay labels for stops, incidents, and segments

The formatting helpers rely on `Intl.NumberFormat` and `Intl.DateTimeFormat`, with fallback formatting when the host runtime does not support the requested locale shape.

## Accessibility Coverage

- skip link to the controls panel
- focus-visible styling across controls and cards
- live-region updates for async compute states, compute traces, and tutorial status
- keyboard-friendly form controls in all major panels
- custom select keyboard support with arrow keys, Home/End, Enter/Space, Escape, and outside-click dismissal
- collapsible cards with `aria-expanded`, `aria-controls`, and a real region wrapper
- accessible info tooltips with `aria-haspopup`, `aria-expanded`, and escape-to-close behavior
- tutorial overlay dialog semantics, keyboard escape handling, and desktop-only gating
- map-level status messages and failure cards announced through polite live regions
- table captions and hidden labels for summary tables that are read by screen readers

Current component-specific coverage:

- `frontend/app/components/Select.tsx` behaves like a listbox and keeps the active option in sync with the selected value
- `frontend/app/components/CollapsibleCard.tsx` preserves region semantics while delaying mount for closed sections
- `frontend/app/components/FieldInfo.tsx` closes on outside click, blur, or Escape and resizes its tooltip to avoid viewport clipping
- `frontend/app/components/TutorialOverlay.tsx` uses dialog semantics, an internal progress meter, and a dedicated checklist structure
- `frontend/app/components/MapView.tsx` keeps marker text readable, exposes status/error overlays, and wires map actions to labeled buttons

## Current Interaction Notes

- Locale selection changes text and number/date formatting only. It does not change routing or scoring.
- The sidebar sections that are commonly toggled in tutorial mode are the same sections that need keyboard and screen-reader friendly locking behavior.
- The map and reporting views now expose their own aria labels so the operator can move through them without relying on visual-only cues.

## Validation

From repo root:

```powershell
pnpm -C frontend build
```

If you are checking a release candidate, also verify the tutorial overlay on desktop and confirm that the `en` and `es` locale toggles still format numbers, dates, and labels correctly in the current browser.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Map Overlays and Tooltips](map-overlays-tooltips.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)

