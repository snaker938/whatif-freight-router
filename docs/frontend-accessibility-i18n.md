# Frontend Accessibility and i18n Readiness

This project now includes a lightweight i18n/accessibility layer aimed at readiness and incremental adoption.

## Locale and formatting

- Locale dictionary lives in `frontend/app/lib/i18n.ts`.
- Formatting helpers live in `frontend/app/lib/format.ts`.
- Current locales:
  - `en` (default)
  - `es` (scaffold)
- Selected locale is persisted in browser storage (`ui_locale_v1`).

## UI behavior

- Language selector is available in the `Setup` card in the sidebar.
- Core panel strings are pulled from the locale dictionary.
- Date/number formatting in key panels uses locale-aware `Intl` wrappers.

## Accessibility hardening

- Skip link added: `Skip to controls panel` (keyboard users can jump to sidebar content).
- Focus-visible styling hardened for interactive controls (`button`, `input`, `select`, `textarea`, link-style buttons).
- Live region added for async status updates:
  - route compute progress
  - scenario comparison result state
  - departure optimization result state
  - duty-chain result state

## Dev verification

```powershell
pnpm -C frontend build
```

Manual checks:
- Tab from page top and verify skip link appears.
- Change locale to `Español` and verify core labels and key numeric/date formats update.
- Trigger compute/compare/departure/duty actions and verify announcements update in the live region.

