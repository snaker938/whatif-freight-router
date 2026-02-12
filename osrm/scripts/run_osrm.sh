# run_osrm.sh
#!/bin/sh
set -eu

PBF="/data/pbf/region.osm.pbf"
OSRM_DIR="/data/osrm"
OSRM_BASE="${OSRM_DIR}/region.osrm"
PROFILE="/profiles/car.lua"
PBF_URL="${REGION_PBF_URL:-}"
OSRM_URL_MARKER="${OSRM_DIR}/.region_pbf_url"
PREPARED_MARKER="${OSRM_DIR}/.prepared.ok"
OSRM_MARKERS="
${OSRM_BASE}.timestamp
${OSRM_BASE}.mldgr
${OSRM_BASE}.cells
"

echo "OSRM boot..."

# Wait for PBF to appear (download container handles it)
if [ ! -f "$PBF" ]; then
  echo "Waiting for $PBF..."
  while [ ! -f "$PBF" ]; do
    sleep 2
  done
fi

mkdir -p "$OSRM_DIR"

# Build once (cached in bind mount)
NEEDS_PREP=0
for MARKER in $OSRM_MARKERS; do
  if [ ! -f "$MARKER" ]; then
    NEEDS_PREP=1
    break
  fi
done

# If we know the current URL, require a matching cache marker.
if [ -n "$PBF_URL" ]; then
  if [ -f "$OSRM_URL_MARKER" ] && [ "$(cat "$OSRM_URL_MARKER" || true)" != "$PBF_URL" ]; then
    NEEDS_PREP=1
  fi
fi

# Backward-compat: existing cache from before marker support.
if [ ! -f "$PREPARED_MARKER" ] && [ "$NEEDS_PREP" -eq 0 ]; then
  if [ -n "$PBF_URL" ] && [ ! -f "$OSRM_URL_MARKER" ]; then
    printf '%s\n' "$PBF_URL" > "$OSRM_URL_MARKER"
  fi
  printf '%s\n' "ok" > "$PREPARED_MARKER"
fi

if [ "$NEEDS_PREP" -eq 1 ]; then
  echo "Preparing OSRM data in $OSRM_DIR ..."

  # Clear stale/incomplete cache artifacts before rebuilding.
  rm -f "${OSRM_DIR}/region.osrm"* "$PREPARED_MARKER"

  # Copy PBF into OSRM output folder so generated .osrm files land in /data/osrm
  cp "$PBF" "${OSRM_DIR}/region.osm.pbf"

  osrm-extract -p "$PROFILE" "${OSRM_DIR}/region.osm.pbf"
  rm -f "${OSRM_DIR}/region.osm.pbf"

  osrm-partition "$OSRM_BASE"
  osrm-customize "$OSRM_BASE"

  if [ -n "$PBF_URL" ]; then
    printf '%s\n' "$PBF_URL" > "$OSRM_URL_MARKER"
  fi
  printf '%s\n' "ok" > "$PREPARED_MARKER"

  echo "OSRM preprocessing complete."
else
  echo "Using cached OSRM data in $OSRM_DIR"
fi

echo "Starting osrm-routed (MLD) on :5000"
exec osrm-routed --algorithm mld "$OSRM_BASE"
