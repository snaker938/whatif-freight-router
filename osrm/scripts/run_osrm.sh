# run_osrm.sh
#!/bin/sh
set -eu

PBF="/data/pbf/region.osm.pbf"
OSRM_DIR="/data/osrm"
OSRM_BASE="${OSRM_DIR}/region.osrm"
PROFILE="/profiles/car.lua"

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
if [ ! -f "$OSRM_BASE" ]; then
  echo "Preparing OSRM data in $OSRM_DIR ..."

  # Copy PBF into OSRM output folder so generated .osrm files land in /data/osrm
  cp "$PBF" "${OSRM_DIR}/region.osm.pbf"

  osrm-extract -p "$PROFILE" "${OSRM_DIR}/region.osm.pbf"
  rm -f "${OSRM_DIR}/region.osm.pbf"

  osrm-partition "$OSRM_BASE"
  osrm-customize "$OSRM_BASE"

  echo "OSRM preprocessing complete."
fi

echo "Starting osrm-routed (MLD) on :5000"
exec osrm-routed --algorithm mld "$OSRM_BASE"
