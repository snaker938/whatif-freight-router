# download_pbf.sh
#!/bin/sh
set -eu

# REGION_PBF_URL must be set
RAW_URL="${REGION_PBF_URL:-}"
if [ -z "$RAW_URL" ]; then
  echo "ERROR: REGION_PBF_URL is not set."
  echo "Fix: create .env with a line like:"
  echo "  REGION_PBF_URL=https://download.geofabrik.de/europe/monaco-latest.osm.pbf"
  exit 2
fi

# Common copy/paste mistake:
#   REGION_PBF_URL=REGION_PBF_URL=https://...
# In that case, the *value* becomes: REGION_PBF_URL=https://...
# Strip that accidental prefix if present.
case "$RAW_URL" in
  REGION_PBF_URL=*)
    RAW_URL="${RAW_URL#REGION_PBF_URL=}"
    ;;
esac

# Strip surrounding quotes if someone added them in .env
case "$RAW_URL" in
  \"*\") RAW_URL="${RAW_URL#\"}"; RAW_URL="${RAW_URL%\"}" ;;
  \'*\') RAW_URL="${RAW_URL#\'}"; RAW_URL="${RAW_URL%\'}" ;;
esac

# Basic sanity check
case "$RAW_URL" in
  http://*|https://*)
    ;;
  *)
    echo "ERROR: REGION_PBF_URL must start with http:// or https://"
    echo "Got: $RAW_URL"
    exit 2
    ;;
esac

URL="$RAW_URL"

mkdir -p /data/pbf
TARGET="/data/pbf/region.osm.pbf"

if [ -f "$TARGET" ]; then
  echo "PBF already present: $TARGET"
  exit 0
fi

echo "Downloading PBF:"
echo "  URL:    $URL"
echo "  TARGET: $TARGET"

curl -L --fail --retry 8 --retry-delay 2 -o "$TARGET" "$URL"

echo "Download complete: $TARGET"
