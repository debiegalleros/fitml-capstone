#!/usr/bin/env bash
# Downloads the catalog photo/garment PNGs (~193MB) from the GitHub Release
# asset and unpacks them into data/catalog/. These are gitignored by design
# (see CLAUDE.md — "images stay ignored"), so Render's free-tier build,
# which has no persistent disk, must fetch them fresh on every deploy.
set -euo pipefail

CATALOG_ASSET_URL="https://github.com/debiegalleros/fitml-capstone/releases/download/catalog-assets-v1/catalog_assets.zip"

cd "$(dirname "$0")/.."

if [ -d "data/catalog/photos" ] && [ -d "data/catalog/garments" ] \
   && [ -n "$(ls -A data/catalog/photos 2>/dev/null)" ]; then
  echo "Catalog assets already present, skipping download."
  exit 0
fi

echo "Fetching catalog assets from $CATALOG_ASSET_URL ..."
curl -sL "$CATALOG_ASSET_URL" -o /tmp/catalog_assets.zip
unzip -q /tmp/catalog_assets.zip -d data/catalog
rm /tmp/catalog_assets.zip
echo "Catalog assets unpacked: $(find data/catalog/photos data/catalog/garments -type f | wc -l) files"
