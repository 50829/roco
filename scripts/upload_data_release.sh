#!/usr/bin/env bash
set -euo pipefail

# Upload a large data archive to a GitHub Release asset via REST API.
# Required auth: export GITHUB_TOKEN=<repo-scoped token>  (or GH_TOKEN)
# Defaults are for this project/repo.
#
# Usage:
#   GITHUB_TOKEN=... ./scripts/upload_data_release.sh
#   GITHUB_TOKEN=... FILE=data_20260520_195735.tar.gz TAG=data-20260520-195735 ./scripts/upload_data_release.sh

REPO="${REPO:-50829/roco}"
FILE="${FILE:-data_20260520_195735.tar.gz}"
TAG="${TAG:-data-20260520-195735}"
TARGET="${TARGET:-mirawind}"
TITLE="${TITLE:-Data archive 20260520 195735}"
NOTES="${NOTES:-Experiment data archive uploaded as a GitHub Release asset.}"
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [ -z "$TOKEN" ]; then
  echo "[ERROR] Missing token. Set GITHUB_TOKEN or GH_TOKEN first." >&2
  echo "Example: export GITHUB_TOKEN=ghp_xxx" >&2
  exit 2
fi
if [ ! -f "$FILE" ]; then
  echo "[ERROR] File not found: $FILE" >&2
  exit 2
fi

SIZE_BYTES="$(stat -c%s "$FILE")"
echo "[release] repo=$REPO tag=$TAG target=$TARGET file=$FILE size=${SIZE_BYTES} bytes"

API="https://api.github.com/repos/$REPO"
CREATE_JSON="$(mktemp)"
RELEASE_JSON="$(mktemp)"
cleanup() { rm -f "$CREATE_JSON" "$RELEASE_JSON"; }
trap cleanup EXIT

python - <<PY > "$CREATE_JSON"
import json
payload = {
    "tag_name": "$TAG",
    "target_commitish": "$TARGET",
    "name": "$TITLE",
    "body": "$NOTES",
    "draft": False,
    "prerelease": False,
}
print(json.dumps(payload))
PY

set +e
HTTP_CODE=$(curl -sS -L -w '%{http_code}' -o "$RELEASE_JSON" \
  -X POST "$API/releases" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d @"$CREATE_JSON")
set -e

if [ "$HTTP_CODE" = "201" ]; then
  echo "[release] created release $TAG"
elif [ "$HTTP_CODE" = "422" ]; then
  echo "[release] release/tag may already exist; fetching existing release by tag $TAG"
  curl -sS -L --fail -o "$RELEASE_JSON" \
    "$API/releases/tags/$TAG" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer $TOKEN" \
    -H "X-GitHub-Api-Version: 2022-11-28"
else
  echo "[ERROR] Failed to create release. HTTP $HTTP_CODE" >&2
  cat "$RELEASE_JSON" >&2
  exit 3
fi

UPLOAD_URL=$(python - <<PY
import json
with open('$RELEASE_JSON') as f:
    data=json.load(f)
print(data['upload_url'].split('{',1)[0])
PY
)
HTML_URL=$(python - <<PY
import json
with open('$RELEASE_JSON') as f:
    data=json.load(f)
print(data.get('html_url',''))
PY
)
ASSET_NAME="$(basename "$FILE")"

echo "[release] uploading asset: $ASSET_NAME"
echo "[release] page: $HTML_URL"

curl -L --fail-with-body \
  -X POST "${UPLOAD_URL}?name=${ASSET_NAME}" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/gzip" \
  --data-binary @"$FILE"

echo
echo "[release] upload complete: $HTML_URL"
