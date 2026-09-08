#!/bin/bash

MARKDOWN_LINK_CHECK=$(which markdown-link-check || echo "")
if [[ -z "$MARKDOWN_LINK_CHECK" ]]; then
    echo "❌ ERROR: markdown-link-check command not found. Install it globally via:"
    exit 1
fi

# Mode selects which links are validated:
#   external (default, used by CI): skip links to the NeMoS docs site. The README on
#     main points to API pages that only exist once this branch's docs are deployed,
#     so checking them on every PR would fail on not-yet-published URLs. The scheduled
#     check-readme-links workflow runs "full" mode to catch genuine drift.
#   full (scheduled job): validate every link, including https://nemos.readthedocs.io/...
MODE="${1:-external}"
BASE_CONFIG=".mlc.external.json"

if [[ "$MODE" == "full" ]]; then
    CONFIG="$BASE_CONFIG"
else
    if ! command -v jq >/dev/null 2>&1; then
        echo "❌ ERROR: jq is required for '$MODE' mode (to derive the CI link-check config)."
        exit 1
    fi
    CONFIG="$(mktemp)"
    trap 'rm -f "$CONFIG"' EXIT
    jq '.ignorePatterns += [{"pattern": "^https://nemos\\.readthedocs\\.io/"}]' "$BASE_CONFIG" > "$CONFIG"
fi

FAILED_FILES=()
FAILED_DETAILS=""

echo "🔍 Checking external Markdown links (mode: $MODE)..."
echo "🔎 Using config: $CONFIG"

check_file() {
    local file="$1"
    echo "📄 Checking $file..."
    local output
    output=$($MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1)
    echo "$output"
    if echo "$output" | grep -q "ERROR:"; then
        FAILED_FILES+=("$file")
        # Collect broken-link lines (bracket lines that are not the ✓ passing marker)
        local broken_lines
        broken_lines=$(echo "$output" | grep -E "^\s+\[" | grep -vF "[✓]")
        FAILED_DETAILS+=$'\n'"=== $file ==="$'\n'"${broken_lines}"$'\n'
    fi
}

# Check root directory
echo "📁 Checking root directory..."
while IFS= read -r -d '' file; do
    check_file "$file"
done < <(find . -maxdepth 1 -name "*.md" -print0)

# Check docs directory (up to 2 levels deep) if it exists
if [[ -d "docs" ]]; then
    echo "📁 Checking docs directory..."
    while IFS= read -r -d '' file; do
        check_file "$file"
    done < <(find docs -maxdepth 2 -name "*.md" -print0)
fi

if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚨 SUMMARY: broken links in ${#FAILED_FILES[@]} file(s):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$FAILED_DETAILS"
    exit 1
else
    echo "✅ All external links passed validation."
fi
