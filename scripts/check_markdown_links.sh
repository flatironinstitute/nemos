#!/bin/bash
set -e

MARKDOWN_LINK_CHECK=$(which markdown-link-check || echo "")
if [[ -z "$MARKDOWN_LINK_CHECK" ]]; then
    echo "❌ ERROR: markdown-link-check command not found. Install it globally via:"
    exit 1
fi

CONFIG=".mlc.external.json"
LOG_FILE=$(mktemp)

echo "🔍 Checking external Markdown links..."
echo "🔎 Using config: $CONFIG"

run_check() {
    local CONFIG=$1

    # Check root directory
    echo "📁 Checking root directory..."
    for file in $(find . -maxdepth 1 -name "*.md"); do
        echo "📄 Checking $file..."
        $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
    done

    # Check docs directory (up to 2 levels deep) if it exists
    if [[ -d "docs" ]]; then
        echo "📁 Checking docs directory..."
        for file in $(find docs -maxdepth 2 -name "*.md"); do
            echo "📄 Checking $file..."
            $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
        done
    fi
}

run_check "$CONFIG"

# Check for errors
if grep -q "ERROR:" "$LOG_FILE"; then
    echo "🚨 Link check failed! Please fix broken links."
    exit 1
else
    echo "✅ All external links passed validation."
fi
#!/bin/bash
set -e

MARKDOWN_LINK_CHECK=$(which markdown-link-check || echo "")
if [[ -z "$MARKDOWN_LINK_CHECK" ]]; then
    echo "❌ ERROR: markdown-link-check command not found. Install it globally via:"
    exit 1
fi

CONFIG=".mlc.external.json"
LOG_FILE=$(mktemp)

echo "🔍 Checking external Markdown links..."
echo "🔎 Using config: $CONFIG"

run_check() {
    local CONFIG=$1

    # Check root directory
    echo "📁 Checking root directory..."
    for file in $(find . -maxdepth 1 -name "*.md"); do
        echo "📄 Checking $file..."
        $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
    done

    # Check docs directory (up to 2 levels deep) if it exists
    if [[ -d "docs" ]]; then
        echo "📁 Checking docs directory..."
        for file in $(find docs -maxdepth 2 -name "*.md"); do
            echo "📄 Checking $file..."
            $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
        done
    fi
}

run_check "$CONFIG"

# Check for errors
if grep -q "ERROR:" "$LOG_FILE"; then
    echo "🚨 Link check failed! Please fix broken links."
    exit 1
else
    echo "✅ All external links passed validation."
fi
