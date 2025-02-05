#!/bin/bash

# Fail script if any command fails
set -e

# Find markdown-link-check path
MARKDOWN_LINK_CHECK=$(which markdown-link-check || echo "")

# If markdown-link-check is not found, print an error and exit
if [[ -z "$MARKDOWN_LINK_CHECK" ]]; then
    echo "❌ ERROR: markdown-link-check command not found. Make sure it is installed globally."
    exit 1
fi

echo "🔍 Checking Markdown links in root directory..."

# Initialize an error flag
ERROR=0
LOG_FILE=$(mktemp)  # Temporary file to store output

# Find all Markdown files in the root directory and check links
for file in $(find . -maxdepth 1 -name "*.md"); do
    echo "📂 Checking $file..."

    # Run markdown-link-check and capture output
    $MARKDOWN_LINK_CHECK "$file" 2>&1 | tee -a "$LOG_FILE"
done

# Check if "ERROR:" appears in the log file
if grep -q "ERROR:" "$LOG_FILE"; then
    echo "🚨 Link check failed! Please fix broken links."
    exit 1
else
    echo "✅ All links are valid."
fi
