#!/bin/bash

# Check for any unallowed absolute links in documentation excluding the badge.
if grep -r "https://nemos.readthedocs.io" docs/ | grep -v "badge"; then
    echo "Error: Unallowed absolute links found in documentation." >&2
    exit 1
else
    echo "No unallowed absolute links found."
fi
