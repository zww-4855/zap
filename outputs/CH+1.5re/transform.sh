#!/bin/bash

ROOT_DIR="./r_vectors"

# Find all files recursively and process them
find "$ROOT_DIR" -type f -print0 | while IFS= read -r -d '' file; do
    echo "Processing: $file"

    tmp=$(mktemp)

    # Preserve header, add separator, and strip complex number formatting
    head -n 2 "$file" > "$tmp"
    echo "++++++++++++++++++++++++++++++" >> "$tmp"
    
    tail -n +3 "$file" | sed -E '
        s/\(//g;
        s/\+0j\)//g;
        s/;//g
    ' >> "$tmp"

    mv "$tmp" "$file"
done

echo "Transformation complete."
