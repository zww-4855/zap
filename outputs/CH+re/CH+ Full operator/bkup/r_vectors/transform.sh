#!/bin/bash

# Define the root directory where R1/R2 folders are located
ROOT_DIR="."

# Find all files in R1/R2 structure (adjust pattern if your files have extensions)
find "$ROOT_DIR" -mindepth 2 -type f | while read -r file; do
    echo "Processing: $file"

    # Create a temporary file
    tmp=$(mktemp)

    # 1. Keep the first line (Header)
    # 2. Add the +++++ line
    # 3. Use sed to:
    #    - Remove '(' and '+0j)'
    #    - Replace ';' with a space
    head -n 1 "$file" > "$tmp"
    echo "++++++++++++++++++++++++++++++" >> "$tmp"
    
    tail -n +2 "$file" | sed -E '
        s/\(//g;            # Remove opening parenthesis
        s/\+0j\)//g;        # Remove +0j and closing parenthesis
        s/;//g              # Remove semicolons (replace with space happens naturally via field spacing)
    ' >> "$tmp"

    # Move temporary file back to original location
    mv "$tmp" "$file"
done

echo "Transformation complete."
