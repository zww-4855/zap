#!/bin/bash

# 1. Clear or create the output file so we don't append to old data
> eom_t.txt

# 2. Iterate through the range 0 to 14
for i in {0..14}; do
    filename="outputs/soln$i"

    # Check if the file exists before attempting to grep
    if [[ -f "$filename" ]]; then
        # Grep the specific line, extract the 6th field, and append to eom_t.txt
        grep "Total, cumulative" "$filename" | awk '{print $6}' >> eom_t.txt
    else
        echo "Warning: $filename not found, skipping."
    fi
done

echo "Extraction complete. Results saved in eom_t.txt"
