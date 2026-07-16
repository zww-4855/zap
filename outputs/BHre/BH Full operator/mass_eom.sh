#!/bin/bash

# 1. Create the output directory if it doesn't exist
mkdir -p outputs

# 2. Loop through every file in the r_vectors directory
for filepath in r_vectors/r1_r2_soln*; do
    
    # Extract just the filename (e.g., r1_r2_soln10)
    filename=$(basename "$filepath")
    
    # Extract the 'solnXX' part (removes 'r1_r2_')
    output_name=${filename#*_*_}
    
    echo "Processing $filename -> outputs/$output_name"

    # 3. Create a temporary python script with the correct filename substituted
    # This searches for 'OUTPUT' and replaces it with the actual filename
    sed "s/OUTPUT/$filename/g" pycc_full.py > temp_run.py

    # 4. Run the temporary python script and pipe output to the directory
    python3 temp_run.py > "outputs/$output_name"

    # 5. Clean up the temporary file
    rm temp_run.py

done

echo "All processing complete."
