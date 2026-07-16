import os
import re

def parse_r_vectors(directory_path, threshold=0.015):
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return

    # Look for files matching the pattern r1_r2_solnXX
    # We use a regex to avoid matching the '_amps' files we are about to create
    file_pattern = re.compile(r'^r1_r2_soln\d+$')
    
    files = [f for f in os.listdir(directory_path) if file_pattern.match(f)]
    
    if not files:
        print("No matching files found in the directory.")
        return

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        data_dict = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the header separator to start parsing data
        start_parsing = False
        for line in lines:
            if "+++++" in line:
                start_parsing = True
                continue
            
            if start_parsing:
                # Split by the pipe character
                if '|' in line:
                    key_part, val_part = line.split('|')
                    
                    # Clean up key (e.g., '10^ 0 7^ 3') and value
                    key = key_part.strip()
                    try:
                        value = float(val_part.strip())
                        data_dict[key] = value
                    except ValueError:
                        # Skip lines that don't have a valid float coefficient
                        continue

        # Filter the dictionary and prepare content for the new file
        significant_amps = {k: v for k, v in data_dict.items() if abs(v) > threshold}

        # If we found significant amplitudes, write them to the new file
        if significant_amps:
            output_filename = f"{filename}_amps"
            output_path = os.path.join(directory_path, output_filename)
            
            with open(output_path, 'w') as out_f:
                out_f.write(f"Significant Amplitudes (|coeff| > {threshold})\n")
                out_f.write("Excitations | Coefficients\n")
                out_f.write("+" * 30 + "\n")
                for k, v in significant_amps.items():
                    out_f.write(f"{k:15} | {v}\n")
            
            print(f"Created: {output_filename} with {len(significant_amps)} entries.")
        else:
            print(f"Skipped: {filename} (No amplitudes > {threshold}).")

if __name__ == "__main__":
    # Specify your subdirectory name here
    target_dir = "./" #"r_vectors"
    parse_r_vectors(target_dir)
