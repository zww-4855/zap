import os
import re

def parse_r_vectors(directory_path, threshold=0.015):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return

    # Provided mapping based on your string
    label_map = {
        '0': 'A1', '1': 'A1', '2': 'A1', '3': 'A1', '4': 'A1', '5': 'A1',
        '6': 'A1', '7': 'A1', '8': 'B1', '9': 'B1',
        '10': 'B2', '11': 'B2'
    }

    def get_labeled_string(excitation_key):
        """
        Transforms '6^ 2' -> 'B1 A1'
        Transforms '10^ 0 7^ 3' -> 'A1 A1 B1 B1'
        """
        # Split the string into individual numbers, ignoring the '^'
        # re.findall identifies all sequences of digits
        numbers = re.findall(r'\d+', excitation_key)
        
        labels = []
        for num in numbers:
            label = label_map.get(num, "??")
            labels.append(label)
            
        return " ".join(labels)

    file_pattern = re.compile(r'^r1_r2_soln\d+$')
    files = [f for f in os.listdir(directory_path) if file_pattern.match(f)]

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        data_dict = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()

        start_parsing = False
        for line in lines:
            if "+++++" in line:
                start_parsing = True
                continue
            
            if start_parsing and '|' in line:
                key_part, val_part = line.split('|')
                key = key_part.strip()
                try:
                    value = float(val_part.strip())
                    data_dict[key] = value
                except ValueError:
                    continue

        # Filter for significant amplitudes
        significant_amps = {k: v for k, v in data_dict.items() if abs(v) > threshold}

        if significant_amps:
            output_filename = f"{filename}_amps"
            output_path = os.path.join(directory_path, output_filename)
            
            with open(output_path, 'w') as out_f:
                out_f.write(f"Significant Amplitudes (|coeff| > {threshold})\n")
                out_f.write(f"{'Excitations':<25} | {'Labeled':<30} | Coefficients\n")
                out_f.write("-" * 85 + "\n")
                
                for k, v in significant_amps.items():
                    labeled_symmetry = get_labeled_string(k)
                    # Output formatting matches your request
                    out_f.write(f"{k:<25} | {labeled_symmetry:<30} | {v}\n")
            
            print(f"Created: {output_filename}")

if __name__ == "__main__":
    # Change this to your actual directory name
    parse_r_vectors("./")
    parse_r_vectors(target_dir)

