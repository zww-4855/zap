import pandas as pd
import os

def get_excitation_ratios(filename):
    records = []
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                clean_line = line.strip()
                #print(clean_line)                
                # Only look at lines that actually contain the delimiter
                if '|' in clean_line:
                    parts = [p.strip() for p in clean_line.split('|')]
                    #print("parts:",parts,len(parts))                    
                    # We expect 3 columns: Excitations, Labeled, Coefficients
                    if len(parts) >= 2:
                        try:
                            # Use the last part as the coefficient
                            val_str = parts[-1].replace(' ', '')
                            coefficient = float(val_str)
                            #print(parts[0],coefficient)        
                            records.append({
                                'excitation': parts[0], 
                                'amplitude': coefficient
                            })
                        except ValueError:
                            # This naturally skips headers/dashes that aren't numbers
                            continue
                            
    except Exception as e:
        print(f"  [!] Error reading {filename}: {e}")
        return None

    if not records:
        # Debug: Uncomment the line below to see which files are coming up empty
        print(f"  [?] No valid data rows found in {filename}")
        return None

    df = pd.DataFrame(records)
    df['sq_amp'] = df['amplitude'] ** 2
    total_sq_sum = df['sq_amp'].sum()

    # Classification logic
    # Singles: 1 caret, Doubles: 2 carets
    r1_sum = df[df['excitation'].str.count(r'\^') == 1]['sq_amp'].sum()
    r2_sum = df[df['excitation'].str.count(r'\^') == 2]['sq_amp'].sum()

    p_r1 = (r1_sum / total_sq_sum) * 100 if total_sq_sum > 0 else 0
    p_r2 = (r2_sum / total_sq_sum) * 100 if total_sq_sum > 0 else 0

    return total_sq_sum, p_r1, p_r2

# --- Directory Loop ---

target_directory = "./" 

print(f"{'Filename':<35} | {'Sum Sq':<10} | {'%R1':<8} | {'%R2':<8}")
print("-" * 70)

for entry in os.listdir(target_directory):
    # Process only .txt or .out files (adjust as needed)
    if os.path.isfile(os.path.join(target_directory, entry)): #and entry.endswith(".txt"):
        #print(entry) 
        result = get_excitation_ratios(entry)
        
        if result:
            total, r1, r2 = result
            print(f"{entry:<35} | {total:<10.4f} | {r1:<8.2f}% | {r2:<8.2f}%")
        else:
            # Optional: notify if a file was skipped
            print(f"{entry:<35} | [No Data Found]")
