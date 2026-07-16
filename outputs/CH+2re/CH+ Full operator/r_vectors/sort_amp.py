import re

raw_output = """
r1_r2_soln11                        | 1.0000     | 21.06   % | 78.94   %
r1_r2_soln16                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln20                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln18                        | 1.0000     | 68.30   % | 31.70   %
r1_r2_soln19                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln21                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln17                        | 1.0000     | 68.30   % | 31.70   %
r1_r2_soln10                        | 1.0000     | 31.70   % | 68.30   %
r1_r2_soln9                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln7                         | 1.0000     | 18.76   % | 81.24   %
r1_r2_soln0                         | 1.0000     | 90.20   % | 9.80    %
r1_r2_soln1                         | 1.0000     | 90.20   % | 9.80    %
r1_r2_soln6                         | 1.0000     | 18.76   % | 81.24   %
r1_r2_soln8                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln15                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln12                        | 1.0000     | 21.06   % | 78.94   %
r1_r2_soln24                        | 1.0000     | 16.64   % | 83.36   %
r1_r2_soln23                        | 1.0000     | 16.64   % | 83.36   %
r1_r2_soln22                        | 1.0000     | 50.83   % | 49.17   %
r1_r2_soln13                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln14                        | 1.0000     | 23.84   % | 76.16   %
r1_r2_soln3                         | 1.0000     | 78.63   % | 21.37   %
r1_r2_soln4                         | 1.0000     | 78.63   % | 21.37   %
r1_r2_soln5                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln2                         | 1.0000     | 99.83   % | 0.17    %
"""

def sort_soln_data(text):
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    # Helper function to extract digits after 'soln'
    def get_soln_number(line):
        match = re.search(r'soln(\d+)', line)
        return int(match.group(1)) if match else -1

    # Sort the list using the extracted number as the key
    sorted_lines = sorted(lines, key=get_soln_number)
    
    # Print the sorted output
    for line in sorted_lines:
        print(line)

sort_soln_data(raw_output)


