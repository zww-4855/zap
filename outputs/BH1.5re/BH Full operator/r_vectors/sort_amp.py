import re

raw_output = """
r1_r2_soln11                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln16                        | 1.0000     | 30.41   % | 69.59   %
r1_r2_soln20                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln18                        | 1.0000     | 25.84   % | 74.16   %
r1_r2_soln19                        | 1.0000     | 87.53   % | 12.47   %
r1_r2_soln21                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln17                        | 1.0000     | 25.84   % | 74.16   %
r1_r2_soln10                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln9                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln7                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln0                         | 1.0000     | 94.63   % | 5.37    %
r1_r2_soln1                         | 1.0000     | 94.63   % | 5.37    %
r1_r2_soln6                         | 1.0000     | 66.06   % | 33.94   %
r1_r2_soln8                         | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln15                        | 1.0000     | 64.76   % | 35.24   %
r1_r2_soln12                        | 1.0000     | 0.00    % | 100.00  %
r1_r2_soln24                        | 1.0000     | 63.04   % | 36.96   %
r1_r2_soln23                        | 1.0000     | 8.78    % | 91.22   %
r1_r2_soln22                        | 1.0000     | 51.28   % | 48.72   %
r1_r2_soln13                        | 1.0000     | 40.35   % | 59.65   %
r1_r2_soln14                        | 1.0000     | 40.35   % | 59.65   %
r1_r2_soln3                         | 1.0000     | 88.34   % | 11.66   %
r1_r2_soln4                         | 1.0000     | 98.47   % | 1.53    %
r1_r2_soln5                         | 1.0000     | 66.06   % | 33.94   %
r1_r2_soln2                         | 1.0000     | 88.34   % | 11.66   %
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


