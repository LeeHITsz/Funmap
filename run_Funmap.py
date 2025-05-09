"""
This script runs the FUNMAP algorithm to analyze SNP data for a specified chromosome and position.

Input files:
- LD matrix: `realdata/input/<chr>_<pos>.ld`
- Annotation file: `realdata/input/<chr>_<pos>.annot`
- Z-score file: `realdata/input/Cholesterol_<chr>_<pos>.zscore`

Output files:
- PIP results: `realdata/output/Cholesterol_<chr>_<pos>_pip.tsv`
- Selected SNP sets: `realdata/output/Cholesterol_<chr>_<pos>_sets.txt`
- Annotation weights: `realdata/output/<chr>_<pos>_weight.tsv`
"""

import numpy as np
import pandas as pd
from funmap import FUNMAP

# Specify the chromosome and position
Chr = 8
Pos = 5

# Total number of individuals in the study
n = 315133

def get_ldname(chr, pos):
    """
    Generate LD file name based on chromosome and position.
    """
    return 'chr' + str(chr) + '_' + str(int(pos * 1e6 + 1)) + '_' + str(int((pos + 3) * 1e6 + 1))

# Load the LD matrix
print("Loading LD matrix...")
R = pd.read_csv(f'realdata/input/{get_ldname(Chr, Pos)}.ld', header=None)
R = R.values

# Load the annotation file
print("Loading annotation file...")
A = pd.read_csv(f'realdata/input/{get_ldname(Chr, Pos)}.annot')
A_name = np.array(A.columns)  # Save annotation column names
A = A.values.astype(float)

# Load the z-score file
print("Loading z-score file...")
z = pd.read_csv(f'realdata/input/Cholesterol_{get_ldname(Chr, Pos)}.zscore', header=None)
snp_name = z.values[:, 0]  # Extract SNP names
z = z.values[:, 1]  # Extract z-scores

# Run the FUNMAP algorithm
print("Running FUNMAP algorithm...")
result = FUNMAP(z, R, A, n, L=10)

# Save PIP results
print("Saving PIP results...")
s_result = pd.DataFrame({'SNP': snp_name, 'PIP': result.pip})
s_result.to_csv(f'realdata/output/Cholesterol_{get_ldname(Chr, Pos)}_pip.tsv', sep='\t', index=False)

# Save selected SNP sets
print("Saving selected SNP sets...")
file = open(f'realdata/output/Cholesterol_{get_ldname(Chr, Pos)}_sets.txt', "w")
file.write(str(result.sets))
file.close()

# Save annotation weights
print("Saving annotation weights...")
w_result = pd.DataFrame(result.mu_w, columns=A_name)
w_result.insert(loc=0, column='sigma_w2', value=result.sigma_w2)
w_result.to_csv(f'realdata/output/{get_ldname(Chr, Pos)}_weight.tsv', sep='\t', index=False)

print("All processing completed successfully.")
