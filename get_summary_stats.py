"""
Download Cholesterol summary statistics data and preprocess it.
"""

import gzip
import subprocess
import pandas as pd

# Specify the chromosome to process
Chr = 8
# Specify the Cholesterol summary statistics file name
file = "30690_raw.gwas.imputed_v3.both_sexes.varorder.tsv.bgz"

# Download the SNPs reference file
cmd1 = ('wget https://broad-ukb-sumstats-us-east-1.s3.amazonaws.com/round2/annotations/variants.tsv.bgz '
        '-O realdata/zscore/variants.tsv.bgz')
result1 = subprocess.run(cmd1, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result1.returncode == 0:
    print("SNPs reference file downloaded successfully.")
else:
    print("Failed to download SNPs reference file.")

# Download the Cholesterol summary statistics file
cmd2 = ('wget https://broad-ukb-sumstats-us-east-1.s3.amazonaws.com/round2/additive-tsvs/' + file + ' '
        '-O realdata/zscore/Cholesterol.tsv.bgz')
result2 = subprocess.run(cmd2, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result2.returncode == 0:
    print("Cholesterol summary statistics file downloaded successfully.")
else:
    print("Failed to download Cholesterol summary statistics file.")

# Read the SNPs reference file (.bgz format)
with gzip.open('realdata/zscore/variants.tsv.bgz', 'rt') as f:
    info = pd.read_csv(f, sep='\t')  # Adjust the separator based on file content

# Read the Cholesterol summary statistics file (.bgz format)
with gzip.open('realdata/zscore/Cholesterol.tsv.bgz', 'rt') as f:
    data = pd.read_csv(f, sep='\t')  # Adjust the separator based on file content

# Process the SNPs reference data
info = info[['variant', 'chr', 'pos', 'ref', 'alt', 'rsid']]
data = data[['variant', 'minor_allele', 'tstat', 'pval']]

# Filter SNPs with single-letter reference and alternate alleles
info = info.loc[info['ref'].map(lambda x: len(x) == 1) & info['alt'].map(lambda x: len(x) == 1)]

# Merge the summary statistics data with the SNPs reference data
data = pd.merge(data, info, how='inner', on='variant')

# Filter data for the specified chromosome
data = data.loc[data['chr'].map(lambda x: str(x) == str(Chr))]

# Remove duplicate SNPs based on rsid and keep the first occurrence
data = data.drop_duplicates(subset=['rsid'], keep='first')

# Save the processed data to a CSV file
data.to_csv(f"realdata/zscore/Cholesterol_chr{Chr}.csv", index=False)
print(f"Processed data for chromosome {Chr} saved successfully.")
