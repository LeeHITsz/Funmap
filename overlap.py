"""
This script processes LD matrices, annotation files, and z-score data for a specified chromosome and position.
It filters and intersects SNPs across multiple datasets and saves the processed results for downstream analysis.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from functools import reduce

# Specify the chromosome and position
Chr = 8
Pos = 5

# Load the annotation file for the specified chromosome
anno = pd.read_csv(f'realdata/annotation/baselineLF_chr{Chr}.annot', sep='\t')
# Drop unnecessary columns
anno.drop(anno[['CHR', 'BP', 'CM']], axis=1, inplace=True)
# Remove duplicate SNPs, keeping the first occurrence
anno.drop_duplicates(subset=['SNP'], keep='first', inplace=True)

# Load the z-score file for the specified chromosome
zscore = pd.read_csv(f'realdata/zscore/Cholesterol_chr{Chr}.csv').drop_duplicates(subset=['rsid'], keep='first')


def get_ldname(chr, pos):
    """
    Generate LD file name based on chromosome and position.
    """
    return 'chr'+str(chr)+'_'+str(int(pos*1e6+1))+'_'+str(int((pos+3)*1e6+1))


def load_ld_npz(ld_prefix):
    """
    Load the LD matrix and SNP metadata from .npz and .gz files.
    """
    # Load the SNP metadata
    gz_file = '%s.gz' % (ld_prefix)
    df_ld_snps = pd.read_table(gz_file, sep='\s+')
    df_ld_snps.rename(columns={'rsid': 'SNP', 'chromosome': 'CHR', 'position': 'BP', 'allele1': 'A1', 'allele2': 'A2'},
                      inplace=True, errors='ignore')
    assert 'SNP' in df_ld_snps.columns
    assert 'CHR' in df_ld_snps.columns
    assert 'BP' in df_ld_snps.columns
    assert 'A1' in df_ld_snps.columns
    assert 'A2' in df_ld_snps.columns
    df_ld_snps.index = df_ld_snps['CHR'].astype(str) + '.' + df_ld_snps['BP'].astype(str) + '.' + df_ld_snps[
        'A1'] + '.' + df_ld_snps['A2']

    # Load the LD matrix
    npz_file = '%s.npz' % (ld_prefix)
    try:
        R = sparse.load_npz(npz_file).toarray()
        R += R.T
    except ValueError:
        raise IOError('Corrupt file: %s' % (npz_file))

    # Create and return the LD matrix and SNP metadata
    df_R = pd.DataFrame(R, index=df_ld_snps.index, columns=df_ld_snps.index)
    return df_R, df_ld_snps


# Load the LD matrix and SNP metadata
LD, LD_info = load_ld_npz('realdata/ld/' + get_ldname(Chr, Pos))
# Remove duplicate SNPs, keeping the first occurrence
LD_info.drop_duplicates(subset=['SNP'], keep='first', inplace=True)
# Filter SNPs with single-letter alleles
LD_info = LD_info.loc[LD_info['A1'].map(lambda x: len(x) == 1) & LD_info['A2'].map(lambda x: len(x) == 1)]

# Select SNPs within the 100MB-200MB range of the 300MB region
LD_info = LD_info.loc[(LD_info['BP'] < ((Pos+2)*1e6)) & (LD_info['BP'] > ((Pos+1)*1e6))]

# Find the intersection of SNPs across the three files
SNP_list = pd.Series(reduce(np.intersect1d, (LD_info['SNP'], anno['SNP'], zscore['rsid'])), name='SNP')

# Filter LD_info to include only SNPs in the intersection
LD_info = LD_info[['SNP', 'A1']]
SNP_index = LD_info[LD_info['SNP'].isin(SNP_list.values)].index
LD_info = pd.merge(LD_info, SNP_list, how='inner', on='SNP')

# Filter the LD matrix to include only the selected SNPs
LD = LD.loc[SNP_index][SNP_index]
# Save the filtered LD matrix to a file
np.savetxt(f'realdata/input/{get_ldname(Chr, Pos)}.ld', LD.values, fmt='%.3f', delimiter=',')

# Filter the annotation file to include only the selected SNPs
anno_pos = pd.merge(anno, SNP_list, how='inner', on='SNP')
anno_pos.drop(anno[['SNP']], axis=1, inplace=True)
# Save the filtered annotation file to a CSV
anno_pos.to_csv(f'realdata/input/{get_ldname(Chr, Pos)}.annot', index=False)

# Merge the z-score file with LD_info to include only the selected SNPs
zscore = pd.merge(zscore, LD_info, how='inner', left_on='rsid', right_on='SNP')

# Adjust the t-statistics for SNPs on the reverse strand
inv_index = (zscore['A1'] != zscore['alt'])
zscore.loc[inv_index, 'tstat'] = -zscore.loc[inv_index, 'tstat']

# Save the z-score data to a file
z_score = pd.DataFrame({'SNP': zscore['rsid'].values, 'z': zscore['tstat'].values})
z_score.to_csv(f'realdata/input/Cholesterol_{get_ldname(Chr, Pos)}.zscore', index=False, header=False)
print("All files have been processed and saved successfully.")
