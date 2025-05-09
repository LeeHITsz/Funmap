"""
Download the baselineLF_v2.2.UKB annotation file and process it.
"""

import subprocess

# Set the chromosome
Chr = 8

# Download the annotation file
cmd1 = (f'wget https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.tar.gz '
        '-O realdata/annotation/baselineLF_v2.2.UKB.tar.gz')
result1 = subprocess.run(cmd1, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result1.returncode == 0:
    print("Annotation file downloaded successfully.")
else:
    print("Failed to download the annotation file.")

# Extract the downloaded tar.gz file
cmd2 = f'tar -zxvf realdata/annotation/baselineLF_v2.2.UKB.tar.gz -C realdata/annotation/'
result2 = subprocess.run(cmd2, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result2.returncode == 0:
    print("Annotation file extracted successfully.")
else:
    print("Failed to extract the annotation file.")

# Decompress the annotation file for the specified chromosome
cmd3 = (f'gzip -d -c realdata/annotation/baselineLF_v2.2.UKB/baselineLF2.2.UKB.{Chr}.annot.gz > '
        f'realdata/annotation/baselineLF_chr{Chr}.annot')
result3 = subprocess.run(cmd3, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result3.returncode == 0:
    print(f"Annotation file for chromosome {Chr} decompressed successfully.")
else:
    print(f"Failed to decompress the annotation file for chromosome {Chr}.")
