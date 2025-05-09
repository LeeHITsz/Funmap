"""
Download UKBB LD files and LD info files using AWS.
Requires AWS CLI to be pre-installed and configured with credentials.
"""

import subprocess


def get_ldname(chr, pos):
    """
    Generate LD file name based on chromosome and position.
    """
    return 'chr' + str(chr) + '_' + str(int(pos * 1e6 + 1)) + '_' + str(int((pos + 3) * 1e6 + 1))


# Set chromosome and position
"""
For example, Chr=8 and Pos=5 correspond to the region on chromosome 8 
from base pair 5000001 to 8000001. The required files are:
- chr8_5000001_8000001.npz (LD file)
- chr8_5000001_8000001.gz (info file)
"""
Chr = 8
Pos = 5

# Set AWS CLI commands and download files
cmd_1 = 'aws s3api get-object --bucket broad-alkesgroup-ukbb-ld ' + \
        '--key UKBB_LD/' + get_ldname(Chr, Pos) + '.gz ' \
        'realdata/LD/' + get_ldname(Chr, Pos) + '.gz'
result_1 = subprocess.run(cmd_1, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result_1.returncode == 0:
    print("LD info file downloaded successfully.")
else:
    print("Failed to download LD info file.")

cmd_2 = 'aws s3api get-object --bucket broad-alkesgroup-ukbb-ld ' + \
        '--key UKBB_LD/' + get_ldname(Chr, Pos) + '.npz ' \
        'realdata/LD/' + get_ldname(Chr, Pos) + '.npz'
result_2 = subprocess.run(cmd_2, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result_2.returncode == 0:
    print("LD file downloaded successfully.")
else:
    print("Failed to download LD file.")
