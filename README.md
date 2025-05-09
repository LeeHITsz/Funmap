# Funmap

Funmap is a unified method to integrate high-dimensional 
functional annotations with fine-mapping.

## Overview
Fine-mapping aims to prioritize causal variants underlying 
complex traits by accounting for the linkage disequilibrium 
of GWAS risk locus. The expanding resources of functional annotations 
serve as auxiliary evidence to improve the power of fine-mapping. 
However, existing fine-mapping methods tend to generate many 
false positive results when integrating a large number of annotations.

Funmap can effectively improve the power of fine-mapping 
by borrowing information from hundreds of functional annotations. 
Meanwhile, it relates the annotation to the causal probability 
with a random effects model that avoids the over-fitting issue, 
thereby producing a well-controlled false positive rate. 
Paired with a fast algorithm, Funmap enables scalable integration 
of a large number of annotations to facilitate prioritizing 
multiple causal SNPs. Our simulations demonstrate that 
Funmap is the only method that produces well-calibrated FDR 
under the setting of high-dimensional annotations while achieving 
better or comparable power gains as compared to existing methods.

## Installation

Funmap was developed under Python 3.9 environment 
but should be compatible with older versions of Python 3. 
The following Python modules are required:

* [numpy](http://www.numpy.org/) (version==1.26.2)
* [scipy](http://www.scipy.org/) (version==1.11.4)
* [pandas](https://pandas.pydata.org/) (version==2.1.3)
* [matplotlib](https://matplotlib.org/) (version==3.8.2)

To install Funmap:

``` shell
git clone https://github.com/LeeHITsz/Funmap.git
cd Funmap
pip install -r requirements.txt 
``` 

We provide two ways to run Funmap:

### Run Funmap from the command line

``` shell 
$> python funmap_cmd.py -h
usage: funmap_cmd.py [-h] --zdir ZDIR --Rdir RDIR --Adir ADIR --n N --save SAVE [--L L] [--iter ITER] [--tol TOL] [--verbose]

Funmap Commands:

optional arguments:
  -h, --help   show this help message and exit
  --zdir ZDIR  path to zscores files
  --Rdir RDIR  path to LD files
  --Adir ADIR  path to annotations files
  --n N        GWAS sample size
  --save SAVE  path to save result
  --L L        the maximum number of causal variables (default is 10)
  --iter ITER  the maximum number of iterations (default is 100)
  --tol TOL    the convergence tolerance (default is 5e-5)
  --verbose    whether to display more information (default is True)
``` 

For example: 
``` shell
$> python funmap_cmd.py --zdir data/zscore.txt --Rdir data/ld.txt --Adir data/anno.txt --n 50000 --save result --verbose
``` 

### Run Funmap within a Jupyter Notebook

Install the funmap package into your local virtual environment:
``` shell
$> python setup.py install
``` 

Use the 'FUNMAP' function in the 'funmap' package to run Funmap:
``` python
from funmap import FUNMAP
result = FUNMAP(z, R, A, n=50000, L=10)
``` 

For a completed example, please refer to
[funmap_example.ipynb](funmap_example.ipynb).

## Input files

Funmap takes in z-scores file, LD file 
and annotations file as inputs.

- **zscore file** contains two mandatory columns: 
variant IDs and z-scores. An example can be found at [data/zscore.txt](data/zscore.txt).

- **LD file** contains Pearson correlation coefficient matrix. 
An example can be found at [data/ld.txt](data/ld.txt).

- **annotations file** contains functional annotation matrix.
An example can be found at [data/anno.txt](data/anno.txt).

Example input files are included in the [data](data) directory.

## Results

- **variant-level PIP** file contains two columns: variant ID and PIP:

``` shell
$> head result/PIP.csv
rs1124048,2.9715906380123336e-05
rs10494829,3.081396542481407e-05
rs4915210,0.0002291531559568405
rs3198583,0.00020109873830131964
rs56368827,3.4685542804724356e-05
rs3738255,0.00013761266975498287
rs296569,3.677616215946866e-05
rs296568,4.581612902221366e-05
rs296567,4.3880889903258335e-05
rs296566,5.091157180914241e-05

``` 

- **set-level summary** file contains the infomation of credible sets.

``` shell
$> head result/sets.txt
{'cs': {'L0': array([259], dtype=int64)}, 'purity':     min_abs_corr  mean_abs_corr  median_abs_corr
L0           1.0            1.0              1.0, 'cs_index': array([0], dtype=int64), 'coverage': 0    1.0
dtype: float64, 'requested_coverage': 0.95}
``` 

Example output files are included in the [result](result) directory.

## A full real data example

Data and code for analysis in the paper can be viewed in [https://github.com/LeeHITsz/Funmap_analysis](https://github.com/LeeHITsz/Funmap_analysis). Here we only introduce the reproduction examples shown in the paper:

Here, we provide a step-by-step tutorial to analyze real data with available LD matrix. We use the example presented in section 3.2 (Figure 5b) of the Funmap paper. We will implement the complete workflow from scratch to perform fine-mapping of SNPs associated with cholesterol levels.

This example focuses on chromosome 8, position 5MB-8MB, which contains potential causal variants associated with cholesterol levels.

### Step 1: Create Directory Structure

First, create the necessary folder structure to organize our data and results:

```bash
mkdir -p realdata/annotation realdata/ld realdata/zscore realdata/input realdata/output
```

### Step 2: Download Required Data

Run the following scripts to download the necessary data files:

- 2.1 Download Annotation Files
Run `get_annotation.py` to download the baseline functional annotation files:

```bash
python get_annotation.py
```

This will download and extract the baselineLF_v2.2.UKB annotation file for chromosome 8.

- 2.2 Download Summary Statistics
Run `get_summary_stats.py` to download cholesterol-related summary statistics:

```bash
python get_summary_stats.py
```

This script downloads both the SNP reference file and cholesterol summary statistics file.

- 2.3 Download LD Files

We suggest using the in-sample LD files. When the in-sample LD files are not available, we suggestusing the UKB reference LD matrices prepared by Prof. Alkes Price's group [https://registry.opendata.aws/ukbb-ld/](https://registry.opendata.aws/ukbb-ld/).

**Before running the next script, make sure you have AWS CLI installed and configured with appropriate credentials.**

```bash
python get_ld.py
```

This downloads the linkage disequilibrium (LD) matrix and related information for chromosome 8, position 5.

### Step 3: Process and Align Data

Run the `overlap.py` script to process and align the zscore, LD matrix, and annotation files:

```bash
python overlap.py
```

This script performs several important tasks:
- Filters SNPs within the specified genomic region
- Finds the intersection of SNPs across all data sources
- Adjusts t-statistics for SNPs on the reverse strand
- Saves processed data to the input directory for fine-mapping

### Step 4: Run Fine-mapping Analysis

Finally, run the Funmap algorithm on the prepared data:

```bash
python run_Funmap.py
```

This script:
- Loads the processed LD matrix, annotation file, and z-scores
- Runs the FUNMAP algorithm with L=10 components
- Saves the results to the output directory, including:
  - PIP (Posterior Inclusion Probability) for each SNP
  - Selected SNP sets
  - Annotation weights

The final output files will be saved in the `realdata/output` directory.

## Reference
Yuekai Li, Jiashun Xiao, Jingsi Ming, Yicheng Zeng, Mingxuan Cai. (2024). [Funmap: integrating high-dimensional functional annotations to improve fine-mapping](https://doi.org/10.1093/bioinformatics/btaf017) _Bioinformatics_
