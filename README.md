**Author**: Katharina Lenhart
*Spice* aims to estimate alternative splicing induced functional changes across genes between two conditions. For details, please refer to my masters thesis in the [DokuWiki](http://core.izn-ffm.intern/wiki/doku.php?id=thesis:master).

# Installation

1. Create a conda enviroment
```bash
conda create -n spice_env python=3.12.2
```
2. Activate environment
```bash
conda activate spice_env
```
3. Install requirements
```bash
pip install -r conda_requirements.txt
```
---

# Required Files

Before you use the functions in this repository you need the following:
1. *Spice* library
  You should have already created a *Spice* library for your desired organism. If not, please refer to the corresponding [GitHub](https://github.com/felixhaidle/spice-nf) repository.
2. Transcript quantification
  At this point, you should have preprocessed your RNA-seq data and performed transcript quantification using *Salmon*. Note: If you choose to use another tool for transcript quantification, you need to make sure that your quantification table contains at least these columns:
    - Name: Denotes the ENSEMBL transcript ID (typically versioned in Salmon).
    - NumReads: Number of estimated reads assigned to a transcript.
    - EffectiveLength: Estimated effective length of each transcript.
    - TPM: Counts normalized to TPM.
3. Metadata table
  This tab-separated table should contain the information about your samples and the paths to you sample-level quantification tables. Needs at least columns:
  - Sample: Denotes the sample ID
  - File: Stores the path to the *Salmon* output for each sample (quant.sf files)
  - some group column: You can name this as you like, for example "Condition". This column denotes to which condition the sample belongs to.

# General information
1. *spice_func.py*: Contains all *Spice*-related functions, including normalization function and the extension of the method described in my thesis.
2. *analysis_func.py*: Contains all function I used for analysis of the Spice method including plotting functions.
3. *plotting_func.py*: Contains the plotting functions for the FA plots and relative expression bar plots.

I recommend performing your analysis in jupyter notebook and import these files:
```python
import spice_func as spice
import analysis_func as af
import plotting_func as pf
```
If you have troubles of identifying how to use the arguments of a function and it is not clear from this documentation, please refer to the docstrings of the functions first. Also if you are unsure about what each function returns, please read the corresponding docstrings.

The following shows how to use *Spice* as implemented in my thesis which uses the functions in the spice_func.py file. There are also other useful functions (analysis_func) suitable for data analysis which usage will not be shown here. If you are unsure about how to use them, you can find examples in the notebooks *msc_method_analysis.ipynb* and *cardiac_data_analysis.ipynb* and can also refer to the docstrings. Otherwise, you can write me an [E-Mail](mailto:k.lenhart@outlook.com).

# Preparation
## Count Normalization
Performs:
  - Effective Length correction
  - Library size correction
  - Median-ratio normalization

```python
scaled_counts, sf, tx_ids, sample_ids = spice.swish_scale_from_metadata(metadata_tsv="path/to/your/metadata.txt", outdir="/path/to/outdir/tx_medianRatio_scaledCounts")
```
The "tx_medianRatio_scaledCounts" folder contains for each sample the normalized count values.
  - per-sample gzipped TSVs: *sample_id.tx_medianRatio_scaledCounts.tsv.gz*.
  - wide gzipped TSV containing all samples: *tx_medianRatio_scaledCounts_per_sample.tsv.gz*.
  - manifest: *manifest.samples.tsv* (contains columns "Sample" and "file"). 

## Create a mapping table
Define a mapping table that maps each ENSEMBL transcript ID to ENSEMBL gene ID. To do so, you need to download the annotation GTF file of your organism on ENSEMBL prior to running this function.

```python
mapping_from_gtf =  spice.mapping_from_gtf(gtf_path="/path/to/your/reference/annotation/file.gtf",
                                            version_suffix="113" # adjust accordingly
                                            )
```
The argument "version_suffix" denotes which ENSEMBL version you used. For my project, it was version 113. Please adjust accordingly.

### Filter out genes and transcripts not in the *Spice* library
Filter the mapping dataframe for genes and transcript present in the *Spice* library. Also extends the mapping table with FAS IDs, transcript and gene symbols and transcript biotypes.

```python
 mapping_new = spice.filter_ID_Map(mapping_df=mapping_from_gtf,
                                  transcript_info=transcript_info,
                                  versioned=True)
```
> **Note:** You will notice that some functions have a "versioned" parameter. If set to "True", it means that your quantification table uses versioned ENSEMBL transcript IDs (as it is the case in *Salmon*). If that is NOT the case for your data, please set the argument versioned=False when you are using functions that have this argument. Default is "True".

I also recommend already running:
```python
allowed_genes, allowed_tx_ver, allowed_tx_unv, ver_map, unver_map, ver_unver_map = spice.make_tx_whitelist_and_maps(mapping_new, versioned=True)
```
ver_map, unver_map, ver_unver_map are useful later.
- ver_map: Dictionary mapping versioned ENSEMBL transcript IDs to ENSEMBL gene IDs.
- unver_map: Dictionary mapping unversioned ENSEMBL transcript IDs to ENSEMBL gene IDs.
- ver_unver_map: Dictionary mapping versioned ENSEMBL transcript IDs to unversioned ENSEMBL transcript IDs

## Define Paths and load Metadata
```python
metadata = pd.read_csv("/path/to/your/metadata.txt", sep="\t")
```
```python
# Path to your Spice library
lib_path = "/path/to/your/spice_lib_whatever_organism/"
# Path to your normalized/scaled counts folder
scaled_counts_path = "/path/to/outdir/tx_medianRatio_scaledCounts"
```

# Core *Spice* Workflow
## Define active transcript (AT) sets per gene
**Sample to condition mapping**
```python
sample_groups = dict(zip(metadata["Sample"], metadata["your_group_col"]))
```
**Transcript Filtering**

```python
active_sets, removed_transcripts, removed_genes = spice.build_fixed_AT(
    countspath=scaled_counts_path,
    sample_groups=sample_groups,
    filtered_mapping=mapping_new,
    tau_abs=10,
    r=2)
```
- tau_abs: Count threshold, default: 10
- r: Minimum number of samples. Default is group size (number of replicates per condition)

## Precompute *ewfd* vectors

```python

gene_cache = spice.precompute_ewfd_fixedAT(
    countspath=scaled_counts_path,
    libpath=lib_path,
    active_sets=active_sets,
    filtered_mapping=mapping_new,
    sample_groups=sample_groups)
```

## Calculate RMSD
> **Note**: We also calculate the JSD for each gene here. Please refer to my master thesis in the Dokuwiki for more details.

```python
# Get condition names
grps = set([gr for sample, gr in sample_groups.items()])
# Calculate observed values
obs_rmsd, jsd, grouped_by_AT = spice.calc_obs_metrics(gene_cache=gene_cache,
                                                          groups=grps_psc)
```

**Build result table**
```python
result_table = spice.build_results_table(gene_cache=gene_cache,
                                         obs_rmsd=obs_rmsd,
                                         jsd=jsd)
```
>**Note**: You can plot RMSD, JSD and MPD histograms across AT-bins using af.plot_distributions_from_result`.

## Calculation of cumulative probabilities
First we perform an exact label permutation procedure to obtain additional RMSD values per gene to densify the underlying AT-level RMSD data.
```python
pooled_by_AT, pooled_by_gene, summary_dict = spice.build_AT_reference_dist(gene_cache)
```
### Merge low-populated AT-bins
Bins that are not represented by enough genes are merged to one bin.
**Define which bins to merge**
```python
bins_to_pool = spice.define_bins_to_merge(summary_dict=summary_dict, num_g=200)
```
> **Note**: Argument num_g defines the minimum number of genes that need to represent an AT-bin. Choose as you wish. Bins which show less than num_g will be merged with subsequent AT-bins and will be denoted by the integer "99". Note that there will only be ONE merged bin.

**Merge bins**
```python
summary_dict_mod, pooled_by_AT_mod, result_table_mod = spice.merge_bins(bins_to_pool=bins_to_pool,
                                                        summary_dict=summary_dict,
                                                        pooled_by_AT=pooled_by_AT,
                                                        result_table=result_table)

```
### Distribution Fitting
We fit three distributions to the data: Beta, Johnson SB and scipy's powerlaw function.
```python
# Fit three distribution functions 
df_fits = spice.fit_distributions_scipy(pooled_by_AT_mod,
                                        models=["beta","johnsonsb","powerlaw"]
)
```
```python
# Select the winner by AIC
winner_params_df = spice.winners_by_AIC(df_fits)
```

>**Note**: You can plot QQ plots using `af.plot_all_bins_QQ` and ECDF vs. CDF using `af.plot_ecdf_with_beta_cdf`. Please note that the following functions assume that the beta distribution was selected as best fit.
### Calculate cumulative probabilities and update result table
```python
scored = spice.add_cum_probs(result_table=result_table_mod,
                             winners_df=winner_params_df,
                             value_col="RMSD",
                             out_col="RMSD_q")
```
> **Note**: The cumulative probabilities is denoted by column "RMSD_q". The "scored" table contains ENSEMBL gene IDs, JSD, MPD, RMSD and cumulative probability values.

# Extend result table with expression data
## Gene expression
To extend the results table with expression data, the following code can be run:
```python
# Calculate total gene expression per sample and average across replicates
# to obtain averaged normalized counts and TPM values per gene per condition
gene_sample_group_expr, mean_expr_per_group = spice.get_gene_expression_per_sample(metadata=metadata,
                                                          gene_cache=gene_cache,
                                                          sample_groups=sample_groups,
                                                          ver_map=ver_map,
                                                          scaled_outdir=scaled_counts_path,
                                                          group_col="your_group_column_in_metadata,
                                                          versioned=True,
                                                          ver_unv_map=ver_unver_map,
                                                          agg="mean",
                                                          include_raw=False)

```
`gene_sample_group_expr` contains for each gene and each sample the summed TPM and summed normalized counts (summed across transcripts of a gene). `mean_expr_per_group` contains the average gene TPM and normalized counts per condition (averaged across replicates).

```python
# Extend result table
scored_ext = spice.add_agg_gene_expression(gene_group_agg=mean_expr_per_group,
                                           result_table=scored)
```
`scored__ext` is an extended result table with added gene- and condition-level TPM and normalized count values averaged across replicates. You can also extend the results table with gene symbols by running:

```python
# Extend result table with gene symbols obtained from mapping table
scored_ext = spice.add_gene_symbols(mapping=mapping_new, result_table=scored_ext)
```
## Transcript expression

To obtain a wide dataframe which contains for each transcript the estimated normalized count values you can run:

```python
tidy_expr = spice.scaledcounts_to_AT_wide_df(
    scaled_outdir=scaled_counts_path,
    gene_cache=gene_cache,
    sample_groups=sample_groups,
    versioned=True,
    filtered_mapping=mapping_new,
    ver_unv_map=ver_unver_map,
    include_raw=False)
```

To obtain the relative expression for each transcript calculated from normalized read counts you need to run:

```python
tidy_rel_expr = spice.gene_cache_to_rel_expr_df(gene_cache=gene_cache)
```

# Save result tables
The `spice.export_tables` function enables the export of the following dataframes:
- removed_tx_table: The removed transcripts after AT set definition obtained from `spice.build_fixed_AT`. Contains columns "transcript_id" and	"reason". "reason" denotes the exclusion reason of this transcript which can be either "tau_abs" or "prevalence_rule".
- result_table: The (extended) result table containing at least columns "GeneID", "MPD", "JSD", "RMSD" and "RMSD_q"
- tidy_expr_table: Normalized counts across active transcripts of each gene. Contains columns "gene" (ENSEMBL gene ID), "Name" (FAS ID),	"txname" (unversioned ENSEMBL transcript ID) and for each sample a column in the format `count_scaled_<sample_id>_<condition>`. Obtained when running `spice.scaledcounts_to_AT_wide_df`.
- tidy_relexpr_table: Relative expression across active transcripts of each gene obtained when running `spice.gene_cache_to_rel_expr_df`. Same columns as for tidy_expr_table but sample-level columns have the format `rel_expr_<sample_id>_<condition>`.

```python
spice.export_tables(outdir="/path/to/your/outdir",
                    removed_tx_table=removed_transcripts,
                    result_table=scored_psc_meso_ext,
                    tidy_expr_table=tidy_expr,
                    tidy_relexpr_table=tidy_rel_expr)
```

# Visualize results
To visualize results, you need to load the *plotting_func.py* file:
```python
import plotting_func as pf
```
## Create relative expression barplots
First, we extend the relative expression dataframe with transcript symbols obtained from the generated mapping dataframe:

```python
tidy_rel_expr_ext = pf.map_id_to_symbol(expr_df=tidy_rel_expr,
                                        mapping=mapping_new)
```
For plotting relative expression barplots for a gene, you can use the function `pf.plot_expr_bars`.
```python
pf.plot_expr_bars(expr_df=tidy_rel_expr_ext,
                   gene="DIDO1", # replace with your gene symbol
                   id_type="symbol", # if gene is a gene argument is a gene symbol, use id_type="symbol", else id_type ="ensembl"
                   mapping=mapping_new, # your mapping dataframe
                   versioned_tr_id=True, 
                   groups=["CondA", "CondB"], # replace with your conditions
                   by="symbol") # use "symbol" if you want transcripts to be displayed by their symbols, else by="ensembl".
```
## Feature architecture plots
First we need to load the FAS index:
```python
fa_index = pf.load_fa_index(lib_path=lib_path)
```
To show the FA, run:

```python
pf.feature_architecture_figure(lib_path=lib_path, 
                               gene_id="ENSG00000174437", # replace with your ENSEMBL gene ID
                               expr_df=tidy_rel_expr_ext, 
                               fa_index=fa_index)
```


