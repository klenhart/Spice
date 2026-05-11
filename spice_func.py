from __future__ import annotations
from typing import Tuple, Dict, List, Optional, Set, Iterable, Literal, Union, Any
import gzip
import re
import itertools
from collections import defaultdict
from pathlib import Path
from scipy import stats
import warnings
import csv
import json
import os
import itertools
from itertools import combinations
import math 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#################### GENERAL HELPER FUNCTIONS ###################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def tidy_scaledcounts_over_samples(scaled_outdir: str | Path,
                                   sample_groups: Dict[str, str],
                                   group_col_name: str = "group",
                                   sample_col_name: str = "sample",
                                   versioned: bool = True,
                                   ver_unv_map: Optional[Dict[str, str]] = None,
                                   keep_cols: Tuple[str, ...] = ("count_scaled", "count_raw")
                                   ) -> pd.DataFrame:
    """
    Read per-sample normalized count tables
    and returns a tidy (long) table with transcript-level counts.

    Input: 
        scaled_outdir: Path to the normalized counts which were generated with swish_scale_from_metadata.
        sample_groups: Mapping sample ID to the condition.
        group_col_name: How to name the column denoting the condition in the output table
        sample_col_name: How to name the column denoting the sample ID in the output table
        versioned: True if your files storing the normalized counts uses versioned transcript IDs.
        ver_unv_map: If versioned=True, the function expects a mapping of 
                     versioned transcript IDs to unversioned. This is obtained after
                     running make_tx_whitelist_and_maps with the versioned parameter.
        keep_cols: Which counts to use.

    Expected per-sample file columns:
      - TXNAME
      - count_scaled

    Output columns:
      - Name: transcript id (unversioned if versioned=True and ver_unv_map provided)
      - sample
      - group
      - count_scaled
      - count_raw (if used)
    """
    scaled_outdir = Path(scaled_outdir)
    # access the manifest file
    manifest_path = scaled_outdir / "manifest.samples.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path, sep="\t")
    need = {"Sample", "file"}
    miss = need - set(manifest.columns)
    if miss:
        raise ValueError(f"manifest.samples.tsv missing columns: {miss}")

    # keep only the samples we have group labels for
    manifest = manifest.loc[manifest["Sample"].astype(str).isin(set(sample_groups.keys()))].copy()
    if manifest.empty:
        raise ValueError("No samples in manifest match sample_groups keys.")

    rows: List[pd.DataFrame] = []
    for _, r in manifest.iterrows():
        sample = str(r["Sample"])
        # access group of sample and path to file of normalized counts
        grp = sample_groups[sample]
        fpath = str(r["file"])

        df = pd.read_csv(fpath, sep="\t")
        if "TXNAME" not in df.columns:
            raise ValueError(f"{fpath} missing column 'TXNAME'")

        # choose which columns to keep if present (raw counts or scaled counts or both)
        present = [c for c in keep_cols if c in df.columns]
        if not present:
            raise ValueError(f"{fpath} has none of {keep_cols}. Found: {list(df.columns)}")

        out = df[["TXNAME", *present]].copy()
        out[sample_col_name] = sample
        out[group_col_name] = grp

        # map to unversioned transcript IDs for consistency with spice library
        if versioned:
            if ver_unv_map is None:
                raise ValueError("versioned=True requires ver_unv_map.")
            out["Name"] = out["TXNAME"].map(ver_unv_map)
        else:
            out["Name"] = out["TXNAME"].astype(str)

        out = out.drop(columns=["TXNAME"])
        out = out.dropna(subset=["Name"])

        rows.append(out)

    tidy = pd.concat(rows, ignore_index=True)

    # ensure numeric values in these columns
    for c in keep_cols:
        if c in tidy.columns:
            tidy[c] = pd.to_numeric(tidy[c], errors="coerce")

    return tidy


def invert_dict(d):
    """
    Swaps keys and values of a dicionary.
    Used by build_gene_tx_df.
    """
    return {v: k for k, v in d.items()}


def build_gene_tx_df(gene_cache: Dict,
                     versioned: bool=True,
                     ver_unv_map: Optional[Dict]=None) -> pd.DataFrame:
    """
    Helper function for get_gene_expression_per_sample.
    Build a (gene, Name) mapping from gene_cache as DataFrame. Ensures
    that only transcripts present in the AT set of each gene is
    considered by get_gene_expression_per_sample.

    Input:
        gene_cache: Output of precompute_ewfd_fixedAT
                    {"ENSG...": {"AT",...,"feats": ["tx1", "tx2", ...], ...},
                        ...}
        versioned: Denotes if the count table uses versioned or unversioned
                   transcript ENSEMBL ids. Default: True
        ver_unv_map: If versioned=True, the function expects a mapping of 
                     versioned transcript IDs to unversioned. This is obtained after
                     running make_tx_whitelist_and_maps with the versioned parameter. 
    
    Returns:
         DataFrame with columns "gene" (gene ID), "Name" (transcript ID)
    """
    rows = []
    for gene_id, data in gene_cache.items():
        txs = data.get("feats", [])
        if versioned:
            unv_ver_map = invert_dict(ver_unv_map)
            for tx in txs:
                rows.append((gene_id, unv_ver_map[tx]))
        else:
            for tx in txs:
                rows.append((gene_id, tx))

    gene_tx_df = pd.DataFrame(rows, columns=["gene", "Name"])
    return gene_tx_df

def tidy_expr_over_samples(metadata: pd.DataFrame,
                           groups: Set | List,
                           tr_gene_map: Dict,
                           group_col: str) -> pd.DataFrame:
    """
    Helper function for get_gene_expression_per_sample.
    Builds a long dataframe of TPM transcript expression
    for each transcript in each sample.

    Input:
        metadata: Metadata dataframe denoting condition, sample id and where
                  to find the quantification table. Requires columns "Sample"
                  for sample ID and "File" storing the path to the sample-specific
                  file
        groups: Set or List of groups which are compared.
        ver_map: Mapping of transcript ID to gene ID. Output of 
                 function make_tx_whitelist_and_maps. Use ver_map
                 if transcripts in the expression table are denoted 
                 by versioned ENSEMBL IDs, use unver_map if not.
        group_col: Defines which column in metadata table contains the
                   group information.
    Returns:
        Long DataFrame of Salmon estimated transcript expression (in TPM) 
        with added columns "sample", "group" and "gene".
    """
    counts = list()
    # Select only the samples which are used in the respective comparison
    metadata_grps = metadata.loc[metadata[group_col].isin(groups)].copy()
    for _, row in metadata_grps.iterrows():
        sample = row["Sample"]
        grp = row[group_col]
        salmon_out = pd.read_csv(row["File"], sep="\t")
        salmon_out["sample"] = sample
        salmon_out["group"] = grp
        salmon_out["gene"] = salmon_out["Name"].map(tr_gene_map)
        salmon_out_clean = salmon_out.dropna(subset="gene")
        counts.append(salmon_out_clean)
    out = pd.concat(counts)
    return out

def _parse_beta_params_beta_only(params_str: str):
    """
    Helper function that parses 'a=..., b=...' from winners_by_AIC table
    saved in column "params". Assumes that the fit was a two-param
    (beta) distribution denoted with "a" and "b".

    Returns parameters a and b as floats.
    """
    # tolerant to spaces
    parts = dict(s.strip().split("=") for s in params_str.replace(" ", "").split(","))
    return float(parts["a"]), float(parts["b"])


def open_maybe_gz(path: str):
    """ Helper function called by mapping_from_gtf."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")

def extract_data_from_json(path: str):
    """
    Extracts data stored in transcript_info.json which can be 
    found in the spice library path.
    """ 
    with open(path, 'r') as file:
        json_f = json.load(file)
    return json_f

def load_fas_index(libpath: Path | str):
    with open(os.path.join(libpath, "fas_data/fas_index.json"), "r") as f:
        fas_index: Dict[str, str] = json.load(f)
    return fas_index


def _strip_ver(x: str) -> str:
    """
    Helper function called by filter_ID_Map.
    Removes version number from ENST ID.
    Returns: Trimmed ENST ID.
    """
    return x.split(".", 1)[0] if isinstance(x, str) else x

def _coerce_r(r, n_samples: int) -> int:
    """error handling"""
    if r is None:
        return max(2, math.ceil(0.5 * n_samples))
    try:
        # scalar ints/floats
        return int(r)
    except Exception:
        try:
            return int(pd.Series(r).iloc[0])
        except Exception:
            return max(2, math.ceil(0.5 * n_samples))

def _is_active_enough(t: str,
                      active_sample_counts: Dict[str, int],
                      r_eff: int) -> bool:
    """ 
    Checks if a specific transcript is present in at least r_eff
    samples. If so, the transcript is considered as active across
    samples and will be added to the AT set of its gene (done
    in build_fixed_AT).
    
    Input: 
        t: Transcript ID (likely versioned ENSEMBL ID, potentially unversioned)
        active_sample_counts: Dictionary that saves in how many samples a specific
                              transcript was declared as active given the detection
                              probability rule.
        r_eff: Number of samples in wich trancript t is active to be considered in
               the AT set of its gene.
    Returns: True (if in AT set) or False (not in AT set)
    """
    cnt = active_sample_counts.get(t, 0)
    if cnt >= r_eff:
        return True
    else:
        return False


def build_fas_id_maps(filtered_mapping: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Creates ensembl transcript ID to FAS ID mapping.
    Called by precompute_ewfd_fixed_AT.

    Notes: 
        - Requires the mapping table to contain columns "ensembl_id",
          "ensembl_transcript_id", "fas_id"
        - Format of out: {ENSG: {ENST: fas_id, ...}}

    """
    need = {"ensembl_id", "ensembl_transcript_id", "fas_id"}
    miss = need - set(filtered_mapping.columns)
    if miss: raise ValueError(f"filtered_df missing columns: {miss}")

    out: Dict[str, Dict] = {}
    df = filtered_mapping.copy()
    for gid, gdf in df.groupby("ensembl_id"):
        mp = dict(zip(gdf["ensembl_transcript_id"], gdf["fas_id"]))
        out[gid] = mp

    return out


def collect_all_kept_ids(active_sets: Dict[str, Dict]) -> Tuple[set, set]:
    """
    Collects the transcript IDs which were selected as the AT set of each
    gene.
    
    Input:
        acitve_sets: Nested dictionary containing transcript information for
                     each gene, calculated by build_fixed_AT.
    Returns:
        kept: Set if kept transcripts

    """
    kept = set()
    # Loops over items in active_sets
    for g, info in active_sets.items():
        for tid, keep in zip(info["ids"], info["mask"].astype(bool)):
            if not keep:
                continue
            # Add kept transcripts to set
            kept.add(str(tid))
    return kept

def invert_fas_adjacency_matrix(fas_adjacency_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculates the invert FAS adjacency matrix in nested dict format to
    turn it into a dissimilarity_matrix.

    Behavior:
        row: Source protein/transcript which is the current row of the matrix.
        col: The target protein which is the current column of the matrix.
        inner: The dictionary mapping from row to all its coll values.
        value: The FAS score from pair row; col.
    
    Returns:
        Dictionary with inverted FAS scores.
    
    Note: Currently, the original FAS scores are calculated and obtainable
          via the fas_score path in the spice library folder. When calculating 
          the EWFD vectors, the function (see below) uses the 1-FAS score.
    """
    return {
        row: {col: 1.0 - value for col, value in inner.items()}
        for row, inner in fas_adjacency_matrix.items()}

def dict_to_matrix(inverted_fas_dict: Dict[str, Dict[str, float]], transcript_order: List[str]) -> np.ndarray:
    # Assumes all transcript_order keys exist in inverted_fas_dict
    return np.array([
        [inverted_fas_dict[row][col] for col in transcript_order]
        for row in transcript_order
    ], dtype=float)


def estimate_diversity(inverted_fas: Dict[str, Dict[str, float]]) -> tuple[float, float]:
        """
        Mean-pariwise dissimilarity (MPD) and standard deviation of dissimilarity matrix
        containing 1-complement FAS scores.
        """
        keys = list(inverted_fas.keys())
        num_isoforms = len(keys)
        if num_isoforms < 2:
            return 0.0, 0.0

        values = []
        for i in keys:
            for j in keys:
                if i != j:
                    values.append(inverted_fas[i][j])

        n = len(values)
        if n == 0:
            return 0.0, 0.0

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        sd = math.sqrt(variance)

        return mean, sd


def calculate_ewfd(gene_fas_dists: Dict[str, Dict[str, float]],
                   rel_expressions: List[float],
                   transcript_ids: List[str]) -> List[float]:
    """
    Calculates expression-weighted-functional-disturbance (EWFD) vectors for each gene.

    Input:
        - gene_fas_dists: Dict[str, Dict[str, float]]: Ordered FAS score matrix for a Gene
                            in form of a dictionary. Usually obtained after intialization of
                            EWFDAssembler instance.
        - rel_expressions: List[float]: List containing relative expression values of tran-
                            scripts of that gene.
        - transcript_ids: List[str]): List containing transcript ids.

    Returns:
        ewfd_list: List[float]: List containing ewfd values for each transcript.
    
    Behavior:
        - Effectively performs a Matrix-Vector multiplication with the Matrix containing pairwise
            FAS scores between selected transcripts/proteins of a gene and the Vector containing the
            relative expression values for each transcript.
        - Uses 1-FAS score which reflects FA dissimilarity (FAS-score represents FA similarity)

    """
    # Initiate ewfd list/vector with zeros
    ewfd_list: List[float] = [0.0] * len(transcript_ids)
    # Performs effectively a Matrix-Vector multiplication with 1-FAS scores
    for s, seed_id in enumerate(transcript_ids):
        for q, query_id in enumerate(transcript_ids):
            ewfd_list[s] += rel_expressions[q] * (1 - gene_fas_dists[seed_id][query_id])

    return ewfd_list


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
######################## PREPROCESSING ##########################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def swish_scale_from_metadata(metadata_tsv: str,
                              outdir: str,
                              sample_col: str = "Sample",
                              file_col: str = "File",
                              quiet: bool = False) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Takes Salmon quant.sf files and calculates Swish-style
    median ratio scaling/normalization. See DOI: 10.1093/nar/gkz622.

    Input:
        - metadata_tsv: Tab-separated table containing at least a sample and 
                        file column; i.e, ["Sample	"File"]. Sample column denotes sample ID and file
                        column the full path to the respective quant.sf file.
        - outdir: Directory to save results
        - sample_col: Name of the column in metadata_tsv denoting the sample. Default: "Sample".
        - file_col:  Name of the column in metadata_tsv storing the path to sample-specific
                     quant.sf files. Default: "File".

    Saves:
        - per-sample gzipped TSVs: <Sample>.tx_medianRatio_scaledCounts.tsv.gz
        - wide gzipped TSV: tx_medianRatio_scaledCounts_per_sample.tsv.gz
        - manifest: manifest.samples.tsv

    Returns:
        - scaled_counts: Final counts per sample
        - sf: scaling factors per sample
        - tx_ids: transcript ids
        - sample_ids: sample ids
    Notes:
        - Function requires that tab-separated quantification table are in Salmon style.
          Columns: [Name	Length	EffectiveLength	TPM	NumReads]

    """

    os.makedirs(outdir, exist_ok=True)

    # Read metadata and collect files
    meta = pd.read_csv(metadata_tsv, sep="\t")
    if sample_col not in meta.columns or file_col not in meta.columns:
        raise ValueError(
            f"metadata must contain columns '{sample_col}' and '{file_col}'. "
            f"Found: {list(meta.columns)}"
        )

    # Keep order exactly as in the metadata file
    sample_ids = meta[sample_col].astype(str).tolist()
    files = meta[file_col].astype(str).tolist()

    # Validate files
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Cannot find these paths:\n" + "\n".join(missing)
        )

    # Load Salmon quantification files for all samples and build matrices

    counts_cols = []
    lens_cols = []
    tx_ids: Optional[list[str]] = None

    for sample, path in zip(sample_ids, files):
        qs = pd.read_csv(path, sep="\t")
        # Because this is a specific function for Salmon quant.sf outputs
        # it will raise an error if the columns names do not match salmons output columns
        required = {"Name", "NumReads", "EffectiveLength"}
        if not required.issubset(qs.columns):
            raise ValueError(
                f"{path} is missing required columns {required}. "
                f"Found: {set(qs.columns)}"
            )

        # Ensure transcript IDs align across all samples
        names = qs["Name"].astype(str).tolist()
        if tx_ids is None:
            tx_ids = names
        else:
            if names != tx_ids:
                raise ValueError(
                    f"Transcript order/IDs differ in {path}. "
                    "All quantification files must have identical transcript lists in the same order."
                )

        counts_cols.append(qs["NumReads"].to_numpy(dtype=float))
        lens_cols.append(qs["EffectiveLength"].to_numpy(dtype=float))

    assert tx_ids is not None
    # Stack to (G x N)
    counts = np.column_stack(counts_cols)
    lengths = np.column_stack(lens_cols)

    # Compute geometric mean depth d from raw library sizes
    lib_raw = counts.sum(axis=0)
    pos = lib_raw > 0
    if not np.any(pos):
        d = 1.0
    else:
        d = float(np.exp(np.mean(np.log(lib_raw[pos]))))

    # Length/bias correction
    gm_len = np.exp(np.nanmean(np.log(lengths), axis=1))
    gm_len[~np.isfinite(gm_len)] = 1.0
    b = lengths / gm_len[:, None]
    cts = counts / b

    # Depth scaling
    lib_star = cts.sum(axis=0)
    lib_star[lib_star == 0] = 1.0
    y2 = (cts / lib_star) * d

    # Calculate median-ratio size factors, then final scaled counts:
    logy2 = np.log(y2)
    log_geomeans = np.nanmean(logy2, axis=1)
    use = np.isfinite(log_geomeans)
    if not np.any(use):
        if not quiet:
            print("No rows with finite geometric means; using size factors = 1.")
        sf = np.ones(y2.shape[1], dtype=float)
    else:
        log_geomeans_use = log_geomeans[use]
        logy2_use = logy2[use, :]

        sf = np.empty(y2.shape[1], dtype=float)
        for i in range(y2.shape[1]):
            diffs = logy2_use[:, i] - log_geomeans_use
            diffs = diffs[np.isfinite(diffs)]
            sf[i] = float(np.exp(np.median(diffs))) if diffs.size else 1.0

    scaled_counts = y2 / sf

    # Export per-sample results
    scaled_files = []
    for j, sample in enumerate(sample_ids):
        df = pd.DataFrame(
            {
                "TXNAME": tx_ids,
                "count_scaled": scaled_counts[:, j],
                "count_raw": counts[:, j],
            }
        )
        fn = os.path.join(outdir, f"{sample}.tx_medianRatio_scaledCounts.tsv.gz")
        with gzip.open(fn, "wt") as f:
            df.to_csv(f, sep="\t", index=False)
        scaled_files.append(fn)

    # Export results summarizing all samples (TXNAME and one column per sample)
    wide = pd.DataFrame(scaled_counts, index=tx_ids, columns=sample_ids)
    wide.insert(0, "TXNAME", wide.index)
    fn_wide = os.path.join(outdir, "tx_medianRatio_scaledCounts_per_sample.tsv.gz")
    with gzip.open(fn_wide, "wt") as f:
        wide.to_csv(f, sep="\t", index=False)

    # Export manifest, has "Sample" and "file" column and stores path to normalized count
    # files
    manifest = pd.DataFrame({"Sample": sample_ids, "file": scaled_files})
    fn_manifest = os.path.join(outdir, "manifest.samples.tsv")
    manifest.to_csv(fn_manifest, sep="\t", index=False)

    return scaled_counts, sf, tx_ids, sample_ids


def mapping_from_gtf(gtf_path: str,
                     version_suffix: Union[str, int] = 113,
                     feature: str = "transcript",
                     out_tsv_path: Optional[str] = None):
    """
    Parse an Ensembl-style GTF (optionally .gz) and build a table with columns:
      - ensembl_id: unversioned Ensembl gene ID (gene_id)
      - ensembl_transcript_id: unversioned Ensembl transcript ID (transcript_id)
      - transcript_id_v<version_suffix>: transcript_id + '.' + transcript_version

    Inputs:
        - gtf_path: Path to .gtf or .gtf.gz of target organism
        - version_suffix: ENSEMBL version. Used only for naming the versioned column, e.g. 113 -> transcript_id_v113
        - feature: Which feature type to extract. Default: "transcript"
        - out_tsv_path: If provided, writes a TSV to this path.

    Returns:
      Mapping dataframe.

    Notes:
    - Ensembl GTFs typically store unversioned IDs in gene_id/transcript_id
    and versions separately in gene_version/transcript_version.
    - If transcript_version is missing, the versioned column will be left blank.
    """
    # Name of versioned columns
    vcol = f"transcript_id_v{version_suffix}"

    # Regex to capture key "value" pairs in the attributes column
    attr_re = re.compile(r'(\S+)\s+"([^"]+)"')

    rows = []
    seen = set()  # dedupe on (gene_id, transcript_id, transcript_version)

    with open_maybe_gz(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != feature:
                continue

            attrs = dict(attr_re.findall(parts[8]))
            gene_id = attrs.get("gene_id")
            tx_id = attrs.get("transcript_id")
            tx_ver = attrs.get("transcript_version")  # expected in Ensembl GTF

            if not gene_id or not tx_id:
                continue

            key = (gene_id, tx_id, tx_ver or "")
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "ensembl_id": gene_id,
                "ensembl_transcript_id": tx_id,
                vcol: f"{tx_id}.{tx_ver}" if tx_ver else "",
            })

    if out_tsv_path:
        with open(out_tsv_path, "w", encoding="utf-8") as out:
            out.write("\t".join(["ensembl_id", "ensembl_transcript_id", vcol]) + "\n")
            for r in rows:
                out.write(f"{r['ensembl_id']}\t{r['ensembl_transcript_id']}\t{r[vcol]}\n")

    map_from_gtf = pd.DataFrame(rows)

    return map_from_gtf

def filter_ID_Map(mapping_df: pd.DataFrame,
                  transcript_info:  Dict[str, Dict],
                  versioned: bool=True) -> pd.DataFrame:
    """
    Filters the mapping_from_gtf DataFrame contatining only transcripts and genes
    that are present in the Spice libray. Extends mapping also with FAS ids,
    transcript and gene symbol and transcript biotype

    Input: 
        mapping_df: Mapping dataframe of IDs and biotypes obtained from mapping_from_gtf
        transcript_info: Loaded transcript_info.json file obtained from the spice
                         library path. Note: Genes in this json
                         file summarize all the genes present in the library as well
                         as their corresponding transcripts.
    Returns:
        filtered_df: Mapping dataframe filtered based on present genes and transcripts plus
                    columns: [fas_id, transcript_symbol, transcript_biotype, gene_symbol]
    """
    hard_need = {"ensembl_id"}
    hard_miss = hard_need - set(mapping_df.columns)
    if hard_miss:
        raise ValueError(f"Mapping table missing required columns: {hard_miss}")
    # filter for genes present in the spice library
    filt_by_gene = mapping_df[mapping_df["ensembl_id"].isin(transcript_info.keys())]
    ver_col = None
    if versioned:
        for c in filt_by_gene.columns:
            # extract column in mapping df with versioned transcript ids
            m = re.fullmatch(r"transcript_id_v(\d+)", str(c))
            if m:
                ver_col = c
        if ver_col is None:
            raise ValueError("No column with versioned ensembl transcript ids found. "
                             "Make sure the column name starts with 'transcript_id_v', \
                              followed by ensembl version. E.g: transcript_id_v113")
        if "ensembl_transcript_id" not in filt_by_gene.columns:
            for idx, r in filt_by_gene.iterrows():
                unver_id = _strip_ver(r["transcript_id_v113"])
                filt_by_gene.loc[idx, "ensembl_transcript_id"] = unver_id
    id_maps = {}
    tx_symbol_maps = {}
    g_symbol_maps = {}
    tx_biotype_maps = {}
    # buld tx id to fas id mapping
    for gene in transcript_info.keys():
        # obtain gene symbol from library and save in mapping dict
        g_symbol_maps[gene] = transcript_info[gene]["name"]
        # obtain list of transcripts/proteins for that gene in library
        gene_tx_info = transcript_info[gene]["transcripts"]
        fas_ids = gene_tx_info.keys()
        # somewhere here I need to check if the gene that the transcript belongs to is also like that in the mapping table
        for fas_id in fas_ids:
            # if fas_id for a NMD transcript, key "transcript_id" is missing
            if "ENST" in fas_id:
                tx_id = fas_id
            # if fas_id for a protein coding transcript, we obtain the ensembl transcript id
            else:
                tx_id = gene_tx_info[fas_id]["transcript_id"]

            id_maps[tx_id] = fas_id
            tx_symbol_maps[tx_id] = gene_tx_info[fas_id]["transcript_name"]
            tx_biotype_maps[tx_id] = gene_tx_info[fas_id]["biotype"]
    # filter by transcripts
    filt_by_tx = filt_by_gene[filt_by_gene["ensembl_transcript_id"].isin(id_maps.keys())]
    # Append fas ids, symbols and transcript biotype to mapping df
    filt_by_tx["fas_id"] = filt_by_tx["ensembl_transcript_id"].map(id_maps)
    filt_by_tx["transcript_symbol"] = filt_by_tx["ensembl_transcript_id"].map(tx_symbol_maps)
    filt_by_tx["transcript_biotype"] = filt_by_tx["ensembl_transcript_id"].map(tx_biotype_maps)
    filt_by_tx["gene_symbol"] = filt_by_tx["ensembl_id"].map(g_symbol_maps)

    return filt_by_tx


def make_tx_whitelist_and_maps(filtered_df: pd.DataFrame,
                               versioned: bool=True) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
    """
    Takes as input a filtered dataframe which contains ensemble ids
    (versioned and unversioned) as well as biotypes for transcripts
    and genes which are included in the spice library.

    Input: 
        -filtered_df: Filtered pandas df obtained by function filter_ID_Map.
        - versioned: If the mapping dataframe contains also versioned IDs.

    Returns:
        allowed genes: set containing unversioned ensembl gene IDs
        allowed_tx_ver: set contatining versioned ensembl transcript IDs
        allowed_tx_unv: set containting unversioned ensembl transcript IDs
        ver_map: dictionary mapping versioned transcript IDs to ensemnle Gene ID
        unver_map: dictionary mapping unversioned transcript IDs to ensemnle Gene ID
        ver_unv_map: dictionary mapping versioned transcript IDs to unversioned transcript IDs
    
    Notes:
        - Requires that filter_ID_Map has been run before on the GTF retrieved mapping
        table.
        - Call this in the beginning of the pipeline because it is needed for later functions
        - Will also be called by build_fixed_AT, percompute_ewfd_fixed_AT

    """
    # Keep only the columns we need
    hard_need = {"ensembl_id", "ensembl_transcript_id", "fas_id"}
    hard_miss = hard_need - set(filtered_df.columns)
    if hard_miss:
        raise ValueError(f"Filtered mapping df is missing required columns: {hard_miss}")
    # Whitelists (drop NAs)
    allowed_tx_unv  = set(filtered_df["ensembl_transcript_id"].astype(str))   # ENST without version
    allowed_genes   = set(filtered_df["ensembl_id"].astype(str))
    unver_map = dict(zip(filtered_df["ensembl_transcript_id"].astype(str),
                        filtered_df["ensembl_id"].astype(str)))
    ver_col = None
    if versioned:
        for c in filtered_df.columns:
            # extract column in mapping df with versioned transcript ids
            m = re.fullmatch(r"transcript_id_v(\d+)", str(c))
            if m:
                ver_col = c
        allowed_tx_ver  = set(filtered_df[ver_col].astype(str))     # ENST with version
        ver_map = dict(zip(filtered_df[ver_col].astype(str),
                           filtered_df["ensembl_id"].astype(str)))
        ver_unv_map = dict(zip(filtered_df[ver_col].astype(str),
                           filtered_df["ensembl_transcript_id"].astype(str)))
    else:
        allowed_tx_ver = None
        ver_map = None
        ver_unv_map = None

    return allowed_genes, allowed_tx_ver, allowed_tx_unv, ver_map, unver_map, ver_unv_map


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############################# CORE ##############################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

######################## Build AT Sets ##########################

def build_fixed_AT(countspath: str | Path,
                   sample_groups: Dict[str, str],
                   filtered_mapping: pd.DataFrame,
                   versioned: bool = True,
                   tau_abs: float = 10.0,
                   r: Optional[int] = None
                   ) -> Dict[str, Dict]:
    """
    Define a fixed active transcript (AT) set per gene using per-sample
    scaled counts.

      - For each transcript and sample, we call it 'active in that sample'
        if count_scaled >= tau_abs.
      - For each transcript across samples, we keep it in the AT set if it is
        active in at least r samples (prevalence rule).
      - Genes with only one transcript in library or less than two AT transcripts are removed

    Inputs:
        countspath: Folder containing:
                  - manifest.samples.tsv  (Sample, file -> metadata table for the scaled count
                  tables per sample)
                  - sampleX.tx_medianRatio_scaledCounts.tsv.gz
                    with columns: TXNAME, count_scaled, (optional:count_raw)
                Note: If you use another type of normalization/scaling make sure these columns
                      are present. Otherwise you will run in an error.
        sample_groups: {sample_id: condition_name}
        filtered_mapping: mapping dataframe with at least obtained from function
                          filter_ID_Map (needs to be run before this)
        versioned: Defines if the count table uses versioned or unversioned ensembl transcript
                   ids. E.g. Salmon uses versioned IDs. Important because spice library uses 
                   unversioned IDs. Also needs to be set for function filter_ID_Map which creates
                   also a mapping of versioned to unversioned IDs. Default: True; set to false if 
                   your count table uses unversioned IDs in TXNAME.
        tau_abs: minimum scaled count to call a transcript active in a sample.
        r: minimum number of samples in which transcript must be active to
           be included in AT set (if not given, half of sample count is used or 
           a miniumum of 2).

    Returns:
        active_sets: {ENSG: {"ids": [tx ids],
                             "mask": bool mask of same length,
                             "AT": number of active transcripts,
                             "active_samples": #samples active per tx}}
        tx_removed: dataframe (transcript_id, reason) for transcripts not in AT set
                    "reason" might be:
                        - tau_abs: the transcript didn't reach set tau_abs in any sample
                        - prevalence_rule: the transcript reached tau_abs in some samples
                                            but not enough, defined by r
        g_removed: dataframe (gene_id, reason) for genes which were removed during this
                   process
                   "reason" might be:
                    - numTx: This gene has only one transcript in the spice library
                    - numAT: After AT filtering, this gene contains only one AT.
    """
    countspath = Path(countspath)
    manifest = pd.read_csv(countspath / "manifest.samples.tsv", sep="\t")
    manifest_grps = manifest.loc[manifest['Sample'].isin(sample_groups.keys())].copy()

    # Build whitelists/maps of genes and transcripts present in the library
    allowed_genes, allowed_tx_ver, allowed_tx_unv, ver_gene_map, unver_gene_map, ver_unv_map = \
        make_tx_whitelist_and_maps(filtered_mapping, versioned=versioned)

    active_sample_counts = defaultdict(int)   # TX -> num samples active

    # Loop through samples, use count_scaled only
    for _, row in manifest_grps.iterrows():
        sample = row["Sample"]
        f = row["file"]
        # Read per-sample scaled counts
        df = pd.read_csv(f, sep="\t", usecols=["TXNAME", "count_scaled"])
        if df.empty:
            warnings.warn(f"WARNING: File {f} is empty. Sample {sample} will be ignored.")
            continue
        if versioned:
            df["TX_UNV"] = df["TXNAME"].map(ver_unv_map)
        else:
            df["TX_UNV"] = df["TXNAME"]
        # keep only transcripts present in library
        mask = df["TX_UNV"].isin(allowed_tx_unv)
        df = df.loc[mask].copy()
        if df.empty:
            warnings.warn(f"WARNING: No whitelisted transcripts found in {f}. Sample {sample} ignored.")
            continue

        counts = df["count_scaled"].to_numpy(dtype=float, copy=False)
        is_active = counts >= tau_abs
        for tname, flag in zip(df["TX_UNV"].values, is_active):
            if tname not in allowed_tx_unv:
                continue
            if flag:
                active_sample_counts[tname] += 1

    # Build gene -> transcripts list (versioned where possible)
    gene_to_txs = defaultdict(list)
    removed_genes = {} # G -> reason
    for gene in allowed_genes:
        gene_to_txs[gene] = list(filtered_mapping[filtered_mapping["ensembl_id"] == gene]["ensembl_transcript_id"])
        if len(gene_to_txs[gene]) < 2:
            removed_genes[gene] = "numTx"
            del gene_to_txs[gene]

    n_samples = len(manifest_grps)
    r_eff = _coerce_r(r, n_samples)

    removed_transcripts = {} # TX -> reason
    # tau_abs:   transcript never active in any sample (cnt == 0)
    # prevalence_rule: transcript active in some samples, but fewer than r_eff
    for g, txs in gene_to_txs.items():
        for t in txs:
            cnt = active_sample_counts.get(t, 0) 
            if cnt == 0:
                removed_transcripts[t] = "tau_abs"
            elif cnt < r_eff:
                removed_transcripts[t] = "prevalence_rule"
    
    per_gene = {}
    for g, txs in gene_to_txs.items():

        # prevalence rule: active in at least r_eff samples
        prev_keep = [t for t in txs if _is_active_enough(t, active_sample_counts, r_eff)]

        if len(prev_keep) < 2:
            removed_genes[g] = "numAT"
            # skip this gene entirely if < 2 transcripts pass prevalence rule
            continue

        mask = np.array([t in prev_keep for t in txs], dtype=bool)
        per_gene[g] = {
            "ids": txs,
            "mask": mask,
            "AT": int(mask.sum()),
            "active_samples": np.array(
                [active_sample_counts.get(t, 0)
                 for t in txs],
                dtype=int
            )
        }

    tx_removed = (
        pd.Series(removed_transcripts, name="reason")
        .rename_axis("transcript_id")
        .reset_index()
    )
    g_removed = (
        pd.Series(removed_genes, name="reason")
        .rename_axis("gene_id")
        .reset_index()
    )


    return per_gene, tx_removed, g_removed


######################## Calculate EWFD vectors ##########################

def precompute_ewfd_fixedAT(countspath: str | Path,
                            libpath: str | Path,
                            active_sets: Dict[str, Dict],
                            filtered_mapping: pd.DataFrame,
                            sample_groups: Dict[str, str],
                            versioned: bool=True,
                            dtype=np.float32) -> Dict[str, Dict]:
    """
    Calculate per-gene EWFD and relative-expression vectors for every
    sample.


    Behavior:
     - For each gene and sample:
          * Build a vector of absolute scaled counts over its AT transcripts.
          * Normalize within gene → relative-expression vector.
          * Use dissimilarity to compute the EWFD vector.
      - For each gene, store:   
          * "AT": number of active transcripts
          * "fas_ids": ids used in FAS matrix
          * "E": list of (sample, group, EWFD_vector)
          * "R": list of (sample, group, rel_expr_vector)
          * "lambda_max": largest eigenvalue of D^T D for FAS dissimilarity
          * "feats": transcript ids of active transcripts
          * "n_feats": number of transcripts
          * "MPD": mean pairwise dissimilariy
    
    Input:
        countspath: Path to scaled counts
        libpath: Path to spice library
        active_sets: per_gene output from build_fixed_AT
        filtered_mapping: Filtered mapping dataframe, same as used in build_fixed_AT
        sample_groups: {sample_id: condition_name}
        versioned: True (default) if versioned IDs were used.
    
    Returns:
        gene_cache: 
    """
    countspath = Path(countspath)
    fas_index = load_fas_index(libpath=libpath)
    fas_scores_fullpath = os.path.join(libpath, "fas_data/fas_scores")
    manifest = pd.read_csv(countspath / "manifest.samples.tsv", sep="\t")
    manifest_grps = manifest.loc[manifest['Sample'].isin(sample_groups.keys())].copy()
    if versioned:
        allowed_genes, allowed_tx_ver, allowed_tx_unv, ver_gene_map, unver_gene_map, ver_unv_map = \
            make_tx_whitelist_and_maps(filtered_mapping, versioned=versioned)
    # FAS id mapping
    fas_id_map = build_fas_id_maps(filtered_mapping)

    # Collect union of all kept transcript IDs
    kept = collect_all_kept_ids(active_sets)

    # Build a simple per-sample cache of scaled counts
    sample_cache: Dict = {}
    for _, row in manifest_grps.iterrows():
        sample = row["Sample"]
        df = pd.read_csv(row["file"], sep="\t", usecols=["TXNAME", "count_scaled"])
        if df.empty:
            warnings.warn(f"WARNING: File {row['file']} empty for sample {sample}.")
            continue
        if versioned:
            df["TX_UNV"] = df["TXNAME"].map(ver_unv_map)
        else:
            df["TX_UNV"] = df["TXNAME"]
        # Keep only transcripts that appear in any AT set
        m = df["TX_UNV"].isin(kept)
        df = df.loc[m].reset_index(drop=True)
        by_tx = {}
        for tx, val in zip(df["TX_UNV"].astype(str).values,
                           df["count_scaled"].astype(dtype).values):
            by_tx[tx] = float(val)
        sample_cache[sample] = by_tx

    gene_cache = {}
    for g, aset in active_sets.items():
        mask = aset["mask"].astype(bool)
        # if mask.sum() < 2:
        #     continue
        kept_ids = [str(tid) for tid, keep in zip(aset["ids"], mask) if keep]

        # Map transcripts to FAS IDs
        mp = fas_id_map.get(g, {})
        fas_ids = []
        for tid in kept_ids:
            fid = mp.get(tid)
            fas_ids.append(fid)

        fas_file = fas_index.get(g)
        if not fas_file:
            warnings.warn(f"Gene {g} not in FAS index.")
            continue

        with open(Path(fas_scores_fullpath) / fas_file, "r") as f:
            data = json.load(f)
        if g not in data:
            warnings.warn(f"No FAS matrix available for gene {g}.")
            continue

        fas_adj = data[g]  # similarity adjacency

        # Trim adjacency to AT set
        fas_ids_set = set(fas_ids)
        kept_fas_adj = {
            i: {j: inner[j] for j in fas_ids_set if j in inner}
            for i, inner in fas_adj.items()
            if i in fas_ids_set
        }

        # Invert similarity to dissimilarity, calc lambda_max & MPD
        kept_fas_adj_invert = invert_fas_adjacency_matrix(kept_fas_adj)
        # lambda_max = calc_lambda_max_for_gene(kept_fas_adj_invert, fas_ids)
        mpd_of_g, sd_of_g = estimate_diversity(kept_fas_adj_invert)

        # Collect EWFD + relative expression for each sample
        E_list = []
        R_list = []

        for _, row in manifest_grps.iterrows():
            sample = row["Sample"]
            grp = sample_groups.get(sample)
            if sample not in sample_cache:
                continue
            maps = sample_cache[sample]

            # Build absolute count vector for this gene's AT transcripts
            vals = []
            for tid in kept_ids:
                v = maps.get(tid)
                vals.append(v)
            X_abs = np.asarray(vals, dtype=dtype)           # shape (n_feats,)
            total = X_abs.sum()
            if total > 0:
                X_rel = X_abs / total
            else:
                X_rel = np.zeros_like(X_abs)

            # EWFD for this sample (1D vector length n_feats)
            ewfd_vec = np.asarray(
                calculate_ewfd(fas_adj, X_rel.tolist(), fas_ids),
                dtype=dtype
            )
            E_list.append((sample, grp, ewfd_vec))
            R_list.append((sample, grp, X_rel))

        if E_list:
            entry = {
                "AT": int(mask.sum()),
                "fas_ids": fas_ids,
                "E": E_list,
                "R": R_list,
                # "lambda_max": float(lambda_max),
                "feats": kept_ids,
                "n_feats": int(len(kept_ids)),
                "MPD": mpd_of_g
            }
            gene_cache[g] = entry

    return gene_cache

######################## Calculate JSD and RMSD per gene ##########################

def calc_obs_metrics(gene_cache: Dict[str, Dict],
                     groups: Set,
                     base: float = 2.0,
                     eps: float = 1e-12
                     ) -> Tuple[Dict[str, float],
                          Dict[str, float],
                          Dict[str, float],
                          Dict[str, float]]:
    """
    Compute per-gene RMSD and JSD using only per-sample ewfd and relative expression vectors.

    Input:
        gene_cache: output of precoompute_ewfd_fixed_AT
        groups: {sample_id: condition_name}
        base: base for JSD calculation. Default: 2 (results bounded in 0,1)
        eps: to avoid division by zero

    Returns:
        obs: Dictionary, RMSD per gene
        jsd: Dictionary, JSD per gene
        grouped_by_AT: Dictionary of format {"AT": {"rmsd": List, "genes": List}}

    """
    obs, jsd = {}, {}
    grouped_by_AT = defaultdict(lambda: {"rmsd": [], "genes": []})
    grp1, grp2 = groups

    log_base = np.log(base)

    for g, obj in gene_cache.items():
        AT = obj["AT"]
        grouped_by_AT[AT]["genes"].append(g)
        # split ewfd and relative expression vectors by group
        EA = [E for _, grp, E in obj["E"] if grp == grp1]
        EB = [E for _, grp, E in obj["E"] if grp == grp2]
        RA = [R for _, grp, R in obj["R"] if grp == grp1]
        RB = [R for _, grp, R in obj["R"] if grp == grp2]
        if not EA or not EB:
            continue
        # make a stack of ewfd and rel. expr. vectors per condition
        EA_ = np.stack(EA, axis=0).astype(np.float64, copy=False)# (nA, #feats)
        EB_ = np.stack(EB, axis=0).astype(np.float64, copy=False)   # (nB, #feats)
        RA_ = np.stack(RA, axis=0).astype(np.float64, copy=False)   # (nA, #feats)
        RB_ = np.stack(RB, axis=0).astype(np.float64, copy=False)   # (nB, #feats)

        # calculate RMSD of ewfd vectors and final mean RMSD
        diff_E = EA_[:, None, :] - EB_[None, :, :]  # (nA, nB, F)
        rmsd_pair = np.sqrt(np.mean(diff_E * diff_E, axis=-1))  # (nA, nB)
        obs_g = float(rmsd_pair.mean())
        grouped_by_AT[AT]["rmsd"].append(obs_g)

        # calculate JSD of relative expression vectors
        RA_norm = RA_ / np.maximum(RA_.sum(axis=-1, keepdims=True), eps)
        RB_norm = RB_ / np.maximum(RB_.sum(axis=-1, keepdims=True), eps)

        PA = RA_norm[:, None, :] # (nA, 1, #feats)
        PB = RB_norm[None, :, :] # (1, nB, #feats)
        M = 0.5 * (PA + PB)

        ratio_PM = np.where(PA > 0, PA / np.maximum(M, eps), 1.0)
        kl_PM = np.sum(np.where(PA > 0,
                                PA * (np.log(ratio_PM) / log_base),
                                0.0),
                       axis=-1) # (nA, nB)

        ratio_QM = np.where(PB > 0, PB / np.maximum(M, eps), 1.0)
        kl_QM = np.sum(np.where(PB > 0,
                                PB * (np.log(ratio_QM) / log_base),
                                0.0),
                       axis=-1) # (nA, nB)

        jsd_pairs = 0.5 * (kl_PM + kl_QM) # (nA, nB)
        jsd_g = float(jsd_pairs.mean())

        obs[g] = obs_g
        jsd[g] = jsd_g

    return obs, jsd, dict(grouped_by_AT)

################### Build initial results table ################

def build_results_table(gene_cache: Dict[str, Dict],
                        obs_rmsd: Dict[str, float],
                        jsd: Dict[str, float]):
    """
    Generates the result table containing the observed RMSD, number of AT,
    JSD between relative expression vectors and MPD of AT transcripts.

    Input:
        gene_cache: For each gene contains information about AT and MPD
        obs_rmsd: Stores Observed RMSD values.
        jsd: Stores Observed JSD values.

    Returns:
        Result table as dataframe with columns "GeneID", "AT", "MPD",
        "JSD", "RMSD"
    """
    items = defaultdict(list)
    for gene, data in gene_cache.items():
        if gene not in obs_rmsd:
            continue
        items["GeneID"].append(gene)
        items["AT"].append(data["AT"])
        items["MPD"].append(data.get("MPD"))
        items["JSD"].append(jsd[gene])
        items["RMSD"].append(obs_rmsd[gene])

    return pd.DataFrame.from_dict(items)

################### Create additional RMSD values per AT-bin ################

def build_AT_reference_dist(gene_cache: Dict[str, Dict]
                            ) -> Tuple[Dict[int, list], Dict[int, Dict[str, list]], Dict[int, Dict[str, list]]]:
    """
    Performs exact label permutation to create additonal RMSD values
    per AT bin using sample-specific ewfd vectors.


    For each gene:
      - Extract sample-specific ewfd vectors
      - Defines n = #samples, nA = #original group A samples.
      - For every combination A of size nA, B = complement:
          * Compute RMSD across all cross-group pairs of ewfd vectors.
          * Store the resulting generated RMSD value

    Returns:
        pooled_by_AT: {AT_bin: [T_perm values pooled over genes]}
        pooled_by_gene: {gene: [T_perm values for this gene]}
        bin_nulls: {AT_bin: {"rmsd": [values], "genes": [genes per value]}}
    """
    pooled_by_AT = defaultdict(list)
    pooled_by_gene = defaultdict(list)
    summary_dict = defaultdict(lambda: {"rmsd": [], "genes": []})

    labeling_cache = {}

    for g, obj in gene_cache.items():
        # E stores sample ID, the condition the ewfd vector belongs to
        # and sample specific ewfd vector
        samples, groups, E_arrays = zip(*obj["E"])
        groups = np.array(groups, dtype=object)
        uniq = np.unique(groups)
        if len(uniq) != 2:
            continue
        # count how many samples
        n = len(groups)
        # count how many samples belong to the first condition
        nA = int((groups == uniq[0]).sum())
        # the key in labeling cache allows to avoid recomputing all_A and all_B each
        # time
        key = (n, nA)
        if key not in labeling_cache:
            # generates every possible way to choose nA samples out of n
            all_A = [np.array(A, dtype=int) for A in itertools.combinations(range(n), nA)]
            # creates the nB complement samples for all_A
            all_B = [np.setdiff1d(np.arange(n), A, assume_unique=True) for A in all_A]
            labeling_cache[key] = list(zip(all_A, all_B))
        AB_pairs = labeling_cache[key]

        # stack ewfd arrays
        X = np.stack(E_arrays, axis=0).astype(np.float64, copy=False)  # (#samples, #features)

        for A_idx, B_idx in AB_pairs:
            # selects the samples assigned to group A and B given the indices
            # in AB_pairs
            EA = X[A_idx, :]
            EB = X[B_idx, :]
            # standard RMSD calculation with mean across (synthetic) cross-condition sampl
            # comparisons
            diff = EA[:, None, :] - EB[None, :, :]    # (nA, nB, F)
            rmsd_pair = np.sqrt(np.mean(diff * diff, axis=-1))  # (nA, nB)
            Tperm = float(rmsd_pair.mean())

            AT = obj["AT"]
            # saves for each AT bin all generated RMSD values
            pooled_by_AT[AT].append(Tperm)
            # saces per gene all generated RMSD values
            pooled_by_gene[g].append(Tperm)
            # same style as grouped_by_AT output from calc_obs_metrics
            summary_dict[AT]["rmsd"].append(Tperm)
            summary_dict[AT]["genes"].append(g)
            

    return dict(pooled_by_AT), dict(pooled_by_gene), dict(summary_dict)

################### Merging of AT reference distributions ################

def define_bins_to_merge(summary_dict: Dict,
                         num_g: int=100) -> List:
    """
    Function that returns the bin IDs of AT-bins
    that for which AT reference distributions are merged
    based on a threshold on the number of (unique) genes
    within an AT bin.
    """
    bin_to_gene_map = defaultdict(int)

    for bin, data in summary_dict.items():
        bin_to_gene_map[bin] = data["genes"]

    bins_to_pool = list()

    for key, data in bin_to_gene_map.items():
        if len(set(data)) < num_g:
            bins_to_pool.append(key)

    return bins_to_pool

def merge_bins(bins_to_pool: list[int],
               summary_dict: Dict[int, Dict[str, list]],
               pooled_by_AT: Dict[int, list],
               result_table: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Accepts a list of bins for which AT reference RMSD values should be pooled.
    Modifies the summary_dict and pooled_by_AT dictionaries (obtained from output
    of build_AT_reference_dist) and the result table (output of calc_obs_metrics)
    accordingly.

    Input:
        bins_to_pool: Output of define_bins_to_merge
        summary_dict, pooled_by_AT: Output of build_AT_reference_dist
        result_table: Output of calc_obs_metrics
    
    Returns:
        Modified summary_dict, pooled_by_AT and result table.

    Notes:
        This function assumes that the data exhibits a pattern where
        the number of genes decreases across AT bins where merging results in ONE
        merged bin which will be denoted with integer 99.
        If your data does not exhibit this property, it makes sense to modify this
        function to enable merging of a low-poupulated bin to a subsequent or preceding
        AT bin, whereas this decision might be based on distance values if possible.
    """
    # Avoid in-place changes
    summary_dict_mod = summary_dict.copy()
    pooled_by_AT_mod = pooled_by_AT.copy()
    result_table_mod = result_table.copy()

    # Aggregate values and genes of bins to merge
    values_to_pool = []
    genes = []

    for AT in bins_to_pool:
        values_to_pool += pooled_by_AT.get(AT)
        genes += summary_dict.get(AT)["genes"]
        # remove this AT bin
        summary_dict_mod.pop(AT)
        pooled_by_AT_mod.pop(AT)

    # Merged bins will be indicated by 99
    summary_dict_mod.update({99: {"rmsd": values_to_pool, "genes": genes}})
    pooled_by_AT_mod.update({99: values_to_pool})

    # Modify result dataframe
    result_table_mod.loc[result_table_mod["AT"].isin(bins_to_pool), "AT"] = 99

    return summary_dict_mod, pooled_by_AT_mod, result_table_mod


########### Fitting parametric distribution functions ###########

def _clip01(x: np.ndarray) -> np.ndarray:
    """
    Helper function helping with clipping RMSD values at boundaries.
    """
    return np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)

# def _empirical_quantiles(x: np.ndarray, qs=(0.9, 0.95, 0.99)) -> Tuple[float, ...]:
#     """
#     Compute empirical quantiles for discrete AT-bin nulls using 'higher' interpolation,
#     so returned values are actual observations (no interpolation between ties).
#     """
#     x = np.asarray(x, float)
#     x = x[np.isfinite(x)]
#     return tuple(np.quantile(x, qs, method="higher"))

def _aic(logL: float, k: int) -> float:
    """Akaike Information Criterion with k free params."""
    return 2*k - 2*logL

def winners_by_AIC(fit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the best fit according to AIC.
    Returns one row per bin with the AIC winner and MAE_q.
    """
    # Find fit with the lowest AIC value
    idx = fit_df.groupby("AT")["AIC"].idxmin()
    return (fit_df.loc[idx, ["AT", "model", "AIC", "params"]] # removed here "MAE_q"
                  .sort_values("AT")
                  .reset_index(drop=True))

def fit_distributions_scipy(pooled_by_AT: Dict[int, List[float]],
                            models: Optional[List[str]] = None
                            ) -> pd.DataFrame:
    """
    Fit SciPy distributions per bin (all on [0,1]) and return a tidy DataFrame.
    This function computes the log likelihood, the AIC to the original data,
    the params of each fitted model determined by ML.

    Input:
        pooled_by_AT : Dictionary saving AT-bin level reference distributions. Possibly
                       modified to contain also merged bins (indicated by 99). Output
                       of either calc_obs_metrics (grouped_by_AT) or merge_bins (summary_dict).
        models : list of models among, optional "beta","johnsonsb", "powerlaw" or all (default)

    Returns:
        DataFrame with columns:
        'AT','model','n','logL','AIC','AICc','params','q90_fit','q95_fit','q99_fit',
        'q90_emp','q95_emp','q99_emp','err_q90','err_q95','err_q99','MAE_q'
    """
    if models is None:
        models = ["beta", "johnsonsb", "powerlaw"]
    rows = []

    for AT, vals in sorted(pooled_by_AT.items()):
        # Clip values at boundaries
        x = _clip01(np.asarray(vals, float))
        x = x[np.isfinite(x)]
        n = x.size
        if n == 0:
            continue

        # calculatee empirical quantiles for reference
        # q90_emp, q95_emp, q99_emp = _empirical_quantiles(x)
    
        # Beta (k=2)
        if "beta" in models:
            # Fit beta distibution
            p = stats.beta.fit(x, floc=0, fscale=1)
            logL = float(np.sum(stats.beta.logpdf(x, *p)))
            k = 2
            # Calculate quantile cutoffs
            # q90,q95,q99 = stats.beta.ppf([0.9,0.95,0.99], *p)
            rows.append(dict(AT=AT, model="beta", n=n, logL=logL,
                             AIC=_aic(logL,k),
                             params=f"a={p[0]:.6g}, b={p[1]:.6g}"))
                            #  q90_fit=q90, q95_fit=q95, q99_fit=q99,
                            #  q90_emp=q90_emp, q95_emp=q95_emp, q99_emp=q99_emp))

        # Johnson SB (k=2)
        if "johnsonsb" in models:
            p = stats.johnsonsb.fit(x, floc=0, fscale=1)
            logL = float(np.sum(stats.johnsonsb.logpdf(x, *p)))
            k = 2
            q90,q95,q99 = stats.johnsonsb.ppf([0.9,0.95,0.99], *p)
            rows.append(dict(AT=AT, model="johnsonsb", n=n, logL=logL,
                             AIC=_aic(logL,k),
                             params=f"a={p[0]:.6g}, b={p[1]:.6g}"))
                            #  q90_fit=q90, q95_fit=q95, q99_fit=q99,
                            #  q90_emp=q90_emp, q95_emp=q95_emp, q99_emp=q99_emp))

        # Power-function ("powerlaw") (k=1)
        if "powerlaw" in models:
            p = stats.powerlaw.fit(x, floc=0, fscale=1)
            logL = float(np.sum(stats.powerlaw.logpdf(x, *p)))
            k = 1
            # q90,q95,q99 = stats.powerlaw.ppf([0.9,0.95,0.99], *p)
            rows.append(dict(AT=AT, model="powerlaw", n=n, logL=logL,
                             AIC=_aic(logL,k),
                             params=f"a={p[0]:.6g}"))
                            #  q90_fit=q90, q95_fit=q95, q99_fit=q99,
                            #  q90_emp=q90_emp, q95_emp=q95_emp, q99_emp=q99_emp))

    fit_df = pd.DataFrame(rows).sort_values(["AT", "model"]).reset_index(drop=True)

    # Add absolute errors & MAE across the three quantiles
    # for q in (90,95,99):
    #     fit_df[f"err_q{q}"] = (fit_df[f"q{q}_fit"] - fit_df[f"q{q}_emp"]).abs()
    # fit_df["MAE_q"] = fit_df[["err_q90","err_q95","err_q99"]].mean(axis=1)
    return fit_df

############# Calculate cumulative probabilities ##############

def add_cum_probs(result_table: pd.DataFrame,
                  winners_df: pd.DataFrame,
                  value_col: str = "RMSD",
                  out_col: str = "RMSD_q",
                  clip_eps: float = 1e-12) -> pd.DataFrame:
    """
    Calculates cumulative probabilities of observed RMSD values in column "RMSD" of the
    results table using the fitted beta distribution parameters.

    Input:
        result_table: Result DataFrame obtained from func build_results_table or merge_bins
                      (depends on whether you merged bins on which bins you fitted the
                        distributiom)
                      Must contain at least columns "AT" and the column "RMSD"
        winners_df: DataFrame from winners_by_AIC with columns ['AT','model','params']
                    where model MUST be 'beta' for all rows.
        value_col: Column to score, should be "RMSD" (default)
        out_col: Name of the column which stores the cumulative probabilities (default: "RMSD_q")
        clip_eps: Small epsilon to avoid evaluating CDF exactly at 0 or 1.
    
    Returns:
        out: Modified result_table (DataFrame) containing the calculated cumulative
             probilities for each row.
    
    Note:
        The function EXPECTS that beta was the best fit. Please modify if not.
    """
    ab_by_AT = {}
    for _, r in winners_df.iterrows():
        # Obtain beta parameters
        a, b = _parse_beta_params_beta_only(r["params"])
        ab_by_AT[r["AT"]] = (a, b)

    # Prepare output columns
    out = result_table.copy()
    out[out_col] = np.nan

    for AT, idx in out.groupby("AT", sort=False).groups.items():
        if AT not in ab_by_AT:
            continue
        a, b = ab_by_AT[AT]
        vals = out.loc[idx, value_col].astype(float).to_numpy()
        # Clip to (0,1) for numeric safety (floating point errors)
        vals = np.clip(vals, clip_eps, 1.0 - clip_eps)
        # Calculate cum.prob. using beta params for
        # each observed RMSD in each AT bin
        p = stats.beta.cdf(vals, a, b, loc=0, scale=1.0)
        # Add new column to the result df
        out.loc[idx, out_col] = p

    return out

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#################### HELPFUL OUTPUT DATAFRAMES ##################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def gene_cache_to_rel_expr_df(gene_cache: Dict[str, Dict[str, Any]],
                              r_key: str = "R",
                              fas_key: str = "fas_ids",
                              feats_key: str = "feats",
                              col_prefix: str = "rel_expr",
                              dtype: str = "float32") -> pd.DataFrame:
    """
    Creates a dataframe of relative expression values from gene_cache. Shows columns:
        - gene: ENSEMBL gene ID
        - Name: FAS ID of a transcript
        - txname: ENSEMBL transcript ID (unversioned) for transcripts
        - one column per sample in format {col_prefix}_{sample_id}_{condition},
          stores relative expression per transcript obtained from gene_cache
    
    Input:
        gene_cache: Output from precompute_ewfd_fixed_AT
        r_key: Key in gene_cache that stores sample_id, condition name and relative
               expression vector. Use default.
        fas_key: Key in gene_cache that stores FAS IDs of active transcripts
        feats_kex: Key in gene_cache that stores ENSEMBL transcript IDs of active transcripts
        col_prefix: Prefix for the sample-specific relative expression vector columns.

    """
    rows = []
    all_cols = set()

    for gene, gdat in gene_cache.items():
        # Retieve for each gene in gene_cache fas_ids, transcript ids and
        # sample-specific relative expression vectors
        fas_ids = list(gdat.get(fas_key, []))
        feats = list(gdat.get(feats_key, []))
        r_list = list(gdat.get(r_key, []))
        n = len(fas_ids)
        base = []
        for i in range(n):
            base.append({
                "gene": gene,
                "Name": fas_ids[i],
                "txname": feats[i],
            })
        # Add sample columns with relative expression vectors per sample
        for sample_id, condition, rel_arr in r_list:
            col = f"{col_prefix}_{sample_id}_{condition}"
            all_cols.add(col)

            for i in range(n):
                base[i][col] = float(rel_arr[i])

        rows.extend(base)

    df = pd.DataFrame(rows)
    rel_cols = sorted(all_cols)
    df[rel_cols] = df[rel_cols].astype(dtype)
    # Order columns
    df = df[["gene", "Name", "txname", *rel_cols]]

    return df


def get_gene_expression_per_sample(metadata: pd.DataFrame,
                                   gene_cache: Dict,
                                   sample_groups: Dict[str, str],
                                   group_col: str,
                                   ver_map: Dict,
                                   scaled_outdir: str | Path,
                                   versioned: bool = True,
                                   ver_unv_map: Optional[Dict[str, str]] = None,   # versioned -> unversioned
                                   agg: Optional[Literal["mean", "median"]] = None,
                                   include_raw: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Sums transcript level TPM and scaled count vales to get the gene expression value for
    each gene in each sample.
    Returns per-(gene,sample,group) sums for TPM and scaled counts in a dataframe.
    If agg is provided (either mean or median), another dataframe will be generated containing
    for each gene and each condition the gene expression estimates (TPM and scaled counts)
    averaged across within-condition samples.
    
    Input:
        metadata: Metadata dataframe containing the sample information
        gene_cache: Output of precompute_ewfd_fixed_AT
        sample_groups: Mapping of samples to conditions
        group_col: Column that denotes the conditions, required for calling tidy_expr_over_samples
        ver_map: Mapping of unversioned/versioned (unver_map or ver_map output of make_tx_whitelist_and_maps)
                 transcript IDs to genes
        scaled_outdir: Path to the normalized counts which were generated with swish_scale_from_metadata
        versioned: If the quantification tool used versioned transcript IDs
        ver_unv_map: Mapping of versioned transcript IDs to unversioned. Provide
                     if versioned == True
        agg: How to aggregate within-condition gene expression values. Either
             ["mean", "median"]. "mean" calculates the mean gene expression (for TPM and
             normalized counts, optionally raw counts) across within-condition
             replicate samples.
        include_raw: True if you want the raw counts also displayed in the df

    """
    groups = set(sample_groups.values())
    # create long dataframe denoting TPM expression of transcripts in each sample
    # obtained from Salmon quantification files
    tidy_sal = tidy_expr_over_samples(
        metadata=metadata,
        groups=groups,
        tr_gene_map=ver_map,
        group_col=group_col,
    )

    # AT transcripts in VERSIONED space for TPM join
    if versioned:
        if ver_unv_map is None:
            raise ValueError("versioned=True requires ver_unv_map.")
        # Create a df where genes are mapped to versioned transcript IDs ("gene" and "Name" cols)
        gene_tx_ver = build_gene_tx_df(
            gene_cache=gene_cache,
            versioned=True,
            ver_unv_map=ver_unv_map,   # your build_gene_tx_df inverts internally; OK if it does
        )
    else:
        # Create a df where genes are mapped to unversioned transcript IDs ("gene" and "Name" cols)
        gene_tx_ver = build_gene_tx_df(gene_cache, versioned=False)

    tidy_sal_f = tidy_sal.merge(gene_tx_ver, on=["gene", "Name"], how="inner")
    # calculate the sum of transcript-level TPM estimates to obtain gene expression TPM across samples
    tpm_sum = (
        tidy_sal_f
        .groupby(["gene", "sample", "group"], as_index=False)["TPM"]
        .sum()
        .rename(columns={"TPM": "TPM_sum"})
    )

    # Do the same for but using the normalized counts
    keep_cols = ("count_scaled", "count_raw") if include_raw else ("count_scaled",)
    tidy_sc = tidy_scaledcounts_over_samples(
        scaled_outdir=scaled_outdir,
        sample_groups=sample_groups,
        versioned=versioned,
        ver_unv_map=ver_unv_map,
        keep_cols=keep_cols,
    )

    # AT transcripts using UNVERSIONED Ids for scaled counts join
    gene_tx_unv = build_gene_tx_df(
        gene_cache=gene_cache,
        versioned=False,   # IMPORTANT: gene_cache feats are unversioned
    ).rename(columns={"Name": "tx_unv"})

    tidy_sc_f = tidy_sc.rename(columns={"Name": "tx_unv"}).merge(
        gene_tx_unv, on=["tx_unv"], how="inner"
    )

    sc_cols = [c for c in keep_cols if c in tidy_sc_f.columns]
    sc_sum = (
        tidy_sc_f
        .groupby(["gene", "sample", "group"], as_index=False)[sc_cols]
        .sum()
        .rename(columns={c: f"{c}_sum" for c in sc_cols})
    )

    # Merge TPM and scaled-count sums
    gene_sample_group = (
        tpm_sum.merge(sc_sum, on=["gene", "sample", "group"], how="outer")
    )
    # 
    if agg is None:
        return gene_sample_group, None

    # Create another table where gene expression values (both scaled counts and TPM)
    # are averaged across condition replicates.
    agg_dict = {"TPM_sum": agg}
    rename_dict = {"TPM_sum": f"TPM_{agg}"}
    for c in sc_cols:
        agg_dict[f"{c}_sum"] = agg
        rename_dict[f"{c}_sum"] = f"{c}_{agg}"

    gene_group_agg = (
        gene_sample_group
        .groupby(["gene", "group"], as_index=False)
        .agg(agg_dict)
        .rename(columns=rename_dict)
    )

    return gene_sample_group, gene_group_agg

def add_agg_gene_expression(gene_group_agg: pd.DataFrame,
                            result_table: pd.DataFrame) -> pd.DataFrame:
    """
    Extend result table with aggregated gene expression values
    obtained from get_gene_expression_per_sample.

    Input:
        gene_group_agg: Dataframe output of get_gene_expression_per_sample
                        run with "agg" parameter. Contains gene expression values
                        across conditions. Requires "gene", "group" and one or more
                        valuue columns (e.g. "TPM_mean", "count_scaled_mean", "count_raw_mean")
        result_table: Dataframe output from add_cum_probs or build_results_table
    Returns:
        Result table with added columns informing about the aggregated gene expression
        per condition.
    """
    if gene_group_agg is None or gene_group_agg.empty:
        return result_table.copy()

    if not {"gene", "group"}.issubset(gene_group_agg.columns):
        raise ValueError("gene_group_agg must have columns {'gene','group'}")

    value_cols = [c for c in gene_group_agg.columns if c not in {"gene", "group"}]
    if not value_cols:
        return result_table.copy()

    wide = (
        gene_group_agg
        .pivot(index="gene", columns="group", values=value_cols)
    )
    wide.columns = [f"{val}_{grp}" for (val, grp) in wide.columns]
    wide = wide.reset_index()

    out = (
        result_table
        .merge(wide, left_on="GeneID", right_on="gene", how="left")
        .drop(columns=["gene"])
        .drop_duplicates()
    )
    return out

def scaledcounts_to_wide_df(scaled_outdir: str | Path,
                            sample_groups: Dict[str, str],
                            versioned: bool = True,
                            ver_unv_map: Optional[Dict[str, str]] = None,
                            include_raw: bool = False) -> pd.DataFrame:
    """
    Helper function for scaledcounts_to_AT_wide_df.
    Create a wide transcript-level scaled-count dataframe with:
      - rows: transcript ID (unversioned)
      - cols: count_scaled_<sample>_<group> (and optionally count_raw_...)

    Input:
        scaled_outdir: Path to the files containing normalized counts, generated
                       when running swish_scale_from_metadata
        sample_groups: Dictionary mapping each sample ID to condition.
        versioned: True if the quantification table contains versioned ENST IDs.
        ver_unv_map: Only provide if versioned == True. Contains mapping of versioned
                     to unversioned ENST IDs
        include_raw: True if you want to include columns for the raw counts 
                     per-(transcript, sample).
    """
    keep_cols = ("count_scaled", "count_raw") if include_raw else ("count_scaled",)
    tidy = tidy_scaledcounts_over_samples(
        scaled_outdir=scaled_outdir,
        sample_groups=sample_groups,
        versioned=versioned,
        ver_unv_map=ver_unv_map,
        keep_cols=keep_cols,
    )
    if tidy.empty:
        return pd.DataFrame(columns=["Name"])

    tidy["sample_group"] = tidy["sample"] + "_" + tidy["group"]

    # pivot each measure separately and merge
    frames = []
    for meas in keep_cols:
        if meas not in tidy.columns:
            continue
        wide = tidy.pivot(index="Name", columns="sample_group", values=meas)
        wide.columns = [f"{meas}_{c}" for c in wide.columns]
        frames.append(wide)

    out = pd.concat(frames, axis=1).reset_index()
    out.columns.name = None
    return out

def scaledcounts_to_AT_wide_df(scaled_outdir: str | Path,
                               gene_cache: Dict[str, Dict[str, Any]],
                               sample_groups: Dict[str, str],
                               filtered_mapping: pd.DataFrame,
                               versioned: bool = True,
                               ver_unv_map: Optional[Dict[str, str]] = None,
                               include_raw: bool = False) -> pd.DataFrame:
    """
    Build a wide transcript-level matrix of scaled/normalized counts restricted to
    AT transcripts in gene_cache, using FAS IDs.

    Returns:
        Dataframe with columns:
            - gene: ENSEMBL gene IDs
            - Name: FAS ID of transcript in column txname.
            - txname: transcript id (unversioned ENST)  [optional, controlled by keep_txname]
            - count_scaled_<sample>_<group> (and optionally count_raw_<sample>_<group>)

    Input:
        scaled_outdir: Path to the files containing normalized counts, generated
                       when running swish_scale_from_metadata
        gene_cache: Output of precompute_ewfd_fixed_AT
        sample_groups: Dictionary mapping each sample ID to condition.
        filtered_mapping: Mapping table filtered to gene and transcripts present in 
                          Spice library (output of filter_ID_Map)
        versioned: True if the quantification table contains versioned ENST IDs.
        ver_unv_map: Only provide if versioned == True. Contains mapping of versioned
                     to unversioned ENST IDs
        include_raw: True if you want to include columns for the raw counts 
                     per-(transcript, sample).
    """
    # Create table with columns txname (unversioned transcript ID), count_scaled_<sample>_<group> per sample
    wide_tx = scaledcounts_to_wide_df(
        scaled_outdir=scaled_outdir,
        sample_groups=sample_groups,
        versioned=versioned,
        ver_unv_map=ver_unv_map,
        include_raw=include_raw,
    )
    # rename column "Name" to "txname"
    wide_tx = wide_tx.rename(columns={"Name": "txname"}) 

    # Resrict the set of transcripts to the AT sets of genes in gene_cache
    # gene_tx_df contains columns: ["gene","txname"] where "Name" is transcript id (unversioned ENST)
    gene_tx_df = build_gene_tx_df(
        gene_cache=gene_cache,
        versioned=False,
        ver_unv_map=ver_unv_map,
    ).rename(columns={"Name": "txname"})

    # Map gene, txname -> fas_id via mapping table
    need = {"ensembl_id", "ensembl_transcript_id", "fas_id"}
    miss = need - set(filtered_mapping.columns)
    if miss:
        raise ValueError(f"filtered_mapping missing columns: {miss}")

    fas_map = (
        filtered_mapping[["ensembl_id", "ensembl_transcript_id", "fas_id"]]
        .rename(columns={"ensembl_id": "gene", "ensembl_transcript_id": "txname"})
        .drop_duplicates()
    )

    gene_tx_fas = gene_tx_df.merge(fas_map, on=["gene", "txname"], how="left")
    gene_tx_fas = gene_tx_fas.rename(columns={"fas_id": "Name"})  # matches gene_cache_to_rel_expr_df
    out = gene_tx_fas.merge(wide_tx, on="txname", how="left")
    wide_cols = [c for c in out.columns if c.startswith("count_scaled_") or c.startswith("count_raw_")]
    base_cols = ["gene", "Name", "txname"]
    out = out[base_cols + wide_cols]

    return out

def add_gene_symbols(mapping: pd.DataFrame,
                     result_table: pd.DataFrame):
    """
    Extends the result dataframe with gene symbols
    obtained from the mapping dataframe.

    Input:
        mapping: DataFrame containing the mappings of gene IDs to
                 transcript IDs, symbols etc. (output of filter_ID_Map)
        result_table: Table obtained after running the full result generation
                      pipeline (either before or after cumulative prob. calculation)
    Returns:
        DataFrame with columns as in result_table with additonal "gene_symbol"
        column.
    """
    mapping_short = mapping[["ensembl_id", "gene_symbol"]]
    extended_results = (
        result_table.merge(
            mapping_short,
            left_on="GeneID",
            right_on="ensembl_id",
            how="left",
        )
    .drop(columns="ensembl_id") 
    ).drop_duplicates()
    return extended_results

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#################### EXPORTING RESULTS ##########################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def export_tables(outdir: str | Path,
                  removed_tx_table: Optional[pd.DataFrame] = None,
                  result_table: Optional[pd.DataFrame] = None,
                  tidy_expr_table: Optional[pd.DataFrame] = None,
                  tidy_relexpr_table: Optional[pd.DataFrame] = None) -> None:
    if all(df is None for df in (removed_tx_table, result_table, tidy_expr_table)):
        raise TypeError("No submitted DataFrames to save.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if removed_tx_table is not None:
        outfile_rmtx = outdir/"removed_transcripts.tsv"
        removed_tx_table.to_csv(outfile_rmtx, index=False, sep="\t")
    if result_table is not None:
        outfile_res = outdir/"spice_result.tsv"
        result_table_sorted = result_table.sort_values("RMSD_q", ascending=False)
        result_table_sorted.to_csv(outfile_res, index=False, sep="\t")
    if tidy_expr_table is not None:
        outfile_ex = outdir/"ATx_scCounts_per_gene.tsv"
        tidy_expr_table.to_csv(outfile_ex, index=False, sep="\t")
    if tidy_relexpr_table is not None:
        outfile_ex = outdir/"ATx_relexpr_per_gene.tsv"
        tidy_relexpr_table.to_csv(outfile_ex, index=False, sep="\t")


# removed_genes_psc_meso.to_csv("/home/katharina/msc/spice/psc_vs_meso2/rmvGenes.tsv", sep="\t", index=False)
# removed_genes_meso_cardmeso.to_csv("/home/katharina/msc/spice/meso_vs_cardmeso2/rmvGenes.tsv", sep="\t", index=False)
# removed_genes_cardmeso_cm.to_csv("/home/katharina/msc/spice/cardmeso_vs_cm2/rmvGenes.tsv", sep="\t", index=False)
# # add this to export_tables