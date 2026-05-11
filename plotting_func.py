from typing import List, Literal, List, Dict, Optional
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_fa_gene(lib_path, gene_id, fa_index):
    if gene_id not in fa_index:
        raise KeyError(f"No FA entry for gene {gene_id!r} fas_index.json")
    file_id = f"{fa_index[gene_id]:09d}"    
    fa_path = os.path.join(lib_path, "fas_data", "architectures", f"{file_id}.json")
    with open(fa_path) as f:
        data = json.load(f)
    return data[gene_id]

def load_fa_index(lib_path):
    with open(os.path.join(lib_path, "fas_data/architectures/index.json")) as f:
        fa_index = json.load(f)["genes"]
    return fa_index

def se(x):
    x = x.dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

def map_id_to_symbol(expr_df: pd.DataFrame,
                     mapping: pd.DataFrame):
    tr_map_symb = dict(zip(mapping["fas_id"], mapping["transcript_symbol"]))
    g_map_symb = dict(zip(mapping["ensembl_id"], mapping["gene_symbol"]))
    expr_df["gene_symbol"] = expr_df["gene"].map(g_map_symb)
    expr_df["transcript_symbol"] = expr_df["Name"].map(tr_map_symb)
    return expr_df

# FA plot functions where obtained from Spice dashboard created by Blümel

def organize_fa_data(fa_data):
    length, features = fa_data['length'], fa_data['fmap']
    final_fa_data = {}
    path = list(features.keys())
    for i in path:
        instance = features[i]
        tool = instance[0].split('_')[0]
        if tool not in final_fa_data:
            final_fa_data[tool] = {}
        if instance[0] not in final_fa_data[tool]:
            final_fa_data[tool][instance[0]] = {'x': [], 'y': []}
        final_fa_data[tool][instance[0]]['x'].append(instance[1])
        final_fa_data[tool][instance[0]]['x'].append(instance[2])
        final_fa_data[tool][instance[0]]['x'].append(None)

    lane = 1
    for tool in final_fa_data:
        for feature in final_fa_data[tool]:
            last = [-1]
            entry = 0
            while entry < len(final_fa_data[tool][feature]['x']):
                if final_fa_data[tool][feature]['x'][entry] is None:
                    final_fa_data[tool][feature]['y'].append(None)
                    entry += 1
                else:
                    tmp = True
                    x = 0
                    while tmp:
                        if x >= len(last):
                            final_fa_data[tool][feature]['y'].extend([lane + x, lane + x])
                            last.append(final_fa_data[tool][feature]['x'][entry+1])
                            tmp = False
                        elif final_fa_data[tool][feature]['x'][entry] > last[x]:
                            final_fa_data[tool][feature]['y'].extend([lane + x, lane + x])
                            last[x] = final_fa_data[tool][feature]['x'][entry+1]
                            tmp = False
                        else:
                            x += 1
                    entry += 2
            lane = lane + len(last)
    return final_fa_data, length, lane


def create_fa_plot_input_multi(fa_data, length, isoforms):
    # step size along x axis
    stepsize = 1
    maxlen = 1
    for le in length:
        if le is not None and le > maxlen:
            maxlen = le
    if maxlen >= 100:
        stepsize = int(maxlen / 100)

    # collect all tools across isoforms
    tools = set()
    for iso in isoforms:
        if fa_data.get(iso) is None:
            fa_data[iso] = {}
        tools.update(fa_data[iso].keys())

    n = len(isoforms)
    x = [[] for _ in range(n)]
    y = [[] for _ in range(n)]
    labels = [[] for _ in range(n)]

    for tool in tools:
        features = set()
        for iso in isoforms:
            if tool not in fa_data[iso]:
                fa_data[iso][tool] = {}
            features.update(fa_data[iso][tool].keys())

        # fill per feature/per isoform
        for feature in features:
            for i, iso in enumerate(isoforms):
                if feature in fa_data[iso][tool]:
                    entry = 0
                    vals = fa_data[iso][tool][feature]
                    while entry < len(vals['x']):
                        start = vals['x'][entry]
                        stop  = vals['x'][entry + 1]
                        y_val = vals['y'][entry]
                        # start point
                        x[i].append(start)
                        y[i].append(y_val)
                        labels[i].append(feature)
                        # intermediate sampling
                        step = start + stepsize
                        while step < stop:
                            x[i].append(step)
                            y[i].append(y_val)
                            labels[i].append(feature)
                            step += stepsize
                        # end + gap
                        x[i].extend([stop, None])
                        y[i].extend([y_val, y_val])
                        labels[i].extend([feature, feature])
                        entry += 3
                else:
                    # keep arrays same length even if isoform lacks this feature
                    x[i].append(None)
                    y[i].append(None)
                    labels[i].append(feature)

    # add baseline/non-coding shape
    isoform_labels = []
    for i, le in enumerate(length):
        if le is not None:
            x[i].extend([0, le])
            y[i].extend([0, 0])
            labels[i].extend(['Protein Length', 'Protein Length'])
            isoform_labels.append(isoforms[i])
        else:
            # draw the “crossed box” for non-coding like blümel
            x[i].extend([0, maxlen, 0, 0, maxlen, 0])
            y[i].extend([-0.25, 1.25, 1.25, -0.25, -0.25, 1.25])
            labels[i].extend(['Non-coding'] * 6)
            isoform_labels.append(isoforms[i] + ' (Non-Coding)')

    return x, y, labels, maxlen, isoform_labels


def create_fa_plot_multi(x, y, labels, maxlen, isoforms, line_size, lanes):
    """
    Generalized create_fa_plot for any number of isoforms (one row per isoform).
    """
    n = len(isoforms)
    fig = make_subplots(
        rows=n,
        cols=1,
        specs=[[{'type': 'scatter'}] for _ in range(n)],
        subplot_titles=isoforms,
        vertical_spacing=0.1
    )
    fig.update_layout(height=250 * n)
    dfs = [
        pd.DataFrame({'x': x[i], 'y': y[i], 'labels': labels[i], 'fids': labels[i]})
        for i in range(n)
    ]

    # color palette taken from first isoform
    n_colors = len(dfs[0]['labels'].unique())
    if n_colors <= 1:
        colors = ['blue']
    else:
        colors = px.colors.sample_colorscale(
            'Rainbow', [k / (n_colors - 1) for k in range(n_colors)]
        )

    tmpfigs = [
        px.line(df, x='x', y='y', color='labels',
                custom_data=("fids",), color_discrete_sequence=colors)
        for df in dfs
    ]

    used_legend_names = set()
    for row, tmpfig in enumerate(tmpfigs, start=1):
        for tr in tmpfig.data:
            name = tr['name']
            tr['line']['width'] = line_size

            if name == 'Protein Length':
                tr['line']['color'] = 'black'
                tr['line']['width'] = 2
            elif name == 'Non-coding':
                tr['line']['color'] = 'black'
                tr['line']['width'] = 2
                tr['showlegend'] = False

            # only show each legend entry once
            if name in used_legend_names:
                tr['showlegend'] = False

            fig.add_trace(tr, row=row, col=1)
            used_legend_names.add(name)

    fig.update_traces(connectgaps=False)

    for i in range(1, n + 1):
        fig.update_yaxes(
            showticklabels=False,
            range=[-0.5, lanes[i-1] + 0.5],
            row=i, col=1
        )
        fig.update_xaxes(range=[0, maxlen], row=i, col=1)

    fig.update_layout(yaxis_title='', font=dict(size=14))
    return fig


def feature_architecture_figure(lib_path: Path | str,
                                gene_id: str,
                                expr_df: pd.DataFrame,
                                fa_index: Dict,
                                isoforms: Optional[List] = None,
                                line_width=2):
    """
    Wrapper for generating the FA plots (main plotting code was derived from the Spice Dashboard).
    """
    fa_gene = load_fa_gene(lib_path=lib_path, gene_id=gene_id, fa_index=fa_index)

    # choose which isoforms to plot
    if isoforms is None:
        isoforms = (
            expr_df.loc[expr_df["gene"] == gene_id, "Name"]
            .dropna()
            .unique()
            .tolist()
        )

    data = {}
    length = []
    lanes = []

    for iso in isoforms:
        if iso in fa_gene:
            # isoform has a feature architecture
            fa_struct = fa_gene[iso]
            d, le, lane_count = organize_fa_data(fa_struct)
            data[iso] = d
            length.append(le)
            lanes.append(lane_count)
        else:
            # no FA for this isoform -> will be drawn as NMD cross like blümel did
            data[iso] = None
            length.append(None)
            lanes.append(1)

    x, y, labels, maxlen, isoform_labels = create_fa_plot_input_multi(
        data, length, isoforms
    )
    fig = create_fa_plot_multi(
        x, y, labels, maxlen, isoform_labels, line_width, lanes
    )
    return fig


def plot_expr_bars(expr_df: pd.DataFrame,
                   mapping: pd.DataFrame,
                   groups: List,
                   gene: str,
                   id_type: Literal["ensembl", "symbol"] = "ensembl",
                   versioned_tr_id: Literal[True, False] = True,
                   how: Literal["TPM", "rel"] = "rel",
                   by: Literal["ensembl", "symbol"] = "ensembl",
                   condition_colors: Optional[Dict[str, str]] = None):
    """
    Plot transcript expression bars with standard errors
    for one gene across conditions.

    Inputs:
        expr_df: Expression dataframe.
        mapping: Mapping dataframe used by map_id_to_symbol().
        groups: List of condition names to detect in sample column names.
        gene: Gene ENSG ID or symbol, depending on "id_type".
        id_type: Whether "gene" is an ensembl gene ID or a gene symbol 
                  ( "ensembl", "symbol")
        versioned_tr_id: If 
        by: Whether transcripts should be labeled by ensembl transcript ID or symbol
             ("ensembl", "symbol").
        condition_colors: Optional dictionary mapping condition name to matplotlib color, e.g.
                           {"PSC": "tab:orange", "mesoderm": "#E69F00"}
    """

    no_symb_annot = "gene_symbol" not in expr_df.columns
    typ_ens_by_symb = (id_type == "ensembl") and (by == "symbol")
    typ_ens_by_ens = (id_type == "ensembl") and (by == "ensembl")
    typ_symb_by_ens = (id_type == "symbol") and (by == "ensembl")
    typ_symb_by_symb = (id_type == "symbol") and (by == "symbol")

    if typ_ens_by_ens:
        expr_df_gene = expr_df[expr_df["gene"] == gene]
        gene_id = gene
    
    elif typ_ens_by_symb:
        expr_df_gene = expr_df[expr_df["gene"] == gene]
        if expr_df_gene.empty:
            raise ValueError(f"Gene '{gene}' not found in expr_df.")
        if no_symb_annot:
            expr_df_gene = map_id_to_symbol(expr_df=expr_df_gene,
                                            mapping=mapping,
                                            versioned_tr_id=versioned_tr_id)
            gene_id = expr_df_gene.iloc[0]["gene_symbol"]
        else:
            gene_id = expr_df_gene.iloc[0]["gene_symbol"]
    
    elif typ_symb_by_ens or typ_symb_by_symb:
        gene_id = gene
        if no_symb_annot:
            expr_df = expr_df.copy()
            expr_df = map_id_to_symbol(expr_df=expr_df,
                                       mapping=mapping,
                                       versioned_tr_id=versioned_tr_id)
        expr_df_gene = expr_df[expr_df["gene_symbol"] == gene]

    cols = [c for c in expr_df_gene.columns if c.startswith("rel_expr_")]

    # Make long format
    id_cols = [c for c in expr_df_gene.columns if c not in cols]
    long_df = expr_df_gene.melt(
        id_vars=id_cols,
        value_vars=cols,
        var_name="sample",
        value_name="value"
    )
    
    # Assign condition
    long_df["condition"] = "cond"
    for gr in groups:
        long_df.loc[long_df["sample"].str.contains(gr, regex=False, na=False), "condition"] = gr

    if by == "ensembl":
        if "Name" not in long_df.columns:
            raise KeyError("Column 'Name' not found, but by='ensembl' was requested.")
        summary = (
            long_df
            .groupby(["Name", "condition"], as_index=False)
            .agg(mean_value=("value", "mean"),
                 se_value=("value", se))
        )
        names = summary["Name"].unique()

    elif by == "symbol":
        if "transcript_symbol" not in long_df.columns:
            raise KeyError("Column 'transcript_symbol' not found, but by='symbol' was requested.")
        summary = (
            long_df
            .groupby(["transcript_symbol", "condition"], as_index=False)
            .agg(mean_value=("value", "mean"),
                 se_value=("value", se))
        )
        names = summary["transcript_symbol"].unique()

    conditions = summary["condition"].unique()
    x = np.arange(len(names))
    n_cond = len(conditions)
    width = 0.8 / n_cond
    
    fig, ax = plt.subplots(figsize=(1.5 * len(names), 6))
    
    for i, cond in enumerate(conditions):
        if by == "ensembl":
            sub = summary[summary["condition"] == cond].set_index("Name")
        elif by == "symbol":
            sub = summary[summary["condition"] == cond].set_index("transcript_symbol")

        sub = sub.reindex(names)
        bar_positions = x + (i - (n_cond - 1) / 2) * width

        bar_kwargs = {}
        if condition_colors is not None and cond in condition_colors:
            bar_kwargs["color"] = condition_colors[cond]

        ax.bar(
            bar_positions,
            sub["mean_value"],
            width,
            yerr=sub["se_value"],
            capsize=4,
            label=cond,
            align="center",
            **bar_kwargs
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")

    ax.set_ylabel("Relative expression (mean ± SE)")

    ax.set_title(gene_id)
    ax.set_xlabel("Transcript")
    ax.legend(title="Condition")
    plt.tight_layout()
    plt.show()

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    "Calculates row-wise z-scores"
    x = df.to_numpy(dtype=float)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return pd.DataFrame((x - mu) / sd, index=df.index, columns=df.columns)


def order_samples_by_celltype(sample_names, samples_meta: pd.DataFrame,
                             sample_col="Sample", cond_col="Celltype",
                             desired=("PSC", "mesoderm", "cardiac mesoderm", "CM")):
    """
    Orders samples in the desired order.
    """
    meta = samples_meta.copy()
    meta = meta.set_index(sample_col, drop=True)
    keep = [s for s in sample_names if s in meta.index]
    meta = meta.loc[keep, :].copy()
    # order by desired celltype sequence then by sample name
    desired_list = list(desired)
    meta["_order"] = meta[cond_col].apply(lambda x: desired_list.index(x) if x in desired_list else 999)
    meta["_sample"] = meta.index

    meta = meta.sort_values(["_order", "_sample"], kind="stable")
    ordered = meta.index.tolist()

    return ordered, meta


def build_marker_matrix_from_vst(vst_df: pd.DataFrame,
                                 marker_symbols_order: list,
                                 samples_meta: pd.DataFrame,
                                 ensg_to_symbol: dict = None,
                                 ensg_col="ensembl_id",
                                 symbol_col="gene_symbol",
                                 desired_celltype_order=("PSC", "mesoderm", "cardiac mesoderm", "CM"),
                                 zscore=True):
    """
    Subsets the vst table obtained from DESeq to only show selected 
    markergenes and extend the dataframe by gene symbols.
    """
    df = vst_df.copy()

    # ensure symbol column exists
    df["gene_symbol"] = df["ensembl_id"].map(ensg_to_symbol)

    sample_cols = [c for c in df.columns if c not in {ensg_col, symbol_col}]
    # order the samples in the desired order
    ordered_samples, meta_sorted = order_samples_by_celltype(
        sample_cols, samples_meta,
        sample_col="Sample", cond_col="Celltype",
        desired=list(desired_celltype_order)
    )
    df = df.set_index(symbol_col)
    present = set(df.index)
    missing = [g for g in marker_symbols_order if g not in present]
    if missing:
        print(f"Warning: {len(missing)} marker symbols not found: {missing}")

    markers_present = [g for g in marker_symbols_order if g in present]
    mat = df.loc[markers_present, ordered_samples].astype(float)
    if zscore:
        mat = zscore_rows(mat)

    return mat, meta_sorted


def plot_marker_heatmap(mat: pd.DataFrame, meta_sorted: pd.DataFrame,
                        celltype_col="Celltype", figsize=(12, 10),
                        cmap="RdBu_r", center=0.0,
                        save_path=None):

    conds = meta_sorted.loc[mat.columns, celltype_col].tolist()
    unique = pd.Series(conds).unique()
    palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))
    col_colors = pd.Series(conds, index=mat.columns).map(palette)

    g = sns.clustermap(
        mat,
        row_cluster=False,
        col_cluster=False,
        col_colors=col_colors,
        cmap=cmap,
        center=center,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True
    )

    if save_path is not None:
        g.fig.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()
    return g