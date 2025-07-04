from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="paper", font_scale=1)

def get_conf_int(alpha, lr, X=None, y=None):
    
    """
    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """
    
    X_aux = X.copy()    
    X_aux.insert(0, 'const', 1)        
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - lr.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha/2, dof)        
    gap0 = t_val * np.sqrt(mse.values[0] * var_params[1:])
    gap1 = t_val * np.sqrt(mse.values[1] * var_params[1:])

    return pd.DataFrame({f'{mse.index[0]}': gap0, 
                         f'{mse.index[1]}': gap1})

def model_data(target_genes):
    
    # Load the data
    X = pd.read_csv(snakemake.input.X, sep="\t", index_col=0)
    y = pd.read_csv(snakemake.input.y, sep="\t", index_col=0)
    
    model = LinearRegression(
        fit_intercept=True,
        copy_X=True,        
    )
    
    # Filter the features to only include target genes
    common_target_genes = list(set(X.index).intersection(set(target_genes)).intersection(set(y.index)))
    X = X.loc[common_target_genes].copy()
    y = y.loc[common_target_genes].copy()

    model.fit(X, y)

    return model, X, y

def iterate_target_genes(target_genes_df, dfs, less_padj_threshold=0.05,
                         more_padj_threshold=0, label="SALL4 targets"):
    """Iterate over target genes and return a DataFrame with mean and median SHAP values."""
    target_genes_df = target_genes_df[
        (target_genes_df["padj"] < less_padj_threshold) &
        (target_genes_df["padj"] >= more_padj_threshold)
    ].copy()
    
    up_target_genes = target_genes_df[target_genes_df["log2FoldChange"] > 0].index.tolist()
    down_target_genes = target_genes_df[target_genes_df["log2FoldChange"] < 0].index.tolist()
    
    common_model, X, y = model_data(list(target_genes_df.index))
    up_model, X, y = model_data(up_target_genes)
    down_model, X, y = model_data(down_target_genes)
    
    common_coefficients = pd.DataFrame(common_model.coef_, index=y.columns,
        columns=X.columns).T
    common_gap = get_conf_int(0.05, common_model, X, y)
    common_gap.index = common_coefficients.index    
    common_gap_upper = common_gap + common_coefficients[common_gap.columns]
    common_gap_lower = common_coefficients[common_gap.columns] - common_gap
    common_coefficients = pd.concat([common_coefficients, common_gap_upper, common_gap_lower])
    up_coefficients = pd.DataFrame(up_model.coef_, index=y.columns,
        columns=X.columns).T
    down_coefficients = pd.DataFrame(down_model.coef_, index=y.columns,
        columns=X.columns).T
    common_coefficients["label"] = label
    up_coefficients["label"] = label + " up"
    down_coefficients["label"] = label + " down"
    
    dfs += [common_coefficients] #, up_coefficients, down_coefficients]
    
    return dfs

if __name__ == "__main__":
    
    dfs = []

    target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
    # target_genes_df = target_genes_df[target_genes_df["padj"] < 0.05].copy()

    # for (lpadj, mpadj), label in zip([(0.05, 0), (0.5, 0.05), (0.9, 0.5), (1, 0.9)],
    for (lpadj, mpadj), label in zip([(0.05, 0), (1, 0.05)],
        # ["hyper responsive", "responsive", "moderately responsive", "non-responsive"]):
        ["direct targets", "non-direct targets"]):
        # Iterate over different thresholds for padj values
        dfs += iterate_target_genes(target_genes_df, dfs, less_padj_threshold=lpadj, 
            more_padj_threshold=mpadj, label=label)
    
    df = pd.concat(dfs).reset_index()    
    df["mark"] = df["index"].apply(lambda x: x.partition(" ")[0])
    nfeatures = df["mark"].nunique()
    df["context"] = df["index"].apply(lambda x: x.rpartition(" ")[-1])
    promoter_ordered = df[["index", f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} promoter"]].groupby("index").mean().sort_values(by=f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} promoter").index.tolist()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 0.3*nfeatures), sharey=True)
    sns.barplot(data=df[df["context"] == "promoter"],
                y="mark",
                x=f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} promoter",
                errorbar=("pi", 100),
                hue="label",
                palette=reversed(sns.dark_palette("#3783bb", 2)),                
                estimator="median",
                capsize=0.2, err_kws={'linewidth': 1},
                ax=ax)
    sns.barplot(data=df[df["context"] == "gb"],
                y="mark",
                x=f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} gb",
                hue="label",
                palette=reversed(sns.dark_palette("#3783bb", 2)),
                errorbar=("pi", 100),
                capsize=0.2, err_kws={'linewidth': 1},
                estimator="median",
                ax=bx)

    ax.set_ylabel("")
    ax.legend(loc="best", fontsize=8)
    bx.legend().set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(snakemake.output.coefficients_plot, bbox_inches='tight')