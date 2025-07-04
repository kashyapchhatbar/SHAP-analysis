import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1.2)

dfs, all_dfs = [], []

target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
# target_genes_df = target_genes_df[target_genes_df["padj"] < 0.05].copy()

input_value_dict = {
    "deepshap": snakemake.input.deepshap,
    "kernelshap": snakemake.input.kernelshap,
    "treeshap": snakemake.input.treeshap,
    "linearshap": snakemake.input.linearshap
}

def iterate_target_genes(target_genes_df, dfs, all_dfs, less_padj_threshold=0.05,
                         more_padj_threshold=0, label="SALL4 targets"):
    """Iterate over target genes and return a DataFrame with mean and median SHAP values."""
    target_genes_df = target_genes_df[
        (target_genes_df["padj"] < less_padj_threshold) &
        (target_genes_df["padj"] >= more_padj_threshold)
    ].copy()
    
    for shap_algorithm in ["deepshap", "kernelshap", "treeshap", "linearshap"]:
        for split in range(1, 6):
            if str(split) == "1":
                split_shap_values = input_value_dict[shap_algorithm]
            else:
                split_shap_values = input_value_dict[shap_algorithm].replace("/1/", f"/{split}/")
            df = pd.read_csv(split_shap_values, sep="\t", index_col=0)

            common_genes = list(set(df.index).intersection(set(target_genes_df.index)))

            for i in range(50):
                mean_df = df.loc[common_genes].sample(min(50, len(common_genes))).mean().abs().to_frame(name='value')
                median_df = df.loc[common_genes].sample(min(50, len(common_genes))).median().abs().to_frame(name='value')
                mean_df["split"] = split
                median_df["split"] = split
                mean_df["metric"] = "|mean|"
                median_df["metric"] = "|median|"
                mean_df["target"] = label
                median_df["target"] = label
                mean_df["algorithm"] = shap_algorithm
                median_df["algorithm"] = shap_algorithm
                mean_df["subsample"] = i
                median_df["subsample"] = i
                all_dfs.append(mean_df)
                all_dfs.append(median_df)
        
    return all_dfs

# for (lpadj, mpadj), label in zip([(0.05, 0), (0.5, 0.05), (0.9, 0.5), (1, 0.9)],
for (lpadj, mpadj), label in zip([(0.05, 0), (1, 0.05)],
    # ["hyper responsive", "responsive", "moderately responsive", "non-responsive"]):
    ["direct targets", "non-direct targets"]):
    # Iterate over different thresholds for padj values
    _all_dfs = iterate_target_genes(target_genes_df, dfs, all_dfs, less_padj_threshold=lpadj, 
        more_padj_threshold=mpadj, label=label)
    all_dfs += _all_dfs
    
to_plot_all_df = pd.concat(all_dfs, axis=0).reset_index()
mean_all_df = to_plot_all_df[to_plot_all_df["metric"] == "|mean|"]
median_all_df = to_plot_all_df[to_plot_all_df["metric"] == "|median|"]

mean_all_df.to_csv(snakemake.output.mean, sep="\t", index=False)
median_all_df.to_csv(snakemake.output.median, sep="\t", index=False)