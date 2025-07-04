import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

context = {"gb": "gene body", "promoter": "promoter"}

mean_df = pd.read_csv(snakemake.input.mean, sep="\t")
mean_df["context"] = mean_df["index"].apply(lambda x: context[x.rpartition(" ")[-1]])
mean_df["mark"] = mean_df["index"].apply(lambda x: x.partition(" ")[0])
median_df = pd.read_csv(snakemake.input.median, sep="\t")
median_df["context"] = median_df["index"].apply(lambda x: context[x.rpartition(" ")[-1]])
median_df["mark"] = median_df["index"].apply(lambda x: x.partition(" ")[0])
sns.set(style="ticks", context="paper", font_scale=1.2)


mean_mean_df = mean_df.drop(["split", "metric", "subsample", "index"], axis=1).groupby(["algorithm", "target", "context", "mark"]).mean().reset_index()
median_median_df = median_df.drop(["split", "metric", "subsample", "index"], axis=1).groupby(["algorithm", "target", "context", "mark"]).median().reset_index()

ordered_mean_dfs = []
ordered_median_dfs = []


for c in ["promoter", "gene body"]:
    for a in ["deepshap", "kernelshap", "treeshap", "linearshap"]:    
        temp_df = mean_mean_df[(mean_mean_df["algorithm"] == a) & (mean_mean_df["context"] == c)].pivot(index="mark", columns="target", values="value")
        temp_df = temp_df.reindex(columns=["hyper responsive", "responsive", "moderately responsive", "non-responsive"])
        temp_df.columns = [f"{a} {c} {col}" for col in temp_df.columns]
        if a == "linearshap":
            pass
        else:
            temp_df[f"{a} {c} empty"] = np.nan  # Add empty column for alignment
        ordered_mean_dfs.append(temp_df)
        
        temp_df = median_median_df[(median_median_df["algorithm"] == a) & (median_median_df["context"] == c)].pivot(index="mark", columns="target", values="value")
        temp_df = temp_df.reindex(columns=["hyper responsive", "responsive", "moderately responsive", "non-responsive"])
        temp_df.columns = [f"{a} {c} {col}" for col in temp_df.columns]
        if a == "linearshap":
            pass
        else:
            temp_df[f"{a} {c} empty"] = np.nan  # Add empty column for alignment
        ordered_median_dfs.append(temp_df)

ordered_mean_df = pd.concat(ordered_mean_dfs, axis=1)
ordered_mean_promoter_df = ordered_mean_df.filter(like="promoter")
ordered_mean_gene_body_df = ordered_mean_df.filter(like="gene body")

ordered_median_df = pd.concat(ordered_median_dfs, axis=1)
ordered_median_promoter_df = ordered_median_df.filter(like="promoter")
ordered_median_gene_body_df = ordered_median_df.filter(like="gene body")

vmin_mean = ordered_mean_df.min().min()
vmax_mean = ordered_mean_df.max().max()
vmin_mean = 0
vmax_mean = 0.5
vmin_median = ordered_median_df.min().min()
vmax_median = ordered_median_df.max().max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 1.5 + (0.15 * ordered_mean_promoter_df.shape[0])))
sns.heatmap(ordered_mean_promoter_df, ax=ax, cmap="bone_r", vmin=0, vmax=vmax_mean, linewidths=1, cbar_kws={"shrink": 0.5, "location": "top", "label": "Promoter SHAP value"})
sns.heatmap(ordered_mean_gene_body_df, ax=bx, cmap="pink_r", vmin=0, vmax=vmax_mean, linewidths=1, cbar_kws={"shrink": 0.5, "location": "top", "label": "Gene Body SHAP value"})

ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=0)
bx.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, labelrotation=0)

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel("")
bx.set_xticks([])
bx.set_xticklabels([])
bx.set_ylabel("")
fig.tight_layout()
fig.savefig(snakemake.output.comparison_plot, bbox_inches='tight')
fig.savefig(snakemake.output.comparison_plot.replace('pdf', 'png'), dpi=300, bbox_inches='tight')
