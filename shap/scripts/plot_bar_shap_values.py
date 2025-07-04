import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1.2)

dfs, all_dfs = [], []

target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
# target_genes_df = target_genes_df[target_genes_df["padj"] < 0.05].copy()

all_dfs = []

for split in range(1, 6):
    if str(split) == "1":
        split_shap_values = snakemake.input.shap_values
    else:
        split_shap_values = snakemake.input.shap_values.replace("/1/", f"/{split}/")
    df = pd.read_csv(split_shap_values, sep="\t", index_col=0)

    mean_df = df.mean().to_frame(name='value')
    median_df = df.median().to_frame(name='value')
    mean_df["split"] = split
    median_df["split"] = split
    mean_df["metric"] = "mean"
    median_df["metric"] = "median"
    mean_df["target"] = "target"
    median_df["target"] = "target"            
    all_dfs.append(mean_df)
    all_dfs.append(median_df)

to_plot_all_df = pd.concat(all_dfs, axis=0).reset_index()
mean_all_df = to_plot_all_df[to_plot_all_df["metric"] == "mean"]
median_all_df = to_plot_all_df[to_plot_all_df["metric"] == "median"]

all_ordered_features = mean_all_df[["index", "value"]].groupby("index").mean()["value"].sort_values().index.tolist()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
sns.barplot(data=mean_all_df,
            y="index",
            x="value",
            hue="target",            
            order=all_ordered_features,
            palette=sns.color_palette("RdBu", 4),
            ax=ax)
sns.barplot(data=median_all_df,
            y="index",
            x="value",
            hue="target",            
            order=all_ordered_features,
            palette=sns.color_palette("RdBu", 4),
            ax=bx)

ax.set_title("Mean SHAP values")
bx.set_title("Median SHAP values")
ax.set_xlabel("SHAP value")
bx.set_xlabel("SHAP value")
ax.set_ylabel("Feature")
ax.legend(loc="upper right", fontsize=8)
bx.legend().set_visible(False)
# ax.legend().set_visible(False)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(snakemake.output.summary, bbox_inches='tight')