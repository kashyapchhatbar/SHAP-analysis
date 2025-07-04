import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

context = {"gb": "gene body", "promoter": "promoter"}

mean_df = pd.read_csv(snakemake.input.promoter, sep="\t")
mean_df["context"] = mean_df["index"].apply(lambda x: context[x.rpartition(" ")[-1]])
mean_df["mark"] = mean_df["index"].apply(lambda x: x.partition(" ")[0])
median_df = pd.read_csv(snakemake.input.gb, sep="\t")
median_df["context"] = median_df["index"].apply(lambda x: context[x.rpartition(" ")[-1]])
median_df["mark"] = median_df["index"].apply(lambda x: x.partition(" ")[0])
sns.set(style="ticks", context="paper", font_scale=1)


mean_mean_df = mean_df.drop(["split", "metric", "subsample", "index"], axis=1).groupby(["algorithm", "target", "context", "mark"]).mean().reset_index()
median_median_df = median_df.drop(["split", "metric", "subsample", "index"], axis=1).groupby(["algorithm", "target", "context", "mark"]).mean().reset_index()

ordered_mean_dfs = []
ordered_median_dfs = []


for c in ["promoter", "gene body"]:
    for a in ["deepshap", "kernelshap", "treeshap", "linearshap"]:    
        temp_df = mean_mean_df[(mean_mean_df["algorithm"] == a) & (mean_mean_df["context"] == c)].pivot(index="mark", columns="target", values="value")
        # temp_df = temp_df.reindex(columns=["hyper responsive", "responsive", "moderately responsive", "non-responsive"])
        temp_df = temp_df.reindex(columns=["direct targets", "non-direct targets"])
        temp_df.columns = [f"{a} {c} {col}" for col in temp_df.columns]
        # if a == "linearshap":
        #     pass
        # else:
        #     temp_df[f"{a} {c} empty"] = np.nan  # Add empty column for alignment
        ordered_mean_dfs.append(temp_df)
        
        temp_df = median_median_df[(median_median_df["algorithm"] == a) & (median_median_df["context"] == c)].pivot(index="mark", columns="target", values="value")
        # temp_df = temp_df.reindex(columns=["hyper responsive", "responsive", "moderately responsive", "non-responsive"])
        temp_df = temp_df.reindex(columns=["direct targets", "non-direct targets"])
        temp_df.columns = [f"{a} {c} {col}" for col in temp_df.columns]
        # if a == "linearshap":
        #     pass
        # else:
        #     temp_df[f"{a} {c} empty"] = np.nan  # Add empty column for alignment
        ordered_median_dfs.append(temp_df)

ordered_mean_df = pd.concat(ordered_mean_dfs, axis=1)
ordered_mean_promoter_df = ordered_mean_df.filter(like="promoter").sort_values(by="deepshap promoter direct targets", ascending=True)
_ordered_mean_promoter_df = ordered_mean_promoter_df[[i for i in ordered_mean_promoter_df.columns if not i.startswith("linearshap")]]
ordered_mean_gene_body_df = ordered_mean_df.filter(like="gene body")

ordered_median_df = pd.concat(ordered_median_dfs, axis=1)
ordered_median_promoter_df = ordered_median_df.filter(like="promoter")
ordered_median_gene_body_df = ordered_median_df.filter(like="gene body").loc[list(ordered_mean_promoter_df.index)]
_ordered_median_gene_body_df = ordered_median_gene_body_df[[i for i in ordered_median_gene_body_df.columns if not i.startswith("linearshap")]]

vmin_mean = _ordered_mean_promoter_df.min().min()
vmax_mean = _ordered_mean_promoter_df.max().max()
vmin_median = ordered_median_gene_body_df.min().min()
vmax_median = ordered_median_gene_body_df.max().max()

if ordered_mean_promoter_df.shape[0] > 5:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 0.325 * ordered_mean_promoter_df.shape[0]-0.75))
else:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 0.325 * ordered_mean_promoter_df.shape[0]))
sns.heatmap(_ordered_mean_promoter_df, ax=ax, cmap="OrRd", vmin=vmin_mean, vmax=vmax_mean, linewidths=1, cbar_kws={"shrink": 0.5, "location": "top", "label": "Promoter SHAP value"}, annot=True, fmt=".2f", cbar=False)
sns.heatmap(_ordered_median_gene_body_df, ax=bx, cmap="OrRd", linewidths=1, cbar_kws={"shrink": 0.5, "location": "top", "label": "Gene Body SHAP value"}, annot=True, fmt=".2f", cbar=False, vmin=vmin_mean, vmax=vmax_mean)

ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=0)
bx.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, labelrotation=0)

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel("")
bx.set_xticks([])
bx.set_xticklabels([])
bx.set_ylabel("")
fig.tight_layout()
fig.savefig(snakemake.output.promoter_gb_heatmap, bbox_inches='tight')

nfeatures = mean_df["mark"].nunique()
order_features = ordered_mean_promoter_df.sort_values(by="deepshap promoter direct targets", ascending=True).index.tolist()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 0.8*nfeatures), sharey=True)
sns.barplot(data=mean_df[mean_df["context"] == "promoter"],
            y="mark",
            x="value",
            hue="target",
            estimator="mean",
            errorbar=("sd", 0.5),
            capsize=0.2, err_kws={'linewidth': 1},
            order=order_features,
            palette=reversed(sns.dark_palette("#3783bb", 2)),
            ax=ax)
sns.barplot(data=median_df[median_df["context"] == "gene body"],
            y="mark",
            x="value",
            hue="target",
            estimator="mean",
            errorbar=("sd", 0.5),
            capsize=0.2, err_kws={'linewidth': 1},
            palette=reversed(sns.dark_palette("#3783bb", 2)),
            ax=bx)

ax.set_title("Promoter SHAP values")
bx.set_title("Gene Body SHAP values")
ax.set_xlabel("|mean SHAP value|")
bx.set_xlabel("|mean SHAP value|")
ax.set_ylabel("Feature")
ax.legend(loc="upper right", fontsize=8)
bx.legend().set_visible(False)
# ax.legend().set_visible(False)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(snakemake.output.promoter_gb_barplot, bbox_inches='tight')

for algorithm in ["deepshap", "kernelshap", "treeshap", "linearshap"]:
    temp_mean_df = mean_df[mean_df["algorithm"] == algorithm]
    temp_median_df = median_df[median_df["algorithm"] == algorithm]
    nfeatures = temp_mean_df["mark"].nunique()
    order_features = ordered_mean_promoter_df.sort_values(by=f"{algorithm} promoter direct targets", ascending=True).index.tolist()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 0.3*nfeatures), sharey=True)
    sns.barplot(data=temp_mean_df[temp_mean_df["context"] == "promoter"],
                y="mark",
                x="value",
                hue="target",
                estimator="mean",
                errorbar=("sd", 0.5),
                capsize=0.2, err_kws={'linewidth': 1},
                order=order_features,
                palette=reversed(sns.dark_palette("#3783bb", 2)),
                ax=ax)
    sns.barplot(data=temp_median_df[temp_median_df["context"] == "gene body"],
                y="mark",
                x="value",
                hue="target",
                estimator="mean",
                errorbar=("sd", 0.5),
                capsize=0.2, err_kws={'linewidth': 1},
                palette=reversed(sns.dark_palette("#3783bb", 2)),
                ax=bx)
    ax.set_title(f"{algorithm} Promoter SHAP values")
    bx.set_title(f"{algorithm} Gene Body SHAP values")
    ax.set_xlabel("|mean SHAP value|")
    bx.set_xlabel("|mean SHAP value|")
    ax.set_ylabel("Feature")
    ax.legend(loc="upper right", fontsize=8)
    bx.legend().set_visible(False)
    # ax.legend().set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(snakemake.output[f"promoter_gb_{algorithm}_barplot"], bbox_inches='tight')