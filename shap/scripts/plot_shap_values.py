import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1.2)

dfs, all_dfs = [], []

target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
# target_genes_df = target_genes_df[target_genes_df["padj"] < 0.05].copy()

def iterate_target_genes(target_genes_df, dfs, all_dfs, less_padj_threshold=0.05,
                         more_padj_threshold=0, label="SALL4 targets"):
    """Iterate over target genes and return a DataFrame with mean and median SHAP values."""
    target_genes_df = target_genes_df[
        (target_genes_df["padj"] < less_padj_threshold) &
        (target_genes_df["padj"] >= more_padj_threshold)
    ].copy()
    
    up_target_genes = target_genes_df[target_genes_df["log2FoldChange"] > 0].index.tolist()
    down_target_genes = target_genes_df[target_genes_df["log2FoldChange"] < 0].index.tolist()
    
    for split in range(1, 6):
        if str(split) == "1":
            split_shap_values = snakemake.input.shap_values
        else:
            split_shap_values = snakemake.input.shap_values.replace("/1/", f"/{split}/")
        df = pd.read_csv(split_shap_values, sep="\t", index_col=0)

        up_common_genes = list(set(df.index).intersection(set(up_target_genes)))
        down_common_genes = list(set(df.index).intersection(set(down_target_genes)))
        common_genes = list(set(df.index).intersection(set(target_genes_df.index)))        

        for i in range(50):            
            up_mean_df = df.loc[up_common_genes].sample(min(100, len(up_common_genes))).mean().abs().to_frame(name='value')
            down_mean_df = df.loc[down_common_genes].sample(min(100, len(down_common_genes))).mean().abs().to_frame(name='value')
            up_median_df = df.loc[up_common_genes].sample(min(100, len(up_common_genes))).median().abs().to_frame(name='value')
            down_median_df = df.loc[down_common_genes].sample(min(100, len(down_common_genes))).median().abs().to_frame(name='value')
            mean_df = df.loc[common_genes].sample(min(100, len(common_genes))).mean().abs().to_frame(name='value')
            median_df = df.loc[common_genes].sample(min(100, len(common_genes))).median().abs().to_frame(name='value')
            mean_df["split"] = split
            median_df["split"] = split
            mean_df["metric"] = "|mean|"
            median_df["metric"] = "|median|"
            mean_df["target"] = label
            median_df["target"] = label            
            up_mean_df["split"] = split
            down_mean_df["split"] = split
            up_median_df["split"] = split
            down_median_df["split"] = split
            up_mean_df["metric"] = "|mean|"
            down_mean_df["metric"] = "|mean|"
            up_median_df["metric"] = "|median|"
            down_median_df["metric"] = "|median|"
            up_mean_df["target"] = f"{label} (up-regulated)"
            down_mean_df["target"] = f"{label} (down-regulated)"
            up_median_df["target"] = f"{label} (up-regulated)"
            down_median_df["target"] = f"{label} (down-regulated)"
            dfs.append(up_mean_df)
            dfs.append(down_mean_df)
            dfs.append(up_median_df)
            dfs.append(down_median_df)
            all_dfs.append(mean_df)
            all_dfs.append(median_df)
    
    return dfs, all_dfs

for (lpadj, mpadj), label in zip([(0.05, 0), (0.5, 0.05), (0.9, 0.5), (1, 0.9)],
    ["hyper responsive", "responsive", "moderately responsive", "non-responsive"]):
    # Iterate over different thresholds for padj values
    _dfs, _all_dfs = iterate_target_genes(target_genes_df, dfs, all_dfs, less_padj_threshold=lpadj, 
        more_padj_threshold=mpadj, label=label)
    dfs += _dfs
    all_dfs += _all_dfs
    
to_plot_df = pd.concat(dfs, axis=0).reset_index()
to_plot_all_df = pd.concat(all_dfs, axis=0).reset_index()
mean_all_df = to_plot_all_df[to_plot_all_df["metric"] == "|mean|"]
median_all_df = to_plot_all_df[to_plot_all_df["metric"] == "|median|"]
mean_df = to_plot_df[to_plot_df["metric"] == "|mean|"]
median_df = to_plot_df[to_plot_df["metric"] == "|median|"]

ordered_features = mean_df[mean_df["target"] == "hyper responsive (up-regulated)"][["index", "value"]].groupby("index").mean()["value"].sort_values().index.tolist()
all_ordered_features = mean_all_df[mean_all_df["target"] == "hyper responsive"][["index", "value"]].groupby("index").mean()["value"].sort_values().index.tolist()

mean_circle_df = mean_all_df[["index", "target", "value"]].groupby(["index", "target"]).mean().reset_index()
median_circle_df = median_all_df[["index", "target", "value"]].groupby(["index", "target"]).median().reset_index()
min_mean_value = mean_circle_df["value"].min()
max_mean_value = mean_circle_df["value"].max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(5, max(6, mean_circle_df["index"].nunique()*0.35)), sharey=True)
sns.heatmap(data=mean_circle_df.pivot(values="value", index="index", columns="target")[["hyper responsive", "responsive", "moderately responsive", "non-responsive"]].sort_values(by="hyper responsive"), cmap="bone_r", linewidths=0.5, linecolor='black', ax=ax, cbar=False, vmin=min_mean_value, vmax=max_mean_value)
sns.heatmap(data=median_circle_df.pivot(values="value", index="index", columns="target")[["hyper responsive", "responsive", "moderately responsive", "non-responsive"]].sort_values(by="hyper responsive"), cmap="bone_r", linewidths=0.5, linecolor='black', ax=bx, vmin=min_mean_value, vmax=max_mean_value)
ax.set_title("|Mean| SHAP")
bx.set_title("|Median| SHAP")
ax.set_ylabel("")
ax.set_xlabel("")
bx.set_xlabel("")
bx.set_ylabel("")
fig.tight_layout()
fig.savefig(snakemake.output.circle_summary, bbox_inches='tight')


fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
sns.pointplot(
    data=mean_df,
    y="index",
    x="value",
    hue="target",
    dodge=True,
    marker="o",
    markersize=4,
    estimator="mean",
    err_kws={'linewidth': 1},
    order=ordered_features,
    errorbar=("sd", 0.5),
    hue_order=["hyper responsive (up-regulated)", "responsive (up-regulated)",
               "moderately responsive (up-regulated)", "non-responsive (up-regulated)",
               "non-responsive (down-regulated)", "moderately responsive (down-regulated)",
               "responsive (down-regulated)", "hyper responsive (down-regulated)"],
    linestyles="none",
    palette=sns.color_palette("RdBu", 8),
    ax=ax)

sns.pointplot(
    data=median_df,
    y="index",
    x="value",
    hue="target",
    dodge=True,
    marker="o",
    palette=sns.color_palette("RdBu", 8),
    order=ordered_features,
    markersize=4,
    estimator="median",
    hue_order=["hyper responsive (up-regulated)", "responsive (up-regulated)",
               "moderately responsive (up-regulated)", "non-responsive (up-regulated)",
               "non-responsive (down-regulated)", "moderately responsive (down-regulated)",
               "responsive (down-regulated)", "hyper responsive (down-regulated)"],
    linestyles="none",
    errorbar=("pi", 25),
    err_kws={'linewidth': 1},
    ax=bx)

ax.set_title("|Mean| SHAP values")
bx.set_title("|Median| SHAP values")
ax.set_xlabel("|SHAP value|")
bx.set_xlabel("|SHAP value|")
ax.set_ylabel("Feature")
bx.legend(loc="upper right")
# bx.legend().set_visible(False)
ax.legend().set_visible(False)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(snakemake.output.summary, bbox_inches='tight')

def hide_stripplot_pointplot():    
    sns.stripplot(
        data=mean_all_df,
        y="index",
        x="value",
        hue="target",
        dodge=True,
        marker="o",
        size=2,    
        order=all_ordered_features,
        alpha=0.25,    
        hue_order=["hyper responsive", "responsive", "moderately responsive", "non-responsive"],
        zorder=-10,
        legend=False,
        palette=reversed(sns.dark_palette("#3783bb", 4)),
        ax=ax)
    sns.pointplot(
        data=mean_all_df,
        y="index",
        x="value",
        hue="target",
        dodge=True,
        marker="_",
        markeredgewidth=1,
        estimator="mean",
        errorbar=("sd", 0.5),
        zorder=10,
        markersize=4,
        err_kws={'linewidth': 1},
        order=all_ordered_features,
        hue_order=["hyper responsive", "responsive", "moderately responsive", "non-responsive"],
        linestyles="none",
        palette=reversed(sns.dark_palette("#3783bb", 4)),
        ax=ax)
    sns.stripplot(
        data=median_all_df,
        y="index",
        x="value",
        hue="target",
        dodge=True,
        marker="o",        
        size=2,    
        order=all_ordered_features,
        alpha=0.25,    
        hue_order=["hyper responsive", "responsive", "moderately responsive", "non-responsive"],
        zorder=-10,
        legend=False,
        palette=reversed(sns.dark_palette("#3783bb", 4)),
        ax=bx)
    sns.pointplot(
        data=median_all_df,
        y="index",
        x="value",
        hue="target",
        dodge=True,
        estimator="median",
        errorbar=("pi", 25),
        marker="_",
        markeredgewidth=1,
        zorder=10,
        palette=reversed(sns.dark_palette("#3783bb", 4)),
        order=all_ordered_features,
        markersize=4,
        hue_order=["hyper responsive", "responsive", "moderately responsive", "non-responsive"],
        linestyles="none",
        err_kws={'linewidth': 1},
        ax=bx)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
sns.barplot(data=mean_all_df,
            y="index",
            x="value",
            hue="target",
            estimator="mean",
            errorbar=("sd", 0.5),
            order=all_ordered_features,
            palette=reversed(sns.dark_palette("#3783bb", 4)),
            ax=ax)
sns.barplot(data=median_all_df,
            y="index",
            x="value",
            hue="target",
            estimator="median",
            errorbar=("pi", 25),
            order=all_ordered_features,
            palette=reversed(sns.dark_palette("#3783bb", 4)),
            ax=bx)

ax.set_title("|Mean| SHAP values")
bx.set_title("|Median| SHAP values")
ax.set_xlabel("|SHAP value|")
bx.set_xlabel("|SHAP value|")
ax.set_ylabel("Feature")
ax.legend(loc="upper right", fontsize=8)
bx.legend().set_visible(False)
# ax.legend().set_visible(False)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
bx.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(snakemake.output.all_summary, bbox_inches='tight')