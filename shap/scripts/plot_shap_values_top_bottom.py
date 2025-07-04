import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1.2)

dfs, all_dfs = [], []

up_shap_genes = df.sort_values(by=f"{snakemake.params.degron.split('_')[0].upper()} sum gb", ascending=False).head(len(common_genes)).index.tolist()
down_shap_genes = df.sort_values(by=f"{snakemake.params.degron.split('_')[0].upper()} sum gb", ascending=True).head(len(common_genes)).index.tolist()

for i in range(50):            
    up_shap_mean_df = df.loc[up_shap_genes].sample(min(100, len(up_shap_genes))).mean().abs().to_frame(name='value')
    down_shap_mean_df = df.loc[down_shap_genes].sample(min(100, len(down_shap_genes))).mean().abs().to_frame(name='value')
    up_shap_median_df = df.loc[up_shap_genes].sample(min(100, len(up_shap_genes))).median().abs().to_frame(name='value')
    down_shap_median_df = df.loc[down_shap_genes].sample(min(100, len(down_shap_genes))).median().abs().to_frame(name='value')
    
    up_shap_mean_df["split"] = split
    down_shap_mean_df["split"] = split
    up_shap_median_df["split"] = split
    down_shap_median_df["split"] = split
    up_shap_mean_df["metric"] = "|mean|"
    down_shap_mean_df["metric"] = "|mean|"
    up_shap_median_df["metric"] = "|median|"
    down_shap_median_df["metric"] = "|median|"
    up_shap_mean_df["target"] = f"SHAP top"
    down_shap_mean_df["target"] = f"SHAP bottom"
    up_shap_median_df["target"] = f"SHAP top"
    down_shap_median_df["target"] = f"SHAP bottom"


    dfs.append(up_shap_mean_df)
    dfs.append(down_shap_mean_df)
    dfs.append(up_shap_median_df)
    dfs.append(down_shap_median_df)
            

    
to_plot_df = pd.concat(dfs, axis=0).reset_index()
mean_df = to_plot_df[to_plot_df["metric"] == "|mean|"]
median_df = to_plot_df[to_plot_df["metric"] == "|median|"]

ordered_features = mean_df[mean_df["target"] == "SHAP top"][["index", "value"]].groupby("index").mean()["value"].sort_values().index.tolist()

min_mean_value = mean_circle_df["value"].min()
max_mean_value = mean_circle_df["value"].max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(5, max(6, mean_circle_df["index"].nunique()*0.35)), sharey=True)
sns.heatmap(data=mean_circle_df.pivot(values="value", index="index", columns="target")[["SHAP top", "SHAP bottom"]].sort_values(by="SHAP top"), cmap="bone_r", linewidths=0.5, linecolor='black', ax=ax, cbar=False, vmin=min_mean_value, vmax=max_mean_value)
sns.heatmap(data=median_circle_df.pivot(values="value", index="index", columns="target")[["SHAP top", "SHAP bottom"]].sort_values(by="SHAP top"), cmap="bone_r", linewidths=0.5, linecolor='black', ax=bx, vmin=min_mean_value, vmax=max_mean_value)
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