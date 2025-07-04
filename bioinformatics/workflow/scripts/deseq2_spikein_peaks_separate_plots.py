import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def filter(df, up=True):
    if up:
        return df[(df["padj"]<snakemake.params.padj) & (df["log2FoldChange"]>=snakemake.params.l2fc)]
    else:
        return df[(df["padj"]<snakemake.params.padj) & (df["log2FoldChange"]<=-snakemake.params.l2fc)]

def ax_plot(df, ax):

    df.plot(kind="scatter", x="baseMean", y="log2FoldChange", ax=ax, logx=True, 
            c="grey", edgecolor="white", linewidth=0.25, s=3)
    filter(df).plot(kind="scatter", x="baseMean", y="log2FoldChange", ax=ax, 
                    logx=True, c="red", edgecolor="white", linewidth=0.25, s=5)
    ax.text(0.8, 0.9, f"↑ {len(filter(df))}", transform=ax.transAxes,
            color="red")
    filter(df, up=False).plot(kind="scatter", x="baseMean", y="log2FoldChange", ax=ax, 
                    logx=True, c="red", edgecolor="white", linewidth=0.25, s=5)
    ax.text(0.8, 0.1, f"↓ {len(filter(df, up=False))}", transform=ax.transAxes,
            color="red")

    return ax

def summary_plot(one, two, three, four, five, six, seven, pdf):

    fig, axes = plt.subplots(2, 3, figsize=(12,8), sharex=True, sharey=True)

    for f, ax, l in zip([one, two, three, four, five, six], axes.flatten(),
        ["dTAG 0.5h C2 vs DMSO", "dTAG 2h C1 vs DMSO", "dTAG 2h C2 vs DMSO", "dTAG 2h C1_C2 vs DMSO", "dTAG 3h C2 vs DMSO", "dTAG 9h C2 vs DMSO", "WT vs DMSO C2"]):
        df = pd.read_csv(f, sep="\t", index_col=0)
        ax = ax_plot(df, ax)    
        ax.set_title(l)
    
    fig.tight_layout()
    fig.savefig(pdf)
            

def size_factor_plot(one, two, three, four, five, six, pdf):
    fig, axes = plt.subplots(6, 2, figsize=(8, 4.5), sharex=True, sharey=False)
    for f, ax, l, mouse in zip([one, one, two, two, three, three, four, four, five, five, six, six], axes.flatten(),
        ["dTAG 0.5h C2", "dTAG 0.5h C2", "dTAG 2h C1", "dTAG 2h C1", "dTAG 2h C2", "dTAG 2h C2",
         "dTAG 3h C2", "dTAG 3h C2", "dTAG 9h C2", "dTAG 9h C2", "WT C2", "WT"],
        [True, False, True, False, True, False, True, False, True, False, True,
         False, True, False, True, False]):
        filename = f.split("/")[-1]        
        if mouse:
            df = pd.read_csv(f.replace(filename, "reference_size_factors.tsv"), sep="\t", index_col=0)
        else:
            df = pd.read_csv(f.replace(filename, "size_factors.tsv"), sep="\t", index_col=0)
        
        design = pd.read_csv("config/samples.tsv", sep="\t", index_col=0)
        design["treatment"] = ["DMSO"]*3 + ["dTAG 0.5h C2"]*3 + ["dTAG 3h C2"]*3 + ["dTAG 9h C2"]*3 + ["WT"]*3 + ["DMSO C1"]*3 + ["dTAG 2h C1"]*3 + ["DMSO C2"]*3 + ["dTAG 2h C2"]*3
        
        df['genotype'] = design["treatment"]
        size_factor_name = df.columns[0]
        sns.pointplot(data=df, y="genotype", x=size_factor_name, 
            linestyle='none', errorbar="sd", color="grey", alpha=0.8, zorder=-10, ax=ax)
        plt.setp(ax.collections, alpha=.3) #for the markers
        plt.setp(ax.lines, alpha=.3)       #for the lines
        sns.stripplot(data=df, y="genotype", x=size_factor_name,  hue="genotype", legend=False,
                      palette=sns.color_palette(["black", "dodgerblue"]),
                    s=9, linewidth=1, edgecolor="white", ax=ax, zorder=20)            
        ax.set_xlabel(size_factor_name)
        ax.set_ylabel("")        
        if not mouse:
            ax.set_yticks([])
    
    fig.tight_layout()
    fig.savefig(pdf)
    
def get_difference(one, two):
    one_df = pd.read_csv(one, sep="\t", index_col=0)
    two_df = pd.read_csv(two, sep="\t", index_col=0)
    one_up, one_down = filter(one_df), filter(one_df, up=False)
    two_up, two_down = filter(two_df), filter(two_df, up=False)
    return one_up, one_down, two_up, two_down

def count_summary_plot(one, two, three, four, five, six, seven,
    non_one, non_two, non_three, non_four, non_five, non_six, non_seven, pdf):

    fig, axes = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)

    calibrated_dict = {}
    non_calibrated_dict = {}

    for pair, l in zip([get_difference(one, non_one), get_difference(two, non_two),
                            get_difference(three, non_three), get_difference(four, non_four),
                            get_difference(five, non_five), get_difference(six, non_six),
                            get_difference(seven, non_seven)],               
                            ["dTAG 0.5h C2", "dTAG 2h C1", "dTAG 2h C2", "dTAG 2h C1_C2", 
                             "dTAG 3h C2", "dTAG 9h C2", "WT C2"]):
        one_up, one_down, two_up, two_down = pair
        calibrated_dict[l] = {"up": len(one_up), "down": -len(one_down)}
        non_calibrated_dict[l] = {"up": len(two_up), "down": -len(two_down)}
    
    calibrated_df = pd.DataFrame(calibrated_dict).T
    non_calibrated_df = pd.DataFrame(non_calibrated_dict).T
    calibrated_df.plot(kind="barh", stacked=True, ax=axes[1])
    non_calibrated_df.plot(kind="barh", stacked=True, ax=axes[0])
    axes[0].set_xlim(-calibrated_df.abs().max().max()-5000, calibrated_df.abs().max().max()+5000)    
    axes[1].set_xlabel("Spike-in calibrated")
    axes[0].set_xlabel("Non calibrated")
    
    for ax in axes:
        ax.legend().set_visible(False)        
        ax.set_xticks([0])
        ax.set_xticklabels([])
        ax.grid(which="major", axis="x", linestyle="--", linewidth=0.5,
                color="grey", alpha=0.5)
        for container in ax.containers:
            ax.bar_label(container)
    
    fig.tight_layout()
    fig.savefig(pdf)

summary_plot(snakemake.input.lfc_non_dTAG_05,
             snakemake.input.lfc_non_T2C10_dTAG_2,
             snakemake.input.lfc_non_T1C2_dTAG_2,
             snakemake.input.lfc_non_dTAG_C1_C2,
             snakemake.input.lfc_non_dTAG_3, 
             snakemake.input.lfc_non_dTAG_9,
             snakemake.input.lfc_non_none,
             snakemake.output.non_calibrated)
summary_plot(snakemake.input.lfc_dTAG_05,
             snakemake.input.lfc_T2C10_dTAG_2,
             snakemake.input.lfc_T1C2_dTAG_2,
             snakemake.input.lfc_dTAG_C1_C2,
             snakemake.input.lfc_dTAG_3, 
             snakemake.input.lfc_dTAG_9,
             snakemake.input.lfc_none,
             snakemake.output.summary)
size_factor_plot(snakemake.input.lfc_dTAG_05,
             snakemake.input.lfc_T2C10_dTAG_2,
             snakemake.input.lfc_T1C2_dTAG_2,             
             snakemake.input.lfc_dTAG_3, 
             snakemake.input.lfc_dTAG_9,
             snakemake.input.lfc_none,
             snakemake.output.size_factors)
count_summary_plot(snakemake.input.lfc_dTAG_05,
             snakemake.input.lfc_T2C10_dTAG_2,
             snakemake.input.lfc_T1C2_dTAG_2,
             snakemake.input.lfc_dTAG_C1_C2,
             snakemake.input.lfc_dTAG_3, 
             snakemake.input.lfc_dTAG_9,
             snakemake.input.lfc_none,
             snakemake.input.lfc_non_dTAG_05,
             snakemake.input.lfc_non_T2C10_dTAG_2,
             snakemake.input.lfc_non_T1C2_dTAG_2,
             snakemake.input.lfc_non_dTAG_C1_C2,
             snakemake.input.lfc_non_dTAG_3, 
             snakemake.input.lfc_non_dTAG_9,
             snakemake.input.lfc_non_none,
             snakemake.output.up_down_genes)
