import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
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

context = {"gene body": "gb", "promoter": "promoter"}
degron_dict = {'zc3h4': 'ZC3H4', 'ints11': 'INTS11', 'set1ab': 'SET1A'}

target_genes_df = target_genes_df[target_genes_df["padj"]<0.05].copy()

overlap_shap_df = []
overlap_X_df = []
overlap_shap_corr = []
overlap_X_corr = []

for shap_algorithm in tqdm(["deepshap", "kernelshap", "treeshap", "linearshap"], position=0, desc="SHAP algorithms"):
    for split in tqdm(range(1, 6), position=1, desc="Splits", leave=False):
        if str(split) == "1":
            split_shap_values = input_value_dict[shap_algorithm]
        else:
            split_shap_values = input_value_dict[shap_algorithm].replace("/1/", f"/{split}/")
        df = pd.read_csv(split_shap_values, sep="\t", index_col=0)        
        to_replace = split_shap_values.split("/")[-1]
                    
        common_genes = list(set(df.index).intersection(set(target_genes_df.index)))
        _target_genes_df = target_genes_df.loc[common_genes].copy()
        X = pd.read_csv(split_shap_values.replace(to_replace, "X_test.tsv"), sep="\t", index_col=0)
        X["fc"] = target_genes_df["log2FoldChange"]
        shap_values = [f"{i} {snakemake.wildcards.metric} {context[snakemake.wildcards.context]}" for i in snakemake.params.important]
        df["sum shap"] = df[shap_values].sum(axis=1)
        df["fc"] = target_genes_df["log2FoldChange"]
        
        corr_df = df[shap_values + ["fc"]].loc[common_genes].corr()[["fc"]].drop("fc").reset_index()
        corr_X = X[shap_values + ["fc"]].loc[common_genes].corr()[["fc"]].drop("fc").reset_index()
        corr_df["shap algorithm"] = shap_algorithm
        corr_X["shap algorithm"] = shap_algorithm
        corr_df["split"] = split
        corr_X["split"] = split
        overlap_shap_corr.append(corr_df)
        overlap_X_corr.append(corr_X)
        
        for i in range(len(common_genes)):
            overlap_shap_df.append([shap_algorithm, split, i+1, len(set(df.sort_values(by="sum shap", ascending=False).head(i+1).index).intersection(_target_genes_df.index))])
            overlap_X_df.append([shap_algorithm, split, i+1, len(set(X.sort_values(by=f"{degron_dict[snakemake.wildcards.degron.split('_')[0]]} mean {snakemake.wildcards.context}", ascending=False).head(i+1).index).intersection(_target_genes_df.index))])
        
overlap_shap_df = pd.DataFrame(overlap_shap_df, columns=["shap algorithm", "split", "i", "overlap"])
overlap_X_df = pd.DataFrame(overlap_X_df, columns=["shap algorithm", "split", "i", "overlap"])
overlap_shap_corr = pd.concat(overlap_shap_corr, axis=0)
overlap_X_corr = pd.concat(overlap_X_corr, axis=0)

overlap_shap_df.to_csv(snakemake.output.shap_overlap, sep="\t", index=False)
overlap_X_df.to_csv(snakemake.output.X_overlap, sep="\t", index=False)
overlap_shap_corr.to_csv(snakemake.output.shap_corr, sep="\t", index=False)
overlap_X_corr.to_csv(snakemake.output.X_corr, sep="\t", index=False)