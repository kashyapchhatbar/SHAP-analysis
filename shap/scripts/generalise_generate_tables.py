import pandas as pd
import numpy as np

combinations = {
    "generalise_MEL": "AFF4_BRD4_CDK9_H3K4me3_HEXIM1_RBBP5_DPY30_INTS11_NELFA_ZC3H4_SET1A_H2AK119ub1_H3K27me3_RING1B_SUZ12",
    "generalise_C12LX": "AFF4_BRD4_CDK9_H3K4me3_HEXIM1_RBBP5_DPY30_INTS11_NELFA_ZC3H4_SET1A_H2AK119ub1_H3K27me3_RING1B_SUZ12",
}

# y = "PolII_PolIIS5P_PolIIS2P"
y = "PolII"

samples = pd.read_csv(snakemake.input.samples, sep="\t", header=None)

sum_dfs, mean_dfs = [], []
y_sum_dfs, y_mean_dfs = [], []
for mark in combinations[snakemake.wildcards.combination].split("_"):
    # Iterate over each mark in the selected combination (split by "_")
    for gsm in samples[samples[1].str.startswith(f"{mark}_")][0].values:
        # For each sample GSM ID that starts with the current mark
        df = pd.read_csv(
            f"shap/bigwig_average_cut/{gsm}.bed",
            sep="\t",
            header=None,
            usecols=[0, 3, 5],
            names=["gene_id_scale", "sum", "mean"],
        )
        # Read the .bed file for the sample, selecting relevant columns and naming them

        df["gene_id"] = df["gene_id_scale"].apply(lambda x: x.split("_")[0])
        df["scale"] = df["gene_id_scale"].apply(lambda x: x.split("_")[1])
        # Split the 'gene_id_scale' column into 'gene_id' and 'scale'

        mean_df = df.pivot(index="gene_id", values="mean", columns="scale")
        sum_df = df.pivot(index="gene_id", values="sum", columns="scale")
        mean_df.index = [i.split(".")[0] for i in mean_df.index]
        sum_df.index = [i.split(".")[0] for i in sum_df.index]
        # Ensure the index is cleaned up by removing any ensembl IDs that may have a version suffix
        # Pivot the dataframe to get 'mean' and 'sum' values for each gene_id and scale

        mean_df.columns = [f"{mark} mean {i}" for i in mean_df.columns]
        sum_df.columns = [f"{mark} sum {i}" for i in sum_df.columns]
        # Rename columns to include the mark and value type

        mean_dfs.append(mean_df)
        sum_dfs.append(sum_df)
        # Append the resulting dataframes to the lists for later concatenation


for mark in y.split("_"):
    # For each mark in the y variable (split by "_")
    for gsm in samples[samples[1].str.startswith(f"{mark}_")][0].values:
        # For each sample GSM ID that starts with the current mark
        df = pd.read_csv(
            f"shap/bigwig_average_cut/{gsm}.bed",
            sep="\t",
            header=None,
            usecols=[0, 3, 5],
            names=["gene_id_scale", "sum", "mean"],
        )        
        # Read the .bed file for the sample, selecting relevant columns and naming them

        df["gene_id"] = df["gene_id_scale"].apply(lambda x: x.split("_")[0])
        df["scale"] = df["gene_id_scale"].apply(lambda x: x.split("_")[1])
        # Split the 'gene_id_scale' column into 'gene_id' and 'scale'

        mean_df = df.pivot(index="gene_id", values="mean", columns="scale")
        sum_df = df.pivot(index="gene_id", values="sum", columns="scale")
        mean_df.index = [i.split(".")[0] for i in mean_df.index]
        sum_df.index = [i.split(".")[0] for i in sum_df.index]
        # Ensure the index is cleaned up by removing any ensembl IDs that may have a version suffix
        # Pivot the dataframe to get 'mean' and 'sum' values for each gene_id and scale

        mean_df.columns = [f"{mark} mean {i}" for i in mean_df.columns]
        sum_df.columns = [f"{mark} sum {i}" for i in sum_df.columns]
        # Rename columns to include the mark and value type

        y_mean_dfs.append(mean_df)
        y_sum_dfs.append(sum_df)
        # Append the resulting dataframes to the lists for later concatenation

mean_df = (
    pd.concat(mean_dfs).groupby(level=0).mean().dropna().replace(0, 0.01).apply(np.log10)
)
# Concatenate all mean dataframes, group by gene_id, take the mean across samples,
# fill any missing values with 0.01, and replace zeros with 0.01 to avoid issues with downstream analysis.

sum_df = (
    pd.concat(sum_dfs).groupby(level=0).mean().dropna().replace(0, 0.01).apply(np.log10)
)

# Concatenate all sum dataframes, group by gene_id, mean across samples,
# fill any missing values with 0.01, and replace zeros with 0.01 for consistency.

y_mean_df = (
    pd.concat(y_mean_dfs).groupby(level=0).mean().dropna().replace(0, 0.01).apply(np.log10)
)

y_sum_df = (
    pd.concat(y_sum_dfs).groupby(level=0).mean().dropna().replace(0, 0.01).apply(np.log10)
)

if snakemake.params.standardise:
    # If standardisation is requested, standardise the dataframes
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    mean_df.to_csv(snakemake.output.untransformed_mean_x, sep="\t", header=True, index=True)
    mean_df = pd.DataFrame(scaler.fit_transform(mean_df), index=mean_df.index, columns=mean_df.columns)
    mean_df.to_csv(snakemake.output.mean_x, sep="\t", header=True, index=True)
    
    sum_df.to_csv(snakemake.output.untransformed_sum_x, sep="\t", header=True, index=True)
    sum_df = pd.DataFrame(scaler.fit_transform(sum_df), index=sum_df.index, columns=sum_df.columns)
    sum_df.to_csv(snakemake.output.sum_x, sep="\t", header=True, index=True)
    
    y_mean_df.to_csv(snakemake.output.untransformed_mean_y, sep="\t", header=True, index=True)
    y_mean_df = pd.DataFrame(scaler.fit_transform(y_mean_df), index=y_mean_df.index, columns=y_mean_df.columns)
    y_mean_df.to_csv(snakemake.output.mean_y, sep="\t", header=True, index=True)
    
    y_sum_df.to_csv(snakemake.output.untransformed_sum_y, sep="\t", header=True, index=True)
    y_sum_df = pd.DataFrame(scaler.fit_transform(y_sum_df), index=y_sum_df.index, columns=y_sum_df.columns)
    y_sum_df.to_csv(snakemake.output.sum_y, sep="\t", header=True, index=True)
    # Standardise the dataframes using sklearn's StandardScaler

else:
    # If standardisation is not requested, save the dataframes as they are
    mean_df.to_csv(snakemake.output.untransformed_mean_x, sep="\t", header=True, index=True)
    sum_df.to_csv(snakemake.output.untransformed_sum_x, sep="\t", header=True, index=True)
    y_mean_df.to_csv(snakemake.output.untransformed_mean_y, sep="\t", header=True, index=True)
    y_sum_df.to_csv(snakemake.output.untransformed_sum_y, sep="\t", header=True, index=True)
    
    # Save the final dataframes to the specified output files
    mean_df.to_csv(snakemake.output.mean_x, sep="\t", header=True, index=True)
    sum_df.to_csv(snakemake.output.sum_x, sep="\t", header=True, index=True)
    y_mean_df.to_csv(snakemake.output.mean_y, sep="\t", header=True, index=True)
    y_sum_df.to_csv(snakemake.output.sum_y, sep="\t", header=True, index=True)