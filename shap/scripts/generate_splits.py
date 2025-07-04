import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# Read the x and y dataframes
mean_x = pd.read_csv(snakemake.input.mean_x, sep="\t", index_col=0)
mean_y = pd.read_csv(snakemake.input.mean_y, sep="\t", index_col=0)
sum_x = pd.read_csv(snakemake.input.sum_x, sep="\t", index_col=0)
sum_y = pd.read_csv(snakemake.input.sum_y, sep="\t", index_col=0)

mean_common_genes = list(set(mean_x.index).intersection(mean_y.index))
sum_common_genes = list(set(sum_x.index).intersection(sum_y.index))

# Filter the dataframes to keep only common genes
mean_x = mean_x.loc[mean_common_genes]
mean_y = mean_y.loc[mean_common_genes]
sum_x = sum_x.loc[sum_common_genes]
sum_y = sum_y.loc[sum_common_genes]

kf = KFold(n_splits=snakemake.params.n_splits, shuffle=True, random_state=42)
for k, (train_index, test_index) in enumerate(kf.split(mean_x), start=1):
    mean_x.loc[mean_x.index[train_index]].to_csv(
        f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/X_train.tsv", sep="\t"
    )
    mean_y.loc[mean_y.index[train_index]].to_csv(
        f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/y_train.tsv", sep="\t"
    )
    
    mean_x_val, mean_x_test, mean_y_val, mean_y_test = train_test_split(
        mean_x.loc[mean_x.index[test_index]],
        mean_y.loc[mean_y.index[test_index]],
        test_size=0.5,
        random_state=42,
    )
    mean_x_val.to_csv(f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/X_val.tsv", sep="\t")
    mean_y_val.to_csv(f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/y_val.tsv", sep="\t")
    mean_x_test.to_csv(f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/X_test.tsv", sep="\t")
    mean_y_test.to_csv(f"shap/models/{snakemake.wildcards.combination}/mean/splits/{k}/y_test.tsv", sep="\t")
    # Save the training, validation, and test sets for each fold
    
for k, (train_index, test_index) in enumerate(kf.split(sum_x), start=1):
    sum_x.loc[sum_x.index[train_index]].to_csv(
        f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/X_train.tsv", sep="\t"
    )
    sum_y.loc[sum_y.index[train_index]].to_csv(
        f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/y_train.tsv", sep="\t"
    )
    
    sum_x_val, sum_x_test, sum_y_val, sum_y_test = train_test_split(
        sum_x.loc[sum_x.index[test_index]],
        sum_y.loc[sum_y.index[test_index]],
        test_size=0.5,
        random_state=42,
    )
    sum_x_val.to_csv(f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/X_val.tsv", sep="\t")
    sum_y_val.to_csv(f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/y_val.tsv", sep="\t")
    sum_x_test.to_csv(f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/X_test.tsv", sep="\t")
    sum_y_test.to_csv(f"shap/models/{snakemake.wildcards.combination}/sum/splits/{k}/y_test.tsv", sep="\t")
    # Save the training, validation, and test sets for each fold
    
    
if snakemake.params.pre_standardised:
    for split in range(1, snakemake.params.n_splits + 1):
        for metric in ["mean", "sum"]:
            for xy in ["X", "y"]:
                for ttv in ["train", "val", "test"]:
                    df = pd.read_csv(
                        f"shap/models/{snakemake.wildcards.combination}/{metric}/splits/{split}/{xy}_{ttv}.tsv",
                        sep="\t",
                        index_col=0,
                    )
                    df.to_csv(
                        f"shap/models/{snakemake.wildcards.combination}/{metric}/splits/{split}/{xy}_{ttv}_scaled.tsv",
                        sep="\t",
                    )
    # If the data is already standardized, just copy the files
    

else:
    # Standardize the data
    scaler = StandardScaler()
    for split in range(1, snakemake.params.n_splits + 1):
        for metric in ["mean", "sum"]:
            for xy in ["X", "y"]:
                for ttv in ["train", "val", "test"]:
                    df = pd.read_csv(
                        f"shap/models/{snakemake.wildcards.combination}/{metric}/splits/{split}/{xy}_{ttv}.tsv",
                        sep="\t",
                        index_col=0,
                    )
                    df_scaled = pd.DataFrame(
                        scaler.fit_transform(df),
                        index=df.index,
                        columns=df.columns,
                    )
                    df_scaled.to_csv(
                        f"shap/models/{snakemake.wildcards.combination}/{metric}/splits/{split}/{xy}_{ttv}_scaled.tsv",
                        sep="\t",
                    )
    # Save the scaled dataframes