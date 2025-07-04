from sklearn.linear_model import LinearRegression
import numpy as np
import optuna.storages.journal
import pandas as pd
from sklearn.metrics import r2_score
import optuna
import sys
import shap
import sys

def find_best_params(optuna_storage):
    # Create optuna storage
    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(optuna_storage),
        )
    except FileNotFoundError:
        print(f"File not found: {optuna_storage}")
        print(f"Model may not be tuned yet...")
        sys.exit(1)

    study_name = optuna.study.get_all_study_names(storage=storage)[0]    

    # Load the study
    study = optuna.load_study(storage=storage, study_name=study_name)

    best_params = study.best_params

    return best_params

def model_data(best_params):
    
    # Load the data
    X_train = pd.read_csv(snakemake.input.X_train, sep="\t", index_col=0)
    y_train = pd.read_csv(snakemake.input.y_train, sep="\t", index_col=0)
    y_test = pd.read_csv(snakemake.input.y_test, sep="\t", index_col=0)
    X_test = pd.read_csv(snakemake.input.X_test, sep="\t", index_col=0)
    X_val = pd.read_csv(snakemake.input.X_val, sep="\t", index_col=0)
    
    model = LinearRegression(
        fit_intercept=best_params["fit_intercept"],
        copy_X=best_params["copy_X"],
        positive=best_params["positive"],
    )
    
    X_combined = pd.concat([X_train, X_val, X_test])

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)
    r2_scores = []
    for col in y_test.columns:
        r2 = r2_score(y_test[col], y_pred[col])
        r2_scores.append(r2)
    # Calculate the mean R2 score across all columns     
    print(f"R2 test score: {np.mean(r2_scores):.4f}±{np.std(r2_scores):.4f}")

    return model, X_train, X_test, X_combined, y_test

if __name__ == "__main__":
    
    target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
    sig_target_genes_df = target_genes_df[target_genes_df["padj"] < snakemake.params.padj_threshold].copy()
    # target_genes_df = target_genes_df[target_genes_df["padj"] < snakemake.params.padj_threshold].copy()
    # target_genes_df = target_genes_df[target_genes_df["log2FoldChange"] > snakemake.params.log2fc_threshold].copy()
    target_genes = target_genes_df.index.tolist()
    
    model, X_train, X_test, X_combined, y_test = model_data(find_best_params(snakemake.input.optuna_storage))
    
    coefficients = pd.DataFrame(model.coef_, index=y_test.columns,
                                columns=X_combined.columns).T
    coefficients.to_csv(snakemake.output.linear_coefficients, sep="\t", header=True)

    # common_genes = list(set(X_combined.index).intersection(target_genes))
    # ensure that the target genes are in the training data
    common_genes = list(set(X_test.index).intersection(target_genes))
    X_target_genes = X_combined.loc[common_genes].copy()
    
    # Use SHAP to explain the model predictions
    train_target_genes = list(set(X_train.index).intersection(sig_target_genes_df.index))
    X_train_without_target_genes = X_train.drop(train_target_genes, axis=0, errors='ignore').copy()
    X_train_summary = shap.kmeans(X_train_without_target_genes, snakemake.params.kmeans)
    explainer = shap.LinearExplainer(model, X_train_summary.data)
    
    y_test = pd.read_csv(snakemake.input.y_test, sep="\t", index_col=0)
    mark_column = f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} {snakemake.wildcards.context}"
    try:
        mark_id = y_test.columns.get_loc(mark_column)
    except KeyError:
        print(f"Column '{mark_column}' not found in the data. Available columns: {y_test.columns.tolist()}")
        sys.exit(1)
    
    shap_values = pd.DataFrame(
        np.array(explainer.shap_values(np.array(X_target_genes)))[:, :, mark_id],
        columns=X_target_genes.columns,
        index=X_target_genes.index
    )    

    # Save the SHAP values    
    shap_values.to_csv(snakemake.output.shap_values, sep="\t")