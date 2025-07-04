rule all:
    input:
        expand("shap/models/{combination}/mean/X.tsv", combination=combination),
        expand("shap/models/{combination}/mean/splits/1/X_train.tsv", combination=combination),
        
rule plot:
    input:
        expand("shap/models/{combination}/{metric}/combined_r2.pdf",
            combination=combination,
            metric=["mean", "sum"])

rule generate_tables:
    input:
        samples="shap/config/{combination}.tsv"
    output:
        sum_x="shap/models/{combination}/sum/X.tsv",
        sum_y="shap/models/{combination}/sum/y.tsv",
        mean_x="shap/models/{combination}/mean/X.tsv",
        mean_y="shap/models/{combination}/mean/y.tsv",
        untransformed_sum_x="shap/models/{combination}/sum/untransformed_X.tsv",
        untransformed_sum_y="shap/models/{combination}/sum/untransformed_y.tsv",
        untransformed_mean_x="shap/models/{combination}/mean/untransformed_X.tsv",
        untransformed_mean_y="shap/models/{combination}/mean/untransformed_y.tsv",
    params:
        standardise=False,
    conda:
        "../envs/data.yml"
    script:
        "../scripts/generalise_generate_tables.py"

rule generate_splits:
    input:
        sum_x="shap/models/{combination}/sum/X.tsv",
        sum_y="shap/models/{combination}/sum/y.tsv",
        mean_x="shap/models/{combination}/mean/X.tsv",
        mean_y="shap/models/{combination}/mean/y.tsv",
    output:
        expand("shap/models/{combination}/{metric}/splits/{split}/X_train.tsv", combination=["{combination}"],
               split=[1, 2, 3, 4, 5], metric=["sum", "mean"]),
    params:
        n_splits=5,
        pre_standardised=False
    conda:
        "../envs/data.yml"
    script:
        "../scripts/generate_splits.py"

rule get_model_accuracy:
    input:
        X_train="shap/models/generalise/{metric}/splits/1/X_train_scaled.tsv",
        y_train="shap/models/generalise/{metric}/splits/1/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/1/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/1/y_test_scaled.tsv",
        X_val="shap/models/generalise/{metric}/splits/1/X_val_scaled.tsv",
        y_val="shap/models/generalise/{metric}/splits/1/y_val_scaled.tsv",        
        mlp_optuna="shap/models/generalise/{metric}/splits/1/optuna.storage",
        pytorch_optuna="shap/models/generalise/{metric}/splits/1/pytorch_optuna.storage",
        xgboost_optuna="shap/models/generalise/{metric}/splits/1/xgboost_optuna.storage",
        linear_optuna="shap/models/generalise/{metric}/splits/1/linear_regression_optuna.storage",
    output:
        loss_plot="shap/models/{combination}/{metric}/pytorch_loss_accuracy_history.pdf",
        r2_plot="shap/models/{combination}/{metric}/combined_r2.pdf",
        r2_scores="shap/models/{combination}/{metric}/r2_scores.tsv",
    params:
        device="cuda:0",        
    conda:
        "../envs/torch.yml"
    script:
        "../scripts/plot_generalise_model_accuracy.py"