rule r2_summary:
    input:
        expand("shap/models/{combination}/{metric}/combined_r2.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean", "sum"])

rule get_model_accuracy:
    input:
        X_train="shap/models/{combination}/{metric}/splits/1/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/1/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/1/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/1/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/1/X_val_scaled.tsv",
        y_val="shap/models/{combination}/{metric}/splits/1/y_val_scaled.tsv",        
        mlp_optuna="shap/models/{combination}/{metric}/splits/1/optuna.storage",
        pytorch_optuna="shap/models/{combination}/{metric}/splits/1/pytorch_optuna.storage",
        xgboost_optuna="shap/models/{combination}/{metric}/splits/1/xgboost_optuna.storage",
        linear_optuna="shap/models/{combination}/{metric}/splits/1/linear_regression_optuna.storage",
    output:
        loss_plot="shap/models/{combination}/{metric}/pytorch_loss_accuracy_history.pdf",
        r2_plot="shap/models/{combination}/{metric}/combined_r2.pdf",
        r2_scores="shap/models/{combination}/{metric}/r2_scores.tsv",
    params:
        device="cuda:0",        
    conda:
        "../envs/torch.yml"
    script:
        "../scripts/plot_model_accuracy.py"