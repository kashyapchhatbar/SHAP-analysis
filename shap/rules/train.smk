rule tune:
    input: 
        expand("shap/models/{combination}/{metric}/splits/{split}/best_trial.txt",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5])

rule tune_gpu:
    input: 
        expand("shap/models/{combination}/{metric}/splits/{split}/model.pt",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5])

rule tune_xgboost:
    input: 
        expand("shap/models/{combination}/{metric}/splits/{split}/xgboost_best_trial.txt",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5])

rule tune_linear_regression:
    input: 
        expand("shap/models/{combination}/{metric}/splits/{split}/linear_regression_best_trial.txt",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5])

rule tune_params:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val.tsv"        
    output:
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/optuna.storage",
        best_trial="shap/models/{combination}/{metric}/splits/{split}/best_trial.txt",
    threads: 12
    params:
        n_trials=100
    conda:
        "../envs/train.yml"
    script:
        "../scripts/tune_params.py"

rule xgboost_params:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val.tsv"        
    output:
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/xgboost_optuna.storage",
        best_trial="shap/models/{combination}/{metric}/splits/{split}/xgboost_best_trial.txt",
    threads: 12
    params:
        n_trials=100,
    conda:
        "../envs/train.yml"
    script:
        "../scripts/xgboost_train.py"

rule torch_params:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val_scaled.tsv",        
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",        
    output:
        model="shap/models/{combination}/{metric}/splits/{split}/model.pt",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/pytorch_optuna.storage",
    params:
        device=lambda wildcards: "cuda:3" if int(wildcards.split) == 5 else f"cuda:{int(wildcards.split)-1}",
        n_trials=100,
    conda:
        "../envs/torch.yml"
    script:
        "../scripts/torch_train.py"

rule linear_regression:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val_scaled.tsv",        
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",  
    output:
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/linear_regression_optuna.storage",
        best_trial="shap/models/{combination}/{metric}/splits/{split}/linear_regression_best_trial.txt",
    params:
        n_trials=5,
    threads: 12,
    conda:
        "../envs/train.yml"
    script:
        "../scripts/linear_regression_params.py"