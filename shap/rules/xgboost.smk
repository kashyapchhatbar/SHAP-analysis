rule tune:
    input: 
        expand("shap/models/{combination}/{metric}/splits/{split}/best_trial.txt",
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