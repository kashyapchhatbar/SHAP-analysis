rule all:
    input:
        expand("shap/models/{combination}/mean/X.tsv", combination=combination),
        expand("shap/models/{combination}/mean/splits/1/X_train.tsv", combination=combination),

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
        "../scripts/generate_tables.py"

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

