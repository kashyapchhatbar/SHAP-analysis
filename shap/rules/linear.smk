rule linear_coefficients:
    input:
        expand("shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_coefficients.pdf", 
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean", "sum"], 
            mark=["PolII"],
            degron=["ints11_degron"])


rule get_linear_coefficients:
    input:
        X="shap/models/{combination}/{metric}/X.tsv",
        y="shap/models/{combination}/{metric}/y.tsv",        
        target_genes="shap/degron_data/{degron}.tsv",
    params:
        padj_threshold=0.05,
        log2fc_threshold=0.25,
        kmeans=70,
    output:
        # linear_coefficients="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_coefficients.tsv",
        # linear_coefficients_up="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_up_coefficients.tsv",
        # linear_coefficients_down="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_down_coefficients.tsv",
        coefficients_plot="shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_coefficients.pdf",
    conda:
        "../envs/linear.yml"
    script:
        "../scripts/linear_regression_coefficients.py"