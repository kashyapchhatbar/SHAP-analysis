rule overlap:
    input:
        expand("shap/models/{combination}/{metric}/overlap/{degron}_{mark}_{context}_shap_overlap.tsv",
               combination=["SET1A_ZC3H4_INTS11_integrated"],
               mark=["PolII"],
                context=["promoter"],
               metric=["mean"],
               degron=["ints11_degron", "zc3h4_degron", "set1ab_degron"])

rule get_promoter_gb_plot:
    input:
        deepshap="shap/models/{combination}/{metric}/splits/1/torch_{degron}_{mark}_{context}_shap_values.tsv",
        kernelshap="shap/models/{combination}/{metric}/splits/1/MLP_{degron}_{mark}_{context}_shap_values.tsv",
        treeshap="shap/models/{combination}/{metric}/splits/1/xgboost_{degron}_{mark}_{context}_shap_values.tsv",
        linearshap="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
    params:
        important=["INTS11", "SET1A", "ZC3H4"],
    output:
        shap_overlap="shap/models/{combination}/{metric}/overlap/{degron}_{mark}_{context}_shap_overlap.tsv",
        X_overlap="shap/models/{combination}/{metric}/overlap/{degron}_{mark}_{context}_X_overlap.tsv",
        shap_corr="shap/models/{combination}/{metric}/overlap/{degron}_{mark}_{context}_shap_corr.tsv",
        X_corr="shap/models/{combination}/{metric}/overlap/{degron}_{mark}_{context}_X_corr.tsv"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/fraction_overlap.py"