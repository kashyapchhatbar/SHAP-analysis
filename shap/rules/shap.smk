rule all:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/MLP_{degron}_{mark}_{context}_shap_values.tsv",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule summary:
    input:
        expand("shap/models/{combination}/{metric}/MLP_{degron}_{mark}_{context}_all_shap_summary.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule deepshap_summary:
    input:
        expand("shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_all_deepshap_summary.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule kernelshap_summary:
    input:
        expand("shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_all_kernelshap_summary.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule kernelexplainer:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/MLP_{degron}_{mark}_{context}_shap_values.tsv",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule treeexplainer:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/xgboost_{degron}_{mark}_{context}_shap_values.tsv",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule treeexplainer_summary:
    input:
        expand("shap/models/{combination}/{metric}/xgboost_{degron}_{mark}_{context}_all_treeexplainer_summary.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule deepshap:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/torch_{degron}_{mark}_{context}_shap_values.tsv",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule kernelshap:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/torch_{degron}_{mark}_{context}_kernel_shap_values.tsv",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule linearexplainer:
    input:
        expand("shap/models/{combination}/{metric}/splits/{split}/linear_regression_{degron}_{mark}_{context}_shap_values.tsv",
            #combination=["one", "two", "three", "four", "five", "six", "seven"],
            combination=["four", "seven"],
            metric=["sum", "mean"],
            split=[1, 2, 3, 4, 5],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule linearexplainer_summary:
    input:
        expand("shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_{context}_all_shap_summary.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule get_shap_values:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/optuna.storage",
    params:
        padj_threshold=0.05,
        log2fc_threshold=0.25,
        kmeans=70,
    output:
        shap_values="shap/models/{combination}/{metric}/splits/{split}/MLP_{degron}_{mark}_{context}_shap_values.tsv",
    conda:
        "../envs/train.yml"
    script:
        "../scripts/get_shap_values.py"

rule get_treeexplainer_values:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/xgboost_optuna.storage",
    params:
        padj_threshold=0.05,
        log2fc_threshold=0.25,
        kmeans=70,
    output:
        shap_values="shap/models/{combination}/{metric}/splits/{split}/xgboost_{degron}_{mark}_{context}_shap_values.tsv",
    conda:
        "../envs/train.yml"
    script:
        "../scripts/xgboost_treeexplainer.py"

rule get_linearexplainer_values:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/linear_regression_optuna.storage",
    params:
        padj_threshold=0.05,
        log2fc_threshold=0.25,
        kmeans=70,
    output:
        shap_values="shap/models/{combination}/{metric}/splits/{split}/linear_regression_{degron}_{mark}_{context}_shap_values.tsv",
        linear_coefficients="shap/models/{combination}/{metric}/splits/{split}/linear_regression_{degron}_{mark}_{context}_coefficients.tsv",
    conda:
        "../envs/train.yml"
    script:
        "../scripts/linear_regression_explainer.py"

rule get_deep_shap_values:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val_scaled.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/pytorch_optuna.storage",
    output:
        shap_values="shap/models/{combination}/{metric}/splits/{split}/torch_{degron}_{mark}_{context}_shap_values.tsv",
    params:
        device=lambda wildcards: "cuda:3" if int(wildcards.split) == 5 else f"cuda:{int(wildcards.split)-1}",
        kmeans=100,
        padj_threshold=0.05,
    conda:
        "../envs/torch.yml"
    script:
        "../scripts/torch_deepshap.py"

rule get_kernel_shap_values:
    input:
        X_train="shap/models/{combination}/{metric}/splits/{split}/X_train_scaled.tsv",
        y_train="shap/models/{combination}/{metric}/splits/{split}/y_train_scaled.tsv",
        X_test="shap/models/{combination}/{metric}/splits/{split}/X_test_scaled.tsv",
        y_test="shap/models/{combination}/{metric}/splits/{split}/y_test_scaled.tsv",
        X_val="shap/models/{combination}/{metric}/splits/{split}/X_val_scaled.tsv",
        y_val="shap/models/{combination}/{metric}/splits/{split}/y_val_scaled.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
        optuna_storage="shap/models/{combination}/{metric}/splits/{split}/pytorch_optuna.storage",
    output:
        shap_values="shap/models/{combination}/{metric}/splits/{split}/torch_{degron}_{mark}_{context}_kernel_shap_values.tsv",
    params:
        device=lambda wildcards: "cuda:3" if int(wildcards.split) == 5 else f"cuda:{int(wildcards.split)-1}",
        kmeans=100,
        padj_threshold=0.05,
    conda:
        "../envs/torch.yml"
    script:
        "../scripts/torch_kernelshap.py"

rule summarise_deepshap_values:
    input:
        shap_values="shap/models/{combination}/{metric}/splits/1/torch_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv"
    output:
        summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_deepshap_summary.pdf",
        all_summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_all_deepshap_summary.pdf",
        circle_summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_circle_deepshap_summary.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/plot_shap_values.py"

rule summarise_kernelshap_values:
    input:
        shap_values="shap/models/{combination}/{metric}/splits/1/torch_{degron}_{mark}_{context}_kernel_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv"
    output:
        summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_kernelshap_summary.pdf",
        all_summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_all_kernelshap_summary.pdf",
        circle_summary="shap/models/{combination}/{metric}/torch_{degron}_{mark}_{context}_all_circle_kernelshap_summary.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/plot_shap_values.py"

rule summarise_treeexplainer_values:
    input:
        shap_values="shap/models/{combination}/{metric}/splits/1/xgboost_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv"
    output:
        summary="shap/models/{combination}/{metric}/xgboost_{degron}_{mark}_{context}_treeexplainer_summary.pdf",
        all_summary="shap/models/{combination}/{metric}/xgboost_{degron}_{mark}_{context}_all_treeexplainer_summary.pdf",
        circle_summary="shap/models/{combination}/{metric}/xgboost_{degron}_{mark}_{context}_all_circle_treeexplainer_summary.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/plot_shap_values.py"

rule summarise_linearexplainer_values:
    input:
        shap_values="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv"
    output:
        summary="shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_{context}_shap_summary.pdf",
        all_summary="shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_{context}_all_shap_summary.pdf",
        circle_summary="shap/models/{combination}/{metric}/linear_regression_{degron}_{mark}_{context}_all_circle_shap_summary.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/plot_shap_values.py"

rule summarise_shap_values:
    input:
        shap_values="shap/models/{combination}/{metric}/splits/1/MLP_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv"
    output:
        summary="shap/models/{combination}/{metric}/MLP_{degron}_{mark}_{context}_shap_summary.pdf",
        all_summary="shap/models/{combination}/{metric}/MLP_{degron}_{mark}_{context}_all_shap_summary.pdf",
        circle_summary="shap/models/{combination}/{metric}/MLP_{degron}_{mark}_{context}_all_circle_shap_summary.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/plot_shap_values.py"

rule summarise_comparison:
    input:
        expand("shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_{stat}.tsv.gz",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            stat=["mean", "median"],
            mark=["PolII"],
            context=["gb", "promoter"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule comparison_plot:
    input:
        expand("shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_plot.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            context=["gb", "promoter"],            
            degron=["set1ab_degron", "set1abzc3h4_degron", "zc3h4_degron", "dpy30_degron", "rbbp5_degron", "ints11_degron"]),
        expand("shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_plot.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule get_comparison_dataframe:
    input:
        deepshap="shap/models/{combination}/{metric}/splits/1/torch_{degron}_{mark}_{context}_shap_values.tsv",
        kernelshap="shap/models/{combination}/{metric}/splits/1/MLP_{degron}_{mark}_{context}_shap_values.tsv",
        treeshap="shap/models/{combination}/{metric}/splits/1/xgboost_{degron}_{mark}_{context}_shap_values.tsv",
        linearshap="shap/models/{combination}/{metric}/splits/1/linear_regression_{degron}_{mark}_{context}_shap_values.tsv",
        target_genes="shap/degron_data/{degron}.tsv",
    output:
        mean="shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_mean.tsv.gz",
        median="shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_median.tsv.gz"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/comparison_dataframe.py"

rule get_comparison_plot:
    input:
        mean="shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_mean.tsv.gz",
        median="shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_median.tsv.gz",
    output:
        comparison_plot="shap/models/{combination}/{metric}/{degron}_{mark}_{context}_comparison_plot.pdf"
    conda:
        "../envs/train.yml"
    script:
        "../scripts/comparison_plot.py"

rule promoter_gb_plot:
    input:
        expand("shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_plot.pdf",
            combination=["SET1A_ZC3H4_INTS11_integrated"],
            metric=["mean"],
            mark=["PolII"],            
            degron=['set1ab_degron', 'set1abzc3h4_degron', 'zc3h4_degron', 'dpy30_degron', 'rbbp5_degron', 'ints11_degron']),

rule get_promoter_gb_plot:
    input:
        promoter="shap/models/{combination}/{metric}/{degron}_{mark}_promoter_comparison_mean.tsv.gz",
        gb="shap/models/{combination}/{metric}/{degron}_{mark}_gb_comparison_median.tsv.gz",
    output:
        promoter_gb_heatmap="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_plot.pdf",
        promoter_gb_barplot="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_barplot.pdf",
        promoter_gb_deepshap_barplot="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_deepshap_barplot.pdf",
        promoter_gb_kernelshap_barplot="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_kernelshap_barplot.pdf",
        promoter_gb_treeshap_barplot="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_treeshap_barplot.pdf",
        promoter_gb_linearshap_barplot="shap/models/{combination}/{metric}/promoter_gb_{degron}_{mark}_linearshap_barplot.pdf",
    conda:
        "../envs/train.yml"
    script:
        "../scripts/promoter_gb_plot.py"