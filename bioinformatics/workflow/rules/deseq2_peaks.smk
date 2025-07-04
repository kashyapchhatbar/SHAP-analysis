import numpy as np

rule deseq2_peaks:
    input:
        expand("results/deseq2_peaks/{feature}/{dtag}/{spike}/calibrated_dds.Rds", 
        dtag=["dTAG_05", "dTAG_3", "dTAG_9", "none", "T1C2", "T2C10", "dTAG_C1_C2"], feature=["intergenic"], spike=["spikein", "extra_spikein"]),

rule generate_count_table_peaks:
    input:
        "results/cutnrun/SALL4_CnR_SG_intergenic_peaks_SAF_counts.txt"
    output:
        "results/cutnrun/reference.intergenic.txt"
    run:
        import pandas as pd
        df = pd.read_csv(input[0], sep="\t", comment="#", index_col=0)
        df = df.drop(["Chr", "Start", "End", "Strand", "Length"], axis=1)
        df.columns = [i.split("/")[-1].split(".bam")[0] for i in df.columns]
        df.to_csv(output[0], sep="\t")

rule deseq2_spikein_peaks:
    input:
        reference="results/cutnrun/reference.{feature}.txt",
        spikein="results/deseq2/{spike}.exons.txt",
        design="config/designs/{dtag}.tsv"
    params:
        threshold=10,
        out_dir="results/deseq2_peaks/{feature}/{dtag}/{spike}",
        treatment="{dtag}",
        wt="DMSO",
    output:
        "results/deseq2_peaks/{feature}/{dtag}/{spike}/calibrated_dds.Rds",
        "results/deseq2_peaks/{feature}/{dtag}/{spike}/calibrated_{dtag}_vs_DMSO_lfc.tsv",
        "results/deseq2_peaks/{feature}/{dtag}/{spike}/{dtag}_vs_DMSO_lfc.tsv"
    conda: "../envs/deseq2.yml"
    threads: 12
    script:
        "../scripts/deseq2_spikein.R"

rule deseq2_spikein_peaks_plots:
    input:
        lfc_dTAG_05="results/deseq2_peaks/{feature}/dTAG_05/{spike}/calibrated_dTAG_05_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_05="results/deseq2_peaks/{feature}/dTAG_05/{spike}/dTAG_05_vs_DMSO_lfc.tsv",
        lfc_dTAG_3="results/deseq2_peaks/{feature}/dTAG_3/{spike}/calibrated_dTAG_3_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_3="results/deseq2_peaks/{feature}/dTAG_3/{spike}/dTAG_3_vs_DMSO_lfc.tsv",
        lfc_dTAG_9="results/deseq2_peaks/{feature}/dTAG_9/{spike}/calibrated_dTAG_9_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_9="results/deseq2_peaks/{feature}/dTAG_9/{spike}/dTAG_9_vs_DMSO_lfc.tsv",
        lfc_none="results/deseq2_peaks/{feature}/none/{spike}/calibrated_none_vs_DMSO_lfc.tsv",
        lfc_non_none="results/deseq2_peaks/{feature}/none/{spike}/none_vs_DMSO_lfc.tsv",
        lfc_T2C10_dTAG_2="results/deseq2_peaks/{feature}/T2C10/{spike}/calibrated_T2C10_vs_DMSO_lfc.tsv",
        lfc_T1C2_dTAG_2="results/deseq2_peaks/{feature}/T1C2/{spike}/calibrated_T1C2_vs_DMSO_lfc.tsv",
        lfc_non_T2C10_dTAG_2="results/deseq2_peaks/{feature}/T2C10/{spike}/T2C10_vs_DMSO_lfc.tsv",
        lfc_non_T1C2_dTAG_2="results/deseq2_peaks/{feature}/T1C2/{spike}/T1C2_vs_DMSO_lfc.tsv",
        lfc_dTAG_C1_C2="results/deseq2_peaks/{feature}/dTAG_C1_C2/{spike}/calibrated_dTAG_C1_C2_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_C1_C2="results/deseq2_peaks/{feature}/dTAG_C1_C2/{spike}/dTAG_C1_C2_vs_DMSO_lfc.tsv"
    output:
        summary="results/deseq2_peaks/plots/{feature}/{spike}/calibrated_summary_lfc_plots.pdf",
        non_calibrated="results/deseq2_peaks/plots/{feature}/{spike}/summary_lfc_plots.pdf",
        up_down_genes="results/deseq2_peaks/plots/{feature}/{spike}/up_down_genes.pdf",
        size_factors="results/deseq2_peaks/plots/{feature}/{spike}/size_factors.pdf"
    params:
        padj=0.05,
        l2fc=np.log2(1.01),
    conda:
        "../envs/matplotlib.yml"
    script:
        "../scripts/deseq2_spikein_peaks_separate_plots.py"

rule peaks_plots:
    input: expand("results/deseq2_peaks/plots/{feature}/{spike}/up_down_genes.pdf", feature=["intergenic"], spike=["spikein", "extra_spikein"])
