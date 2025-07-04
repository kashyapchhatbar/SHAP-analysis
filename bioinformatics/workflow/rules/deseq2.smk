import numpy as np

rule deseq2:
    input:
        expand("results/deseq2/{feature}/{dtag}/{spike}/calibrated_dds.Rds", 
        dtag=["RBBP5_8h", "RBBP5_24h", "DPY30_8h", "DPY30_24h", "INTS11_2h", "INTS11_8h"], feature=["genes"], spike=["spikein"]),

rule generate_count_table_exons:
    input:
        saf="results/deduplicated/reference/exons.saf",
        txts=expand("results/deduplicated/reference/{sample}_SAF_exons.txt", sample=samples)
    output:
        reference="results/deseq2/reference.exons.txt",
        spikein="results/deseq2/spikein.exons.txt",
        extra_spikein="results/deseq2/extra_spikein.exons.txt"
    script:
        "../scripts/generate_count_table.py"

rule generate_count_table_introns:
    input:
        saf="results/deduplicated/reference/introns.saf",
        txts=expand("results/deduplicated/reference/{sample}_SAF_introns.txt", sample=samples)
    output:
        reference="results/deseq2/reference.introns.txt",
        spikein="results/deseq2/spikein.introns.txt",
        extra_spikein="results/deseq2/extra_spikein.introns.txt"
    script:
        "../scripts/generate_count_table.py"

rule generate_count_table_genes:
    input:
        saf="results/deduplicated/reference/genes.saf",
        txts=expand("results/deduplicated/reference/{sample}_SAF_genes.txt", sample=samples)
    output:
        reference="results/deseq2/reference.genes.txt",
        spikein="results/deseq2/spikein.genes.txt",
        extra_spikein="results/deseq2/extra_spikein.genes.txt"
    script:
        "../scripts/generate_count_table.py"

rule generate_count_table_genes_enhancers:
    input:
        saf="results/deduplicated/reference/genes_enhancers.saf",
        txts=expand("results/deduplicated/reference/{sample}_SAF_genes_enhancers.txt", sample=samples)
    output:
        reference="results/deseq2/reference.genes_enhancers.txt",
        spikein="results/deseq2/spikein.genes_enhancers.txt",
        extra_spikein="results/deseq2/extra_spikein.genes_enhancers.txt"
    script:
        "../scripts/generate_count_table.py"

rule generate_tpm:
    input:
        reference="results/deseq2/reference.{feature}.txt",
        genes="resources/regions/reference/genes.bed"
    output:
        "results/deseq2/reference.{feature}.tpm.txt"
    run:
        import pandas as pd
        count_df = pd.read_csv(input.reference, sep="\t", index_col=0).drop("Chr", axis=1)
        gene_df = pd.read_csv(input.genes, sep="\t", header=None, index_col=3).loc[count_df.index]
        gene_df["length"] = gene_df[2] - gene_df[1]
        count_df = count_df.apply(lambda x: x / gene_df["length"])
        count_df = 1e6 * (count_df / count_df.sum())
        count_df.to_csv(output[0], sep="\t")

rule generate_enhancers_tpm:
    input:
        reference="results/deseq2/reference.genes_enhancers.txt",
        genes="resources/regions/reference/genes_enhancers.bed"
    output:
        "results/deseq2/reference.genes_enhancers.tpm.txt"
    run:
        import pandas as pd
        count_df = pd.read_csv(input.reference, sep="\t", index_col=0).drop("Chr", axis=1)
        gene_df = pd.read_csv(input.genes, sep="\t", header=None, index_col=3).loc[count_df.index]
        gene_df["length"] = gene_df[2] - gene_df[1]
        count_df = count_df.apply(lambda x: x / gene_df["length"])
        count_df = 1e6 * (count_df / count_df.sum())
        count_df.to_csv(output[0], sep="\t")


rule deseq2_spikein_feature:
    input:
        reference="results/deseq2/reference.{feature}.txt",
        spikein="results/deseq2/{spike}.exons.txt",
        design="config/designs/{dtag}.tsv"
    params:
        threshold=10,
        out_dir="results/deseq2/{feature}/{dtag}/{spike}",
        treatment="{dtag}",
        wt="DMSO",
    output:
        "results/deseq2/{feature}/{dtag}/{spike}/calibrated_dds.Rds",
        "results/deseq2/{feature}/{dtag}/{spike}/calibrated_{dtag}_vs_DMSO_lfc.tsv",
        "results/deseq2/{feature}/{dtag}/{spike}/{dtag}_vs_DMSO_lfc.tsv"
    conda: "../envs/deseq2.yml"
    threads: 12
    script:
        "../scripts/deseq2_spikein.R"

rule deseq2_spikein_feature_plots:
    input:
        lfc_dTAG_05="results/deseq2/{feature}/dTAG_05/{spike}/calibrated_dTAG_05_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_05="results/deseq2/{feature}/dTAG_05/{spike}/dTAG_05_vs_DMSO_lfc.tsv",
        lfc_dTAG_3="results/deseq2/{feature}/dTAG_3/{spike}/calibrated_dTAG_3_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_3="results/deseq2/{feature}/dTAG_3/{spike}/dTAG_3_vs_DMSO_lfc.tsv",
        lfc_dTAG_9="results/deseq2/{feature}/dTAG_9/{spike}/calibrated_dTAG_9_vs_DMSO_lfc.tsv",
        lfc_non_dTAG_9="results/deseq2/{feature}/dTAG_9/{spike}/dTAG_9_vs_DMSO_lfc.tsv",
        lfc_none="results/deseq2/{feature}/none/{spike}/calibrated_none_vs_DMSO_lfc.tsv",
        lfc_non_none="results/deseq2/{feature}/none/{spike}/none_vs_DMSO_lfc.tsv",
        lfc_T2C10_dTAG_2="results/deseq2/{feature}/T2C10/{spike}/calibrated_T2C10_vs_DMSO_lfc.tsv",
        lfc_T1C2_dTAG_2="results/deseq2/{feature}/T1C2/{spike}/calibrated_T1C2_vs_DMSO_lfc.tsv",
        lfc_non_T2C10_dTAG_2="results/deseq2/{feature}/T2C10/{spike}/T2C10_vs_DMSO_lfc.tsv",
        lfc_non_T1C2_dTAG_2="results/deseq2/{feature}/T1C2/{spike}/T1C2_vs_DMSO_lfc.tsv",
    output:
        summary="results/deseq2/plots/{feature}/{spike}/calibrated_summary_lfc_plots.pdf",
        non_calibrated="results/deseq2/plots/{feature}/{spike}/summary_lfc_plots.pdf",
        up_down_genes="results/deseq2/plots/{feature}/{spike}/up_down_genes.pdf",
        size_factors="results/deseq2/plots/{feature}/{spike}/size_factors.pdf"
    params:
        padj=0.05,
        l2fc=np.log2(1.1),
    conda:
        "../envs/matplotlib.yml"
    script:
        "../scripts/deseq2_spikein_separate_plots.py"

rule plots:
    input: expand("results/deseq2/plots/{feature}/{spike}/up_down_genes.pdf", feature=["exons", "introns"], spike=["spikein", "extra_spikein"])
