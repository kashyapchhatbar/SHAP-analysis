chip_samples = ["GSM7187834", "GSM5513895", "GSM5513891", "GSM5513878", "GSM5513889"]

rule combined_heatmap:
    input: expand("results/heatmaps/{gene}_combined.heatmap.pdf", gene=["INTS11_2h", "INTS11_8h", "SET1AB_2h", "Z4_2h", "SET1AB_Z4_2h", "RBBP5_8h", "DPY30_8h"])

rule get_sig_genes:
    input:
        lfc="results/deseq2/{gene}_vs_DMSO_lfc.tsv",
        bed="resources/regions/reference/genes.bed"
    output:
        sig="results/heatmaps/{gene}_sig_genes.bed",
        non_sig="results/heatmaps/{gene}_non_sig_genes.bed"
    script:
        "../scripts/get_sig_genes.py"

rule combined_compute_matrix:
    input:
        bws=expand("results/log2_chip_control/{sample}.bw", sample=chip_samples),
        sig_genes="results/heatmaps/{gene}_sig_genes.bed",
        non_sig_genes="results/heatmaps/{gene}_non_sig_genes.bed"
    output:
        "results/heatmaps/{gene}_combined.matrix.gz"
    conda:
        "../envs/deeptools.yml"
    shell:
        """
        computeMatrix scale-regions -S {input.bws} -R {input.sig_genes} {input.non_sig_genes} \
            -a 2000 -b 3000 -m 10000 -o {output} -p 128
        """

rule combined_plot_heatmap:
    input:
        "results/heatmaps/{gene}_combined.matrix.gz"
    output:
        "results/heatmaps/{gene}_combined.heatmap.pdf"
    conda:
        "../envs/deeptools.yml"
    shell:
        """
        plotHeatmap --boxAroundHeatmaps yes --samplesLabel INTS11 DPY30 RBBP5 PolII S5P-PolII \
            -m {input} -o {output} --regionsLabel "Sig genes" "Non-sig genes" --heatmapHeight 8 \
            --averageTypeSummaryPlot median --plotType se --colorMap viridis --zMin 0 --zMax 5
        """
