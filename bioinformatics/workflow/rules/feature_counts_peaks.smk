rule download_liftover:
    params:
        url="https://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/mm10ToMm39.over.chain.gz"
    output:
        gz=temp("resources/liftover/mm10ToMm39.over.chain.gz"),
        chain="resources/liftover/mm10ToMm39.over.chain"
    shell:
        """
        wget {params.url} -O {output.gz}
        gzip -cd {output.gz} > {output.chain}
        """

rule liftover:
    input:
        chain="resources/liftover/mm10ToMm39.over.chain",
        bed="resources/cutnrun/SALL4_CnR_SG_merged_peaks_mm10.bed"
    output:
        mm39="resources/cutnrun/SALL4_CnR_SG_merged_peaks_mm39.bed",
        unmapped="resources/cutnrun/SALL4_CnR_SG_merged_peaks_mm10_unmapped.bed"
    conda:
        "../envs/liftover.yml"
    shell:
        "liftOver {input.bed} {input.chain} {output.mm39} {output.unmapped}"

rule liftover_convert:
    input:
        bed="resources/cutnrun/SALL4_CnR_SG_merged_peaks_mm39.bed",
        genome="resources/genomes/concatenated.fa.fai"
    output:
        bed="resources/cutnrun/SALL4_CnR_SG_merged_peaks_concatenated.bed",
    run:
        import pandas as pd
        genome = pd.read_csv(input.genome, sep="\t", header=None)
        df = pd.read_csv(input.bed, sep="\t", header=None)        
        df[0] = "reference_" + df[0]
        df = df[df[0].isin(genome[0])]
        df.to_csv(output.bed, sep="\t", header=False, index=False)

rule intergenic_peaks:
    input:
        bed="resources/cutnrun/SALL4_CnR_SG_merged_peaks_concatenated.bed",
        genes="resources/regions/concatenated/genes.bed"
    output:
        "resources/cutnrun/SALL4_CnR_SG_intergenic_peaks_concatenated.bed"
    conda:
        "../envs/bedtools.yml"
    shell:
        """
        bedtools subtract -a {input.bed} -b {input.genes} -A | bedtools \
            slop -b 3000 -i /dev/stdin -g resources/genomes/concatenated.fa.fai \
            > {output}
        """

rule generate_peaks_saf:
    input:
        bed="resources/cutnrun/SALL4_CnR_SG_intergenic_peaks_concatenated.bed"
    output:
        saf="resources/cutnrun/SALL4_CnR_SG_intergenic_peaks_concatenated.saf"
    shell:
        """
        awk '{{print $4"\t"$1"\t"$2"\t"$3"\t+"}}' {input.bed} > {output.saf}
        """

rule feature_counts_peaks_saf:
    input:
        saf="resources/cutnrun/SALL4_CnR_SG_intergenic_peaks_concatenated.saf",
        bam=expand("results/deduplicated/concatenated/{sample}.bam", sample=samples)
    output:        
        counts="results/cutnrun/SALL4_CnR_SG_intergenic_peaks_SAF_counts.txt"    
    conda:
        "../envs/subread.yml"
    threads: 24
    shell:
        """        
        featureCounts -a {input.saf} -F SAF -o {output.counts} -p \
            -T {threads} --countReadPairs -B -C {input.bam}
        """
