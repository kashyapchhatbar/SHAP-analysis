rule bedtools_multicov:
    input:
        bed="resources/regions/{genome}/{feature}.bed",
        bams=expand("results/deduplicated/{genome}/{sample}.bam", genome=['{genome}'], sample=samples)
    output:
        "results/deduplicated/{genome}/multicov_{feature}.tsv"
    conda:
        "../envs/bedtools.yml"
    shell:
        "bedtools multicov -s -p -bams {input.bams} -bed {input.bed} > {output}"

rule generate_saf:
    input:
        bed="resources/regions/{genome}/{feature}.bed"
    output:
        saf="results/deduplicated/{genome}/{feature}.saf"
    shell:
        """
        awk '{{print $4"\t"$1"\t"$2"\t"$3"\t"$6}}' {input.bed} > {output.saf}
        """

rule feature_counts_saf:
    input:
        saf="results/deduplicated/{genome}/{feature}.saf",
        bam="results/deduplicated/{sample}.bam"
    output:        
        counts="results/deduplicated/{genome}/{sample}_SAF_{feature}.txt"    
    conda:
        "../envs/subread.yml"
    shell:
        """        
        featureCounts -a {input.saf} -F SAF -o {output.counts} -p \
            --countReadPairs -B -C {input.bam}
        """

rule bedtools_count:
    input: expand("results/deduplicated/{genome}/multicov_{feature}.tsv", genome=['reference'], feature=['introns', 'exons'])

rule saf:
    input: expand("results/deduplicated/{genome}/{sample}_SAF_{feature}.txt", genome=['reference'], feature=['introns', 'exons', 'genes', 'genes_enhancers'], sample=samples)