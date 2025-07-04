rule alignment_summary:
    input:
        # BAM aligned, splicing-aware, to reference genome
        bam="results/deduplicated/{genome}/{sample}.bam",
        # Reference genome
        ref="resources/genomes/concatenated.fa",
        # Annotation file containing transcript, gene, and exon data
        refflat="resources/genePred/{genome}.txt",
        rRNA_intervals="resources/genePred/{genome}_rRNA_genes.interval_list",
    output:
        "results/qc/rnaseq_metrics/{genome}/{sample}.txt",
    params:
        # strand is optional (defaults to NONE) and pertains to the library preparation
        # options are FIRST_READ_TRANSCRIPTION_STRAND, SECOND_READ_TRANSCRIPTION_STRAND, and NONE
        strand="NONE",
        # optional additional parameters, for example,
        extra="--VALIDATION_STRINGENCY STRICT --RIBOSOMAL_INTERVALS resources/genePred/{genome}_rRNA_genes.interval_list",
    log:
        "logs/picard/rnaseq-metrics/{genome}/{sample}.log",
    wrapper:
        "v3.13.0/bio/picard/collectrnaseqmetrics"

rule multiqc_genome:
    input:
        expand("results/qc/rnaseq_metrics/{genome}/{sample}.txt", genome=['{genome}'], sample=samples),
    output:
        "report/{genome}/multiqc.html",
        directory("report/{genome}/multiqc_data"),
    params:
        extra="--verbose",  # Optional: extra parameters for multiqc.
    log:
        "logs/multiqc/{genome}.log",
    wrapper:
        "file://resources/wrappers/multiqc"

rule multiqc:
    input:
        expand("report/{genome}/multiqc.html", genome=['reference', 'spikein', 'extra_spikein'])