rule bowtie2:
    input:
        fq1="fastq/{sample}_1.fq.gz",
        fq2="fastq/{sample}_2.fq.gz",
        idx=multiext(
            "resources/genomes/reference",
            ".1.bt2",
            ".2.bt2",
            ".3.bt2",
            ".4.bt2",
            ".rev.1.bt2",
            ".rev.2.bt2",
        ),
        ref="resources/genomes/reference.fa", #Required for CRAM output
    output:
        cram="results/sorted/{sample}.bam",
        fq1="results/unmapped/{sample}_1.fq.gz",
        fq2="results/unmapped/{sample}_2.fq.gz"
    params:
        un="results/unmapped/{sample}_%.fq.gz",
        index="resources/genomes/reference"
    log:
        "logs/bowtie2/{sample}.log",    
    threads: 24
    conda:
        "../envs/bowtie2.yml"
    shell:
        """
        bowtie2 --no-unal --un-conc-gz {params.un} --sensitive-local --no-mixed \
        --no-discordant -p {threads} -x {params.index} -1 {input.fq1} \
        -2 {input.fq2} | samtools sort -@ {threads} -m 6G --reference \
        {input.ref} --write-index -o {output.cram} -
        """

rule bowtie2_se:
    input:
        fq1="fastq/{sample}_1.fq.gz",        
        idx=multiext(
            "resources/genomes/reference",
            ".1.bt2",
            ".2.bt2",
            ".3.bt2",
            ".4.bt2",
            ".rev.1.bt2",
            ".rev.2.bt2",
        ),
        ref="resources/genomes/reference.fa", #Required for CRAM output
    output:
        cram="results/se_sorted/{sample}.bam",        
    params:        
        index="resources/genomes/reference"
    log:
        "logs/bowtie2/{sample}.log",    
    threads: 24
    conda:
        "../envs/bowtie2.yml"
    shell:
        """
        bowtie2 --no-unal --sensitive-local --no-mixed \
        --no-discordant -p {threads} -x {params.index} -U {input.fq1} \
        | samtools sort -@ {threads} -m 6G --reference \
        {input.ref} --write-index -o {output.cram} -
        """


rule deduplicate_pe:
    input:
        bam="results/sorted/{sample}.bam"        
    output:
        bam="results/deduplicated/{sample}.bam",
        stats="results/deduplicated/{sample}.stats"
    conda:
        "../envs/samtools.yml"
    threads: 12
    shell:
        """
        samtools collate -@ 12 -O {input.bam} \
            | samtools fixmate -m - - | samtools \
            sort -@ 12 -m 4G - | samtools markdup --write-index -r -@ 12 -f \
            {output.stats} - {output.bam}
        """

# rule deduplicate_se:
#     input:
#         bam="results/se_sorted/{sample}.bam"        
#     output:
#         bam="results/se_deduplicated/{sample}.bam",
#         stats="results/se_deduplicated/{sample}.stats"
#     conda:
#         "../envs/samtools.yml"
#     threads: 12
#     shell:
#         """
#         samtools collate -@ 12 -O {input.bam} \
#             | samtools fixmate -m - - | samtools \
#             sort -@ 12 -m 4G - | samtools markdup --write-index -r -@ 12 -f \
#             {output.stats} - {output.bam}
#         """

rule map:
    input: expand("results/sorted/{sample}.bam", sample=samples)

rule map_se:
    input: expand("results/se_sorted/{sample}.bam", sample=se_samples)