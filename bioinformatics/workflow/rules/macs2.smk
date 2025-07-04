import yaml

gsms = pd.read_csv(config['gsms'], header=None, sep="\t")[0].tolist()

rule fold_change:
    input: expand("results/bigwig_average/{sample}.bed", sample=gsms)

rule fold_change_cut:
    input: expand("results/bigwig_average_cut/{sample}.bed", sample=gsms)

rule fold_change_enhancers:
    input: expand("results/bigwig_average_enhancers/{sample}.bed", sample=gsms)

rule filter_blacklist:
    input: 
        bam="results/sorted/{sample}.bam",
        blacklist="resources/genomes/reference_blacklist.bed"
    output: "results/filtered/{sample}.bam"
    conda: "../envs/bedtools.yml"
    shell: "bedtools intersect -v -abam {input.bam} -b {input.blacklist} > {output}"

rule macs2:
    input: 
        chip="results/filtered/{sample}.bam",
    conda: "../envs/macs2.yml"
    threads: 2    
    output: temp("results/log2_chip_control/{sample}_treat_pileup.bdg"),
        temp("results/log2_chip_control/{sample}_control_lambda.bdg")
    shell: "macs2 callpeak -B -g mm -t {input.chip} --outdir results/log2_chip_control -n {wildcards.sample}"
    
rule macs2_bdg:
    input: treatment="results/log2_chip_control/{sample}_treat_pileup.bdg",
        control="results/log2_chip_control/{sample}_control_lambda.bdg"
    output: temp("results/log2_chip_control/temp.{sample}")
    conda: "../envs/macs2.yml"
    threads: 2
    shell: "macs2 bdgcmp -t {input.treatment} -c {input.control} -m FE -o {output}"
    
rule sort:
    input: "results/log2_chip_control/temp.{sample}"
    output: temp("results/log2_chip_control/{sample}.sorted.bedgraph")
    threads: 6
    shell: "sort --parallel={threads} -S 5% -k1,1 -k2,2n {input} > {output}"
    
rule bigwig:
    input:
        bedgraph="results/log2_chip_control/{sample}.sorted.bedgraph",
        chrlen="resources/genomes/reference.chrlen"
    output: "results/log2_chip_control/{sample}.bw"
    conda: "../envs/ucsc_tools.yml"
    threads: 1
    shell: "bedGraphToBigWig {input.bedgraph} {input.chrlen} {output}"

rule bigwig_average_over_bed:
    input: 
        bw="results/log2_chip_control/{sample}.bw",
        quartiles="resources/regions/reference/genes_quartiles.bed"
    output: "results/bigwig_average/{sample}.bed"
    conda: "../envs/ucsc_tools.yml"
    threads: 1
    shell: "bigWigAverageOverBed {input.bw} {input.quartiles} {output}"

rule bigwig_average_over_bed_cut:
    input: 
        bw="results/log2_chip_control/{sample}.bw",
        quartiles="resources/regions/reference/genes_promoter_gb.bed"
    output: "results/bigwig_average_cut/{sample}.bed"
    conda: "../envs/ucsc_tools.yml"
    threads: 1
    shell: "bigWigAverageOverBed {input.bw} {input.quartiles} {output}"

rule bigwig_average_over_bed_enhancers:
    input: 
        bw="results/log2_chip_control/{sample}.bw",
        quartiles="resources/regions/reference/genes_enhancers_promoter_gb.bed"
    output: "results/bigwig_average_enhancers/{sample}.bed"
    conda: "../envs/ucsc_tools.yml"
    threads: 1
    shell: "bigWigAverageOverBed {input.bw} {input.quartiles} {output}"