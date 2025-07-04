rule bowtie2_index:
    input:
        ref="resources/genomes/{genome}.fa",
    output:
        multiext(
            "resources/genomes/{genome}",
            ".1.bt2",
            ".2.bt2",
            ".3.bt2",
            ".4.bt2",
            ".rev.1.bt2",
            ".rev.2.bt2",
        ),
    log:
        "logs/bowtie2_build/{genome}_build.log"    
    params:
        extra="",  # optional parameters
    threads: 24
    wrapper:
        "v2.2.1/bio/bowtie2/build"

rule download_genome:    
    output:
        ref="resources/genomes/{genome}.fa",
        gtf="resources/genomes/{genome}.gtf",
        fasta_gz=temp("resources/download/{genome}.fa.gz"),
        gtf_gz=temp("resources/download/{genome}.gtf.gz")
    threads: 32
    params:    
        fasta_url=lambda wildcards: config["genome_links"][config["genomes"][wildcards.genome]]["fasta"],
        gtf_url=lambda wildcards: config["genome_links"][config["genomes"][wildcards.genome]]["gtf"]
    log:
        "logs/download_genome/{genome}.log",
    shell:
        """
        wget {params.fasta_url} -O {output.fasta_gz}
        wget {params.gtf_url} -O {output.gtf_gz}
        gzip -cd {output.fasta_gz} > {output.ref}
        gzip -cd {output.gtf_gz} > {output.gtf}
        """

ruleorder: download_genome > bowtie2_index

rule reference_genomes:
    input: expand("resources/genomes/{genome}{ext}", genome=["reference"], ext=[".1.bt2", "/SA"])

rule blacklist:
    output:
        "resources/genomes/reference_blacklist.bed"
    params:
        url=config["genome_links"][config["genomes"]["reference"]]["blacklist"],
        gz="resources/reference_blacklist.bed.gz"
    shell:
        """
        wget {params.url} -O {params.gz}
        gzip -cd {params.gz} > {output}
        """

rule gtfToGenePred_CollectRnaSeqMetrics_genome:
    input:
        # annotations containing gene, transcript, exon, etc. data in GTF format
        "resources/genomes/{genome}.gtf"
    output:
        temp("resources/{genome}.temp")
    log:
        "logs/gtfToGenePred.{genome}.PicardCollectRnaSeqMetrics.log",
    params:
        convert_out="PicardCollectRnaSeqMetrics",
        extra="-genePredExt -geneNameAsName2",  # optional parameters to pass to gtfToGenePred
    wrapper:
        "v3.13.0/bio/ucsc/gtfToGenePred"

rule modify_chromosome:
    input:
        "resources/{genome}.temp",
    output:
        "resources/genePred/{genome}.txt"
    run:
        import pandas as pd
        df = pd.read_csv(input[0], sep="\t", header=None)
        df[2] = f"{wildcards.genome}_" + df[2]
        df.to_csv(output[0], sep="\t", header=False, index=False)

rule genome_fa_fai:
    input:
        "resources/genomes/{genome}.fa"
    output:
        "resources/genomes/{genome}.fa.fai"
    conda:
        "../envs/samtools.yml"
    shell:
        """
        samtools faidx {input}
        """

rule genome_chr_len:
    input:
        "resources/genomes/{genome}.fa.fai"
    output:
        "resources/genomes/{genome}.chrlen"    
    shell:
        """
        cut -f 1,2 {input} > {output}
        """


rule organism_chr_bed:
    input: 
        "resources/genomes/{genome}.fa.fai"
    output:
        "resources/genomes/{genome}_chrs.bed"
    run: 
        import pandas as pd
        df = pd.read_csv(f"{input[0]}", sep="\t", header=None, usecols=[0,1])
        df[2] = 0
        if wildcards.genome == "concatenated":
            df[0] = df[0].apply(lambda x: f"{x}")
        else:
            df[0] = df[0].apply(lambda x: f"{wildcards.genome}_{x}")
        df[[0,2,1]].to_csv(output[0], sep="\t", index=None, header=None)

ruleorder: organism_chr_bed > genome_fa_fai

rule gencode_regions:
    input:
        "resources/genomes/{genome}.gtf"
    output:
        "resources/regions/{genome}/exons.bed",
        "resources/regions/{genome}/introns.bed",
        "resources/regions/{genome}/genes.bed"
    params:
        output="resources/regions/{genome}",
        script="workflow/scripts/create_regions_from_gencode.R"
    conda:
        "../envs/genomicfeatures.yml"
    shell:
        "Rscript {params.script} {input} {params.output}"

rule bed_quartiles:
    input:
        "resources/regions/{genome}/genes.bed",
        "resources/genomes/{genome}.chrlen"
    output:
        "resources/regions/{genome}/genes_quartiles.bed"
    script:
        "../scripts/bed_quartiles.py"

rule bed_cut_quartiles:
    input:
        "resources/regions/{genome}/genes.bed",
        "resources/genomes/{genome}.chrlen"
    output:        
        "resources/regions/{genome}/genes_cut_quartiles.bed"
    script:
        "../scripts/bed_cut_quartiles.py"

rule bed_gene_body:
    input:
        "resources/regions/{genome}/genes.bed",
        "resources/genomes/{genome}.chrlen"
    output:        
        "resources/regions/{genome}/genes_promoter_gb.bed"
    script:
        "../scripts/bed_promoter_gb.py"

rule bed_enhancers_gene_body:
    input:
        "resources/regions/{genome}/genes_enhancers.bed",
        "resources/genomes/{genome}.chrlen"
    output:        
        "resources/regions/{genome}/genes_enhancers_promoter_gb.bed"
    script:
        "../scripts/bed_promoter_gb.py"