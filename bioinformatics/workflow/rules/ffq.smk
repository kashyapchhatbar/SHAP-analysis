rule download:
    input: expand("fastq/download_links/{gsm}_fastq.txt", gsm=gsms)

rule ffq:
    output:
        "fastq/download_links/{gsm}_fastq.txt"
    conda:
        "../envs/ffq.yml"
    shell:
        """ffq --ftp {wildcards.gsm} | jq -r '.[] | .url' > {output}
        sleep 10
        """

rule ftp_download_pe:
    input:
        "fastq/download_links/{gsm}_fastq.txt"
    output:
        one="fastq/{gsm}_1.fq.gz",
        two="fastq/{gsm}_2.fq.gz"
    conda:
        "../envs/ftp.yml"
    script:
        "../scripts/ftp_download.py"
