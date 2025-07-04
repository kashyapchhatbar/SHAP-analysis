def concatenate_genomes(reference, spikein, extra_spikein, concatenated):
    oH = open(concatenated, "wt")
    
    for genome, name in zip([reference, spikein], ["reference", "spikein"]):
        with open(genome, "rt") as iH:
            for line in iH:
                if line.startswith(">"):
                    print(f">{name}_{line[1:]}", file=oH, end="")
                else:
                    print(f"{line}", file=oH, end="")

    if extra_spikein:
        with open(extra_spikein, "rt") as iH:
            for line in iH:
                if line.startswith(">"):
                    print(f">extra_spikein_{line[1:]}", file=oH, end="")
                else:
                    print(f"{line}", file=oH, end="")

    oH.close()
    
concatenate_genomes(snakemake.input.reference, snakemake.input.spikein,
                    snakemake.params.extra_spikein, snakemake.output[0])