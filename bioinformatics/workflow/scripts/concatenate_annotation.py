def concatenate_annotation(reference, spikein, extra_spikein, concatenated):
    oH = open(concatenated, "wt")
    
    for annotation, name in zip([reference, spikein], ["reference", "spikein"]):
        with open(annotation, "rt") as iH:
            for line in iH:
                if not line.startswith("#"):
                    print(f"{name}_{line}", file=oH, end="")
                else:
                    pass
    
    if extra_spikein:
        with open(extra_spikein, "rt") as iH:
            for line in iH:
                if not line.startswith("#"):
                    print(f"extra_spikein_{line}", file=oH, end="")
                else:
                    pass
    
    oH.close()
    
concatenate_annotation(snakemake.input.reference, snakemake.input.spikein,
                    snakemake.params.extra_spikein, snakemake.output[0])