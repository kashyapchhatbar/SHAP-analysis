import pandas as pd

def annotate(lfc, bed, nuc):
    bed = pd.read_csv(bed, sep="\t", header=None)
    nuc = pd.read_csv(nuc, sep="\t", index_col=3)
    bed_copy = bed.set_index(3).copy()
    bed_copy[7] = bed_copy[2] - bed_copy[1]
    lfc["gene_length_kb"] = bed_copy[7] / 1000
    lfc["gene_name"] = bed_copy[6].apply(lambda x: x.split(",")[0])
    lfc["gene_biotype"] = bed_copy[6].apply(lambda x: x.split(",")[1])
    lfc["gene_locus"] = bed_copy.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1)
    lfc["AT"] = nuc["8_pct_at"]
    
    return lfc

lfc = pd.read_csv(snakemake.input[0], sep="\t", index_col=0)
lfc = annotate(lfc, snakemake.input[1], snakemake.input[2])
lfc.to_csv(snakemake.output[0], sep="\t")
lfc.to_excel(snakemake.output[1])