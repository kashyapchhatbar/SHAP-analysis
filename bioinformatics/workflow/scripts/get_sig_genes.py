import pandas as pd
import numpy as np

lfc = pd.read_csv(snakemake.input.lfc, sep='\t', index_col=0)
bed = pd.read_csv(snakemake.input.bed, sep='\t', header=None)

sig_lfc = lfc[(lfc['padj'] < 0.05) & (abs(lfc['log2FoldChange']) >= np.log2(1.5))].copy()
non_sig_lfc = lfc[(lfc['padj'] >= 0.05) | (abs(lfc['log2FoldChange']) < np.log2(1.5))].sample(
    n=sig_lfc.shape[0], random_state=42)

sig_bed = bed[bed[3].isin(sig_lfc.index)]
non_sig_bed = bed[bed[3].isin(non_sig_lfc.index)]

sig_bed.to_csv(snakemake.output.sig, sep='\t', header=False, index=False)
non_sig_bed.to_csv(snakemake.output.non_sig, sep='\t', header=False, index=False)