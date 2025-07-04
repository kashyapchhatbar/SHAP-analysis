import pandas as pd

def get_promoter(row, description="promoter"):    
    if row[5] == "-":
        start = row[2] - 1000
        end = row[2] + 1000
    elif row[5] == "+":
        start = row[1] - 1000
        end = row[1] + 1000
    row[1] = start
    row[2] = end
    if row[1] < 1:
        row[1] = 1
    if chrom_sizes.get(row[0], None):
        if row[2] > chrom_sizes[row[0]]:
            row[2] = chrom_sizes[row[0]] - 1
    row[len(row)] = description
    return row

def get_quartile(row, q=1):    
    length = row[2] - row[1] - 1000
    if length <= 0:
        return get_promoter(row, description=f"q{q}")
    if row[5] == "-":                
        start = row[2] - 1000 - int(q * (length/4))
        end = row[2] + 1000
    elif row[5] == "+":
        start = row[1] - 1000        
        end = row[1] + 1000 + int(q * (length/4))
    row[1] = start
    row[2] = end
    if row[1] < 1:
        row[1] = 1
    if chrom_sizes.get(row[0], None):
        if row[2] > chrom_sizes[row[0]]:
            row[2] = chrom_sizes[row[0]] - 1
    row[len(row)] = f"q{q}"
    return row

def get_cut_quartile(row, qq=1):
    length = row[2] - row[1] - 1000
    if length <= 0:
        return get_promoter(row.copy(), description=f"q{qq}")
    if qq == 1:
        oldrow = get_promoter(row.copy(), description=f"q{qq}")
    else:
        oldrow = get_quartile(row.copy(), q=qq-1)
        
    if row[5] == "-":                
        start = row[2] - 1000 - int(qq * (length/4))
        end = oldrow[1]
    elif row[5] == "+":
        start = oldrow[2]
        end = row[1] + 1000 + int(qq * (length/4))
    row[1] = start
    row[2] = end
    if row[1] < 1:
        row[1] = 1
    if chrom_sizes.get(row[0], None):
        if row[2] > chrom_sizes[row[0]]:
            row[2] = chrom_sizes[row[0]] - 1
    row[len(row)] = f"q{qq}"
    return row

def bed_cut_quartiles(bed, size, bed_quartiles):
    genes = pd.read_csv(bed, sep="\t", header=None)
    chrom_sizes = pd.read_csv(size, sep="\t", header=None, index_col=0)[1].to_dict()
    genes = genes[genes[0].isin(chrom_sizes.keys())].copy()
    genes[6] = genes[2] - genes[1]
    genes = genes[genes[6]>2000].drop(6, axis=1)
    quartiles = [genes.apply(get_promoter, axis=1)]
    for qq in [1,2,3,4]:
        quartiles.append(genes.apply(get_cut_quartile, qq=qq, axis=1))
    gene_scales = pd.concat(quartiles).reset_index().drop("index", axis=1)
    gene_scales[3] = gene_scales[3] + "_" + gene_scales[6]
    gene_scales.sort_values(by=[0,1])[[0,1,2,3]].to_csv(bed_quartiles, sep="\t", header=None, index=None)

chrom_sizes = pd.read_csv(snakemake.input[1], sep="\t", header=None, index_col=0)[1].to_dict()
bed_cut_quartiles(snakemake.input[0], snakemake.input[1], snakemake.output[0])