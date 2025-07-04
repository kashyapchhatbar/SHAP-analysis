import pandas as pd

def detect_source(x):
    if x.startswith("spikein"):
        return "spikein"
    elif x.startswith("extra_spikein"):
        return "extra_spikein"
    else:
        return "reference"

count_dfs = []
for txt in snakemake.input.txts:
    temp_df = pd.read_csv(txt, index_col=0, comment="#", sep="\t", usecols=[0,6])
    temp_df.columns = [temp_df.columns[-1].split("/")[-1].split(".bam")[0]]
    count_dfs.append(temp_df)
temp_df = pd.read_csv(txt, index_col=0, sep="\t", comment="#", usecols=[0,1])
temp_df["genome"] = temp_df["Chr"].apply(detect_source)
df = pd.concat(count_dfs + [temp_df], axis=1)        
for genome, outfile in zip(["reference", "spikein", "extra_spikein"],
    [snakemake.output.reference, snakemake.output.spikein, snakemake.output.extra_spikein]):
    df[df["genome"]==genome].drop("genome", axis=1).fillna(0).to_csv(outfile, sep="\t")