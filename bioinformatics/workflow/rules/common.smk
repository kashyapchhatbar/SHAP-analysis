import pandas as pd
import pathlib

if pathlib.Path(config['gsms']).exists():
    gsms = list(pd.read_csv(config["gsms"], header=None, sep="\t")[0].values)
    download_ffq_rule = True
else:
    download_ffq_rule = False
samples = list(pd.read_csv(config["samples"], sep="\t")["sample_name"].values)
se_samples = list(pd.read_csv(config["se_samples"], sep="\t")["sample_name"].values)