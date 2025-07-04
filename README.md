[![DOI](https://img.shields.io/badge/bioR%CF%87iv%20DOI-10.1101%2F2025.01.30.635704%20-BC2635)](https://doi.org/10.1101/2025.01.30.635704)

## Modeling transcription with explainable AI uncovers context-specific epigenetic gene regulation at promoters and gene bodies

Transcriptional regulation involves complex interactions with chromatin-associated proteins, but disentangling these mechanistically remains challenging. Here, we generate deep learning models to predict RNA Pol-II occupancy from chromatin-associated protein profiles in unperturbed conditions. We evaluate the suitability of Shapley Additive Explanations (SHAP), a widely used explainable AI (XAI) approach, to infer functional relevance and analyse regulatory mechanisms across diverse datasets. We aim to validate these insights using data from degron-based perturbation experiments. Remarkably, genes ranked by SHAP importance predict direct targets of perturbation even from unperturbed data, enabling inference without costly experimental interventions. Our analysis reveals that SHAP not only predicts differential gene expression but also captures the magnitude of transcriptional changes. We validate the cooperative roles of SET1A and ZC3H4 at promoters and uncover novel regulatory contributions of ZC3H4 at gene bodies in influencing transcription. Cross-dataset validation uncovers unexpected connections between ZC3H4, a component of the Restrictor complex, and INTS11, part of the Integrator complex, suggesting crosstalk mediated by H3K4me3 and the SET1/COMPASS complex in transcriptional regulation. These findings highlight the power of integrating predictive modelling and experimental validation to unravel complex context-dependent regulatory networks and generate novel biological hypotheses.

### Repository Structure

- `data/`: Contains all processed ChIP-seq datasets required for analysis.
- `shap/`: Contains all models, model splits, computed SHAP values, snakemake workflows and python scripts for model training and SHAP analysis.
- `bioinformatics/`: Snakemake workflows for performing bioinformatics pre-processing from raw fastq reads to processed data required for regression analysis.

### Prerequisites

Install `snakemake` and execute workflows by using `--use-conda` to handle all dependencies

Feel free to reach out via the repository's issue tracker if you encounter any issues or have questions about the analysis.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
