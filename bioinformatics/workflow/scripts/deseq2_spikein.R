library(BiocParallel)
library(DESeq2)
library(ggplot2)
register(MulticoreParam(snakemake@threads))

deseq_function <- function(reference_counts, spikein_counts, 
  design, nuc, threshold, out_dir, treatment, wt){
  
  out_prefix <- paste(out_dir, "/", sep="")
  
  control_genes = row.names(spikein_counts)
  design = read.csv(design, header=TRUE, sep="\t", row.names=1)
  print(dim(design))

  print("reference_counts")
  print(dim(reference_counts))

  coef = paste("treatment_", treatment, "_vs_", wt, sep="")
  contrast = c("treatment", treatment, wt)


  reference_dds = DESeqDataSetFromMatrix(countData = reference_counts, colData = design, 
                                         design = ~ jointly_handled + treatment)
  reference_dds = reference_dds[rowSums(counts(reference_dds)) > threshold,]
  
  print("reference_dds")
  print(dim(reference_dds))

  reference_dds$treatment <- relevel(reference_dds$treatment, ref = wt)  
  print("reference_dds")
  print(dim(reference_dds))

  reference_dds = DESeq(reference_dds, parallel=TRUE)
  # reference_lfc = results(reference_dds, contrast=contrast, alpha=0.05, parallel=TRUE)
  reference_lfc = lfcShrink(reference_dds, coef=coef, parallel=TRUE)
  reference_lfc_df = as.data.frame(reference_lfc)
  reference_lfc_df_filename = paste(out_prefix, "calibrated_", treatment,
                                    "_vs_", wt, "_lfc.tsv", sep="")
  write.table(reference_lfc_df, file=reference_lfc_df_filename, sep="\t")

  rld <- rlog(reference_dds)
  pcaData <- plotPCA(rld, intgroup=c("treatment"), returnData=TRUE)
  percentVar <- round(100 * attr(pcaData, "percentVar"))
  ggplot(pcaData, aes(PC1, PC2, color=treatment)) +
    geom_point(size=3) +
    xlab(paste0("PC1: ",percentVar[1],"% variance")) +
    ylab(paste0("PC2: ",percentVar[2],"% variance")) + 
    coord_fixed()
  pcaplot_pdf <- paste(out_prefix, "calibrated_PCA.pdf", sep="")
  ggsave(pcaplot_pdf)

  rld_df = as.data.frame(assay(rld))
  rld_df_filename = paste(out_prefix, "calibrated_rld.tsv", sep="")
  write.table(rld_df, file=rld_df_filename, sep="\t")
  
  counts_df = as.data.frame(counts(reference_dds, normalized=TRUE))
  counts_df_filename = paste(out_prefix, "calibrated_counts.tsv", sep="")
  write.table(counts_df, file=counts_df_filename, sep="\t")

  saveRDS(reference_dds, file=paste(out_prefix, "calibrated_dds.Rds", sep=""))

  reference_dds = DESeqDataSetFromMatrix(countData = reference_counts, colData = design, 
                                    design = ~ treatment + jointly_handled)

  print("reference_dds")
  print(dim(reference_dds))

  reference_dds = reference_dds[rowSums(counts(reference_dds)) > threshold,]
  reference_dds$treatment <- relevel(reference_dds$treatment, ref = wt)
  
  print("reference_dds")
  print(dim(reference_dds))

  reference_dds = DESeq(reference_dds, parallel=TRUE)
  # reference_lfc = results(reference_dds, contrast=contrast, alpha=0.05, parallel=TRUE)
  reference_lfc = lfcShrink(reference_dds, coef=coef, parallel=TRUE)
  reference_lfc_df = as.data.frame(reference_lfc)
  reference_lfc_df_filename = paste(out_prefix, treatment,
                                "_vs_", wt, "_lfc.tsv", sep="")
  write.table(reference_lfc_df, file=reference_lfc_df_filename, sep="\t")

  rld <- rlog(reference_dds)
  pcaData <- plotPCA(rld, intgroup=c("treatment"), returnData=TRUE)
  percentVar <- round(100 * attr(pcaData, "percentVar"))
  ggplot(pcaData, aes(PC1, PC2, color=treatment)) +
    geom_point(size=3) +
    xlab(paste0("PC1: ",percentVar[1],"% variance")) +
    ylab(paste0("PC2: ",percentVar[2],"% variance")) + 
    coord_fixed()
  pcaplot_pdf <- paste(out_prefix, "PCA.pdf", sep="")
  ggsave(pcaplot_pdf)

  rld_df = as.data.frame(assay(rld))
  rld_df_filename = paste(out_prefix, "rld.tsv", sep="")
  write.table(rld_df, file=rld_df_filename, sep="\t")
  
  counts_df = as.data.frame(counts(reference_dds, normalized=TRUE))
  counts_df_filename = paste(out_prefix, "counts.tsv", sep="")
  write.table(counts_df, file=counts_df_filename, sep="\t")

  size_factors = as.data.frame(sizeFactors(reference_dds))
  write.table(size_factors, file=paste(out_prefix, "reference_size_factors.tsv", sep=""), sep="\t")
  saveRDS(reference_dds, file=paste(out_prefix, "dds.Rds", sep=""))
}

generate_tables <- function(reference, spikein, design){
  design = read.csv(design, header=TRUE, sep="\t", row.names=1)
  
  reference_counts = read.csv(reference, sep="\t", header = TRUE, 
                              row.names = 1, check.names = FALSE)
  reference_counts = reference_counts[,rownames(design)]
  
  spikein_counts = read.csv(spikein, sep="\t", header = TRUE, 
                            row.names = 1, check.names = FALSE)
  spikein_counts = spikein_counts[,rownames(design)]
  return(list(reference_counts, spikein_counts))
}

reference_spikein = generate_tables(snakemake@input[['reference']],
  snakemake@input[['spikein']], snakemake@input[['design']])

deseq_function(reference_spikein[[1]], reference_spikein[[2]],               
               snakemake@input[['design']], snakemake@input[['nuc']],
               snakemake@params[['threshold']], snakemake@params[['out_dir']],
               snakemake@params[['treatment']],
               snakemake@params[['wt']])
