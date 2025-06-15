
## The code pre-processes the dataset used in Fig2

## Acinar dataset 
Directly downloaded through the R-package of Psupertime. 

R code:
```r
setwd('./Fig2_TemproalVAE_against_benchmark_methods')

# make Acinar cell dataset ------------------------------------------------------------
suppressPackageStartupMessages({
	library('psupertime')
	library('SingleCellExperiment')
})
knitr::opts_chunk$set(collapse = TRUE,comment = "#>",package.startup.message = FALSE)

# load the data Acinar cells: total 8 donors
data(acinar_hvg_sce)

write.csv(acinar_hvg_sce@assays[[".->data"]]@listData[["logcounts"]], file = "data_fromPsupertime/acinar_hvg_sce_X.csv", row.names = TRUE)
write.csv(acinar_hvg_sce$donor_age,file = "data_fromPsupertime/acinar_hvg_sce_Y.csv", row.names = FALSE)
```
## Embryonic beta cells 
Data can access on [GEO GSE87375](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87375) and the developmental stage of beta cells includes **E17.5, P0, P3, P9, P15, P18 and P60**. **We labelled E17.5 as -1**, which is an embryonic stage before the other stage, and **the other stage as 0, 3, 9, 15, 18, 60**. Then we pre-processed data by the pre-processing function from Psupertime.
```python
# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@Author  ：awa121
@Date    ：2023/9/24 21:05

"""

from Fig2_TemproalVAE_against_benchmark_methods.pypsupertime import Psupertime
import anndata
import pandas as pd
import os
import numpy as np

os.chdir('./Fig2_TemproalVAE_against_benchmark_methods')

def main():
    # for mouse embryonic beta cells dataset:
    result_file_name = "embryoBeta"
    data_org = pd.read_csv('data_fromPsupertime/GSE87375_Single_Cell_RNA-seq_Gene_TPM.txt', index_col=0, sep="\t").T
    # get beta cell
    cell_id = data_org.index[1:]
    cell_beta_id = [i for i in cell_id if i[0] == "b"]
    data_org.columns = data_org.iloc[0]
    data_org = data_org[1:]
    data_org = data_org.loc[:, ~data_org.columns.duplicated()]

    adata = anndata.AnnData(data_org)
    adata.var_names_make_unique()
    adata = adata[cell_beta_id].copy()
    temp_time = np.array(adata.obs_names)
    temp_time = [eval(i.split("_")[0].replace("bP", "").replace("bE17.5", "-1")) for i in temp_time] # note bE17.5 is before birth

    adata.obs["time"] = temp_time
    import scanpy as sc
    # sc.pl.highest_expr_genes(adata, n_top=20, )
    sc.pp.filter_genes(adata, min_cells=25)

    adata.var['ERCC'] = adata.var_names.str.startswith('ERCC-')  # annotate the group of mitochondrial genes as 'ERCC'
    adata = adata[:, ~adata.var.ERCC]
    adata.var['RP'] = adata.var_names.str.startswith('RP')  # annotate the group of mitochondrial genes as 'RP'
    adata = adata[:, ~adata.var.RP]

    # sc.pp.calculate_qc_metrics(adata, qc_vars=['ERCC'], percent_top=None, log1p=False, inplace=True)
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_ERCC'],jitter=0.4, multi_panel=True)
    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    preprocessing_params = {"select_genes": "hvg", "log": True}
    # to get hvg gene of humanGermline dataset
    tp = Psupertime(n_jobs=1, n_folds=5,preprocessing_params=preprocessing_params)
    adata_hvg = tp.preprocessing.fit_transform(adata.copy())
    del tp

    hvg_gene_df = pd.DataFrame(adata_hvg.var_names)
    hvg_gene_df = hvg_gene_df.rename(columns={'Symbol': 'gene_name'})
    hvg_gene_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index=True)

    x_df = data_org.loc[adata_hvg.obs_names]
    x_df = x_df[hvg_gene_df["gene_name"]]
    x_df = x_df.T
    x_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_X.csv', index=True)

    y_df = pd.DataFrame(adata_hvg.obs.time)
    y_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_Y.csv', index=True)

    print("Finish save files.")


if __name__ == '__main__':
    main()
```

## Human germline F 
Dataset can access on [GEO GSE86146](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86146). We downloaded the raw count dataset and selected Female cells and pre-processed data by the function from Psupertime. 

1 use following R code to combine .txt.gz files:
```r
setwd('./Fig2_TemproalVAE_against_benchmark_methods')
# make Human germline dataset ------------------------------------------------------------

folder_path <- "data_fromPsupertime/GSE86146"

file_list <- list.files(folder_path, pattern = "\\.txt\\.gz$", full.names = TRUE)

merged_data <- NULL
for (file_path in file_list) {
  data <- read.table(file_path, header = TRUE, sep = "\t", quote = "",row.names=1)

  if (is.null(merged_data)) {
    merged_data <- data
  } else {
    merged_data <- cbind(merged_data, data)
  }
}

write.csv(merged_data, file = "data_fromPsupertime/humanGermline_X.csv", row.names = TRUE)
# make label dataframe
new_data <- data.frame()

col_names <- colnames(merged_data)

for (i in 1:length(col_names)) {
  col_name <- col_names[i]
  col_name_parts <- unlist(strsplit(col_name, "_"))

  new_row <- data.frame(
    "sex" = col_name_parts[1],
    "time" = as.numeric(gsub("W", "", col_name_parts[2])),
    row.names = col_name
  )

  new_data <- rbind(new_data, new_row)
}

print(new_data)

write.csv(new_data,file = "data_fromPsupertime/humanGermline_Y.csv", row.names = TRUE)
```
2 Remove male cell，re-generate data file：
```python

# -*-coding:utf-8 -*-
"""
@Author  ：awa121
@Date    ：2023/9/22 18:20

"""

from Fig2_TemproalVAE_against_benchmark_methods.pypsupertime import Psupertime
import anndata
import pandas as pd
import os
import numpy as np

os.chdir('./Fig2_TemproalVAE_against_benchmark_methods')


def main():
    # for Human Germline dataset:
    result_file_name="humanGermline"
    data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
    data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv',index_col=0)
    # only use female cell
    data_y_df = data_y_df[data_y_df['sex'] != 'M']
    data_x_df=data_x_df.loc[data_y_df.index]
    data_y_df=data_y_df["time"]
    preprocessing_params = {"select_genes": "hvg", "log": True}

    # START HERE
    adata = anndata.AnnData(data_x_df)
    adata.obs["time"] = data_y_df
    print(f"Input Data: n_genes={adata.n_vars}, n_cells={adata.n_obs}")

    # to get hvg gene of humanGermline dataset
    tp = Psupertime(n_jobs=1, n_folds=5,
                    preprocessing_params=preprocessing_params)
    adata_hvg = tp.preprocessing.fit_transform(adata.copy())
    hvg_gene=pd.DataFrame(adata_hvg.var_names,columns=["gene_name"])
    hvg_gene.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index=True)

    x_df = data_x_df.loc[adata_hvg.obs_names]
    x_df = x_df[hvg_gene["gene_name"]]
    x_df = x_df.T
    x_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_X.csv', index=True)

    y_df = pd.DataFrame(adata_hvg.obs.time)
    y_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_Y.csv', index=True)

    print("Finish save files.")

if __name__ == '__main__':
    main()


```

