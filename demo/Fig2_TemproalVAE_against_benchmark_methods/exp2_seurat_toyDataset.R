rm(list = ls())
setwd("demo/Fig2_TemproalVAE_against_benchmark_methods")
library(Seurat)
print(packageVersion("Seurat"))
library(SeuratData)
library(ggplot2)
library(readr)

dataset_list = c("acinarHVG", "embryoBeta", "humanGermline")
for (dataset in dataset_list) {
    # 或者使用基本的 R 函数 read.csv
    data_x_df <- read.csv(sprintf('preprocessedData_ByPypsupertime_forSeurat/%s_X.csv', dataset), row.names = 1)
    data_x_df = t(data_x_df)
    data_y_df <- read_csv(sprintf('preprocessedData_ByPypsupertime_forSeurat/%s_Y.csv', dataset))

    #data_y_df <- as.numeric(data_y_df$time)
    seurat_object <- CreateSeuratObject(counts = as.matrix(data_x_df), project = "ExampleProject")
    seurat_object = UpdateSeuratObject(seurat_object)
    seurat_object$time = as.character(data_y_df$time)

    results_path <- file.path(getwd(), paste0("seurat", "_results"))

    # Check if the directory exists
    if (!dir.exists(results_path)) {
        # Create the directory
        dir.create(results_path)
    }

    donor_list <- unique(seurat_object$time)
    kFold_test_result_df <- data.frame(time = character(0), pseudotime = character(0))

    # Assuming 'adata' is a SingleCellExperiment object and 'donor_list' is available
    for (donor in donor_list) {
        print(donor)

        train_seurat_object <- subset(seurat_object, time != donor)
        print(dim(train_seurat_object))
        # Creating test data for the current donor
        test_seurat_object <- subset(seurat_object, time == donor)
        print(dim(test_seurat_object))
        # Move one cell from test to train to occupy the test time
        # Assuming 'metadata' is the slot where your data's metadata is stored
        one_test_cell <- test_seurat_object[, 1:5]  # Get first 5 cells
        print(dim(one_test_cell))
        test_seurat_object <- test_seurat_object[, 6:nrow(test_seurat_object)]  # Exclude first 5 cells
        print(dim(test_seurat_object))
        # Combining one_test_cell back into train_adata
        train_seurat_object <- merge(train_seurat_object, y = one_test_cell, add.cell.ids = c("Sample1", "Sample2"), project = "CombinedProject")
        print(dim(train_seurat_object)) # Concatenating data along columns


        train_seurat_object <- NormalizeData(train_seurat_object)
        train_seurat_object <- FindVariableFeatures(train_seurat_object)
        train_seurat_object <- ScaleData(train_seurat_object)

        if (dim(test_seurat_object)[2] < 6) {

            anchors <- FindTransferAnchors(reference = train_seurat_object,
                                           query = test_seurat_object,
                                           dims = 1:30,
                                           k.anchor = 4,
                                           k.score = 4)
            predictions <- TransferData(anchorset = anchors,
                                        refdata = train_seurat_object$time,
                                        k.weight = 4)
        }else {
            anchors <- FindTransferAnchors(reference = train_seurat_object,
                                           query = test_seurat_object,
                                           dims = 1:30, k.score = 10)
            predictions <- TransferData(anchorset = anchors,
                                        refdata = train_seurat_object$time,
                                        k.weight = 6)
        }


        test_seurat_object <- AddMetaData(test_seurat_object, metadata = predictions)
        test_seurat_object$prediction.match <- test_seurat_object$predicted.id == test_seurat_object$time
        table(test_seurat_object$prediction.match)

        temp_test_result_df <- data.frame(time = test_seurat_object$time,
                                          pseudotime = test_seurat_object$predicted.id,
                                          row.names = colnames(test_seurat_object))
        kFold_test_result_df = rbind(kFold_test_result_df, temp_test_result_df)


    }
    write.csv(kFold_test_result_df,
              sprintf('%s/%s_seurat_result.csv', results_path, dataset),
              row.names = TRUE,
              quote = FALSE)


}

preprocess_parameters <- function(dataset) {
    cat(sprintf("for dataset %s.\n", dataset))

    if (dataset %in% c("embryoBeta", "humanGermline")) {
        data_x_df <- read_csv(sprintf('data_fromPsupertime/%s_X.csv', dataset))
        row.names(data_x_df) <- data_x_df[[1]]
        data_x_df <- data_x_df[, -1]

        hvg_gene_list <- read_csv(sprintf('data_fromPsupertime/%s_gene_list.csv', dataset))
        data_x_df <- data_x_df[hvg_gene_list$gene_name,]

        data_y_df <- read_csv(sprintf('data_fromPsupertime/%s_Y.csv', dataset))
        data_y_df <- as.numeric(data_y_df$time)

        preprocessing_params <- list(select_genes = "all", log = TRUE)
    } else if (dataset == "acinarHVG") {
        data_x_df <- read_csv('data_fromPsupertime/acinar_hvg_sce_X.csv')
        row.names(data_x_df) <- data_x_df[[1]]
        data_x_df <- data_x_df[, -1]
        data_y_df <- read_csv('data_fromPsupertime/acinar_hvg_sce_Y.csv')
        data_y_df <- as.numeric(data_y_df$x)
        preprocessing_params <- list(select_genes = "all", log = FALSE)
    }
    library('psupertime')
    library('SingleCellExperiment')
    adata_org <- SingleCellExperiment(assays = list(counts = as.matrix(data_x_df)))
    adata_org$time = data_y_df
    cat(sprintf("Input Data: n_genes=%d, n_cells=%d\n", ncol(adata_org), nrow(adata_org)))

    return(list(adata = adata_org, data_x_df = data_x_df, data_y_df = data_y_df))
}
