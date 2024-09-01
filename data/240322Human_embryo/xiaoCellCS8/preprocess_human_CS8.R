rm(list = ls())
file_name="/mnt/yijun/nfs_share/awa_project/awa_github/TemporalVAE/data/240322Human_embryo/xiaoCellCS8/cs8_human_embryo.rds"
save_dir="/mnt/yijun/nfs_share/awa_project/awa_github/TemporalVAE/data/240322Human_embryo/xiaoCellCS8/"

print(save_dir)
print(file_name)
print(paste("preprocess for ", file_name, sep = " "))

if (!dir.exists(save_dir)) {
  dir.create(save_dir)
} else {
  print("Dir already exists!")
}


data_orginal = readRDS(file_name)

data_count = data_orginal[['RNA']]@counts
data_count <- as.data.frame(data_count)

cellname_vs_time = data.frame("cell" = colnames(data_orginal),
                              "clusters"=data_orginal@meta.data$clusters,
                              "time" = 18.5,
                              "x"=data_orginal@meta.data$x,
                              "y"=data_orginal@meta.data$y
                              )


write.table(
  data_count,
  paste(save_dir, "data_count.csv", sep = ""),
  row.names = TRUE,
  col.names = TRUE,
  sep = "\t"
)


write.table(
  cellname_vs_time,
  paste(save_dir, "cell_info.csv", sep = ""),
  row.names = FALSE,
  col.names = TRUE,
  sep = "\t"
)
print(paste("Save at", save_dir, sep = " "))
print(paste("Finish preprocess for ", file_name, sep = " "))
