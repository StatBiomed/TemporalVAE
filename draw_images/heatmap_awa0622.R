library(ggplot2)
library(reshape2)

# 读取CSV文件
data <- read.csv("/mnt/dandancao/public/for_yijun/avg_expr.csv", header = TRUE, row.names = 1)

# 转换数据框为矩阵
mat <- as.matrix(data)
top_gene=FindAllMarkers(object = data, only.pos = TRUE, 
               min.pct = 0.25, 
               thresh.use = 0.25)

library(dplyr) 
top3 <- sce.markers %>% group_by(cluster) %>% top_n(3, avg_log2FC)
DoHeatmap(sce ,top3$gene,size=3)
library(dplyr) 
top3 <- sce.markers %>% group_by(cluster) %>% top_n(3, avg_log2FC)
sce.all <- ScaleData(sce,features =  top3$gene)  
library(paletteer) 
color <- c(paletteer_d("awtools::bpalette"),
           paletteer_d("awtools::a_palette"),
           paletteer_d("awtools::mpalette"))
unique(sce.all$celltype)
ord = c('Naive CD4 T' ,'Memory CD4 T', 'CD8 T', 'NK', 
        'CD14+ Mono', 'FCGR3A+ Mono' ,'DC',  'B','Platelet')
sce.all$celltype = factor(sce.all$celltype ,levels = ord)
ll = split(top10$gene,top10$cluster)
ll = ll[ord]
rmg=names(table(unlist(ll))[table(unlist(ll))>1])
ll = lapply(ll, function(x) x[!x %in% rmg])
library(ggplot2)
DoHeatmap(sce.all,
          features = unlist(ll),
          group.by = "celltype",
          assay = 'RNA',
          group.colors = color,label = F)+
  scale_fill_gradientn(colors = c("white","grey","firebrick3"))

ggsave(filename = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/draw_images/marker_pheatmap.pdf",units = "cm",width = 36,height = 42)
