library(Seurat)
library(ggplot2)
library(tidyverse)
obj=readRDS('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/draw_images/obj_5000.rds')
obj=subset(obj,subset=Minor.CellType!="Undetermined")
Idents(obj)=obj$Minor.CellType

gene=read.table('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/draw_images/gene.txt',header=T,sep="\t",quote='',stringsAsFactors = F)

vag_exp=AverageExpression(obj,assays = "RNA",features = gene$gene,group.by = "Minor.CellType",slot="data")

celltype=unique(FetchData(obj,vars = c("Major.CellType","Minor.CellType")))

celltype$Major.CellType=factor(celltype$Major.CellType,levels=c("T/NK cells","B lymphocytes",
                                                                "Myeloid cells","MAST cells",
                                                                "Endothelial cells","Epithelial cells","Fibroblasts"))
celltype=celltype[order(celltype$Major.CellType),]
celltype$Minor.CellType=factor(celltype$Minor.CellType,levels = unique(celltype$Minor.CellType))

gene=gene[!duplicated(gene$gene),]

gene$gene=factor(gene$gene,levels = gene$gene)

dat=base::apply(vag_exp$RNA,1,function(x) (x-mean(x))/sd(x))%>%t()%>%as.data.frame()%>%rownames_to_column('Gene')%>%reshape2::melt()
dat$Gene=factor(dat$Gene,levels = rev(unique(dat$Gene)))
dat$variable=factor(dat$variable,levels = levels(celltype$Minor.CellType))
dat=dat[order(dat$Gene),]

heatmap_color=RColorBrewer::brewer.pal(name = "RdBu",n=11)
#heatmap_color=heatmap_color[c(-11)]

pal=rev(colorRampPalette(heatmap_color)(500))
label1=levels(celltype$Minor.CellType)[seq(1,length(levels(celltype$Minor.CellType)),by=2)]
label2=levels(celltype$Minor.CellType)[seq(2,length(levels(celltype$Minor.CellType)),by=2)]

label1=levels(celltype$Minor.CellType)[seq(1,length(levels(celltype$Minor.CellType)),by=2)]
label2=levels(celltype$Minor.CellType)[seq(2,length(levels(celltype$Minor.CellType)),by=2)]

dat[dat$value>3,'value']=3
dat[dat$value < -2.5,'value']=-2.5
p1=ggplot(dat,aes(as.numeric(variable),Gene,fill=value))+
geom_tile()+
scale_y_discrete(expand = c(0,0))+
scale_x_continuous(expand = c(0,0),
                       breaks = seq(1,length(levels(celltype$Minor.CellType)),by=2),
                       labels = label1,
                      sec.axis =dup_axis(breaks = seq(2,length(levels(celltype$Minor.CellType)),by=2),
                                        labels = label2))+
    scale_fill_gradientn(colors=pal,limits=c(-2.5,3),name="Z Score")+
    geom_hline(yintercept=as.numeric(cumsum(rev(table(gene$group)[-1]))+.5),linetype=2)+
    geom_vline(xintercept=as.numeric(cumsum(table(celltype$Major.CellType))+.5),linetype=2)+
    theme(text = element_text(face="bold"),
          axis.title = element_blank(),
          axis.ticks=element_blank(),
         axis.text.y=element_blank(),
         axis.text.x.top=element_text(angle=90,hjust=0,vjust=.5),
         axis.text.x.bottom=element_text(angle=90,hjust=1,vjust=.5))
p2=ggplot(gene,aes(x,y,fill=group))+
geom_tile()+
geom_text(aes(label=gene),family="serif",fontface="italic")+
scale_y_continuous(expand = c(0,0))+
scale_fill_brewer(palette = "Pastel2")+
theme(text=element_text(),
      axis.text = element_blank(),
      axis.title =element_blank(),
      axis.ticks = element_blank(),
      legend.position="none")+
scale_x_continuous(expand = c(0,0))+geom_hline(yintercept = as.numeric(cumsum(rev(table(gene$group)[-1]/3)))+.5,color="white")

library(patchwork)

pic.heatmap=p2+p1+plot_layout(ncol = 2, widths  = c(1, 3))

pdf("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/draw_images/test.pdf",w=10.5,h=10)
pic.heatmap & theme(plot.margin = margin(0,0,0,0))
dev.off()
