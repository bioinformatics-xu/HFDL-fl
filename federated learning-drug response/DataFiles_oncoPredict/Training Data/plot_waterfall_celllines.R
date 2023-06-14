library(ggplot2)
library(gridExtra)
library(readxl)
library(dplyr)
library(stats)
## GDSC data
# read data
GDSC2_Expr <- readRDS(file='GDSC2_Expr (RMA Normalized and Log Transformed).rds')
GDSC2_Res <- readRDS(file = 'GDSC2_Res.rds')
#GDSC2_Res <- exp(GDSC2_Res)

CGC_genes <- readxl::read_xlsx('Table-S3.xlsx',sheet = 'CGC')
GDSC2_Expr_CGC <- GDSC2_Expr
#GDSC2_Expr_CGC <- GDSC2_Expr[rownames(GDSC2_Expr) %in% CGC_genes$`Gene Symbol`,]

scale_01 <- function(x){
  x = (x-min(x))/(max(x)-min(x))
  return(x)
}


# record the na number
na_num <- data.frame(drug_name=colnames(GDSC2_Res), na_num = 0)
for (i in 1:ncol(GDSC2_Res)) {
  na_num$na_num[i]<- length(which(is.na(GDSC2_Res[,i]))) 
}

# remove the drugs missing more than 10% sample response
na_num_efficient <- na_num[which(na_num$na_num < round(0.1*nrow(GDSC2_Res))),]
na_num_efficient <- na_num_efficient[order(na_num_efficient$na_num),]

# combine the gene expression feature and the drug response labels
GDSC2_Expr_CGC <- t(GDSC2_Expr_CGC)
GDSC2_Expr_CGC <- GDSC2_Expr_CGC[order(rownames(GDSC2_Expr_CGC)),]
GDSC2_Res <- GDSC2_Res[order(rownames(GDSC2_Res)),]
GDSC2_Res_efficient <- GDSC2_Res[,colnames(GDSC2_Res) %in% na_num_efficient$drug_name[na_num_efficient$na_num < 50]]
GDSC2_Expr_CGC_efficient <- cbind(GDSC2_Expr_CGC,GDSC2_Res_efficient)

# compensate the response data 
# refer to the paper 
exp_dist <- as.matrix(dist(GDSC2_Expr_CGC))
K = 100
for (i in (ncol(GDSC2_Expr_CGC)+1):ncol(GDSC2_Expr_CGC_efficient)) {
  na_sample_label <- which(is.na(GDSC2_Expr_CGC_efficient[,i]))
  na_samle_name <- rownames(GDSC2_Expr_CGC_efficient)[na_sample_label]
  if(length(na_samle_name) > 0){
    for (j in 1:length(na_samle_name)){
      sample_j <- na_samle_name[j]
      sample_j_dist <- exp_dist[rownames(exp_dist) %in% sample_j,]
      #nearest_samples <- names(sample_j_dist)[order(sample_j_dist)[2:(K+1)]]
      dist_res_frame <- data.frame(dist = sample_j_dist[order(sample_j_dist)[2:(K+1)]], res = GDSC2_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i])
      dist_res_frame <- dist_res_frame[!is.na(dist_res_frame$res),]
      compensate_value <- sum((1/dist_res_frame$dist)*dist_res_frame$res)/sum(1/dist_res_frame$dist)
      GDSC2_Expr_CGC_efficient[na_sample_label[j],i] <- compensate_value
      #GDSC2_Expr_CGC_efficient[na_sample_label[j],i] <- mean(GDSC2_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i], na.rm = T)
      
    }
  }
}

GDSC2_Expr_CGC_feature <- GDSC2_Expr_CGC_efficient[,1:ncol(GDSC2_Expr_CGC)]
GDSC2_response <- GDSC2_Expr_CGC_efficient[,(ncol(GDSC2_Expr_CGC)+1):ncol(GDSC2_Expr_CGC_efficient)]

GDSC2_response_class <- GDSC2_response
GDSC2_response <- apply(GDSC2_response, 2, scale_01)
#hist(GDSC2_response[,8])
#GDSC2_response <- exp(GDSC2_response)

# Plot
drug_names <- c('AZD7762', 'PLX4720', 'Olaparib','Dasatinib', 'Docetaxel', 'Linsitinib','Fluorouracil')
drug_names_GDSC <- c('AZD7762_1022', 'PLX-4720_1036', 'Olaparib_1017', 'Dasatinib_1079', 'Docetaxel_1007', 'Linsitinib_1510', '5-Fluorouracil_1073')

GDSC2_target_drugs <- subset(GDSC2_response, select=drug_names_GDSC)

colnames(GDSC2_target_drugs) <- drug_names

filter_outliers <- function(dataVector){
  Q1 <- quantile(dataVector, 0.25)
  Q3 <- quantile(dataVector, 0.75)
  IQR <- Q3 - Q1
  dataVector_new <- dataVector[which(dataVector >= Q1 - 3*IQR & dataVector <= Q3 + 3*IQR)]
  return(dataVector_new)
}


# 创建数据框列表
df_list <- list(
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,1])), y = sort(filter_outliers(GDSC2_target_drugs[,1]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,2])), y = sort(filter_outliers(GDSC2_target_drugs[,2]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,3])), y = sort(filter_outliers(GDSC2_target_drugs[,3]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,4])), y = sort(filter_outliers(GDSC2_target_drugs[,4]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,5])), y = sort(filter_outliers(GDSC2_target_drugs[,5]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,6])), y = sort(filter_outliers(GDSC2_target_drugs[,6]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(GDSC2_target_drugs[,7])), y = sort(filter_outliers(GDSC2_target_drugs[,7]),decreasing = TRUE))
)
PCC_list <- c(round(cor(df_list[[1]]$x,df_list[[1]]$y),4),round(cor(df_list[[2]]$x,df_list[[2]]$y),4),
              round(cor(df_list[[3]]$x,df_list[[3]]$y),4),
              round(cor(df_list[[4]]$x,df_list[[4]]$y),4),round(cor(df_list[[5]]$x,df_list[[5]]$y),4),
              round(cor(df_list[[6]]$x,df_list[[6]]$y),4),round(cor(df_list[[7]]$x,df_list[[7]]$y),4))

# 绘制多个散点图并保存到列表中
p_list <- list()
for (i in seq_along(df_list)) {
  p_list[[i]] <- ggplot(df_list[[i]], aes(x = x, y = y)) +
    geom_point(color = "#1F3A93") +
    scale_y_continuous(limits = c(min(df_list[[i]]$y), max(df_list[[i]]$y))) +  # 设置y轴范围
    labs(x=colnames(GDSC2_target_drugs)[i],y="")+
    theme_bw()
}

# 将多个散点图拼接成一个大图
p_GDSC <- grid.arrange(grobs = p_list, nrow = 1)

ggsave("waterfallplot_GDSC.pdf", p_GDSC, width = 12, height = 3)


## CTRP data
# read data
CTRP_Expr <- readRDS(file='CTRP2_Expr (TPM, not log transformed).rds')
CTRP_Res <- readRDS(file = 'CTRP2_Res.rds')

CGC_genes <- readxl::read_xlsx('Table-S3.xlsx',sheet = 'CGC')
CTRP_Expr_CGC <- CTRP_Expr
#CTRP_Expr_CGC <- CTRP_Expr[rownames(CTRP_Expr) %in% CGC_genes$`Gene Symbol`,]


# record the na number
na_num <- data.frame(drug_name=colnames(CTRP_Res), na_num = 0)
for (i in 1:ncol(CTRP_Res)) {
  na_num$na_num[i]<- length(which(is.na(CTRP_Res[,i]))) 
}

# remove the drugs missing more than 10% sample response
na_num_efficient <- na_num[which(na_num$na_num < round(0.1*nrow(CTRP_Res))),]
na_num_efficient <- na_num_efficient[order(na_num_efficient$na_num),]

# combine the gene expression feature and the drug response labels
CTRP_Expr_CGC <- t(CTRP_Expr_CGC)
CTRP_Expr_CGC <- CTRP_Expr_CGC[order(rownames(CTRP_Expr_CGC)),]
CTRP_Res <- CTRP_Res[order(rownames(CTRP_Res)),]
CTRP_Res_efficient <- CTRP_Res[,colnames(CTRP_Res) %in% na_num_efficient$drug_name[na_num_efficient$na_num < 60]]
CTRP_Expr_CGC_efficient <- cbind(CTRP_Expr_CGC,CTRP_Res_efficient)

# compensate the response data 
# refer to the paper 
exp_dist <- as.matrix(dist(CTRP_Expr_CGC))
K = 100
for (i in (ncol(CTRP_Expr_CGC)+1):ncol(CTRP_Expr_CGC_efficient)) {
  na_sample_label <- which(is.na(CTRP_Expr_CGC_efficient[,i]))
  na_samle_name <- rownames(CTRP_Expr_CGC_efficient)[na_sample_label]
  if(length(na_samle_name) > 0){
    for (j in 1:length(na_samle_name)){
      sample_j <- na_samle_name[j]
      sample_j_dist <- exp_dist[rownames(exp_dist) %in% sample_j,]
      #nearest_samples <- names(sample_j_dist)[order(sample_j_dist)[2:(K+1)]]
      dist_res_frame <- data.frame(dist = sample_j_dist[order(sample_j_dist)[2:(K+1)]], res = CTRP_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i])
      dist_res_frame <- dist_res_frame[!is.na(dist_res_frame$res),]
      compensate_value <- sum((1/dist_res_frame$dist)*dist_res_frame$res)/sum(1/dist_res_frame$dist)
      CTRP_Expr_CGC_efficient[na_sample_label[j],i] <- compensate_value
      #GDSC2_Expr_CGC_efficient[na_sample_label[j],i] <- mean(GDSC2_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i], na.rm = T)
    }
  }
}

CTRP_Expr_CGC_feature <- CTRP_Expr_CGC_efficient[,1:ncol(CTRP_Expr_CGC)]
CTRP_response <- CTRP_Expr_CGC_efficient[,(ncol(CTRP_Expr_CGC)+1):ncol(CTRP_Expr_CGC_efficient)]
CTRP_response <- apply(CTRP_response, 2, scale_01)

# Plot
drug_names <- c('AZD7762', 'PLX4720', 'Olaparib','Dasatinib', 'Docetaxel', 'Linsitinib','Fluorouracil')
drug_names_CTRP <- c('AZD7762', 'PLX-4720', 'olaparib', 'dasatinib', 'docetaxel:tanespimycin (2:1 mol/mol)','linsitinib','fluorouracil')

CTRP_target_drugs <- subset(CTRP_response, select=drug_names_CTRP)
colnames(CTRP_target_drugs) <- drug_names

# 创建数据框列表
df_list <- list(
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,1])), y = sort(filter_outliers(CTRP_target_drugs[,1]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,2])), y = sort(filter_outliers(CTRP_target_drugs[,2]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,3])), y = sort(filter_outliers(CTRP_target_drugs[,3]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,4])), y = sort(filter_outliers(CTRP_target_drugs[,4]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,5])), y = sort(filter_outliers(CTRP_target_drugs[,5]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,6])), y = sort(filter_outliers(CTRP_target_drugs[,6]),decreasing = TRUE)),
  data.frame(x = 1:length(filter_outliers(CTRP_target_drugs[,7])), y = sort(filter_outliers(CTRP_target_drugs[,7]),decreasing = TRUE))
)
PCC_list <- c(round(cor(df_list[[1]]$x,df_list[[1]]$y),4),round(cor(df_list[[2]]$x,df_list[[2]]$y),4),
              round(cor(df_list[[3]]$x,df_list[[3]]$y),4),
              round(cor(df_list[[4]]$x,df_list[[4]]$y),4),round(cor(df_list[[5]]$x,df_list[[5]]$y),4),
              round(cor(df_list[[6]]$x,df_list[[6]]$y),4),round(cor(df_list[[7]]$x,df_list[[7]]$y),4))

# 绘制多个散点图并保存到列表中
p_list <- list()
for (i in seq_along(df_list)) {
  p_list[[i]] <- ggplot(df_list[[i]], aes(x = x, y = y)) +
    geom_point(color = "#1F3A93") +
    scale_y_continuous(limits = c(min(df_list[[i]]$y), max(df_list[[i]]$y))) +  # 设置y轴范围
    labs(x=colnames(CTRP_target_drugs)[i],y="")+
    theme_bw()
}

# 将多个散点图拼接成一个大图
p_CTRP <- grid.arrange(grobs = p_list, nrow = 1)

ggsave("waterfallplot_CTRP.pdf", p_CTRP, width = 12, height = 3)




