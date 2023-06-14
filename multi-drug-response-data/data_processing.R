
GDSC_Expr <- read.csv("./GDSC_DATASET_S1-S12/Table_S1_GDSC_Gene_expression.csv", check.names = FALSE)
GDSC_Res <- read.csv("./GDSC_DATASET_S1-S12/Table_S7_GDSC_Drug_response_AUC.csv", check.names = FALSE)

CCLE_Expr <- read.csv("./CCLE_DATASET_S13-S26/Table_S13_CCLE_Gene_expression.csv", check.names = FALSE)
CCLE_Res <- read.csv("./CCLE_DATASET_S13-S26/Table_S20_CCLE_Drug_response_AUC.csv", check.names = FALSE)

NCI60_Expr <- read.csv("./NCI60_DATASET_S27-S39/Table_S27_NCI60_Gene_expression.csv", check.names = FALSE)
NCI60_Res <- read.csv("./NCI60_DATASET_S27-S39/Table_S33_NCI60_Drug_response_GI50.csv", check.names = FALSE)


intersect(colnames(CCLE_Res),colnames(GDSC_Res))


# record the na number
na_num <- data.frame(drug_name=colnames(GDSC_Res), na_num = 0)
for (i in 1:ncol(GDSC_Res)) {
  na_num$na_num[i]<- length(which(is.na(GDSC_Res[,i]))) 
}

# remove the drugs missing more than 10% sample response
na_num_efficient <- na_num[which(na_num$na_num < round(0.1*nrow(GDSC_Res))),]
na_num_efficient <- na_num_efficient[order(na_num_efficient$na_num),]

# combine the gene expression feature and the drug response labels
#GDSC_Expr <- GDSC_Expr[order(rownames(GDSC_Expr)),]
#GDSC_Res <- GDSC_Res[order(rownames(GDSC_Res)),]
GDSC_Res_efficient <- GDSC_Res[,colnames(GDSC_Res) %in% na_num_efficient$drug_name[na_num_efficient$na_num <= 30]]
GDSC_Expr_efficient <- cbind(GDSC_Expr,GDSC_Res_efficient)

# compensate the response data 
# refer to the paper 
exp_dist <- as.matrix(dist(GDSC_Expr))
K = 100
for (i in (ncol(GDSC_Expr)+1):ncol(GDSC_Expr_efficient)) {
  na_sample_label <- which(is.na(GDSC_Expr_efficient[,i]))
  na_samle_name <- rownames(GDSC_Expr_efficient)[na_sample_label]
  if(length(na_samle_name) > 0){
    for (j in 1:length(na_samle_name)){
      sample_j <- na_samle_name[j]
      sample_j_dist <- exp_dist[rownames(exp_dist) %in% sample_j,]
      #nearest_samples <- names(sample_j_dist)[order(sample_j_dist)[2:(K+1)]]
      dist_res_frame <- data.frame(dist = sample_j_dist[order(sample_j_dist)[2:(K+1)]], res = GDSC_Expr_efficient[order(sample_j_dist)[2:(K+1)],i])
      dist_res_frame <- dist_res_frame[!is.na(dist_res_frame$res),]
      compensate_value <- sum((1/dist_res_frame$dist)*dist_res_frame$res)/sum(1/dist_res_frame$dist)
      GDSC_Expr_efficient[na_sample_label[j],i] <- compensate_value
      #GDSC2_Expr_CGC_efficient[na_sample_label[j],i] <- mean(GDSC2_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i], na.rm = T)
      
    }
  }
}

GDSC_Expr_feature <- GDSC_Expr_efficient[,1:ncol(GDSC_Expr)]
GDSC_response <- GDSC_Expr_efficient[,(ncol(GDSC_Expr)+1):ncol(GDSC_Expr_efficient)]

GDSC_response_class <- GDSC_response
res_median <- apply(GDSC_response, 2, median)
for (j in 1:ncol(GDSC_response)) {
  GDSC_response_class[which(GDSC_response[,j] >= as.numeric(res_median[j])),j] <- 0
  GDSC_response_class[which(GDSC_response[,j] < as.numeric(res_median[j])),j] <- 1
}

CGC_genes <- readxl::read_xlsx('Table-S3.xlsx',sheet = 'CGC')
GDSC_Expr_CGC_feature <- GDSC_Expr_feature[,colnames(GDSC_Expr_feature) %in% CGC_genes$`Gene Symbol`]

write.csv(GDSC_Expr_CGC_feature,"GDSC_Expr_CGC_feature.csv")
write.csv(GDSC_response,"GDSC_response.csv")
write.csv(GDSC_response_class,"GDSC_response_class.csv")


CTRP_Expr <- read.csv("./CTRP_DATASET_S40-S43/Table_S40_CTRP_Gene_expression.csv", check.names = FALSE)
CTRP_Res <- read.csv("./CTRP_DATASET_S40-S43/Table_S41_CTRP_Drug_response_AUC.csv", check.names = FALSE)

# record the na number
na_num <- data.frame(drug_name=colnames(CTRP_Res), na_num = 0)
for (i in 1:ncol(CTRP_Res)) {
  na_num$na_num[i]<- length(which(is.na(CTRP_Res[,i]))) 
}

# remove the drugs missing more than 10% sample response
na_num_efficient <- na_num[which(na_num$na_num < round(0.1*nrow(CTRP_Res))),]
na_num_efficient <- na_num_efficient[order(na_num_efficient$na_num),]

# combine the gene expression feature and the drug response labels
#GDSC_Expr <- GDSC_Expr[order(rownames(GDSC_Expr)),]
#GDSC_Res <- GDSC_Res[order(rownames(GDSC_Res)),]
CTRP_Res_efficient <- CTRP_Res[,colnames(CTRP_Res) %in% na_num_efficient$drug_name[na_num_efficient$na_num <= 30]]
CTRP_Expr_efficient <- cbind(CTRP_Expr,CTRP_Res_efficient)

# compensate the response data 
# refer to the paper 
exp_dist <- as.matrix(dist(CTRP_Expr))
K = 100
for (i in (ncol(CTRP_Expr)+1):ncol(CTRP_Expr_efficient)) {
  na_sample_label <- which(is.na(CTRP_Expr_efficient[,i]))
  na_samle_name <- rownames(CTRP_Expr_efficient)[na_sample_label]
  if(length(na_samle_name) > 0){
    for (j in 1:length(na_samle_name)){
      sample_j <- na_samle_name[j]
      sample_j_dist <- exp_dist[rownames(exp_dist) %in% sample_j,]
      #nearest_samples <- names(sample_j_dist)[order(sample_j_dist)[2:(K+1)]]
      dist_res_frame <- data.frame(dist = sample_j_dist[order(sample_j_dist)[2:(K+1)]], res = CTRP_Expr_efficient[order(sample_j_dist)[2:(K+1)],i])
      dist_res_frame <- dist_res_frame[!is.na(dist_res_frame$res),]
      compensate_value <- sum((1/dist_res_frame$dist)*dist_res_frame$res)/sum(1/dist_res_frame$dist)
      CTRP_Expr_efficient[na_sample_label[j],i] <- compensate_value
      #GDSC2_Expr_CGC_efficient[na_sample_label[j],i] <- mean(GDSC2_Expr_CGC_efficient[order(sample_j_dist)[2:(K+1)],i], na.rm = T)
      
    }
  }
}

CTRP_Expr_feature <- CTRP_Expr_efficient[,1:ncol(CTRP_Expr)]
CTRP_response <- CTRP_Expr_efficient[,(ncol(CTRP_Expr)+1):ncol(CTRP_Expr_efficient)]

CTRP_response_class <- CTRP_response
res_median <- apply(CTRP_response, 2, median)
for (j in 1:ncol(CTRP_response)) {
  CTRP_response_class[which(CTRP_response[,j] >= as.numeric(res_median[j])),j] <- 0
  CTRP_response_class[which(CTRP_response[,j] < as.numeric(res_median[j])),j] <- 1
}

CGC_genes <- readxl::read_xlsx('Table-S3.xlsx',sheet = 'CGC')
CTRP_Expr_CGC_feature <- CTRP_Expr_feature[,colnames(CTRP_Expr_feature) %in% CGC_genes$`Gene Symbol`]

write.csv(CTRP_Expr_CGC_feature,"CTRP_Expr_CGC_feature.csv")
write.csv(CTRP_response,"CTRP_response.csv")
write.csv(CTRP_response_class,"CTRP_response_class.csv")

common_drug <- intersect(colnames(GDSC_response_class),colnames(CTRP_response_class))
GDSC_response_class_common <- subset(GDSC_response_class, select = common_drug)
CTRP_response_class_common <- subset(CTRP_response_class, select = common_drug)

common_drug <- intersect(colnames(GDSC_response_class),colnames(CTRP_response_class))
GDSC_response_common <- subset(GDSC_response, select = common_drug)
CTRP_response_common <- subset(CTRP_response, select = common_drug)

write.csv(GDSC_response_common, "GDSC_response_common.csv")
write.csv(CTRP_response_common, "CTRP_response_common.csv")

common_CGC_genes <- intersect(colnames(GDSC_Expr_CGC_feature), colnames(CTRP_Expr_CGC_feature))
GDSC_Expr_CGC_feature_common <- subset(GDSC_Expr_CGC_feature, select = common_CGC_genes)
CTRP_Expr_CGC_feature_common <- subset(CTRP_Expr_CGC_feature, select = common_CGC_genes)

common_genes <- intersect(colnames(GDSC_Expr_feature), colnames(CTRP_Expr_feature))
GDSC_Expr_feature_common <- subset(GDSC_Expr_feature, select = common_genes)
CTRP_Expr_feature_common <- subset(CTRP_Expr_feature, select = common_genes)



write.csv(GDSC_Expr_feature_common, "GDSC_Expr_feature_common.csv")
write.csv(CTRP_Expr_feature_common, "CTRP_Expr_feature_common.csv")

write.csv(GDSC_response_class_common, "GDSC_response_class_common.csv")
write.csv(CTRP_response_class_common, "CTRP_response_class_common.csv")


## GDSC data
# read data
GDSC2_Expr <- readRDS(file='GDSC2_Expr (RMA Normalized and Log Transformed).rds')
GDSC2_Res <- readRDS(file = 'GDSC2_Res.rds')
GDSC2_Res <- exp(GDSC2_Res)

CGC_genes <- readxl::read_xlsx('Table-S3.xlsx',sheet = 'CGC')
GDSC2_Expr_CGC <- GDSC2_Expr
#GDSC2_Expr_CGC <- GDSC2_Expr[rownames(GDSC2_Expr) %in% CGC_genes$`Gene Symbol`,]


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
res_median <- apply(GDSC2_response, 2, median)
for (j in 1:ncol(GDSC2_response)) {
  GDSC2_response_class[which(GDSC2_response[,j] >= as.numeric(res_median[j])),j] <- 0
  GDSC2_response_class[which(GDSC2_response[,j] < as.numeric(res_median[j])),j] <- 1
}

GDSC2_Expr_CGC_feature <- GDSC2_Expr_CGC_feature[,colnames(GDSC2_Expr_CGC_feature) %in% CGC_genes$`Gene Symbol`]

write.csv(GDSC2_Expr_CGC_feature,"GDSC2_Expr_CGC_feature.csv")
write.csv(GDSC2_response,"GDSC2_response.csv")
write.csv(GDSC2_response_class,"GDSC2_response_class.csv")



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

CTRP_response_class <- CTRP_response
res_median <- apply(CTRP_response, 2, median)
for (j in 1:ncol(CTRP_response)) {
  CTRP_response_class[which(CTRP_response[,j] >= as.numeric(res_median[j])),j] <- 0
  CTRP_response_class[which(CTRP_response[,j] < as.numeric(res_median[j])),j] <- 1
}

CTRP_Expr_CGC_feature <- CTRP_Expr_CGC_feature[,colnames(CTRP_Expr_CGC_feature) %in% CGC_genes$`Gene Symbol`]

#write.csv(CTRP_Expr_CGC_feature,"CTRP_Expr_feature.csv")
write.csv(CTRP_Expr_CGC_feature,"CTRP_Expr_CGC_feature.csv")
write.csv(CTRP_response,"CTRP_response.csv")
write.csv(CTRP_response_class,"CTRP_response_class.csv")


CTRP_response_class_common <- subset(CTRP_response_class, select = c('AZD7762', 'PLX-4720', 'olaparib', 'dasatinib', 'docetaxel:tanespimycin (2:1 mol/mol)'))
colnames(CTRP_response_class_common) <- c('AZD7762', 'PLX4720', 'Olaparib', 'Dasatinib', 'Docetaxel')

GDSC2_response_class_common <- subset(GDSC2_response_class, select = c('AZD7762_1022', 'PLX-4720_1036', 'Olaparib_1017', 'Dasatinib_1079', 'Docetaxel_1007'))
colnames(GDSC2_response_class_common) <- c('AZD7762', 'PLX4720', 'Olaparib', 'Dasatinib', 'Docetaxel')

write.csv(CTRP_response_class_common,"CTRP_response_class_common.csv")
write.csv(GDSC2_response_class_common,"GDSC2_response_class_common.csv")

intersect_genes <- intersect(colnames(GDSC2_Expr_CGC_feature),colnames(CTRP_Expr_CGC_feature))
GDSC2_Expr_CGC_feature_common <- subset(GDSC2_Expr_CGC_feature, select = intersect_genes) 
CTRP_Expr_CGC_feature_common <- subset(CTRP_Expr_CGC_feature, select = intersect_genes) 

write.csv(CTRP_Expr_CGC_feature_common,"CTRP_Expr_CGC_feature_common.csv")
write.csv(GDSC2_Expr_CGC_feature_common,"GDSC2_Expr_CGC_feature_common.csv")



#AZD6482
#AZD7762
#PLX4720
#Olaparib
#Dasatinib
#Docetaxel

#CTRP_Expr_CGC_feature <- read.csv('CTRP_Expr_CGC_feature.csv',row.names = 1)
#CTRP_Expr_CGC_feature <- scale(CTRP_Expr_CGC_feature, center = T, scale = T)
#write.csv(CTRP_Expr_CGC_feature,'CTRP_Expr_CGC_feature_scale.csv')

# a = dist_res_frame$dist[1]
# b = dist_res_frame$dist[2]
# c = dist_res_frame$dist[3]
# ar = dist_res_frame$res[1]
# br = dist_res_frame$res[2]
# cr = dist_res_frame$res[3]
# ((1/a)*ar + (1/b)*br + (1/c)*cr)/(1/a+1/b+1/c)




