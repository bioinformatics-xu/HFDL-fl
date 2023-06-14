library(readxl)
## GDSC data
# read data
GDSC2_Expr <- readRDS(file='GDSC2_Expr (RMA Normalized and Log Transformed).rds')
GDSC2_Res <- readRDS(file = 'GDSC2_Res.rds')
#GDSC2_Res <- exp(GDSC2_Res)

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
#hist(GDSC2_response[,8])
GDSC2_response <- exp(GDSC2_response)

threshold_GDSC = 4
dist2d <- function(a,b,c) {
  v1 <- b - c
  v2 <- a - b
  m <- cbind(v1,v2)
  d <- abs(det(m))/sqrt(sum(v1*v1))
} 

for (j in 1:ncol(GDSC2_response)) {
  GDSC2_response_j <- GDSC2_response[,j]
  cor(c(1:length(sort(GDSC2_response_j))),sort(GDSC2_response_j))
  if(cor(c(1:length(sort(GDSC2_response_j))),sort(GDSC2_response_j))>0.95){
    b2 <- c(1,sort(GDSC2_response_j)[1])
    c2 <- c(length(GDSC2_response_j),sort(GDSC2_response_j)[length(GDSC2_response_j)])
    max_k <- 1
    d2_init <- 0
    for (k in 2:(length(GDSC2_response_j)-1)) {
      a2 <- c(k,sort(GDSC2_response_j)[k])
      d2 <- dist2d(a2,b2,c2)
      if(d2 > d2_init){
        max_k <- k
        d2_init <- d2
      }
    }
    median_j <- sort(GDSC2_response_j)[k]
    
    GDSC2_response_class[,j] <- 1
    GDSC2_response_class[which(GDSC2_response[,j] > median_j*threshold_GDSC),j] <- 0
    GDSC2_response_class[which(GDSC2_response[,j] < median_j/threshold_GDSC),j] <- 2
    if(length(which(GDSC2_response[,j] > median_j*threshold_GDSC))<length(GDSC2_response_j)*0.05){
      GDSC2_response_class[order(x=GDSC2_response_j,decreasing=TRUE)[1:length(GDSC2_response_j)*0.05],j] <- 0
    }
    if(length(which(GDSC2_response[,j] < median_j/threshold_GDSC))<length(GDSC2_response_j)*0.05){
      GDSC2_response_class[order(x=GDSC2_response_j,decreasing=FALSE)[1:length(GDSC2_response_j)*0.05],j] <- 2
    }
    
  }else{
    median_j <- median(GDSC2_response_j)
    
    GDSC2_response_class[,j] <- 1
    GDSC2_response_class[which(GDSC2_response[,j] > median_j*threshold_GDSC),j] <- 0
    GDSC2_response_class[which(GDSC2_response[,j] < median_j/threshold_GDSC),j] <- 2
    if(length(which(GDSC2_response[,j] > median_j*threshold_GDSC))<length(GDSC2_response_j)*0.05){
      GDSC2_response_class[order(x=GDSC2_response_j,decreasing=TRUE)[1:length(GDSC2_response_j)*0.05],j] <- 0
    }
    if(length(which(GDSC2_response[,j] < median_j/threshold_GDSC))<length(GDSC2_response_j)*0.05){
      GDSC2_response_class[order(x=GDSC2_response_j,decreasing=FALSE)[1:length(GDSC2_response_j)*0.05],j] <- 2
    }
  }

}

GDSC2_Expr_CGC_feature <- GDSC2_Expr_CGC_feature[,colnames(GDSC2_Expr_CGC_feature) %in% CGC_genes$`Gene Symbol`]

write.csv(GDSC2_Expr_CGC_feature,"GDSC2_Expr_CGC_feature.csv")
write.csv(GDSC2_response,"GDSC2_response.csv")
write.csv(GDSC2_response_class,"GDSC2_response_class_waterfall.csv")



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
hist(CTRP_response[,3])

CTRP_response_class <- CTRP_response
plot(x=seq(1,length(sort(CTRP_response[,3]))),y=sort(CTRP_response[,3]))

dist2d <- function(a,b,c) {
  v1 <- b - c
  v2 <- a - b
  m <- cbind(v1,v2)
  d <- abs(det(m))/sqrt(sum(v1*v1))
} 

threshold_CTRP = 1.2

scale_01 <- function(x){
  x = (x-min(x))/(max(x)-min(x))
  return(x)
}
CTRP_response_save <- CTRP_response
CTRP_response <- apply(CTRP_response, 2, scale_01)


for (j in 1:ncol(CTRP_response)) {
  CTRP_response_j <- CTRP_response[,j]
  if(cor(c(1:length(sort(CTRP_response_j))),sort(CTRP_response_j))>0.95){
    b2 <- c(1,sort(CTRP_response_j)[1])
    c2 <- c(length(CTRP_response_j),sort(CTRP_response_j)[length(CTRP_response_j)])
    d2_list = c()
    for (k in 2:(length(CTRP_response_j)-1)) {
      a2 <- c(k,sort(CTRP_response_j)[k])
      d2 <- dist2d(a2,b2,c2)
      d2_list <- c(d2_list,d2)
    }
    median_j <- sort(CTRP_response_j)[which.max(d2_list)+1]
    
    CTRP_response_class[,j] <- 1
    CTRP_response_class[which(CTRP_response[,j] > median_j*threshold_CTRP),j] <- 0
    CTRP_response_class[which(CTRP_response[,j] < median_j/threshold_CTRP),j] <- 2
    if(length(which(CTRP_response[,j] > median_j*threshold_CTRP))<length(CTRP_response_j)*0.05){
      CTRP_response_class[order(x=CTRP_response_j,decreasing=TRUE)[1:length(CTRP_response_j)*0.05],j] <- 0
    }
    if(length(which(CTRP_response[,j] < median_j/threshold_CTRP))<length(CTRP_response_j)*0.05){
      CTRP_response_class[order(x=CTRP_response_j,decreasing=FALSE)[1:length(CTRP_response_j)*0.05],j] <- 2
    }
    
  }else{
    print("non linear")
    median_j <- median(CTRP_response_j)
    
    CTRP_response_class[,j] <- 1
    CTRP_response_class[which(CTRP_response[,j] > median_j*threshold_CTRP),j] <- 0
    CTRP_response_class[which(CTRP_response[,j] < median_j/threshold_CTRP),j] <- 2
    if(length(which(CTRP_response[,j] > median_j*threshold_CTRP))<length(CTRP_response_j)*0.05){
      CTRP_response_class[order(x=CTRP_response_j,decreasing=TRUE)[1:length(CTRP_response_j)*0.05],j] <- 0
    }
    if(length(which(CTRP_response[,j] < median_j/threshold_CTRP))<length(CTRP_response_j)*0.05){
      CTRP_response_class[order(x=CTRP_response_j,decreasing=FALSE)[1:length(CTRP_response_j)*0.05],j] <- 2
    }
  }
  
}


CTRP_Expr_CGC_feature <- CTRP_Expr_CGC_feature[,colnames(CTRP_Expr_CGC_feature) %in% CGC_genes$`Gene Symbol`]

#write.csv(CTRP_Expr_CGC_feature,"CTRP_Expr_feature.csv")
write.csv(CTRP_Expr_CGC_feature,"CTRP_Expr_CGC_feature.csv")
write.csv(CTRP_response,"CTRP_response.csv")
write.csv(CTRP_response_class,"CTRP_response_class_waterfall.csv")


CTRP_response_class_common <- subset(CTRP_response_class, select = c('AZD7762', 'PLX-4720', 'olaparib', 'linsitinib','fluorouracil'))
#CTRP_response_class_common <- subset(CTRP_response_class, select = c('AZD7762', 'PLX-4720', 'olaparib', 'dasatinib', 'docetaxel:tanespimycin (2:1 mol/mol)','linsitinib','fluorouracil'))

colnames(CTRP_response_class_common) <- c('AZD7762', 'PLX4720', 'Olaparib', 'Linsitinib','Fluorouracil')


xxx <- cbind(CTRP_response_class_common[,colnames(CTRP_response_class_common) %in% "Fluorouracil"],CTRP_response_save[,colnames(CTRP_response_save) %in% "fluorouracil"])
xxxx <- xxx[order(xxx[,2]),]

GDSC2_response_class_common <- subset(GDSC2_response_class, select = c('AZD7762_1022', 'PLX-4720_1036', 'Olaparib_1017', 'Linsitinib_1510', '5-Fluorouracil_1073'))
#GDSC2_response_class_common <- subset(GDSC2_response_class, select = c('AZD7762_1022', 'PLX-4720_1036', 'Olaparib_1017', 'Dasatinib_1079', 'Docetaxel_1007', 'Linsitinib_1510', '5-Fluorouracil_1073'))
colnames(GDSC2_response_class_common) <- c('AZD7762', 'PLX4720', 'Olaparib', 'Linsitinib','Fluorouracil')


write.csv(CTRP_response_class_common,"CTRP_response_class_common_waterfall.csv")
write.csv(GDSC2_response_class_common,"GDSC2_response_class_common_waterfall.csv")

intersect_genes <- intersect(colnames(GDSC2_Expr_CGC_feature),colnames(CTRP_Expr_CGC_feature))
GDSC2_Expr_CGC_feature_common <- subset(GDSC2_Expr_CGC_feature, select = intersect_genes) 
CTRP_Expr_CGC_feature_common <- subset(CTRP_Expr_CGC_feature, select = intersect_genes) 

write.csv(CTRP_Expr_CGC_feature_common,"CTRP_Expr_CGC_feature_common.csv")
write.csv(GDSC2_Expr_CGC_feature_common,"GDSC2_Expr_CGC_feature_common.csv")

intersect_genes <- intersect(colnames(GDSC2_Expr_CGC),colnames(CTRP_Expr_CGC))
GDSC2_Expr_feature_common <- subset(GDSC2_Expr_CGC, select = intersect_genes) 
CTRP_Expr_feature_common <- subset(CTRP_Expr_CGC, select = intersect_genes) 

write.csv(CTRP_Expr_feature_common,"CTRP_Expr_feature_common.csv")
write.csv(GDSC2_Expr_feature_common,"GDSC2_Expr_feature_common.csv")


j=0
j=j+1
print(colnames(GDSC2_response_class_common)[j])
table(GDSC2_response_class_common[,j])
table(CTRP_response_class_common[,j])

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




