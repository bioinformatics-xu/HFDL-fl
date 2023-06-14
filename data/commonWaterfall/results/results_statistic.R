

## table global results
best_criterion <- "Global_train_acc"


gm_list <- c(1,2,3)
drug_alg_row <- 1


# 按照指定的评价指标，找到最优的round， 提取特定gmma, algorithm, drug对应的准确率，macro-F1，weighted-F1
result_file <- paste("./gamma=",gm_list[1],"/results_global.csv",sep = "")
result_global <- read.csv(result_file, check.names = FALSE)
row_n <- length(gm_list)*length(unique(result_global$algorithm))*length(unique(result_global$drug_name))
global_statistic <- data.frame(gamma=rep(NA,row_n),algorithm=rep(NA,row_n),drug_name=rep(NA,row_n),
                               Global_test_acc=rep(NA,row_n),macro_F1=rep(NA,row_n),weighted_F1=rep(NA,row_n))
best_round_list <- data.frame(gamma=rep(gm_list,each=row_n/length(gm_list)),algorithm=rep(NA,row_n),drug_name=rep(NA,row_n),best_round=NA)

for (gm in gm_list) {
  result_file <- paste("./gamma=",gm,"/results_global.csv",sep = "")
  result_global <- read.csv(result_file, check.names = FALSE)

  for (drug in unique(result_global$drug_name)) {
    for (alg in unique(result_global$algorithm)) {
      result_drug_alg <- result_global[(result_global$drug_name %in% drug & result_global$algorithm %in% alg),]
      best_round <- result_drug_alg$round[which.max(result_drug_alg[[best_criterion]])]  
      print(c(drug, alg))
      global_statistic[drug_alg_row,] <- subset(result_drug_alg, select = c("gamma","algorithm","drug_name","Global_test_acc","macro_F1","weighted_F1"))[which(result_drug_alg$round == best_round),]
      best_round_list[drug_alg_row,] <- c(subset(result_drug_alg,select = c("gamma","algorithm","drug_name"))[which(result_drug_alg$round == best_round),],best_round)
      print(subset(result_drug_alg, select = c("gamma","algorithm","drug_name","Global_test_acc","macro_F1","weighted_F1"))[which(result_drug_alg$round == best_round),])
      drug_alg_row <- drug_alg_row + 1
    }
  }
}

# 过滤掉效果不好、不可用的两种药物
global_statistic_filter <- global_statistic[!global_statistic$drug_name %in% c("Dasatinib", "Docetaxel"),]

# 提取每个gamma参数及每种算法下所有数据集的平均准确率、macro-F1和weighted-F1，用于确定选择哪个gamma和哪个算法
alg_list <- sort(c("fedavg", "fednova", "fedprox","scaffold"))
gm_list <- c(1,2,3)
criterion_list <- c("Global_test_acc","macro_F1","weighted_F1")

gamma_summary <- data.frame(gamma=rep(gm_list,each=length(alg_list)),algorithm=rep(alg_list,length(gm_list)),Global_test_acc=rep(NA,length(gm_list)*length(alg_list)),
                            macro_F1=rep(NA,length(gm_list)*length(alg_list)),weighted_F1=rep(NA,length(gm_list)*length(alg_list)))

for (gm in c(1,2,3)) {
  results_gm <- global_statistic_filter[global_statistic_filter$gamma %in% gm,]
  for (ct in criterion_list) {
    avg_acc_gm <- tapply(results_gm[[ct]], results_gm$algorithm, FUN=mean)
    gamma_summary[which(gamma_summary$gamma %in% gm),which(colnames(gamma_summary) %in% ct)] <- avg_acc_gm[order(names(avg_acc_gm))]
  }
}


# 选择好gamma和算法后，比较联邦模型和每个client的准确率等指标
## table local results
gm=2
alg = "fedavg"
result_file <- paste("./gamma=",gm,"/results_clientdata.csv",sep = "")
result_client <- read.csv(result_file, check.names = FALSE)
result_client_best_gm_alg <- result_client[(result_client$gamma %in% 2) & (result_client$algorithm %in% alg),]

result_client_best_gm_alg <- result_client_best_gm_alg[!(result_client_best_gm_alg$drug_name %in% c("Dasatinib", "Docetaxel")),]
drug <- "PLX4720"
j=2
drug <- unique(result_client_best_gm_alg$drug_name)[j]

for (drug in unique(result_client_best_gm_alg$drug_name)){
  result_drug <- result_client_best_gm_alg[result_client_best_gm_alg$drug_name %in% drug,]
  best_round <- best_round_list$best_round[(best_round_list$gamma == gm)&(best_round_list$algorithm==alg)&(best_round_list$drug_name==drug)] 
  result_drug_best <- result_drug[result_drug$round==best_round,]
}

for (drug in unique(result_client_best_gm_alg$drug_name)) {
  for (alg in unique(result_client_best_gm_alg$algorithm)) {
    result_drug_alg <- result_client_best_gm_alg[(result_client_best_gm_alg$drug_name %in% drug & result_client_best_gm_alg$algorithm %in% alg),]
    best_round <- result_drug_alg$round[which.max(result_drug_alg[[best_criterion]])]  
    print(c(drug, alg))
    global_statistic[drug_alg_row,] <- subset(result_drug_alg, select = c("gamma","algorithm","drug_name","Global_test_acc","macro_F1","weighted_F1"))[which(result_drug_alg$round == best_round),]
    best_round_list[drug_alg_row,] <- c(subset(result_drug_alg,select = c("gamma","algorithm","drug_name"))[which(result_drug_alg$round == best_round),],best_round)
    print(subset(result_drug_alg, select = c("gamma","algorithm","drug_name","Global_test_acc","macro_F1","weighted_F1"))[which(result_drug_alg$round == best_round),])
    drug_alg_row <- drug_alg_row + 1
  }
}



GDSC <- read.csv("GDSC2_Expr_feature_common.csv", check.names = FALSE,row.names = 1)
response <- read.csv("GDSC2_response_class_common_waterfall.csv", check.names = FALSE,row.names=1)
GDSC_drug <- cbind(response[,1],GDSC)
colnames(GDSC_drug)[1] <- "res_flag" 
write.csv(GDSC_drug,file = "GDSC_AZD7762.csv")

