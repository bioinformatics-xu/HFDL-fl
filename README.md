# HFDL-fl

## Introduction

Predicting anticancer drug sensitivity on distributed data sources using federated deep learning

We verified the feasibility of applying horizontal federated learning to collaborative drug sensitivity prediction based on distributed data sources. Combining the class imbalance property inherent to the prediction, a horizontal federated deep learning model with focal loss function was proposed, denoted as HFDL-fl.

## Access

HFDL-fl is free for non-commerical use only.

## Run

The run_DRP.py is to run HFDL-fl (experiments.py) on five drugs from GDSC and CTRP with gamma = 1,2,3 and aggregation algorithms fedavg, fednova, and scaffold.

## Data

The files in data directory are the demo data, which are not responsible for the performance. The preprocessed data form GDSC and CTRP databases can be download from the following link: https://1drv.ms/u/s!AoH29XiJLYEbdaJnPQj2vDRRBaw?e=3lggLg .


## Cite

If you find this repository useful, please cite paper:

Xu Xiaolu, Qi Zitong, Han Xiumei, Xu Aiguo, Geng Zhaohong, He Xinyu, Ren Yonggong, Duo Zhaojun. Predicting anticancer drug sensitivity on distributed data sources using federated deep learning. Heliyon (2023), 9(8).

## Reference

Li Qinbin, Diao Yiqun, Chen Quan, He Bingsheng. Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) (pp.965-978).

## Developer

Xiaolu Xu 

lu.xu@lnnu.edu.cn

Liaoning Normal University.

