# HFDL-fl

## Introduction

Predicting anticancer drug sensitivity on distributed data sources using federated deep learning

We verified the feasibility of applying horizontal federated learning to collaborative drug sensitivity prediction based on distributed data sources. Combining the class imbalance property inherent to the prediction, a horizontal federated deep learning model with focal loss function was proposed, denoted as HFDL-fl.

## Reference
Xu Xiaolu, Qi Zitong, Han Xiumei, Xu Aiguo, Geng Zhaohong, He Xinyu, Ren Yonggong, Duo Zhaojun. Predicting anticancer drug sensitivity on distributed data sources using federated deep learning. Heliyon (2023), 9(8).

## Access

HFDL-fl is free for non-commerical use only.

## Run

The run_DRP.py is to run HFDL-fl (experiments.py) on five drugs from GDSC and CTRP with gamma = 1,2,3 and aggregation algorithms fedavg, fednova, and scaffold.

## Data

The files in data directory are the demo data, which are not responsible for the performance. The preprocessed data form GDSC and CTRP databases can be download from the following link: https://1drv.ms/u/s!AoH29XiJLYEbdaJnPQj2vDRRBaw?e=3lggLg .


## Cite

This project is developed based on Non-NIID, if you find this repository useful, please cite paper:

```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```

## Developer

Xiaolu Xu 

lu.xu@lnnu.edu.cn

Liaoning Normal University.

