import hydra
import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure, DataInfo
from libreco.algorithms import SVDpp
# remove unnecessary tensorflow logging
import os
import sys
#file_dir = os.path.dirname(__file__)
sys.path.append(".")
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):

    # load data & split train/eval/test
    data = pd.read_csv(cfg.data_path, sep=",",
                       header=0).reset_index()
    data = data.rename(columns={"user_id": "user", "id": "item", "rating": "label"})
    data = data[["user", "item", "label"]]
    train_data, eval_data, test_data = random_split(
        data, multi_ratios=cfg.split_ratio)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    print(data_info)   # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    # load model
    cfg.model.data_info = data_info
    model = hydra.utils.instantiate(cfg.model)

    # train model
    model.fit(train_data, verbose=cfg.verbose, eval_data=eval_data,
              metrics=cfg.metrics)

    print("\n", "="*25, "Testing Model & Data", "="*25)
    data_info.save(path="model_path")
    model.save(path="model_path", model_name="svdpp_model", manual=True,
                inference_only=True)


    print("\n", "="*25, "Testing Load Model ", "="*25)
    # important to reset graph if model is loaded in the same shell.
    tf.compat.v1.reset_default_graph()
    # load data_info
    data_info = DataInfo.load("model_path")
    # load model, should specify the model name, e.g., DeepFM
    model = SVDpp.load(path="model_path", model_name="svdpp_model",
                        data_info=data_info, manual=True)

if __name__ == "__main__":
    main()

"""
├── configs              <- Hydra configuration files
│   ├── dataset          <- Dataset configs
│   │   ├── pure.yaml 
│   │   └── pure25.yaml 
│   │
│   ├── model            <- Model configs
│   │   ├── svdpp.yaml 
│   │   └── svd.yaml
│   │
│   └── config.yaml      <- Main config for training
│
└── train-hydra.py       <- Training file
"""
