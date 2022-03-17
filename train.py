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

def main():
    data = pd.read_csv("data/data_for_pure.csv", sep=",",
                       header=0).reset_index()
    data = data.rename(columns={"user_id": "user", "id": "item", "rating": "label"})
    data = data[["user", "item", "label"]]
    #data["time"] = 909090909

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(
        data, multi_ratios=[0.9, 0.05, 0.05])

    train_data, data_info = DatasetPure.build_trainset(train_data)

    eval_data = DatasetPure.build_evalset(eval_data)

    test_data = DatasetPure.build_testset(test_data)
    print(data_info)   # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    svdpp = SVDpp(task="rating", data_info=data_info, embed_size=16,
                n_epochs=10, lr=0.001, reg=None, batch_size=256)

    # monitor metrics on eval_data during training
    svdpp.fit(train_data, verbose=2, eval_data=eval_data,
              metrics=["rmse", "mae", "r2"])
    

    print("\n", "="*25, "Testing Model & Data", "="*25)
    data_info.save(path="model_path")
    svdpp.save(path="model_path", model_name="svdpp_model", manual=True,
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
