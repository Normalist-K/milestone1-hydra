from argparse import ArgumentParser
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

def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    # dataset
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--split_ratio', default=[0.9, 0.05, 0.05])
    # SVDpp
    parser.add_argument('--task', type=str, default='rating')
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    # train
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--metrics', default=['rmse', 'mae', 'r2'])
    args = parser.parse_args(argv)
    return args

def main(args):
    # load data & split train/eval/test
    data = pd.read_csv(args.data_path, sep=",",
                       header=0).reset_index()
    data = data.rename(columns={"user_id": "user", "id": "item", "rating": "label"})
    data = data[["user", "item", "label"]]
    train_data, eval_data, test_data = random_split(
        data, multi_ratios=args.split_ratio)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    print(data_info)   # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    # load model
    svdpp = SVDpp(task=args.task, data_info=data_info, 
                  embed_size=args.embed_size,
                  n_epochs=args.n_epochs, lr=args.lr, 
                  reg=args.reg, batch_size=args.batch_size)
    # train model
    svdpp.fit(train_data, verbose=args.verbose, 
              eval_data=eval_data, metrics=args.metrics)


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
    main(parse_args())
