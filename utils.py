import json
import pandas as pd
import numpy as np

import os
import glob

from sklearn.utils import shuffle


from cfg import CFG_of_data


def data_parser(CFG_of_data, movie_dict, user_dict):
    """
    input1 : CFG info of data [dict]
    input2 : one user's feature data [dict]
    input3 : one user's info [dict] 
    
    output : one user's preprocessed feature [dict]
    """
    feature_dict = {}
    feature_list = CFG_of_data["use_feature"]

    ############### error handle ##############
    if "message" in movie_dict:
        return None

    ################ data_dict ################ 
    if "id" in feature_list:
        # str -> boolean
        feature_dict["id"] = movie_dict["id"]

    if "tmdb_id" in feature_list:
        # str -> boolean
        feature_dict["tmdb_id"] = movie_dict["tmdb_id"]

    # if "adult" in feature_list:
    #     # str -> boolean
    #     feature_dict["adult"] = (movie_dict["adult"] == "True")

    # if "genres" in feature_list:
    #     genres_id = [int(i['id']) for i in movie_dict["genres"]][:3]
        
    #     # genres_id.sort()
    #     feature_dict["genre1"] = genres_id[0] if len(genres_id) > 0 else "missing"
    #     feature_dict["genre2"] = genres_id[1] if len(genres_id) > 1 else "missing"
    #     feature_dict["genre3"] = genres_id[2] if len(genres_id) > 2 else "missing"

    # if "original_language" in feature_list:
    #     feature_dict["original_language"] = movie_dict["original_language"]

    # if "popularity" in feature_list:
    #     # str -> float
    #     feature_dict["popularity"] = float(movie_dict["popularity"])

    # if "release_date" in feature_list:
    #     # only use year
    #     year = movie_dict["release_date"].split("-")[0]
    #     feature_dict["release_date"] = int(year)

    # if "runtime" in feature_list:
    #     feature_dict["runtime"] = movie_dict["runtime"]

    # if "vote_average" in feature_list:
    #     # str -> float
    #     feature_dict["vote_average"] = float(movie_dict["vote_average"])

    # if "vote_count" in feature_list:
    #     # str -> int
    #     feature_dict["vote_count"] = int(movie_dict["vote_count"])

    ################ user_dict ################ 
    if "user_id" in feature_list:
        feature_dict["user_id"] = user_dict["user_id"]

    # if "age" in feature_list:
    #     feature_dict["age"] = user_dict["age"]

    # if "occupation" in feature_list:
    #     feature_dict["occupation"] = user_dict["occupation"]

    # if "gender" in feature_list:
    #     feature_dict["gender"] = user_dict["gender"]

    return feature_dict
        

def files_to_pandas(CFG_of_data, data_path, return_only_rate=False):
    """
    input1 : CFG info of data [dict]
    input2 : data folder path [str]
    input3 : return only rating row [bool]

    output : train data Dataframe [pandas]
    """
    feature_dict_list = []
    data_list = glob.iglob(data_path + '/movie/*.json')
    
    for movie_data_path in data_list:
        
        user_id = os.path.basename(movie_data_path).split(".json")[0].split("_")[-1]
        user_data_path = data_path + "/user/{}.json".format(user_id)
        movie_name = os.path.basename(movie_data_path).split(".json")[0][2:]
        rating_data_path = data_path + "/rating/R_{}.json".format(movie_name)
    
        rating_flag = os.path.isfile(rating_data_path)

        with open(movie_data_path, 'r') as f:
            movie_data = json.load(f)
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
        if rating_flag:
            with open(rating_data_path) as f:
                rating = int(json.load(f)['rate'][:1])
        else:
            if return_only_rate:
                continue
            rating = np.nan
        feature_dict = data_parser(CFG_of_data, movie_data, user_data)
        #### error handle ####
        if feature_dict == None:
            continue
        ######################
        feature_dict['rating'] = rating

        feature_dict_list.append(feature_dict)

    df = pd.DataFrame(feature_dict_list)
    column_for_drop = list(df.columns)
    column_for_drop.remove('rating') 
    # column_for_drop.remove('genre1')
    # column_for_drop.remove('genre2')
    # column_for_drop.remove('genre3')
    df.drop_duplicates(subset=column_for_drop, keep='last', inplace=True)

    return df
    

if __name__ == "__main__":
    # returne all
    df = files_to_pandas(CFG_of_data, "./data", True)
    df.to_csv("./data/data_for_pure.csv", index=False)

    portion_list = [25,50,75]
    total_len = len(df)
    df = shuffle(df)
    df.reset_index(inplace=True)
    for portion in portion_list:
        df_portion = df[:portion]
        df_portion.to_csv("./data/data_for_pure_{}%.csv".format(portion), index=False)
    

