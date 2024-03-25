import json
import requests

import numpy as np
from statistics import mean
import enrichment
import pandas as pd
import utilsforminds.helpers as helpers
import utilsforminds
import group_finder
import other_models_tester
import load_dataset
import pickle

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # try:
    #     ip = requests.get("http://checkip.amazonaws.com/")
    # except requests.RequestException as e:
    #     # Send some context about this error to Lambda Logs
    #     print(e)

    #     raise e
    print("Lambda Function")

    with open("." + "/aws_test.txt", "a") as txtFile:
        txtFile.write('Lambda Write Result\n')
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "hello world",
            # "location": ip.text.replace("\n", "")
        }),
    }

def lambda_task():
    dataset_kind = "traffic"
    num_samples = 100

    base_dir = "."
    img_split = [2, 2]
    static_data, static_features_encoded, dynamic_data, dynamic_features_region, dynamic_imgs_path, target_labels, similarities, features_range_dict = load_dataset.load_austin(path_to_austin_image_info= f"{base_dir}/austin/austin_data_with_speeds.csv", path_to_images= f"{base_dir}/austin/Austin Images", path_to_satellite_images= f"{base_dir}/austin/austin_satellite_images/content/images", img_split= img_split, prediction_interval= 60, num_records = 5, num_bags = num_samples)

    data_dict_list = [
        {'target_to_predict': [target_labels], 'static_arr': static_data, "dynamic_arr_dict": dynamic_data},
    ]
    group_idc_lst = [[], []] ## In this example, 0~3-th features are in the first group, and 4~10-th features are in the second group.
    for i in range(len(static_features_encoded)):
        if static_features_encoded[i] != "satellite":
            group_idc_lst[0].append(i)
        else:
            group_idc_lst[1].append(i)
    dataset_specific_kwargs= {"static_features_encoded": static_features_encoded, "dynamic_features_region": dynamic_features_region, "dynamic_imgs_path": dynamic_imgs_path, "img_split": img_split, "similarities": similarities, "features_range_dict": features_range_dict}

    # gamma_lst = [1e-1, 1e-4, 1e-2, 1e-3, 1e-1, 1e-1, 1e-1, 1e-7, 1e-7]
    gamma_lst = [1e-5, 1e-5, 1e-4, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5]
    mu_init_val_lst = [10., 10., 1e+4, 1e+4]
    rho_dict = {'rho_1': 1.25, 'rho_2':1.25, 'rho_3':1.35, 'rho_4':1.35}

    # # %% Predict Last
    for data_dict in data_dict_list:
        memo= f"num_samples= {num_samples}"
        for mode in data_dict.keys():
            if mode[:5] != "ravlt" and data_dict[mode] is not None:
                memo= memo+ f"_{mode.split('_')[0]}"
        for p in [0.5]:
            enrichment_inst = enrichment.Enrichment(gamma_lst = gamma_lst, rho_dict = rho_dict, static_group_idc_lst = group_idc_lst, r_0 = 60, r_1 = 40, p = p, training_set_ratio = 0.8, reweight_gammas = True, reweight_mu = True, mu_init_val_lst = mu_init_val_lst, debug = True, pick_keep_probs_dict_dict= {'train': {-1: 0.5}, 'test': {-1: 0.5}}, baseline_idx= -1, dataset_specific_kwargs= dataset_specific_kwargs, memo= memo, if_save_array = True, dataset_kind= dataset_kind, two_loops = False, x_timepoint_idx= -1, y_timepoint_idx= -1, if_save_enrichment = True, use_similarities= (False and dataset_kind == "traffic"), **data_dict)

            # enrichment_inst.fit(keep_previous_ratio=0.8, limit_iter=[5, 10], use_before_ADMM = False, svr_kernel = "rbf", svr_params = {"gamma": 1 / 60.})
            enrichment_inst.fit(keep_previous_ratio=0., limit_iter=[2, 4], use_before_ADMM = False, auto_reweight_mu = False, svr_kernel = "poly", svr_params = {"gamma": 1 / 40., "coef0": 0., "degree": 2})

    dir_name = f"{enrichment_inst.new_directory_name}"
    enrichment_inst = pickle.load(open(f"{dir_name}/enrichment.pickle", "rb"))
    if enrichment_inst.dataset_kind == "traffic": add_dict = {"img_paths": enrichment_inst.img_paths}
    else: add_dict = {}

    other_models_tester_inst = other_models_tester.Other_models_tester(dir_name + '/data_array/X_all.npy', dir_name + '/data_array/target_arr_training.npy', dir_name + '/data_array/target_arr_test.npy', dir_name + '/W_array/W_all.npy', dir_name + '/info_dct.pkl', dir_name + '/data_array/target_arr_original.npy', debug = False, min_max_enriched = True, features_range_dict= enrichment_inst.dataset_specific_kwargs["features_range_dict"] if "features_range_dict" in enrichment_inst.dataset_specific_kwargs.keys() else None, add_dict= add_dict)
    other_models_tester_inst.lasso_predict()
    other_models_tester_inst.SVR_predict()
    other_models_tester_inst.ridge_predict()
    other_models_tester_inst.linearRegression_predict()
    
    if False: ## Post-visualize the enrichment results
        dir_num = 154
        dir_name = f'./out/result_{dir_num}'
        enrichment_inst = pickle.load(open(f"{dir_name}/enrichment.pickle", "rb"))
        enrichment.plot_results(enrichment= enrichment_inst, iteration_tot= -1)

if __name__ == "__main__":
    lambda_task()