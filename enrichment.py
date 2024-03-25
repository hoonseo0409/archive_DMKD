import numpy as np
import pandas as pd
import helpers
import copy
import scipy.linalg
from tqdm import tqdm
import os
import pandas
import inspect
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import csv
from copy import deepcopy
import other_models_tester
import utilsforminds
import scipy.optimize as optimize
from sklearn import svm
from numpy.linalg import inv, pinv
import utilsforminds
from random import sample
import plotly.express as px
import plotly.graph_objects as go
import shutil
import json
from time import time

@utilsforminds.decorators.redirect_function(utilsforminds.math)
def is_converged(loss_lst, consecutive_trends = 3, comparison_ratio = 1.0, check_start = 1, debug = False):
    """
        The smaller comparison_ratio and the larger consecutive_trends result the stricter checking.
    """
    #%% For debugging
    if debug and len(loss_lst) > 1:
        return True

    if len(loss_lst) < consecutive_trends + check_start + 1:
        return False
    else:
        gradient_avg = 0
        for i in range(check_start, len(loss_lst) - consecutive_trends):
            gradient_avg += abs(loss_lst[i + 1] - loss_lst[i])
        gradient_avg = gradient_avg / (len(loss_lst) - consecutive_trends - check_start)
        for past in range(1, consecutive_trends):
            if (abs(loss_lst[- past] - loss_lst[- past - 1]) > gradient_avg * comparison_ratio):
                return False
        return True

def trace_ADC(A, D, C):
    """Calculate A @ D @ C more efficiently.

    Examples
    --------
    (trace_ADC(np.array([[1, 2], [3, 4]]), np.array([[-1, 0], [0, 2]]), np.array([[1, 2], [2, 1]])))
        : 9.0
    """

    assert(A.shape[0] == C.shape[1] and A.shape[1] == D.shape[0] and D.shape[1] == C.shape[0])
    assert(D[1, 0] == 0. and D[0, 1] == 0.)
    summed = 0.
    # for i in range(A.shape[0]):
    #     summed += A[i, :] @ (B @ C[:, i])
    # diag = np.diag(D)
    # for i in range(A.shape[0]):
    #     summed += np.sum(A[i, :] * diag * C[:, i])
    # return summed

    tmp = D @ C
    for i in range(A.shape[0]):
        summed += A[i, :] @ tmp[:, i]
    return summed

def mu_reweight_condition(loss : list, threshold = 10e+10):
    if len(loss) < 2:
        return False
    elif loss[-1] * loss[-2] < 0: ## signs are different
        return False
    elif abs(loss[-1]) < threshold or abs(loss[-2]) < threshold:
        return False
    else:
        return True

class Enrichment(object):
    def __init__(self, static_arr, target_to_predict, gamma_lst, rho_dict, static_group_idc_lst, dynamic_arr_dict, r_0 = 30, r_1 = 30, p = 1.0, training_set_ratio = 0.8, init_transform = lambda x: x, reweight_gammas = False, reweight_mu = False, mu_init_val_lst = [1.0, 1.0, 100., 200.], debug = False, pick_keep_probs_dict_dict= {'train': {}, 'test': {}}, baseline_idx= 0, dataset_specific_kwargs= None, small_delta = 1e-20, memo= "", if_save_array = False, dataset_kind = "ADNI", two_loops = False, x_timepoint_idx= 0, y_timepoint_idx= 0, if_save_enrichment= False, use_similarities = False, if_exact = False):
        """
        
        Parameters
        ----------
        vbm_arr : numpy array
            MRI image with vbm mode. Shape is (the number of patients, length of MRI image vector, the number of timelines)
        fs_arr : numpy array
            MRI image with fs mode. Shape is (the number of patients, length of MRI image vector, the number of timelines)
        static_arr : numpy array
            DNA data as SNP. Shape is (the number of patients, length of DNA)
        target_to_predict : numpy array
            Cognitive scores. Shape is (the number of patients, the number of cognitive score modality, the number of timelines).
        p : float
            0.5 ~ 2.0 recommended, try 0.5 first.
        pick_keep_probs_dict_dict: dict
            Dictionary of probabilities to keep the x data on specific time point.
        y_timepoint_idx: int
            Index of time point of cognitive score labels known. For test sets, cognitive scores of all the time points are unknown.
        baseline_idx: int
            Specific time point to be regarded as baseline.
        """

        for key in ['rho_1', 'rho_2', "rho_3", "rho_4"]:
            assert(rho_dict[key] > 1 and rho_dict[key] < 2 and p > 0 and p <= 2)
        
        ## Set num_participants
        if static_arr is not None:
            num_participants = static_arr.shape[0]
        else:
            num_participants = next(iter(self.dynamic_arr_dict.values())).shape[0]
        for data in dynamic_arr_dict.values():
            assert(num_participants == data.shape[0]) ## Check data shape is consistent with num_participants.
        for target in target_to_predict:
            assert(target["data_arr"].shape[0] == num_participants)
        
        self.rho_dict = copy.deepcopy(rho_dict)
        self.p = p
        self.mu_init_val_lst = copy.deepcopy(mu_init_val_lst)
        self.training_set_ratio = training_set_ratio
        self.debug = debug
        self.if_save_array = if_save_array
        self.dataset_kind = dataset_kind
        self.dataset_specific_kwargs = dataset_specific_kwargs
        self.if_save_enrichment = if_save_enrichment
        self.use_similarities = use_similarities
        self.if_exact = if_exact

        self.two_loops = two_loops
        self.x_timepoint_idx = x_timepoint_idx
        self.y_timepoint_idx = y_timepoint_idx

        ## Merge target labels
        self.target_labels = []
        self.target_labels_readable_dict = {}
        self.target_arr = []
        for target in target_to_predict:
            self.target_labels = self.target_labels + target["name_raw"]
            for name_raw_idx in range(len(target["name_raw"])):
                self.target_labels_readable_dict[target["name_raw"][name_raw_idx]] = target["name_readable"][name_raw_idx]
            self.target_arr.append(target["data_arr"])
        self.target_arr = np.concatenate(self.target_arr, axis = 1)

        if static_arr is not None:
            static_reordered_idx = []
            self.group_split_idc_lst = [0]
            reordered_static_info_list = []

            if dataset_kind == "ADNI": 
                from outer_sources.clean_labels import VBMDataCleaner
                from outer_sources.roi_map import VBMRegionOfInterestMap
                from outer_sources.roi_map import FSRegionOfInterestMap
                static_info_excel_df = pd.read_excel(dataset_specific_kwargs["path_to_static_info_excel"], sheet_name = "CanSNPs_Top40Genes_Annotation")
            elif dataset_kind == "traffic":
                static_features_group = [(feature_encoded.split('_is')[0] if "_is" in feature_encoded else feature_encoded) for feature_encoded in self.dataset_specific_kwargs["static_features_encoded"]]

            split_idx = 0
            for idc_lst, group_idx in zip(static_group_idc_lst, range(len(static_group_idc_lst))): ## for indices(in original SNPs array) of each group
                split_idx += len(idc_lst) ## partition index
                self.group_split_idc_lst.append(split_idx) ## append partition index
                static_reordered_idx = static_reordered_idx + idc_lst ## append indices(in original static array) of this partition
                for idx in idc_lst:
                    if dataset_kind == "ADNI":
                        reordered_static_info_list.append([idx, group_idx] + list(static_info_excel_df.loc[idx, ["SNP", "chr", "AlzGene", "location"]]))
                    elif dataset_kind == "traffic":
                        reordered_static_info_list.append([idx, group_idx] + [self.dataset_specific_kwargs["static_features_encoded"][idx], static_features_group[idx]])

            #%% Reorder static data following groups
            if dataset_kind == "ADNI":
                col_names = ["index_original", "identified_group", "SNP", "chr", "AlzGene", "location"]
                self.reordered_statics_info_df = pd.DataFrame(reordered_static_info_list, columns = col_names)
                self.static_labels = self.reordered_statics_info_df['SNP'].tolist()
                series_obj = self.reordered_statics_info_df.apply(lambda x: True if x['chr'] != 'X' else False, axis = 1)
                self.idc_chr_not_X_list = list(series_obj[series_obj == True].index)
                self.reordered_statics_info_df = self.reordered_statics_info_df.loc[self.idc_chr_not_X_list, :]
                self.colors_list = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"]
                # alzgene_color_dict = {'MTHFR': "red", 'ECE1': 'navy', 'CHRNB2': 'lightgreen', 'CR1': 'coral', 'LDLR': 'lavender', 'IL1A': 'brown', 'IL1B': 'gold', 'BIN1': 'violet', 'CCR2': 'green', }
                # self.reordered_statics_info_df['color_chr'] = self.reordered_statics_info_df["chr"].apply(lambda x: self.colors_statics[x - 1])
                for group_column in ['chr', 'AlzGene', 'location', 'identified_group']:
                    self.reordered_statics_info_df = utilsforminds.helpers.add_column_conditional(self.reordered_statics_info_df, group_column, self.colors_list, new_column_name= group_column + "_colors")

                # self.reordered_statics_info_df['color_AlzGene'] = self.reordered_statics_info_df["AlzGene"].apply(lambda x: self.colors_statics[x - 1])
            elif dataset_kind == "traffic":
                self.reordered_statics_info_df = pd.DataFrame(reordered_static_info_list, columns = ["idx_encoded", "idx_original", "encoded", "original"])
                self.static_labels = self.reordered_statics_info_df['encoded'].tolist()
                self.colors_list = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"]
                for group_column in ["encoded", "original"]:
                    self.reordered_statics_info_df = utilsforminds.helpers.add_column_conditional(self.reordered_statics_info_df, group_column, self.colors_list, new_column_name= group_column + "_colors")

                
            
            self.static_reordered_idx = np.array(static_reordered_idx) ## 1204 length vector with indices re-arranged with the partitions(groups)
            self.static_arr = np.zeros(static_arr.shape)
            for i in range(self.static_arr.shape[0]):
                self.static_arr[i] = np.copy(static_arr[i, static_reordered_idx]) ## re-ordered snps along the partitions.
            self.is_static_given = True
        else:
            self.static_arr = np.zeros((num_participants, 2))
            self.group_split_idc_lst = [0, 2]
            self.is_static_given = False

        ## Shuffle the participants order
        self.idc_patients_shuffled = list(range(num_participants))
        np.random.shuffle(self.idc_patients_shuffled)
        self.dynamic_arr_dict = deepcopy(dynamic_arr_dict)
        for key in self.dynamic_arr_dict.keys():
            self.dynamic_arr_dict[key] = self.dynamic_arr_dict[key][self.idc_patients_shuffled]
        if static_arr is not None:
            self.static_arr = self.static_arr[self.idc_patients_shuffled]
        self.target_arr_tot = self.target_arr[self.idc_patients_shuffled]
        self.num_of_training_set = int(training_set_ratio * len(self.idc_patients_shuffled))
        self.target_arr_training = self.target_arr_tot[:self.num_of_training_set, :, :]
        self.target_arr_test = self.target_arr_tot[self.num_of_training_set:, :, :]
        self.Y_l = self.target_arr_training[:, :, y_timepoint_idx].transpose() # 0 for baseline

        #%% Set sizes
        if len(self.dynamic_arr_dict) > 0:
            self.num_followups = next(iter(self.dynamic_arr_dict.values())).shape[2]
        else:
            self.num_followups = 2
        self.delta = small_delta
        self.gamma_lst = copy.deepcopy(gamma_lst)
        self.gamma_lst.insert(0, "dummy") # to change order from 0, 1, 2, ... to 1, 2, 3, ...
        self.r_0 = r_0
        self.r_1 = r_1
        self.d_static = self.static_arr.shape[1]
        self.n = num_participants
        self.c = self.target_arr_tot.shape[1]
        self.l = self.target_arr_training.shape[0]
        static_arr_stacked = np.stack([self.static_arr] * self.num_followups, axis=2) # static_arr is constant respect to time, copy to the number of time stamps

        ## Set feature labels
        if "FS" in self.dynamic_arr_dict.keys():
            self.fs_labels = pd.read_csv(f"./data/adni_data/fs_atlas_labels.csv")
            self.fs_labels_only_names_arr = self.fs_labels.values[:, 0]
            self.fs_labels_only_names_list = self.fs_labels_only_names_arr.tolist()
        if "VBM" in self.dynamic_arr_dict.keys():
            vbm_cleaner = VBMDataCleaner()
            vbm_cleaner.load_data(f"./data/adni_data/longitudinal imaging measures_VBM_mod_final.xlsx")
            self.vbm_labels = vbm_cleaner.clean()
            # self.y_labels = ["BL_RAVLT_TOTAL", "BL_RAVLT30", "BL_RAVLT30_RECOG"]
            # self.y_labels_readable_dict = {'BL_RAVLT_TOTAL': 'RAVLT TOTAL', 'BL_RAVLT30': 'RAVLT 30', 'BL_RAVLT30_RECOG': 'RAVLT RECOG'}

        ## Concatenate modality
        self.X_all = []
        self.total_labels = []
        self.mode_length_dict = dict()
        self.mode_length_list = []
        self.is_dynamic_bools = []
        self.modality_name = ""
        mode_split_idx = 0
        for key in self.dynamic_arr_dict.keys():
            num_features = self.dynamic_arr_dict[key].shape[1]
            self.X_all.append(self.dynamic_arr_dict[key])
            if key == "FS":
                self.total_labels += self.fs_labels_only_names_list
            elif key == "VBM":
                self.total_labels += self.vbm_labels
            else:
                self.total_labels += ["Unknown" for i in range(num_features)]
            self.mode_length_dict[key] = [mode_split_idx, mode_split_idx + num_features]
            self.mode_length_list.append([mode_split_idx, mode_split_idx + num_features])
            mode_split_idx += num_features
            self.is_dynamic_bools.append(True)
            self.modality_name += f"{key} +"

        if static_arr is not None:
            num_features = self.static_arr.shape[1]
            self.X_all.append(static_arr_stacked)
            if self.dataset_kind in ["ADNI", "traffic"]:
                self.total_labels += self.static_labels
            else:
                self.total_labels += ["Unknown" for i in range(num_features)]
            self.mode_length_dict["static"] = [mode_split_idx, mode_split_idx + num_features]
            self.mode_length_list.append([mode_split_idx, mode_split_idx + num_features])
            mode_split_idx += num_features
            self.is_dynamic_bools.append(False)
            self.modality_name += f"{key} +"

        self.modality_name = self.modality_name[:-2]
        self.X_all = np.concatenate(self.X_all, axis= 1) ## (num_bags, dims, num_instances)
        self.total_labels = deepcopy(self.total_labels)

        self.d = self.X_all.shape[1]
        self.K = len(self.mode_length_list)

        if self.is_static_given and self.dataset_kind == "ADNI":
            for i in range(len(self.idc_chr_not_X_list)):
                self.idc_chr_not_X_list[i] += self.mode_length_dict['static'][0]

        ## Variable init
        if reweight_gammas:
            self.gamma_lst[1] = self.gamma_lst[1] / (self.n * self.d * self.num_followups)
            self.gamma_lst[2] = self.gamma_lst[2] / (max(1, sum(self.is_dynamic_bools)) * self.r_0 * self.n)
            self.gamma_lst[3] = self.gamma_lst[3] / (self.d_static * self.n)
            self.gamma_lst[4] = self.gamma_lst[4] / (self.K * self.r_1 * self.n)
            self.gamma_lst[5] = self.gamma_lst[5] / (self.c * self.l * self.l)
            self.gamma_lst[6] = self.gamma_lst[6] / (self.r_0 * self.K * self.r_1)
            self.gamma_lst[7] = self.gamma_lst[7] / (self.d_static * self.r_1)
            self.gamma_lst[8] = self.gamma_lst[8] / (self.d * self.r_0 * self.n)
            self.gamma_lst[9] = self.gamma_lst[9] / (self.d * self.r_0 * self.n)
            if self.use_similarities: self.gamma_lst[10] = self.gamma_lst[10] / (self.r_0 * self.r_0 * self.n)

        self.mu_dict = {}
        self.mu_dict['mu_1'] = [1.] * self.n
        self.mu_dict['mu_2'] = [1.] * self.n
        self.mu_dict['mu_3'] = [1.] * (self.K + 1)
        self.mu_dict['mu_4'] = [1.] * (self.K + 1)
        self.init_mu_dict(self.mu_init_val_lst)
        if reweight_mu:
            for i in range(self.n):
                self.mu_dict['mu_1'][i] = self.mu_dict['mu_1'][i] / (self.n * self.n)
                self.mu_dict['mu_2'][i] = self.mu_dict['mu_2'][i] / (self.d * self.n)
            for g in range(self.K):
                self.mu_dict['mu_4'][g] = self.mu_dict['mu_4'][g] / (self.r_1 * self.r_1)
                self.mu_dict['mu_3'][g] = self.mu_dict['mu_3'][g] / (self.r_0 * self.r_1)
        
        # self.X_all = self.X_all[self.idc_patients_shuffled] ## vbm, fs, vbm are already shuffled
        item_length_dict_dict= {'train': {}, 'test': {}}
        random_pick_items_dict_dict = {'train': {}, 'test': {}}
        for idx in range(self.num_of_training_set):
            item_length_dict_dict['train'][idx]= self.X_all.shape[2]
        for idx in range(self.num_of_training_set, self.n):
            item_length_dict_dict['test'][idx]= self.X_all.shape[2]
        random_pick_items_dict_dict['train']= helpers.random_pick_items(item_length_dict_dict['train'], pick_keep_probs_dict_dict['train'])
        random_pick_items_dict_dict['test']= helpers.random_pick_items(item_length_dict_dict['test'], pick_keep_probs_dict_dict['test'])
        for key in ['train', 'test']:
            random_pick_items_dict_dict[key] = {pid: sorted(random_pick_items_dict_dict[key][pid]) for pid in random_pick_items_dict_dict[key].keys()}

        if len(self.dynamic_arr_dict) > 0:
            ## self.X_all.shape== (412, 1314, 4)
            ## self.X_follows[0].shape== (1314, 4)
            ## self.X_baseline.shape== (1314, 412)
            ## self.X_SNP.shape== (1224, 412)
            self.X_follows= []
            for idx in range(self.num_of_training_set):
                self.X_follows.append(self.X_all[idx, :, random_pick_items_dict_dict['train'][idx]].transpose())
            for idx in range(self.num_of_training_set, self.n):
                self.X_follows.append(self.X_all[idx, :, random_pick_items_dict_dict['test'][idx]].transpose())
            self.X_baseline= []
            for idx in range(self.n):
                self.X_baseline.append(self.X_follows[idx][:, baseline_idx])
            self.X_baseline= np.stack(self.X_baseline, axis= 1)
            if self.debug and 0 not in pick_keep_probs_dict_dict['test'].keys():
                last_participant_idx= max(list(random_pick_items_dict_dict['test'].keys())) ## == self.n
                last_timepoint_idx= max(random_pick_items_dict_dict['test'][last_participant_idx])
                assert(np.allclose(self.X_follows[last_participant_idx][:, last_timepoint_idx], self.X_all[last_participant_idx, :, last_timepoint_idx]))
                assert(np.allclose(self.X_baseline[:, last_participant_idx], self.X_follows[last_participant_idx][:, baseline_idx]))
        else:
            self.X_baseline = self.X_all[:, :, 0].transpose()
            self.X_follows = []
            for i in range(self.n):
                self.X_follows.append(self.X_all[i, :, 1:])
        self.X_static = np.copy(self.static_arr).transpose()

        # self.init_mu_dict(self.mu_init_val_lst)
        # if reweight_mu:
        #     raise Exception("Deprecated option")
        #     self.mu_dict['mu_1'] = self.mu_dict['mu_1'] / (self.c * self.l)
        #     for i in range(self.n):
        #         self.mu_dict['mu_2'][i] = self.mu_dict['mu_2'][i] / (self.d * self.r_1 * self.r_1)
        #         self.mu_dict['mu_4'][i] = self.mu_dict['mu_4'][i] / (self.d * self.r_1)
        #     self.mu_dict['mu_3'] = self.mu_dict['mu_3'] / (self.r_2 * self.c)
            
        self.init_trainables(init_transform = init_transform)

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dict = {arg: values[arg] for arg in args}
        self.param_dict_str_init = helpers.paramDictToStr(param_dict)
        self.param_dict_str_init = self.param_dict_str_init + f"gamma_lst : {gamma_lst}\n"

        self.best_score_dict = None
        self.total_RMSE_dct_with_raw = None
        self.total_RMSE_dct_list = []

        #%% create dirs and save params info and meta data
        self.new_directory_name = helpers.getNewDirectoryName('out/', 'result_', root_dir = './')
        self.new_directory_name = './out/' + self.new_directory_name
        os.mkdir(self.new_directory_name + '/')
        os.mkdir(self.new_directory_name + '/W_array/')
        os.mkdir(self.new_directory_name + '/data_array/')
        os.mkdir(self.new_directory_name + '/plots/')
        os.mkdir(self.new_directory_name + '/statics/')
        os.mkdir(self.new_directory_name + '/features/')
        if self.dataset_kind == "traffic":
            os.mkdir(self.new_directory_name + '/dynamic/')
        elif self.dataset_kind == "ADNI":
            os.mkdir(self.new_directory_name + '/brain_maps/')
        if self.if_save_array:
            np.save(self.new_directory_name + '/data_array/target_arr_training.npy', self.target_arr_training)
            np.save(self.new_directory_name + '/data_array/target_arr_test.npy', self.target_arr_test)
            np.save(self.new_directory_name + '/data_array/target_arr_original.npy', self.target_arr)
        with open(self.new_directory_name + "/summary.txt", "a") as txtFile:
            txtFile.write('----- __init__() parameters -----\n')
            txtFile.write(self.param_dict_str_init)
            txtFile.write(f'--- memo ---\n{memo}\n')
        self.info_dct = {}
        self.info_dct['num_of_training_set'] = self.num_of_training_set
        self.info_dct['idc_patients_shuffled'] = self.idc_patients_shuffled
        self.info_dct['r_0'] = self.r_0
        self.info_dct['r_1'] = self.r_1
        self.info_dct['d'] = self.d
        self.info_dct['target_label'] = deepcopy(self.target_labels)
        self.info_dct['target_labels_readable_dict'] = deepcopy(self.target_labels_readable_dict)
        self.info_dct['static_group_split_idc_lst'] = self.group_split_idc_lst
        self.info_dct['dataset_kind'] = self.dataset_kind
        file_pkl = open(self.new_directory_name + '/info_dct.pkl', "wb")
        pickle.dump(self.info_dct, file_pkl)
        file_pkl.close()

        if self.dataset_kind == "traffic":
            self.img_paths = []
            for bag_idx in range(self.X_all.shape[0]):
                bag = {"static": self.X_static[:, bag_idx]}
                # for inst_idx in range(self.X_all[bag_idx].shape[0]):
                for modality in self.dataset_specific_kwargs["dynamic_imgs_path"].keys():
                    bag[modality] = self.dataset_specific_kwargs["dynamic_imgs_path"][modality][bag_idx]
                self.img_paths.append(bag)
            with open(self.new_directory_name + '/img_paths.pkl', 'wb') as handle:
                pickle.dump(self.img_paths, handle)
        print('init done')
    
    def fit(self, keep_previous_ratio = 0.0, limit_iter = [10, 100], use_before_ADMM = False, plot_factorization_prediction = False, auto_reweight_mu = False, auto_reweight_gamma = False, svr_kernel = "rbf", svr_params = {}):
        """
        
        SVR Kernel
        ----------
        svr_kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        svr_params :
            For poly: gamma = 1 / self.r_1, coef0 = 0., degree = 3
        """
        if self.debug: now = time()
        assert(keep_previous_ratio >= 0.0 and keep_previous_ratio < 1.0)
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dict = {arg: values[arg] for arg in args}
        with open(self.new_directory_name + "/summary.txt", "a") as txtFile:
            txtFile.write('----- fit() parameters -----\n')
            txtFile.write(helpers.paramDictToStr(param_dict))
        #%% Set temporary variables
        # WT_cross_X = np.zeros((self.r_1, self.n))
        # for i in range(self.n):
        #     WT_cross_X[:, i] = self.before['W'][i].transpose() @ self.X_baseline[:, i]

        self.W_p1p = np.concatenate(self.before['W'], axis=1)
        self.W_p2p = []
        for i in range(self.n):
            self.W_p2p.append(self.before['W'][i].transpose())
        self.W_p2p = np.concatenate(self.W_p2p, axis=1)

        #%% Set constant calculation
        self.XXT_lst = []
        self.x_i_x_iT_lst = []
        for i in range(self.n):
            self.XXT_lst.append(self.X_follows[i] @ self.X_follows[i].transpose())
            self.x_i_x_iT_lst.append(np.outer(self.X_baseline[:, i], self.X_baseline[:, i]))

        ## Set SVR similarity function
        if svr_kernel in ["rbf"]:
            def get_similarity_matrix(G):
                similarity_matrix = np.zeros((self.l, self.l))
                for i in range(self.l):
                    for j in range(self.l):
                        similarity_matrix[i, j] = np.sum((G[:, i] - G[:, j]) ** 2)
                return similarity_matrix
        elif svr_kernel in ["poly"]:
            def get_similarity_matrix(G):
                similarity_matrix = np.zeros((self.l, self.l))
                for i in range(self.l):
                    for j in range(self.l):
                        similarity_matrix[i, j] = np.sum(G[:, i] * G[:, j])
                return similarity_matrix
        else:
            raise Exception(NotImplementedError)
        ## Set SVR kernel
        if svr_kernel == "rbf":
            assert(svr_params["gamma"] > 0)
            def kernel_SVR(s):
                return np.exp(- svr_params["gamma"] * s)
        elif svr_kernel == "poly":
            def kernel_SVR(s):
                return (svr_params["gamma"] * s + svr_params["coef0"]) ** svr_params["degree"]
        else:
            raise Exception(NotImplementedError)

        #%% Init metric dictionaries
        iteration_tot = 0
        is_converged_smoothness = False
        iteration_smoothness = 0

        self.gammas_loss = [[] for i in range(len(self.gamma_lst))]
        self.gammas_loss[0] = "dummy"

        self.loss_dct_smoothness = {}
        self.loss_dct_smoothness[r'$objective$'] = []
        self.loss_dct_smoothness[r'$\Vert W_i^TW_i - I\Vert _F$'] = []
        self.loss_dct_smoothness[r'$\Vert B_i = W_i \Vert _F$'] = []
        self.loss_dct_smoothness[r'$\Vert H_g^TH_g - I\Vert _F$'] = []
        self.loss_dct_smoothness[r'$\Vert A_g = H_g \Vert _F$'] = []
        self.converged_check_dct_smoothness = {}
        self.converged_check_dct_smoothness['dual_converged'] = []
        self.converged_check_dct_smoothness['W_i^TW_i=I_converged'] = []
        self.converged_check_dct_smoothness['B_i=W_i_converged'] = []
        self.converged_check_dct_smoothness['H_g^TH_g=I_converged'] = []
        self.converged_check_dct_smoothness['A_g=H_g_converged'] = []

        is_converged_ADMM = False
        iteration_ADMM = 0
        self.loss_dct_ADMM = {}
        self.loss_dct_ADMM[r'$objective$'] = []
        self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'] = []
        self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'] = []
        self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'] = []
        self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'] = []

        self.loss_dct_ADMM_keep = deepcopy(self.loss_dct_ADMM)

        self.converged_check_dct_ADMM = {}
        self.converged_check_dct_ADMM['dual_converged'] = []
        self.converged_check_dct_ADMM['W_i^TW_i=I_converged'] = []
        self.converged_check_dct_ADMM['B_i=W_i_converged'] = []
        self.converged_check_dct_ADMM['H_g^TH_g=I_converged'] = []
        self.converged_check_dct_ADMM['A_g=H_g_converged'] = []
        if self.debug: print(f"Init took: {time() - now}.")

        while(not is_converged_smoothness or iteration_tot < limit_iter[0]):
            self.update_smoothness_matrices()
                            
            while(not is_converged_ADMM or (not self.two_loops and iteration_tot < limit_iter[0])):
                #%% Solve ADMM
                if use_before_ADMM:
                    self.update_ADMM(self.after, self.before, keep_previous_ratio = keep_previous_ratio, auto_reweight_mu = auto_reweight_mu, svr_kernel = svr_kernel, svr_params = svr_params)
                else:
                    self.update_ADMM(self.after, self.after, keep_previous_ratio = keep_previous_ratio, auto_reweight_mu = auto_reweight_mu, svr_kernel = svr_kernel, svr_params = svr_params)

                #%% Set temporary variables
                # for i in range(self.n):
                #     WT_cross_X[:, i] = self.after['W'][i].transpose() @ self.X_baseline[:, i]
                self.W_p1p = np.concatenate(self.after['W'], axis=1)
                self.W_p2p = []
                for i in range(self.n):
                    self.W_p2p.append(self.after['W'][i].transpose())
                self.W_p2p = np.concatenate(self.W_p2p, axis=1)
                # B_p2p = []
                # for i in range(self.n):
                #     B_p2p.append(self.after['B'][i].transpose())
                # B_p2p = np.concatenate(B_p2p, axis=1)

                #%% Calculating dual loss
                print("Print: ADMM Lagrangian loss")
                loss_dual = 0.
                tmp_loss = 0
                for i in range(self.n):
                    tmp_loss += self.gamma_lst[1] * trace_ADC((self.X_follows[i] - self.after['W'][i] @ self.after['W'][i].transpose() @ self.X_follows[i]).transpose(), self.diagonals['D_1'][i], (self.X_follows[i] - self.after['B'][i] @ self.after['B'][i].transpose() @ self.X_follows[i]))
                self.gammas_loss[1].append(tmp_loss)
                print(f'gamma_1 loss: {self.gammas_loss[1]}')
                loss_dual += tmp_loss
                
                tmp_loss = 0
                WkT_cross_Xk = self.WkT_cross_Xk(self.after['W'])
                for k in range(1, self.K + 1):
                    if self.is_dynamic_bools[k - 1]:
                        tmp_loss += self.gamma_lst[2] * trace_ADC((WkT_cross_Xk[k] - self.after['H'][k] @ self.after['G'][k]).transpose(), self.diagonals['D_2'][k], WkT_cross_Xk[k] - self.after['A'][k] @ self.after['G'][k])
                self.gammas_loss[2].append(tmp_loss)
                print(f'gamma_2 loss: {self.gammas_loss[2]}')
                loss_dual += tmp_loss

                # tmp = self.X_SNP - self.after['H'][0] @ self.after['G'][0]
                tmp_loss = self.gamma_lst[3] * trace_ADC((self.X_static - self.after['H'][0] @ self.after['G'][0]).transpose(), self.diagonals['D_3'], self.X_static - self.after['A'][0] @ self.after['G'][0])
                self.gammas_loss[3].append(tmp_loss)
                print(f'gamma_3 loss: {self.gammas_loss[3]}')
                loss_dual += tmp_loss

                tmp_loss = self.gamma_lst[4] * trace_ADC((self.after['G_m'] - self.after['G'][0]).transpose(), self.diagonals['D_4'][0], self.after['G_m'] - self.after['G'][0])
                for k in range(1, self.K + 1):
                    if self.is_dynamic_bools[k - 1]:
                        tmp = self.after['G_m'] - self.after['G'][k]
                        tmp_loss += self.gamma_lst[4] * trace_ADC(tmp.transpose(), self.diagonals['D_4'][k], tmp)
                self.gammas_loss[4].append(tmp_loss)
                print(f'gamma_4 loss: {self.gammas_loss[4]}')
                loss_dual += tmp_loss

                ## ADD SVR loss HERE ##
                tmp_loss = 0.
                for o in range(self.c):
                    tmp_loss += (self.gamma_lst[5] / 2) * self.after["alpha-alpha*"][o, :] @ kernel_SVR(get_similarity_matrix(self.after["G_m"][:, :self.l])) @ self.after["alpha-alpha*"][o, :].transpose()
                self.gammas_loss[5].append(tmp_loss)
                print(f'gamma_5 loss: {self.gammas_loss[5]}')
                loss_dual += tmp_loss

                tmp_loss = 0
                for k in range(1, self.K + 1):
                    if self.is_dynamic_bools[k - 1]:
                        tmp_loss += self.gamma_lst[6] * trace_ADC(self.after["H"][k], self.diagonals['D_6'][k], self.after["A"][k].transpose())
                self.gammas_loss[6].append(tmp_loss)
                print(f'gamma_6 loss: {self.gammas_loss[6]}')
                loss_dual += tmp_loss

                tmp_loss = self.gamma_lst[7] * trace_ADC(self.after["H"][0].transpose(), self.diagonals['D_7'], self.after["A"][0])
                self.gammas_loss[7].append(tmp_loss)
                print(f'gamma_7 loss: {self.gammas_loss[7]}')
                loss_dual += tmp_loss

                #%% Below two tmp calculations may require very large memory space.
                # tmp = W_p1p.transpose() @ self.diagonals['D_7'] @ W_p1p
                # tmp_loss = 0
                # tmp = self.gamma_lst[8] * self.W_p1p.transpose()
                # tmp_2 = self.diagonals['D_8'] @ self.W_p1p
                # for i in range(self.r_0 * self.n):
                #     # loss_primal += self.gamma_lst[6][0] * tmp[i, i]
                #     tmp_loss += tmp[i, :] @ tmp_2[:, i]

                tmp_loss = 0.
                tmp_diag = np.diagonal(self.diagonals['D_8']) * self.gamma_lst[8]
                for i in range(self.r_0 * self.n):
                    tmp_loss += np.sum(self.after["W"][i // self.r_0][:, i % self.r_0] * tmp_diag * self.after["B"][i // self.r_0][:, i % self.r_0])
                self.gammas_loss[8].append(tmp_loss)
                print(f"gamma_8 loss: {self.gammas_loss[8]}")
                loss_dual += tmp_loss
                
                # tmp = W_p2p.transpose() @ self.diagonals['D_8'] @ B_p2p
                # tmp_loss = 0
                # tmp = self.gamma_lst[9] * self.W_p2p.transpose()
                # tmp_2 = self.diagonals['D_9'] @ self.W_p2p
                # for i in range(self.d * self.n):
                #     # loss_primal += self.gamma_lst[6][1] * tmp[i, i]
                #     tmp_loss += tmp[i, :] @ tmp_2[:, i]

                tmp_loss = 0.
                tmp_diag = np.diagonal(self.diagonals["D_9"]) * self.gamma_lst[9]
                for i in range(self.d * self.n):
                    tmp_loss += np.sum(self.after["W"][i // self.d][i % self.d, :] * tmp_diag * self.after["B"][i // self.d][i % self.d, :])
                self.gammas_loss[9].append(tmp_loss)
                print(f'gamma_9 loss: {self.gammas_loss[9]}')
                loss_dual += tmp_loss

                if self.use_similarities:
                    tmp_loss = 0.
                    for i in range(self.n):
                        tmp = self.gamma_lst[10] * self.after["W"][i].transpose() @ self.X_follows[i] @ self.diagonals['D_10'][i] @ self.X_follows[i].transpose() @ self.after["W"][i]
                        for j in range(self.r_0):
                            tmp_loss += tmp[j, j]
                    self.gammas_loss[10].append(tmp_loss)
                    print(f'gamma_10 loss: {self.gammas_loss[10]}')
                    loss_dual += tmp_loss

                self.loss_dct_ADMM[r'$objective$'].append(loss_dual)
                
                #%% Calculating primal loss
                # Mean difference
                # self.loss_dct_ADMM['F_l=Y_l'].append(np.mean(np.abs(self.after['F'][:, :self.l] - self.Y_l)))
                # tmp = 0.
                # for i in range(self.n):
                #     tmp += np.mean(np.abs(self.after['W'][i].transpose() @ self.after['W'][i] - np.eye(self.r_1))) / self.d
                # self.loss_dct_ADMM['W_i^TW_i=I'].append(tmp / self.n)
                # self.loss_dct_ADMM['A=U'].append(np.mean(np.abs(self.after['A'] - self.after['U'])))
                # tmp = 0.
                # for i in range(self.n):
                #     tmp += np.mean(np.abs(self.after['B'][i] - self.after['W'][i]))
                # self.loss_dct_ADMM['B_i=W_i'].append(tmp / self.n)

                # Frobenious norm difference
                tmp = 0.
                for i in range(self.n):
                    tmp += np.sum((self.after['W'][i].transpose() @ self.after['W'][i] - np.eye(self.r_0)) ** 2) / self.d
                self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'].append(tmp)
                tmp = 0.
                for i in range(self.n):
                    tmp += np.sum((self.after['B'][i] - self.after['W'][i]) ** 2)
                self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'].append(tmp)
                tmp = 0.
                for g in range(0, self.K + 1):
                    if g == 0 or self.is_dynamic_bools[g - 1]:
                        tmp += np.sum((self.after['H'][g].transpose() @ self.after['H'][g] - np.eye(self.r_1)) ** 2)
                self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'].append(tmp)
                tmp = 0.
                for g in range(0, self.K + 1):
                    if g == 0 or self.is_dynamic_bools[g - 1]:
                        tmp += np.sum((self.after['A'][g] - self.after['H'][g]) ** 2)
                self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'].append(tmp)

                if auto_reweight_gamma and len(self.gammas_loss[1]) >= 2:
                    raise Exception("Deprecated")
                    avg_loss_diff = 0.
                    for i in range(1, len(self.gamma_lst)):
                        avg_loss_diff += abs(self.gammas_loss[i][-1] - self.gammas_loss[i][-2])
                    avg_loss_diff /= len(self.gamma_lst) - 1
                    for i in range(1, len(self.gamma_lst)):
                        if mu_reweight_condition(self.gammas_loss[i]):
                            self.gamma_lst[i] *= avg_loss_diff / abs(self.gammas_loss[i][-1] - self.gammas_loss[i][-2])
                for key in self.loss_dct_ADMM.keys():
                    self.loss_dct_ADMM_keep[key].append(self.loss_dct_ADMM[key][-1])

                if self.two_loops:
                    is_converged_params = {"consecutive_trends" : 1, "comparison_ratio" : 1.0}
                else:
                    is_converged_params = {"consecutive_trends" : 3, "comparison_ratio" : 1.0}
                #%% Check convergence for ADMM
                self.converged_check_dct_ADMM['dual_converged'].append(is_converged(self.loss_dct_ADMM[r'$objective$'], debug = self.debug, **is_converged_params))
                self.converged_check_dct_ADMM['W_i^TW_i=I_converged'].append(is_converged(self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'], debug = self.debug, **is_converged_params))
                self.converged_check_dct_ADMM['B_i=W_i_converged'].append(is_converged(self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'], debug = self.debug, **is_converged_params))
                self.converged_check_dct_ADMM['H_g^TH_g=I_converged'].append(is_converged(self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'], debug = self.debug, **is_converged_params))
                self.converged_check_dct_ADMM['A_g=H_g_converged'].append(is_converged(self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'], debug = self.debug, **is_converged_params))

                ## copy after to before
                # for i in range(self.n):
                #     self.before['W'][i] = np.copy(self.after['W'][i])
                #     self.before['B'][i] = np.copy(self.after['B'][i])
                #     self.before['Lambda_2_lst'][i] = np.copy(self.after['Lambda_2_lst'][i])
                #     self.before['Lambda_4_lst'][i] = np.copy(self.after['Lambda_4_lst'][i])
                # self.before['U'] = np.copy(self.after['U'])
                # self.before['F'] = np.copy(self.after['F'])
                # self.before['H_1'] = np.copy(self.after['H_1'])
                # self.before['H_2'] = np.copy(self.after['H_2'])
                # self.before['G_1'] = np.copy(self.after['G_1'])
                # self.before['G_2'] = np.copy(self.after['G_2'])
                # self.before['A'] = np.copy(self.after['A'])
                # self.before['Lambda_1'] = np.copy(self.after['Lambda_1'])
                # self.before['Lambda_3'] = np.copy(self.after['Lambda_3'])
                self.before = deepcopy(self.after)

                iteration_ADMM += 1
                iteration_tot += 1
                print(f'\n ADMM Iteration {iteration_ADMM} is done.')

                for key in self.loss_dct_ADMM.keys():
                    print(f'{key}: {self.loss_dct_ADMM[key]}')
                for key in self.converged_check_dct_ADMM.keys():
                    print(f'{key}: {self.converged_check_dct_ADMM[key][-1]}')
                is_converged_ADMM = True
                for key in self.converged_check_dct_ADMM.keys():
                    if not self.converged_check_dct_ADMM[key][-1]:
                        is_converged_ADMM = False
                if not self.two_loops:
                    break
            
            if self.two_loops:
                self.init_mu_dict(self.mu_init_val_lst)
                self.loss_dct_smoothness[r'$objective$'].append(self.loss_dct_ADMM[r'$objective$'][-1])
                self.loss_dct_smoothness[r'$\Vert W_i^TW_i - I\Vert _F$'].append(self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'][-1])
                self.loss_dct_smoothness[r'$\Vert B_i = W_i \Vert _F$'].append(self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'][-1])
                self.loss_dct_smoothness[r'$\Vert H_g^TH_g - I\Vert _F$'].append(self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'][-1])
                self.loss_dct_smoothness[r'$\Vert A_g = H_g \Vert _F$'].append(self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'][-1])

                #%% Check convergence for smoothness
                self.converged_check_dct_smoothness['dual_converged'].append(is_converged(self.loss_dct_smoothness[r'$objective$'], consecutive_trends = 1, comparison_ratio = 2.0, debug = self.debug))
                self.converged_check_dct_smoothness['W_i^TW_i=I_converged'].append(is_converged(self.loss_dct_smoothness[r'$\Vert W_i^TW_i - I\Vert _F$'], consecutive_trends = 1, comparison_ratio = 2.0, debug = self.debug))
                self.converged_check_dct_smoothness['B_i=W_i_converged'].append(is_converged(self.loss_dct_smoothness[r'$\Vert B_i = W_i \Vert _F$'], consecutive_trends = 1, comparison_ratio = 2.0, debug = self.debug))
                self.converged_check_dct_smoothness['H_g^TH_g=I_converged'].append(is_converged(self.loss_dct_smoothness[r'$\Vert H_g^TH_g - I\Vert _F$'], consecutive_trends = 1, comparison_ratio = 2.0, debug = self.debug))
                self.converged_check_dct_smoothness['A_g=H_g_converged'].append(is_converged(self.loss_dct_smoothness[r'$\Vert A_g = H_g \Vert _F$'], consecutive_trends = 1, comparison_ratio = 2.0, debug = self.debug))

                is_converged_ADMM = False
                iteration_ADMM = 0
                self.loss_dct_ADMM = {}
                self.loss_dct_ADMM[r'$objective$'] = []
                self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'] = []
                self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'] = []
                self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'] = []
                self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'] = []
                self.converged_check_dct_ADMM['dual_converged'] = []
                self.converged_check_dct_ADMM['W_i^TW_i=I_converged'] = []
                self.converged_check_dct_ADMM['B_i=W_i_converged'] = []
                self.converged_check_dct_ADMM['H_g^TH_g=I_converged'] = []
                self.converged_check_dct_ADMM['A_g=H_g_converged'] = []

                iteration_smoothness += 1
                print(f'\n smoothness Iteration {iteration_smoothness} is done.')
                for key in self.loss_dct_smoothness.keys():
                    print(f'{key}: {self.loss_dct_smoothness[key]}')
                for key in self.converged_check_dct_smoothness.keys():
                    print(f'{key}: {self.converged_check_dct_smoothness[key][-1]}')
                is_converged_smoothness = True
                for key in self.converged_check_dct_smoothness.keys():
                    if not self.converged_check_dct_smoothness[key][-1]:
                        is_converged_smoothness = False

            if is_converged_ADMM or (self.two_loops and is_converged_smoothness) or iteration_tot == limit_iter[1]:
                if self.if_save_enrichment: pickle.dump(self, file = open(self.new_directory_name + "/enrichment.pickle", "wb"))
                total_RMSE_dct = plot_results(enrichment = self, iteration_tot= iteration_tot)

            if iteration_tot >= limit_iter[0] and ((not self.two_loops and is_converged_ADMM) or (self.two_loops and is_converged_smoothness) or (iteration_tot >= limit_iter[1])):
                with open(self.new_directory_name + "/best.txt", "a") as txtFile:
                    for label_mode in total_RMSE_dct.keys():
                        txtFile.write(f'--- {label_mode} ---\n')
                        for metric in total_RMSE_dct[label_mode].keys():
                            if metric not in ['models', 'std']:
                                txtFile.write(f'- {metric} -\n')
                                for model, idx in zip(total_RMSE_dct[label_mode]['models'], range(len(total_RMSE_dct[label_mode]['models']))):
                                    txtFile.write(f"{model}: mean: {self.best_score_dict[label_mode][metric][model]['mean']}, std: {self.best_score_dict[label_mode][metric][model]['std']}\n")
                # np.save(self.new_directory_name + '/W_array/W_all_end.npy', np.stack(self.after['W'], axis = 0))
                print(f'Enrichment Training Finished, result saved to {self.new_directory_name}')
                break
    
    def init_mu_dict(self, mu_init_val_lst = [1., 1., 1., 1.]):
        self.mu_dict = {}
        self.mu_dict['mu_1'] = [mu_init_val_lst[0]] * self.n
        self.mu_dict['mu_2'] = [mu_init_val_lst[1]] * self.n
        self.mu_dict['mu_3'] = [mu_init_val_lst[2]] * (self.K + 1)
        self.mu_dict['mu_4'] = [mu_init_val_lst[3]] * (self.K + 1)
    
    def init_trainables(self, init_transform = lambda x: x):
        self.before = {}
        self.before['W'] = []
        self.before['B'] = []
        self.before['A'] = [init_transform(np.random.rand(self.d_static, self.r_1))] # A_0
        self.before["Lambda_1"] = []
        self.before["Lambda_2"] = []
        self.before["Lambda_3"] = [init_transform(np.random.rand(self.r_1, self.r_1))] # Lambda_3[0]
        self.before["Lambda_4"] = [init_transform(np.random.rand(self.d_static, self.r_1))] # Lambda_4[0]
        for patientsIdx in range(self.n):
            self.before['W'].append(init_transform(np.random.rand(self.d, self.r_0)))
            self.before['B'].append(init_transform(np.random.rand(self.d, self.r_0)))
            self.before["Lambda_1"].append(init_transform(np.random.rand(self.r_0, self.r_0)))
            self.before["Lambda_2"].append(init_transform(np.random.rand(self.d, self.r_0)))
        self.before['H'] = [init_transform(np.random.rand(self.d_static, self.r_1))] # H_0
        self.before['G'] = [init_transform(np.random.rand(self.r_1, self.n))] # G_0
        for k in range(self.K):
            if self.is_dynamic_bools[k]:
                self.before['H'].append(init_transform(np.random.rand(self.r_0, self.r_1)))
                self.before['G'].append(init_transform(np.random.rand(self.r_1, self.n)))
                self.before["A"].append(init_transform(np.random.rand(self.r_0, self.r_1)))
                self.before["Lambda_3"].append(init_transform(np.random.rand(self.r_1, self.r_1)))
                self.before["Lambda_4"].append(init_transform(np.random.rand(self.r_0, self.r_1)))
            else:
                self.before['H'].append("dummy")
                self.before['G'].append("dummy")
                self.before["A"].append("dummy")
                self.before["Lambda_3"].append("dummy")
                self.before["Lambda_4"].append("dummy")
        # self.before['H_0'] = init_transform(np.random.rand(self.d_static, self.r_1))
        # self.before['G_0'] = init_transform(np.random.rand(self.r_1, self.n))
        self.before["G_m"] = init_transform(np.random.rand(self.r_1, self.n))
        self.before["alpha-alpha*"] = init_transform(np.random.rand(self.c, self.l))
        
        self.after = {}
        self.after['W'] = []
        self.after['B'] = []
        self.after['A'] = [init_transform(np.random.rand(self.d_static, self.r_1))] # A_0
        self.after["Lambda_1"] = []
        self.after["Lambda_2"] = []
        self.after["Lambda_3"] = [init_transform(np.random.rand(self.r_1, self.r_1))] # Lambda_3[0]
        self.after["Lambda_4"] = [init_transform(np.random.rand(self.d_static, self.r_1))] # Lambda_4[0]
        for patientsIdx in range(self.n):
            self.after['W'].append(init_transform(np.random.rand(self.d, self.r_0)))
            self.after['B'].append(init_transform(np.random.rand(self.d, self.r_0)))
            self.after["Lambda_1"].append(init_transform(np.random.rand(self.r_0, self.r_0)))
            self.after["Lambda_2"].append(init_transform(np.random.rand(self.d, self.r_0)))
        self.after['H'] = [init_transform(np.random.rand(self.d_static, self.r_1))] # H_0
        self.after['G'] = [init_transform(np.random.rand(self.r_1, self.n))] # G_0
        for k in range(self.K):
            if self.is_dynamic_bools[k]:
                self.after['H'].append(init_transform(np.random.rand(self.r_0, self.r_1)))
                self.after['G'].append(init_transform(np.random.rand(self.r_1, self.n)))
                self.after["A"].append(init_transform(np.random.rand(self.r_0, self.r_1)))
                self.after["Lambda_3"].append(init_transform(np.random.rand(self.r_1, self.r_1)))
                self.after["Lambda_4"].append(init_transform(np.random.rand(self.r_0, self.r_1)))
            else:
                self.after['H'].append("dummy")
                self.after['G'].append("dummy")
                self.after["A"].append("dummy")
                self.after["Lambda_3"].append("dummy")
                self.after["Lambda_4"].append("dummy")
        # self.after['H_0'] = init_transform(np.random.rand(self.d_static, self.r_1))
        # self.after['G_0'] = init_transform(np.random.rand(self.r_1, self.n))
        self.after["G_m"] = init_transform(np.random.rand(self.r_1, self.n))
        self.after["alpha-alpha*"] = init_transform(np.random.rand(self.c, self.l))

        if not self.is_static_given:
            self.gamma_lst[3] = 0.
            self.gamma_lst[7] = 0.
            self.mu_dict["mu_3"][0] = self.delta
            self.mu_dict["mu_4"][0] = self.delta
        
        self.diagonals = {}
        self.diagonals['D_1'] = []
        for i in range(self.n):
            self.diagonals['D_1'].append(np.zeros((self.d, self.d)))

        self.diagonals['D_4'] = [np.zeros((self.r_1, self.r_1))] # For G_0, add one more.
        self.diagonals['D_2'] = ["dummy"]
        self.diagonals['D_6'] = ["dummy"]
        for mode_idx in range(self.K):
            if self.is_dynamic_bools[mode_idx]:
                self.diagonals['D_4'].append(np.zeros((self.r_1, self.r_1)))
            else:
                self.diagonals['D_4'].append("dummy")
            self.diagonals['D_2'].append(np.zeros((self.r_0, self.r_0)))
            self.diagonals['D_6'].append(np.zeros((self.r_1, self.r_1)))
        self.diagonals['D_3'] = np.zeros((self.d_static, self.d_static))
        self.diagonals['D_7'] = np.zeros((self.d_static, self.d_static))
        self.diagonals['D_8'] = np.zeros((self.d, self.d))
        self.diagonals['D_9'] = np.zeros((self.r_0, self.r_0))
        if self.use_similarities:
            self.diagonals['D_10'] = []
            for i in range(self.n):
                self.diagonals['D_10'].append(np.zeros((self.X_follows[i].shape[1], self.X_follows[i].shape[1])))

        self.converged_check_dct = {}
        self.converged_check_dct['dual_converged'] = []
        self.converged_check_dct['W_i^TW_i=I_converged'] = []
        self.converged_check_dct['B_i=W_i_converged'] = []
        self.converged_check_dct['H_g^TH_g=I_converged'] = []
        self.converged_check_dct['A_g=H_g_converged'] = []

    def update_smoothness_matrices(self):
        if self.debug: now = time()
        ## Set temporaries
        WkT_cross_Xk_temp = self.WkT_cross_Xk(self.before['W'])
        ## Set diagonals
        for i in range(self.n):
            tmp = (self.X_follows[i] - self.before['W'][i] @ self.before['W'][i].transpose() @ self.X_follows[i]) ** 2.
            for j in range(self.d):
                self.diagonals['D_1'][i][j, j] = (self.p / 2) * (np.sum(tmp[j, :]) + self.delta) ** (self.p/2 - 1)
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                tmp = (WkT_cross_Xk_temp[k] - self.before["H"][k] @ self.before["G"][k]) ** 2.
                for j in range(self.r_0):
                    self.diagonals['D_2'][k][j, j] = (self.p / 2) * (np.sum(tmp[j, :]) + self.delta) ** (self.p/2 - 1)
        tmp = (self.X_static - self.before["H"][0] @ self.before["G"][0]) ** 2.
        for j in range(self.d_static):
            self.diagonals['D_3'][j, j] = (self.p / 2) * (np.sum(tmp[j, :]) + self.delta) ** (self.p / 2 - 1)
        tmp = (self.before["G_m"] - self.before["G"][0]) ** 2.
        for j in range(self.r_1):
            self.diagonals['D_4'][0][j, j] = (self.p / 2) * (np.sum(tmp[j, :]) + self.delta) ** (self.p / 2 - 1)
        for g in range(1, self.K + 1):
            if self.is_dynamic_bools[g - 1]:
                tmp = (self.before["G_m"] - self.before["G"][g - 1]) ** 2.
                for j in range(self.r_1):
                    self.diagonals['D_4'][g][j, j] = (self.p / 2) * (np.sum(tmp[j, :]) + self.delta) ** (self.p / 2 - 1)
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                tmp = self.before["H"][k] ** 2.
                for j in range(self.r_1):
                    self.diagonals['D_6'][k][j, j] = (self.p / 2) * (np.sum(tmp[:, j]) + self.delta) ** (self.p / 2 - 1)
        for i in range(len(self.group_split_idc_lst) - 1):
            # tmp = (self.p / 2) * (helpers.get_norm_from_matrix(self.before['H'][0][self.group_split_idc_lst[i]:self.group_split_idc_lst[i + 1], :], 2, 2) + self.delta) ** (self.p / 2 - 1)
            tmp = (self.p / 2) * (helpers.get_norm_from_matrix(self.before['H'][0][self.group_split_idc_lst[i]:self.group_split_idc_lst[i + 1], :], 2, 2) ** 2. + self.delta) ** (self.p / 2 - 1)
            for j in range(self.group_split_idc_lst[i], self.group_split_idc_lst[i + 1]):
                self.diagonals['D_7'][j, j] = tmp
        
        #%% Without thresholding
        # self.diagonals['D_7'] = (self.p / 2) * (W_p1p @ W_p1p.transpose() + self.delta * np.eye(self.d)) ** (self.p / 2 - 1)
        # self.diagonals['D_8'] = (self.p / 2) * (W_p2p @ W_p2p.transpose() + self.delta * np.eye(self.r_1)) ** (self.p / 2 - 1)

        #%% zero thresholding to avoid power on negative number
        # tmp = W_p1p @ W_p1p.transpose()
        # self.diagonals['D_7'] = (self.p / 2) * (np.where(tmp >= 0., tmp, self.delta) + self.delta * np.eye(self.d)) ** (self.p / 2 - 1)
        # tmp = W_p2p @ W_p2p.transpose()
        # self.diagonals['D_8'] = (self.p / 2) * (np.where(tmp >= 0., tmp, self.delta) + self.delta * np.eye(self.r_1)) ** (self.p / 2 - 1)

        #%% D_7, D_8 calculation using exponent with SVD
        self.diagonals['D_8'] = (self.p / 2) * helpers.p_exponent_matrix(self.W_p1p @ self.W_p1p.transpose() + self.delta * np.eye(self.d), (self.p / 2 - 1))
        self.diagonals['D_9'] = (self.p / 2) * helpers.p_exponent_matrix(self.W_p2p @ self.W_p2p.transpose() + self.delta * np.eye(self.r_0), (self.p / 2 - 1))
        
        ## D_10
        if self.use_similarities:
            for i in range(self.n):
                n_i = self.X_follows[i].shape[1]
                WX_i = self.before["W"][i].transpose() @ self.X_follows[i]
                sim_weighted = np.zeros((n_i, n_i))
                sim_weighted_sum = np.zeros((n_i, n_i))
                for j in range(n_i):
                    for k in range(n_i):
                        sim_weighted[j, k] = (self.p / 2) * (np.sum((WX_i[:, j] - WX_i[:, k]) ** 2.) + self.delta) ** (self.p / 2 - 1) * self.dataset_specific_kwargs["similarities"][i][j][k]
                    sim_weighted_sum[j, j] = np.sum(sim_weighted[j, :])
                self.diagonals['D_10'][i] = sim_weighted_sum - sim_weighted

        if self.debug: print(f"Diagonal smoothness took: {time() - now}.")
        print("Diagonal matrices are updated.")

    def update_ADMM(self, to, by, keep_previous_ratio = 0., auto_reweight_mu = False, svr_kernel = "rbf", svr_params = {}): # after <= after VS after <= before
        """to: after, by: before
        
        """

        ## Update H_k
        WkT_cross_Xk_temp = self.WkT_cross_Xk(by["W"])
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                to["H"][k] = inv(self.mu_dict["mu_3"][k] * by["A"][k] @ by["A"][k].transpose() + self.mu_dict["mu_4"][k] * np.eye(self.r_0)) @ (self.gamma_lst[2] * self.diagonals["D_2"][k] @ (WkT_cross_Xk_temp[k] - by["A"][k] @ by["G"][k]) @ by["G"][k].transpose() - by['A'][k] @ (self.gamma_lst[6] * self.diagonals["D_6"][k] + by["Lambda_3"][k].transpose()) + (self.mu_dict["mu_3"][k] + self.mu_dict["mu_4"][k]) * by["A"][k] + by["Lambda_4"][k])

        ## Update H_0
        if self.is_static_given:
            # to["H_0"] = scipy.linalg.solve_sylvester((self.gamma_lst[7] / self.gamma_lst[3]) * np.linalg.inv(self.diagonals["D_3"]) @ self.diagonals["D_7"], by["G_0"] @ by["G_0"].transpose(), self.X_static @ by["G_0"].transpose())
            to["H"][0] = inv(self.mu_dict["mu_3"][0] * by["A"][0] @ by["A"][0].transpose() + self.mu_dict["mu_4"][0] * np.eye(self.d_static)) @ (self.gamma_lst[3] * self.diagonals["D_3"] @ (self.X_static - by["A"][0] @ by["G"][0]) @ by["G"][0].transpose() + ((self.mu_dict["mu_4"][0] + self.mu_dict["mu_3"][0]) * np.eye(self.d_static) - self.gamma_lst[7] * self.diagonals["D_7"]) @ by["A"][0] - by["A"][0] @ by["Lambda_3"][0].transpose() + by["Lambda_4"][0])

        ## Update A_k
        WkT_cross_Xk_temp = self.WkT_cross_Xk(by["W"])
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                to["A"][k] = inv(self.mu_dict["mu_3"][k] * by["H"][k] @ by["H"][k].transpose() + self.mu_dict["mu_4"][k] * np.eye(self.r_0)) @ (self.gamma_lst[2] * self.diagonals["D_2"][k] @ (WkT_cross_Xk_temp[k] - by["H"][k] @ by["G"][k]) @ by["G"][k].transpose() - self.gamma_lst[6] * by["H"][k] @ self.diagonals["D_6"][k] + (self.mu_dict["mu_3"][k] + self.mu_dict["mu_4"][k]) * by["H"][k] - by["H"][k] @ by["Lambda_3"][k] - by["Lambda_4"][k])

        ## Update A_0
        if self.is_static_given:
            to["A"][0] = inv(self.mu_dict["mu_3"][0] * by["H"][0] @ by["H"][0].transpose() + self.mu_dict["mu_4"][0] * np.eye(self.d_static)) @ (self.gamma_lst[3] * self.diagonals["D_3"] @ (self.X_static - by["H"][0] @ by["G"][0]) @ by["G"][0].transpose() + ((self.mu_dict["mu_4"][0] + self.mu_dict["mu_3"][0]) * np.eye(self.d_static) - self.gamma_lst[7] * self.diagonals["D_7"]) @ by["H"][0] - by["H"][0] @ by["Lambda_3"][0].transpose() - by["Lambda_4"][0])
        
        ## Update G_k
        WkT_cross_Xk_temp = self.WkT_cross_Xk(by["W"])
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                to["G"][k] = inv((self.gamma_lst[2] / 2) * (by["H"][k].transpose() @ self.diagonals["D_2"][k] @ by["A"][k] + by["A"][k].transpose() @ self.diagonals["D_2"][k] @ by["H"][k]) + self.gamma_lst[4] * self.diagonals["D_4"][k]) @ (self.gamma_lst[4] * self.diagonals["D_4"][k] @ by["G_m"] + (self.gamma_lst[2] / 2) * (by["H"][k] + by["A"][k]).transpose() @ self.diagonals["D_2"][k] @ WkT_cross_Xk_temp[k])
        
        ## Update G_0
        if self.is_static_given:
            to["G_0"] = inv((self.gamma_lst[3] / 2) * (by["H"][0].transpose() @ self.diagonals["D_3"] @ by["A"][0] + by["A"][0].transpose() @ self.diagonals["D_3"] @ by["H"][0]) + self.gamma_lst[4] * self.diagonals["D_4"][0]) @ ((self.gamma_lst[3] / 2) * (by["H"][0] + by["A"][0]).transpose() @ self.diagonals["D_3"] @ self.X_static + self.gamma_lst[4] * self.diagonals["D_4"][0] @ by["G_m"])

        ## Update alpha - alpha*
        for o in range(self.c):
            clf = svm.SVR(kernel = svr_kernel)
            clf.fit(by["G_m"][:, :self.l].transpose(), self.Y_l[o, :])
            dual_coef_filled = []
            j = 0
            for i in range(self.l):
                if i in clf.support_:
                    dual_coef_filled.append(clf.dual_coef_[0][j])
                    j += 1
                else:
                    dual_coef_filled.append(0.)
            assert(clf.support_.shape[0] == j)
            to["alpha-alpha*"][o, :] = np.array(dual_coef_filled)

        ## Update G left
        if svr_kernel == "poly":
            # G_l = by["G"][:, :self.l]
            def kernel_deriv(G):
                summation = np.zeros((self.r_1, self.l))
                for o in range(self.c):
                    summation += 2 * self.gamma_lst[5] * G @ ((svr_params["degree"] * svr_params["gamma"] * (svr_params["gamma"] * G.transpose() @ G + svr_params["coef0"]) ** (svr_params["degree"] - 1.)) * np.outer(by["alpha-alpha*"][o], by["alpha-alpha*"][o]))

                for k in range(0, self.K + 1):
                    if k == 0 or self.is_dynamic_bools[k - 1]:
                        summation += 2 * self.gamma_lst[4] * self.diagonals["D_4"][k] @ (G - by["G"][k][:, :self.l]) ## May miss to add D_ao?
                return summation
        elif svr_kernel == "rbf":
            def kernel_deriv(G):
                assert(G.shape[1] == self.l)
                summation = np.zeros((self.r_1, self.l))
                s_G = np.zeros((G.shape[1], G.shape[1]))
                for i in range(G.shape[1]):
                    for j in range(G.shape[1]):
                        s_G[i, j] = np.sum((G[:, i] - G[:, j]) ** 2.)
                gamma = svr_params["gamma"]

                for o in range(self.c):
                    tmp = G @ (-gamma * np.exp(-gamma * s_G) * np.outer(by["alpha-alpha*"][o], by["alpha-alpha*"][o]))
                    D_ao = np.zeros((self.l, self.l))
                    for q in range(self.l):
                        D_ao[q, q] = np.sum(tmp[:, q])
                    summation += 4 * self.gamma_lst[5] * (tmp - G @ D_ao)
                for g in range(0, self.K + 1):
                    if g == 0 or self.is_dynamic_bools[g - 1]:
                        summation += 2 * self.gamma_lst[4] * self.diagonals["D_4"][g] @ (G - by["G"][g][:, :self.l])
                return summation                    
        else:
            raise Exception(NotImplementedError)
        to["G_m"][:, :self.l] = optimize.newton(kernel_deriv, np.zeros((self.r_1, self.l)))

        ## Update G right
        summation_0 = self.diagonals["D_4"][0]
        summation_1 = self.diagonals["D_4"][0] @ by["G"][0][:, self.l:]
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                summation_0 += self.diagonals["D_4"][k]
                summation_1 += self.diagonals["D_4"][k] @ by["G"][k][:, self.l:]
        to["G_m"][:, self.l:] = np.linalg.inv(summation_0) @ summation_1

        ## Update B_i
        for i in range(self.n):
            WWT = by["W"][i] @ by["W"][i].transpose()
            to["B"][i] = np.linalg.inv(- self.gamma_lst[1] * (self.diagonals["D_1"][i] @ (self.X_follows[i] - WWT @ self.X_follows[i]) @ self.X_follows[i].transpose() + self.X_follows[i] @ (self.X_follows[i] - WWT @ self.X_follows[i]).transpose() @ self.diagonals["D_1"][i]) + self.mu_dict["mu_1"][i] * WWT + self.mu_dict["mu_2"][i] * np.eye(self.d)) @ (- self.gamma_lst[8] * self.diagonals["D_8"] @ by["W"][i] - self.gamma_lst[9] * by["W"][i] @ self.diagonals["D_9"] + (self.mu_dict["mu_2"][i] + self.mu_dict["mu_1"][i]) * by["W"][i] - by["Lambda_2"][i] - by["W"][i] @ by["Lambda_1"][i])
        
        ## Update W_i
        print("Update W_i")
        if self.debug: now = time()
        ## - Set Temporary vars
        D2K_GK = ["dummy"]
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                D2K_GK.append(self.diagonals["D_2"][k] @ (by["H"][k] + by["A"][k]) @ by["G"][k])
            else:
                D2K_GK.append("dummy")
        for i in tqdm(range(self.n)):
            XXT = self.X_follows[i] @ self.X_follows[i].transpose()
            BBT = by['B'][i] @ by['B'][i].transpose()
            XXT_D1i = XXT @ (np.eye(self.d) - BBT) @ self.diagonals["D_1"][i]
            gamma1_mu2iI = -self.gamma_lst[1] * (XXT_D1i + XXT_D1i.transpose()) + self.mu_dict["mu_1"][i] * BBT + self.mu_dict["mu_2"][i] * np.eye(self.d)
            if self.use_similarities: gamma1_mu2iI += 2 * self.gamma_lst[10] * self.X_follows[i] @ self.diagonals['D_10'][i] @ self.X_follows[i].transpose()

            gamma8_lambda2i = - self.gamma_lst[8] * self.diagonals["D_8"] @ by["B"][i] - self.gamma_lst[9] * by["B"][i] @ self.diagonals["D_9"] + self.mu_dict["mu_1"][i] * by["B"][i] - by["B"][i] @ by["Lambda_1"][i].transpose() + self.mu_dict["mu_2"][i] * by["B"][i] + by["Lambda_2"][i]
            if not self.if_exact:
                gamma8_lambda2i = - gamma8_lambda2i

            D_wik = np.zeros((self.d, self.d))
            for k in range(1, self.K + 1):
                if self.is_dynamic_bools[k - 1]:
                    D_wik[self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1], self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1]] = np.outer(self.X_baseline[:, i][self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1]], self.X_baseline[:, i][self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1]])
            
            ## - Update
            for q in range(self.r_0):
                D_wikq = np.copy(D_wik)
                x_wikq = np.copy(self.X_baseline[:, i])
                for k in range(1, self.K + 1):
                    if self.is_dynamic_bools[k - 1]:
                        D_wikq[self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1], self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1]] *= self.diagonals["D_2"][k][q, q]
                        x_wikq[self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1]] *= D2K_GK[k][q, i]
                if self.if_exact:
                    if True:
                        to["W"][i][:, q] = inv(gamma1_mu2iI + 2 * self.gamma_lst[2] * D_wikq) @ (gamma8_lambda2i[:, q] + self.gamma_lst[2] * x_wikq)
                    if False:
                        to["W"][i][:, q] = pinv(gamma1_mu2iI + 2 * self.gamma_lst[2] * D_wikq) @ (gamma8_lambda2i[:, q] + self.gamma_lst[2] * x_wikq)
                else:
                    T = 2 * self.gamma_lst[2] * D_wikq + gamma1_mu2iI
                    U = - self.gamma_lst[2] * x_wikq + gamma8_lambda2i[:, q]
                    delta_wiq = T @ by["W"][i][:, q] + U
                    siq = delta_wiq.transpose() @ (T @ by["W"][i][:, q] + U) / (delta_wiq.transpose() @ T @ delta_wiq)
                    to["W"][i][:, q] = by["W"][i][:, q] - siq * delta_wiq
                        
        if self.debug: print(f"W update took: {time() - now}.")

        ## Update Labmda's
        for i in range(self.n):
            to["Lambda_1"][i] = by["Lambda_1"][i] + self.mu_dict["mu_1"][i] * (by["W"][i].transpose() @ by["B"][i] - np.eye(self.r_0))
            to["Lambda_2"][i] = by["Lambda_2"][i] + self.mu_dict["mu_2"][i] * (by["B"][i] - by["W"][i])
        for g in range(0 if self.is_static_given else 1, self.K + 1):
            if g == 0 or self.is_dynamic_bools[g - 1]:
                to["Lambda_3"][g] = by["Lambda_3"][g] + self.mu_dict["mu_3"][g] * (by["H"][g].transpose() @ by["A"][g] - np.eye(self.r_1))
                to["Lambda_4"][g] = by["Lambda_4"][g] + self.mu_dict["mu_4"][g] * (by["A"][g] - by["H"][g])

        ## Update mu_dict
        for i in range(self.n):
            self.mu_dict['mu_1'][i] = self.rho_dict['rho_1'] * self.mu_dict['mu_1'][i]
            self.mu_dict['mu_2'][i] = self.rho_dict['rho_2'] * self.mu_dict['mu_2'][i]
            if auto_reweight_mu:
                if mu_reweight_condition(self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$']) : self.mu_dict['mu_1'][i] *= self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'][-1] / self.loss_dct_ADMM[r'$\Vert W_i^TW_i - I\Vert _F$'][-2]
                if mu_reweight_condition(self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$']) : self.mu_dict['mu_2'][i] *= self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'][-1] / self.loss_dct_ADMM[r'$\Vert B_i = W_i \Vert _F$'][-2]
        for g in range(0 if self.is_static_given else 1, self.K + 1):
            if g == 0 or self.is_dynamic_bools[g - 1]:
                self.mu_dict['mu_3'][g] = self.rho_dict['rho_3'] * self.mu_dict['mu_3'][g]
                self.mu_dict['mu_4'][g] = self.rho_dict['rho_4'] * self.mu_dict['mu_4'][g]
                if auto_reweight_mu:
                    if mu_reweight_condition(self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$']) : self.mu_dict['mu_3'][g] *= self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'][-1] / self.loss_dct_ADMM[r'$\Vert H_g^TH_g - I\Vert _F$'][-2]
                    if mu_reweight_condition(self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$']) : self.mu_dict['mu_4'][g] *= self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'][-1] / self.loss_dct_ADMM[r'$\Vert A_g = H_g \Vert _F$'][-2]
        #%% Keep some portion of previous values
        if keep_previous_ratio > 0.0:
            for k in range(0, self.K + 1):
                if self.is_dynamic_bools[k - 1]:
                    to['H'][k] = (1 - keep_previous_ratio) * to['H'][k] + keep_previous_ratio * by['H'][k]
                    to['A'][k] = (1 - keep_previous_ratio) * to['A'][k] + keep_previous_ratio * by['A'][k]
                    to['G'][k] = (1 - keep_previous_ratio) * to['G'][k] + keep_previous_ratio * by['G'][k]
                    to['Lambda_3'][k] = (1 - keep_previous_ratio) * to['Lambda_3'][k] + keep_previous_ratio * by['Lambda_3'][k]
                    to['Lambda_4'][k] = (1 - keep_previous_ratio) * to['Lambda_4'][k] + keep_previous_ratio * by['Lambda_4'][k]
            to['G_m'] = (1 - keep_previous_ratio) * to['G_m'] + keep_previous_ratio * by['G_m']
            to['alpha-alpha*'] = (1 - keep_previous_ratio) * to['alpha-alpha*'] + keep_previous_ratio * by['alpha-alpha*']
            for i in range(self.n):
                to['W'][i] = (1 - keep_previous_ratio) * to['W'][i] + keep_previous_ratio * by['W'][i]
                to['B'][i] = (1 - keep_previous_ratio) * to['B'][i] + keep_previous_ratio * by['B'][i]
                to['Lambda_1'][i] = (1 - keep_previous_ratio) * to['Lambda_1'][i] + keep_previous_ratio * by['Lambda_1'][i]
                to['Lambda_2'][i] = (1 - keep_previous_ratio) * to['Lambda_2'][i] + keep_previous_ratio * by['Lambda_2'][i]        

    def WT_cross_X(self, W_lst):
        raise Exception("Deprecated")
        result = np.zeros((self.r_1, self.n))
        for i in range(self.n):
            result[:, i] = W_lst[i].transpose() @ self.X_baseline[:, i]
        return result
    
    def WkT_cross_Xk(self, W_lst):
        result = ["dummy"]
        for k in range(1, self.K + 1):
            if self.is_dynamic_bools[k - 1]:
                result.append(np.zeros((self.r_0, self.n)))
                for i in range(self.n):
                    result[k][:, i] = W_lst[i][self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1], :].transpose() @ self.X_baseline[self.mode_length_list[k-1][0]: self.mode_length_list[k-1][1], i]
            else:
                result.append("dummy")
        return result

    def get_error_between_F_and_true(self):
        raise Exception("Deprecated")
        # label_map = {'BL_RAVLT_TOTAL': 'RAVLT TOTAL', 'BL_RAVLT30': 'RAVLT 30', 'BL_RAVLT30_RECOG': 'RAVLT RECOG'}
        # factorization_RMSE_dict = {}
        # testset_F = self.after['F'][:, self.num_of_training_set:]
        # for i, label in zip(range(len(self.info_dct['ravlt_label'])), self.info_dct['ravlt_label']):
        #     factorization_RMSE_dict[label_map[label]] = utilsforminds.math.get_RMSE(self.ravlt_arr_test[:, i, self.y_timepoint_idx], testset_F[i, :])
        # return factorization_RMSE_dict

        label_map = {'BL_RAVLT_TOTAL': 'RAVLT TOTAL', 'BL_RAVLT30': 'RAVLT 30', 'BL_RAVLT30_RECOG': 'RAVLT RECOG'}
        factorization_RMSE_dict = {"Factorization": {"RMSE": {"Factorization": []},
        "models": []}}
        testset_F = self.after['F'][:, self.num_of_training_set:]
        for i, label in zip(range(len(self.info_dct['ravlt_label'])), self.info_dct['ravlt_label']):
            factorization_RMSE_dict["Factorization"]["RMSE"]["Factorization"].append(utilsforminds.math.get_RMSE(self.ravlt_arr_test[:, i, self.y_timepoint_idx], testset_F[i, :]))
            factorization_RMSE_dict["Factorization"]["models"].append(label_map[label])
        return factorization_RMSE_dict
    
    def draw_brains(self, W_all_arr, patients_group_name_idc_dict):
        # for start_end_idx in patients_split_start_end_idx_lst:
        #     assert(len(start_end_idx) == 2)
        assert(len(W_all_arr.shape) == 3)

        # fs_labels = pd.read_csv(f"{helpers.getExecPath()}/data/adni_data/fs_atlas_labels.csv")
        # fs_labels_only_names_arr = fs_labels.values[:, 0]

        # vbm_cleaner = VBMDataCleaner()
        # vbm_cleaner.load_data(f"{helpers.getExecPath()}/data/adni_data/longitudinal imaging measures_VBM_mod_final.xlsx")
        # vbm_labels = vbm_cleaner.clean()
        for name, idc_lst in patients_group_name_idc_dict.items():

            # path_to_weights = path_to_results_dir + 'W_array/W_all.npy'
            # fs_weights = pd.read_csv("fs_weights.csv").values.flatten()
            # vbm_weights = pd.read_csv("vbm_weights.csv").values.flatten()

            # Create FreeSurfer ROI Map
            if 'FS' in self.mode_length_dict.keys():
                fs_label_weight_list = []
                fs_weights = np.mean(W_all_arr[idc_lst, self.mode_length_dict['FS'][0]:self.mode_length_dict['FS'][1], :], axis = (0, 2))
                fs_roi_map = FSRegionOfInterestMap()
                for index, row in self.fs_labels.iterrows():
                    atlas = row["Atlas"]
                    rois = row[atlas].split("+")
                    [fs_roi_map.add_roi(roi, fs_weights[index], atlas) for roi in rois]
                    fs_label_weight_list.append([rois[0], fs_weights[index]])
                fs_label_weight_list.sort(key = lambda x: x[1], reverse=True)
                with open(f"{self.new_directory_name}/brain_maps/{name}_fs_weights.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(fs_label_weight_list)

                fs_roi_map.build_map(smoothed=True)
                #fs_roi_map.plot(time)
                fs_roi_map.save(f"{self.new_directory_name}/brain_maps/{name}_fs_fig.png", "FS")
                # self.new_directory_name + '/brain_maps/'

            # Create VBM ROI Map
            if 'VBM' in self.mode_length_dict.keys():
                vbm_label_weight_list = []
                vbm_weights = np.mean(W_all_arr[idc_lst, self.mode_length_dict['VBM'][0]:self.mode_length_dict['VBM'][1], :], axis = (0, 2))
                vbm_roi_map = VBMRegionOfInterestMap()
                for label, weight in zip(self.vbm_labels, vbm_weights):
                    vbm_roi_map.add_roi(label, weight)
                    vbm_label_weight_list.append([label, weight])
                vbm_label_weight_list.sort(key = lambda x: x[1], reverse=True)
                with open(f"{self.new_directory_name}/brain_maps/{name}_vbm_weights.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(vbm_label_weight_list)

                vbm_roi_map.build_map(smoothed=True)
                #vbm_roi_map.plot(time)
                vbm_roi_map.save(f"{self.new_directory_name}/brain_maps/{name}_vbm_fig.png", "VBM")
    
def plot_results(enrichment: Enrichment, iteration_tot, num_features_to_plot = 10):
    self = enrichment
    loss_dct_ADMM = copy.deepcopy(self.loss_dct_ADMM_keep)
    converged_check_dct_ADMM = copy.deepcopy(self.converged_check_dct_ADMM)
    if self.two_loops:
        loss_dct_smoothness = copy.deepcopy(self.loss_dct_smoothness)
        converged_check_dct_smoothness = copy.deepcopy(self.converged_check_dct_smoothness)

    # txtFile = open(new_directory_name + "/summary.txt", "a")
    # # txtFile.write(f'gamma_lst: {self.gamma_lst}\n rho_dict: {self.rho_dict}\n r_1: {self.r_1}\n r_2: {self.r_2}\n p: {self.p}\n training_set_ratio: {self.training_set_ratio}\n n: {self.n}\n')
    # txtFile.write('----- __init__() parameters -----\n')
    # txtFile.write(self.param_dict_str_init)

    loss_df_ADMM = pandas.DataFrame.from_dict(loss_dct_ADMM)
    converged_check_df_ADMM = pandas.DataFrame.from_dict(converged_check_dct_ADMM)
    plot_x_grids = list(range(1, len(loss_dct_ADMM[r'$objective$']) + 1))
    for key, value in loss_dct_ADMM.items():
        plt.plot(plot_x_grids, value)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title(key)
        plt.savefig(self.new_directory_name + '/plots' + f'/{key}_ADMM.png')
        plt.clf()
    loss_df_ADMM.to_csv(self.new_directory_name + '/loss_ADMM.csv')
    converged_check_df_ADMM.to_csv(self.new_directory_name + '/convergence_ADMM.csv')

    if self.two_loops:
        loss_df_smoothness = pandas.DataFrame.from_dict(loss_dct_smoothness)
        converged_check_df_smoothness = pandas.DataFrame.from_dict(converged_check_dct_smoothness)
        plot_x_grids = list(range(1, len(loss_dct_smoothness[r'$objective$']) + 1))
        for key, value in loss_dct_smoothness.items():
            plt.plot(plot_x_grids, value)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title(key)
            plt.savefig(self.new_directory_name + '/plots' + f'/{key}_smoothness.png')
            plt.clf()
        loss_df_smoothness.to_csv(self.new_directory_name + '/loss_smoothness.csv')
        converged_check_df_smoothness.to_csv(self.new_directory_name + '/convergence_smoothness.csv')

    if self.if_save_array:
        np.save(self.new_directory_name + '/data_array/X_all.npy', self.X_all)
        for i in range(self.n):
            np.save(self.new_directory_name + f'/W_array/w_{i}.npy', self.after['W'][i])
        W_arr = np.stack(self.after['W'], axis = 0)
        np.save(self.new_directory_name + '/W_array/W_all.npy', W_arr)

    ## Loss plots
    utilsforminds.visualization.plot_multiple_lists({r'$objective$':self.loss_dct_ADMM_keep[r'$objective$']}, f"{self.new_directory_name}/plots/loss_objective_ADMM_{iteration_tot}.eps")
    loss_dct_ADMM.pop(r'$objective$')
    utilsforminds.visualization.plot_multiple_lists(loss_dct_ADMM, f"{self.new_directory_name}/plots/loss_primal_ADMM_{iteration_tot}.eps")
    if self.two_loops:
        utilsforminds.visualization.plot_multiple_lists({r'$objective$':loss_dct_smoothness[r'$objective$']}, f"{self.new_directory_name}/plots/loss_objective_smoothness_{iteration_tot}.eps")
        loss_dct_smoothness.pop(r'$objective$')
        utilsforminds.visualization.plot_multiple_lists(loss_dct_smoothness, f"{self.new_directory_name}/plots/loss_smoothness_{iteration_tot}.eps")

    ## Test current projection
    W_learned = np.stack(self.after['W'], axis = 0)

    # W_mean_intensities = np.mean(W_learned, axis = (0, 2))

    W_mean_intensities = np.mean(np.absolute(W_learned), axis = (0, 2))
    indices_top_n_features_arr = utilsforminds.helpers.get_top_n_indices(array_like = W_mean_intensities, n = num_features_to_plot, from_largest = True)

    if self.dataset_kind == "traffic": add_dict = {"img_paths": self.img_paths}
    else: add_dict = {}
    if self.best_score_dict is None: # first time case
        other_models_tester_inst = other_models_tester.Other_models_tester(self.X_all, self.target_arr_training, self.target_arr_test, W_learned, self.info_dct, self.target_arr, debug = False, min_max_enriched = False, x_timepoint_idx= self.x_timepoint_idx, y_timepoint_idx= self.y_timepoint_idx, features_range_dict= self.dataset_specific_kwargs["features_range_dict"] if "features_range_dict" in self.dataset_specific_kwargs.keys() else None, add_dict= add_dict)
        is_first = True
    else:
        other_models_tester_inst = other_models_tester.Other_models_tester(self.X_all, self.target_arr_training, self.target_arr_test, W_learned, self.info_dct, self.target_arr, debug = False, min_max_enriched = False, raw_enriched= ['enriched'], x_timepoint_idx= self.x_timepoint_idx, y_timepoint_idx= self.y_timepoint_idx, features_range_dict= self.dataset_specific_kwargs["features_range_dict"] if "features_range_dict" in self.dataset_specific_kwargs.keys() else None, add_dict= add_dict)
        is_first = False
    
    ## Run the estimators
    model_color_dict = {"raw": {"Lasso": (1.0, 0., 0.), "SVR": (0.85, 0., 0.), "RR": (0.7, 0., 0.), "LR": (0.55, 0., 0.), "CNN": (0.4, 0., 0.)}, "enriched": {"Lasso": (0., 1., 0.), "SVR": (0., 0.85, 0.), "RR": (0., 0.7, 0.), "LR": (0., 0.55, 0.), "CNN": (0., 0.4, 0.)}}
    model_color_dict_plotly = deepcopy(model_color_dict)
    for model in model_color_dict["raw"].keys():
        for re in ["raw", "enriched"]:
            if False:
                tp = model_color_dict[re][model]
                model_color_dict_plotly[re][model] = "rgba" + str((tp[0] * 255., tp[1] * 255., tp[2] * 255., 1.0))
            else:
                model_color_dict_plotly[re][model] = {"raw": "red", "enriched": "blue"}[re]

    marker_symbol = {"raw": "circle", "enriched": "cross"}
    prediction_dct_total = {}
    # scatter_data = {label: [] for label in other_models_tester_inst.y_dct.keys()}

    # prediction_dct_total["LR"] = other_models_tester_inst.linearRegression_predict()[0]
    # prediction_dct_total["DNN"] = other_models_tester_inst.DNN_predict(grid_param_dict = {})[0]
    grid_param_dict_CNN = {'filters_factor': [0.5], 'kernel_size_factor': [0.5], 'neurons_factor':[0.5], 'drop_rate':[0.2]}
    for direction in ["NORTH"]:
        for name, pred_result in zip(["Lasso", "SVR", "RR", "CNN"], [other_models_tester_inst.lasso_predict(), other_models_tester_inst.SVR_predict(), other_models_tester_inst.ridge_predict(), other_models_tester_inst.CNN_predict(grid_param_dict_CNN)]):
            prediction_dct_total[name] = pred_result[0]
            if True:
                # for label in pred_result[1].keys():
                features_range_dict = self.dataset_specific_kwargs["features_range_dict"]
                scatter_data = []
                for re in ["raw", "enriched"]:
                    labels = other_models_tester_inst.y_dct[f"{direction}_THRU"]['test']
                    x = prediction_dct_total[name][f"{direction}_THRU"][re]
                    scatter_data.append(go.Scatter(x=x, y=labels, name= f"{re}", mode= "markers", marker= dict(color= model_color_dict_plotly[re][name], symbol= marker_symbol[re])))
                fig = go.Figure(data = scatter_data, layout = {"xaxis": {"title": f"Prediction with {name}"}, "yaxis": {"title": "True"}}) ## "title": "My title"
                fig.write_html(self.new_directory_name + f'/plots/{name}_confusion.html')
                fig.write_image(self.new_directory_name + f'/plots/{name}_confusion.png')

    total_RMSE_dct = deepcopy(other_models_tester_inst.total_RMSE_dct)
    if False:
        raise Exception("Deprecated")
        factorization_RMSE_dict = self.get_error_between_F_and_true()
        # for label in factorization_RMSE_dict.keys():
        #     total_RMSE_dct[label]['RMSE']['Enriched Representation'].append(factorization_RMSE_dict[label])
        #     total_RMSE_dct[label]['std']['Enriched Representation'].append(0.)
        #     total_RMSE_dct[label]['RMSE']['Original Representation'].append(0.)
        #     total_RMSE_dct[label]['std']['Original Representation'].append(0.)
        #     total_RMSE_dct[label]['models'].append("Factorization")

    self.total_RMSE_dct_list.append(deepcopy(total_RMSE_dct))
    percent_increase_from_best = 0.
    with open(self.new_directory_name + "/summary.txt", "a") as txtFile:
        txtFile.write(f'\n--- iteration: {iteration_tot} ---\n')
        # calculate mean and std of W^T X
        WTX = []
        for i in range(self.n):
            WTX.append(self.after['W'][i].transpose() @ self.X_all[i, :, 0])
        mean = np.mean(WTX)
        std = np.std(WTX)
        if is_first:
            self.first_prediction_dct_total = deepcopy(prediction_dct_total)
            self.total_RMSE_dct_with_raw = deepcopy(total_RMSE_dct)
            txtFile.write('----- raw representation results -----\n')
            self.best_score_dict = {'iters': iteration_tot}
            for label_mode in total_RMSE_dct.keys():
                txtFile.write(f'--- {label_mode} ---\n')
                self.best_score_dict[label_mode] = {}
                for metric in total_RMSE_dct[label_mode].keys():
                    self.best_score_dict[label_mode][metric] = {}
                    if metric not in ['models', 'std']:
                        txtFile.write(f'- {metric} -\n')
                        for model, idx in zip(total_RMSE_dct[label_mode]['models'], range(len(total_RMSE_dct[label_mode]['models']))):
                            txtFile.write(f"{model}: {total_RMSE_dct[label_mode][metric]['Original Representation'][idx]}\n")
                            self.best_score_dict[label_mode][metric][model] = {'errors': [total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx]], 'mean': 0., 'std': 0.}
        else:
            for model in self.first_prediction_dct_total.keys():
                for label in self.first_prediction_dct_total[model].keys():
                    prediction_dct_total[model][label]["raw"] = deepcopy(self.first_prediction_dct_total[model][label]["raw"])
            for label in self.total_RMSE_dct_list[-1].keys():
                self.total_RMSE_dct_list[-1][label]['RMSE']['Original Representation'] = deepcopy(self.total_RMSE_dct_list[0][label]['RMSE']['Original Representation'])
        txtFile.write(f'----- W^T X statistics of {iteration_tot} -----\n')
        txtFile.write(f'mean : {mean}\nstd : {std}\n')

        txtFile.write(f'----- enriched representation results of {iteration_tot} -----\n')
        for label_mode in total_RMSE_dct.keys():
            txtFile.write(f'--- {label_mode} ---\n')
            for metric in total_RMSE_dct[label_mode].keys():
                if metric not in ['models', 'std']:
                    txtFile.write(f'- {metric} -\n')
                    for model, idx in zip(total_RMSE_dct[label_mode]['models'], range(len(total_RMSE_dct[label_mode]['models']))):
                        txtFile.write(f"{model}: {total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx]}\n")
                        percent_increase_from_best += (total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx] - self.best_score_dict[label_mode][metric][model]['errors'][-1]) / self.best_score_dict[label_mode][metric][model]['errors'][-1]
                        self.total_RMSE_dct_with_raw[label_mode][metric]['Enriched Representation'][idx] = total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx]
    
    if True or percent_increase_from_best <= 0.: # current win for smaller-better-metric
        print(f'Best projection found at iteration: {iteration_tot}')
        # np.save(self.new_directory_name + '/W_array/W_all_best.npy', W_learned)

        ## Save and plot the SNPs
        if self.is_static_given:
            if self.dataset_kind == "traffic":
                weights_statics = np.mean(np.absolute(W_learned)[:, self.mode_length_dict["static"][0] : self.mode_length_dict["static"][1], :], axis = (0, 2))
                self.reordered_statics_info_df["weights"] = weights_statics.tolist()

                ## plot the SNPs
                self.reordered_statics_info_df.to_pickle(self.new_directory_name + "/statics/reordered_statics_info_df.pkl")
                mean_val = self.reordered_statics_info_df['weights'].mean()
                utilsforminds.visualization.plot_top_bars_with_rows(self.reordered_statics_info_df, self.new_directory_name + f"/statics/topbars_{iteration_tot}_4.eps", color_column = 'original_colors', order_by = "weights", group_column = "original", xlabel = "Feature", ylabel = "Weights", num_bars = 10, num_rows = 2, re_range_max_min_proportion = [0.90, 1.10], rotation_xtickers = 45, xticks_fontsize= 10)

        ## plot the dynamics
        if self.dataset_kind == "traffic":
            dynamic_features_region = self.dataset_specific_kwargs["dynamic_features_region"]
            for bag_idx in sample(range(self.X_all.shape[0]), 2):
                os.mkdir(self.new_directory_name + f'/dynamic/{bag_idx}/')
                for inst_idx in range(self.X_all[bag_idx].shape[1]):
                    for direction in ["image_0", "image_5", "image_10"]:
                        heat_map = [[None for i in range(self.dataset_specific_kwargs["img_split"][0])] for j in range(self.dataset_specific_kwargs["img_split"][1])]
                        idx_img = self.mode_length_dict[direction][0]
                        for pos_idx in range(len(dynamic_features_region[direction]["regions"][bag_idx][inst_idx])):
                            pos = dynamic_features_region[direction]["regions"][bag_idx][inst_idx][pos_idx]
                            idx_cut = [idx_img + dynamic_features_region[direction]["idc"][bag_idx][inst_idx][pos_idx][0], idx_img + dynamic_features_region[direction]["idc"][bag_idx][inst_idx][pos_idx][1]]
                            weights_region = W_learned[bag_idx, idx_cut[0] : idx_cut[1] , :]
                            score = weights_region.transpose() @ self.X_all[bag_idx, idx_cut[0] : idx_cut[1], inst_idx]
                            score = np.mean(np.absolute(score))
                            heat_map[pos[1]][pos[0]] = score
                        fig = px.imshow(heat_map, text_auto=False, color_continuous_scale='Blues')
                        fig.write_html(self.new_directory_name + f'/dynamic/{bag_idx}/{inst_idx}_{direction}.html')
                        fig.write_image(self.new_directory_name + f'/dynamic/{bag_idx}/{inst_idx}_{direction}.png')
                        shutil.copyfile(self.dataset_specific_kwargs["dynamic_imgs_path"][direction][bag_idx][inst_idx], self.new_directory_name + f'/dynamic/{bag_idx}/{inst_idx}_{direction}_original.jpg')



        ## plot the features
        for rank in range(len(indices_top_n_features_arr)):
            for target_label_idx in range(len(self.target_labels)):
                x = self.X_all[self.num_of_training_set:, indices_top_n_features_arr[rank], self.x_timepoint_idx]
                y_list = []
                y_list.append({"label": "True Score", "ydata": self.target_arr_test[:, target_label_idx, self.y_timepoint_idx], "color": "black", "linestyle": "dashed"})
                for model in prediction_dct_total.keys():
                    y_list.append({"label": model + " Original", "ydata": prediction_dct_total[model][self.target_labels[target_label_idx]]["raw"], "color": model_color_dict["raw"][model], "linestyle": "solid"})
                for model in prediction_dct_total.keys():
                    y_list.append({"label": model + " Enriched", "ydata": prediction_dct_total[model][self.target_labels[target_label_idx]]["enriched"], "color": model_color_dict["enriched"][model], "linestyle":'solid'})
                utilsforminds.visualization.plot_xy_lines(x = x, y_dict_list = y_list, path_to_save= self.new_directory_name + f"/features/{rank}_{self.total_labels[indices_top_n_features_arr[rank]]}_{self.target_labels[target_label_idx]}_{iteration_tot}.eps", y_label= f"{self.target_labels_readable_dict[self.target_labels[target_label_idx]]} Score", x_label= f"Intensity of {self.total_labels[indices_top_n_features_arr[rank]]}", title= f"Prediction with {self.modality_name}")
                

        self.best_score_dict['iters'] = iteration_tot
        with open(self.new_directory_name + "/best.txt", "a") as txtFile:
            txtFile.write(f'----- iters: {iteration_tot} -----\n')
            for label_mode in total_RMSE_dct.keys():
                txtFile.write(f'--- {label_mode} ---\n')
                for metric in total_RMSE_dct[label_mode].keys():
                    if metric not in ['models', 'std']:
                        txtFile.write(f'- {metric} -\n')
                        for model, idx in zip(total_RMSE_dct[label_mode]['models'], range(len(total_RMSE_dct[label_mode]['models']))):
                            txtFile.write(f"{model}: {total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx]}\n")
                            self.best_score_dict[label_mode][metric][model]['errors'].append(total_RMSE_dct[label_mode][metric]['Enriched Representation'][idx])
                            self.best_score_dict[label_mode][metric][model]['mean'] = np.mean(np.array(self.best_score_dict[label_mode][metric][model]['errors']))
                            self.best_score_dict[label_mode][metric][model]['std'] = np.std(np.array(self.best_score_dict[label_mode][metric][model]['errors']))
    
    other_models_tester_inst.save_result(f'{self.new_directory_name}/plots', suffix = f'{iteration_tot}', replace_result_dict = self.total_RMSE_dct_with_raw)
    if False:
        raise Exception("Deprecated")
        other_models_tester_inst.save_result(f'{self.new_directory_name}/plots', suffix = f'{iteration_tot}', replace_result_dict = factorization_RMSE_dict)
    
    return total_RMSE_dct