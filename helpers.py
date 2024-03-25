import matplotlib
from sys import platform
if platform.lower() != 'darwin' and 'win' in platform.lower():
    matplotlib.use('TkAgg')
else:
    matplotlib.use("MacOSX")
# from matplotlib import rc
# rc('text',usetex=True)
# rc('text.latex', preamble='\usepackage{color}')
import os
import sys
import numpy as np
import csv
import numbers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, MaxNLocator
from random import random
from copy import deepcopy
import tikzplotlib
import utilsforminds
import pandas as pd

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def getExecPath():
    '''
        ref: https://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
    '''
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    return os.path.dirname(sFile)

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def getNewDirectoryName(parentDir, newDir):
    '''
        To get new directory name to save results while avoiding duplication
    '''

    if parentDir[0] != '/':
        parentDir = '/' + parentDir
    if parentDir[-1] != '/':
        parentDir = parentDir + '/'

    assert(getExecPath() + parentDir)

    duplicatedNameNum = 0
    while(os.path.isdir(getExecPath() + parentDir + newDir + str(duplicatedNameNum))):
        duplicatedNameNum = duplicatedNameNum + 1
    newDir = newDir + str(duplicatedNameNum)

    return newDir

@utilsforminds.decorators.redirect_function(utilsforminds.math)
def p_exponent_matrix(arr, p):
    """
    
    Examples
    --------
    self.diagonals['D_7'] = (self.p / 2) * helpers.p_exponent_matrix(W_p1p @ W_p1p.transpose() + self.delta * np.eye(self.d), (self.p / 2 - 1))
    """
    assert(len(arr.shape) == 2)

    #%% Using SVD
    u, s, v = np.linalg.svd(arr, full_matrices=False)
    return u @ np.diag(s ** p) @ v

    #$$ Using Igenvalue decomposition
    # w, v = np.linalg.eig(arr)
    # return v @ np.diag(w ** p) @ v.transpose()

def get_SNP_name_sequence_dict(path_to_csv):
    """
    
    Examples
    --------
    SNP_label_dict = helpers.get_SNP_name_sequence_dict('./data/snp_labels.csv')
    """
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            row_ = row
            line_count += 1
    assert(line_count == 1)
    result_dct = {}
    for i in range(len(row_)):
        result_dct[row_[i]] = i
    return result_dct

def get_adjacency_matrix_from_pairwise_value(SNP_label_dict, SNP_value_df, col_names_pair_vertex_lst, col_name_pairwise_value = 'r^2', threshold_pairwise_value = 0.2):
    assert(len(col_names_pair_vertex_lst) == 2)
    columns_ = SNP_value_df.columns.values
    assert(col_names_pair_vertex_lst[0] in columns_ and col_names_pair_vertex_lst[1] in columns_ and col_name_pairwise_value in columns_)
    number_of_vertices = len(SNP_label_dict)
    adjacency_matrix = np.zeros((number_of_vertices, number_of_vertices))
    for index, row in SNP_value_df.iterrows():
        if row[col_name_pairwise_value] >= threshold_pairwise_value:
            adjacency_matrix[SNP_label_dict[row[col_names_pair_vertex_lst[0]]], SNP_label_dict[row[col_names_pair_vertex_lst[1]]]] = 1
            adjacency_matrix[SNP_label_dict[row[col_names_pair_vertex_lst[1]]], SNP_label_dict[row[col_names_pair_vertex_lst[0]]]] = 1
    return adjacency_matrix

@utilsforminds.decorators.redirect_function(utilsforminds.math)
def get_norm_from_matrix(arr, under_p_1, under_p_2):
    """
    
    Examples
    --------
    testArr_1 = np.array(list(range(6))).reshape(2,3)\n
    print(helpers.get_norm_from_matrix(testArr_1, 2, 2))
        => 7.416198487095663
    """

    summed = 0
    for i in range(arr.shape[0]):
        summed += np.sum(arr[i, :] ** under_p_1) ** (under_p_2 / under_p_1)
    return summed ** (1 / under_p_2)

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def is_small_container(container, length_limit = 20):
    if (type(container) == type({}) or type(container) == type([]) or type(container) == type(tuple([3, 3]))) and len(container) < length_limit:
        if type(container) == type({}):
            for key, value in container.items():
                if not (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) and isinstance(value, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
        if type(container) == type([]) or type(container) == type(tuple(3, 3)):
            for item in container:
                if not (isinstance(item, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
    else:
        False

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def paramDictToStr(param_dict):
    result = ""
    for key in param_dict.keys():
        if (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) or is_small_container(key)) and (isinstance(param_dict[key], (numbers.Number, type('a'), type(True), type(None))) or is_small_container(param_dict[key])):
            result = result + str(key) + " : " + str(param_dict[key]) + "\n"
    return result

@utilsforminds.decorators.redirect_function(utilsforminds.math)
def get_RMSE(arr_1, arr_2, features_range_dict= None):
    assert(arr_1.shape == arr_2.shape)
    if features_range_dict is not None:
        arr_1 = arr_1 * features_range_dict["max-min"] + features_range_dict["min"]
        arr_2 = arr_2 * features_range_dict["max-min"] + features_range_dict["min"]
    return (np.sum((arr_1 - arr_2) ** 2) / arr_1.shape[0]) ** (1/2.)

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def min_max_scale(arr, min_, max_):
    return (arr - min_) / (max_ - min_)

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def load_csv_columns_into_list(path_to_csv: str):
    result_dict = {}
    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = ',')
        headers = csv_reader.fieldnames
        for header in headers:
            result_dict[header] = []
        for line in csv_reader:
            for header in line.keys():
                result_dict[header].append(line[header])
    del result_dict['']
    return result_dict

@utilsforminds.decorators.redirect_function(utilsforminds.visualization)
def plot_bar_charts(path_to_save : str, name_numbers : dict, xlabels : list, xtitle = None, ytitle = None, bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = False, name_errors = None, name_to_show_percentage = None, fontsize = 10, title = None, fix_legend = True):
    """
    
    Parameters
    ----------
        name_numbers : dict
            For example, name_numbers['enriched'] == [0.12, 0.43, 0.12] for RMSE
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]
    """

    ## Set kwargs parameters
    plt_bars_kwargs_dict = {}
    for name in name_numbers.keys():
        plt_bars_kwargs_dict[name] = {}
        if name_errors is not None:
            plt_bars_kwargs_dict[name]['yerr'] = name_errors[name]
        if colors_dict is not None:
            plt_bars_kwargs_dict[name]['color'] = colors_dict[name]

    single_key = next(iter(name_numbers))
    n_groups = len(name_numbers[single_key])
    for numbers in name_numbers.values():
        assert(len(numbers) == n_groups)
    assert(len(xlabels) == n_groups)
    xlabels_copied = deepcopy(xlabels)
    if name_to_show_percentage is not None:
        assert(name_to_show_percentage in name_numbers.keys())
        assert(len(name_numbers) >= 2)
        for i in range(len(xlabels_copied)):
            scores_of_group = []
            for name in name_numbers.keys():
                if name != name_to_show_percentage:
                    scores_of_group.append(name_numbers[name][i])
            mean = np.mean(scores_of_group)
            xlabels_copied[i] += f'({(mean - name_numbers[name_to_show_percentage][i]) * 100. / mean:.2f}%)'
    if bar_width == 'auto':
        bar_width_ = 0.30 * (2 / len(name_numbers))   
    else:
        bar_width_ = bar_width

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)

    rects_list = []
    index_copied = np.copy(index).astype(np.float)
    for name, numbers in name_numbers.items():
        rects_list.append(plt.bar(index_copied, numbers, bar_width_, alpha = alpha, label = name, **plt_bars_kwargs_dict[name]))
        index_copied += bar_width_

    if title is not None:
        plt.title(title)
    if xtitle is not None:
        plt.xlabel(xtitle, fontsize = fontsize)
    if ytitle is not None:
        plt.ylabel(ytitle, fontsize = fontsize)
    # plt.title('Scores by person')
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize)
    if fix_legend:
        numbers_tot = []
        for numbers in name_numbers.values():
            numbers_tot += numbers
        plt.ylim([0., np.max(numbers_tot) * (1. + 0.1 * len(name_numbers))])
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)
    tikzplotlib.save(utilsforminds.visualization.format_path_extension(path_to_save))

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item= True):
    """

    Examples
    --------
    random_pick_items(item_length_dict= 0: 3, 1: 4, 4: 2}, pick_keep_probs_dict= {-1: 0.3, 0: 0.5}, keep_non_prob_item= True)
        Returns
        {0: [2, 1], 1: [1, 2], 4: [1]}
    """

    picked_idc_dict = {}
    for idx_0 in item_length_dict.keys():
        picked_idc_dict[idx_0] = []
        done_idx= []
        for key, prob in pick_keep_probs_dict.items():
            key_positive = key if key>= 0 else item_length_dict[idx_0] + key
            if key_positive in done_idx:
                raise Exception(f"Ambiguous pick with index: {key}")
            done_idx.append(key_positive)
            rand_num= random()
            if rand_num<= prob:
                picked_idc_dict[idx_0].append(key_positive)
        if keep_non_prob_item:
            for idx_1 in range(item_length_dict[idx_0]):
                if idx_1 not in done_idx and idx_1 not in picked_idc_dict[idx_0]:
                    picked_idc_dict[idx_0].append(idx_1)
    return picked_idc_dict

@utilsforminds.decorators.redirect_function(utilsforminds.helpers)
def delete_items_from_list_with_indices(list_to_filter, indices, keep_not_remove = False):
    base_idx = 0
    sorted_indices = sorted(indices, reverse = False)
    sorted_indices_to_remove = []
    if keep_not_remove:
        for i in range(len(list_to_filter)):
            if i not in sorted_indices:
                sorted_indices_to_remove.append(i)
    else:
        sorted_indices_to_remove = deepcopy(sorted_indices)
    list_to_filter_copied = deepcopy(list_to_filter)
    for idx in sorted_indices_to_remove:
        del list_to_filter_copied[idx - base_idx]
        base_idx += 1
    return list_to_filter_copied

@utilsforminds.decorators.redirect_function(utilsforminds.visualization, "plot_group_scatter")
def plot_chromosome(group_df, path_to_save, colors = ["red", "navy", "lightgreen", "lavender", "khaki", "teal", "gold", "violet", "green", "orange", "blue", "coral", "azure", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], tick_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 17, 19, 20, 21]):
    group_df_copied = group_df.copy()
    group_df.sort_values(by = ["chr", "weights"], ascending = [True, True])
    # reordered_SNPs_info_df_copied['color'] = reordered_SNPs_info_df_copied["chr"].apply(lambda x: colors[x])
    ax = group_df_copied.plot.scatter(x = 'index_original', y = 'weights', c = group_df_copied['color'], s = 0.03, figsize = (8, 2), colorbar = False, fontsize = 6, marker = ',')
    counts_snps = [0]
    for chr in range(1, 22):
        series_obj = group_df_copied.apply(lambda x: True if x['chr'] == chr else False, axis = 1)
        counts_snps.append(len(series_obj[series_obj == True].index))
    hori_labels = []
    accumulated = 0
    for i in range(21):
        accumulated += counts_snps[i]
        hori_labels.append(round(counts_snps[i + 1] / 2 + accumulated))
    x_ticks_texts = list(range(1, 22))
    x_ticks_colors = deepcopy(colors)
    x_ticks_indices = list(np.array(tick_indices) - 1)

    hori_labels = delete_items_from_list_with_indices(hori_labels, x_ticks_indices, keep_not_remove = True)
    x_ticks_texts = delete_items_from_list_with_indices(x_ticks_texts, x_ticks_indices, keep_not_remove = True)
    x_ticks_colors = delete_items_from_list_with_indices(x_ticks_colors, x_ticks_indices, keep_not_remove = True)

    x_axis = ax.axes.get_xaxis()
    # x_axis.set_visible(False)
    ax.set_xticks(hori_labels)
    ax.set_xticklabels(x_ticks_texts)
    # ax.tick_params(axis = 'x', colors = x_ticks_colors)
    for i in range(len(x_ticks_colors)):
        ax.get_xticklabels()[i].set_color(x_ticks_colors[i])
    ax.margins(x = 0)
    ax.margins(y = 0)
    plt.xlabel("")
    plt.ylabel("Weights")
    # cbar = plt.colorbar(mappable = ax)
    # cbar.remove()
    # plt.margins(y=0)
    # plt.tight_layout()
    # plt.grid(which = 'major', linestyle='-', linewidth=2)
    plt.savefig(path_to_save, bbox_inches = "tight")
    tikzplotlib.save(utilsforminds.visualization.format_path_extension(path_to_save))

def plot_SNP(reordered_SNPs_info_df, path_to_save : str, bar_width = 'auto', opacity = 0.8, format = 'eps', xticks_fontsize = 6, diagonal_xtickers = False):
    """
    
    Parameters
    ----------
        name_numbers : dict
            For example, name_numbers['enriched'] == [0.12, 0.43, 0.12] for RMSE
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]
    """

    reordered_SNPs_info_df_copied = reordered_SNPs_info_df.copy()
    reordered_SNPs_info_df_copied = reordered_SNPs_info_df_copied.sort_values(by = 'weights', ascending = False)
    top_20_SNPs_names = list(reordered_SNPs_info_df_copied.loc[:, "SNP"][:20])
    top_20_SNPs_weights = list(reordered_SNPs_info_df_copied.loc[:, "weights"][:20])
    top_20_SNPs_colors = list(reordered_SNPs_info_df_copied.loc[:, "color_chr"][:20])
    fig = plt.figure(figsize = (7, 4))

    n_groups = 10

    if bar_width == 'auto':
        bar_width_ = 0.1

    ## create plot
    ax_1 = plt.subplot(2, 1, 1)
    index = np.arange(n_groups)

    ## set range
    min_, max_ = np.min(top_20_SNPs_weights), np.max(top_20_SNPs_weights)
    plt.ylim([0.5 * min_, 1.2 * max_])

    rects_list = []
    plt.bar(np.arange(10), top_20_SNPs_weights[:10], alpha = opacity, color = top_20_SNPs_colors[:10])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10])
    # plt.legend()
    plt.title('Top-20 SNPs')
    for obj in ax_1.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    
    ax_2 = plt.subplot(2, 1, 2)
    plt.ylim([0.5 * min_, 1.2 * max_])
    plt.bar(np.arange(10), top_20_SNPs_weights[10:], alpha = opacity, color = top_20_SNPs_colors[10:])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:])
    for obj in ax_2.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)


if __name__ == "__main__":
    pass
    # for idx in [103, 104, 107, 108, 109, 110, 111]:
    #     list_dict = load_csv_columns_into_list(f'/Users/seohun/projects/python/KDD_factorization_representation/out/result_{idx}/loss.csv')
    #     list_dict_ = {}
    #     list_dict_[r'$\Vert F_l = Y_l \Vert _F$'] = list_dict['F_l=Y_l']
    #     list_dict_[r'$\Vert A = U \Vert _F$'] = list_dict['A=U']
    #     list_dict_[r'$\Vert B_i = W_i \Vert _F$'] = list_dict['B_i=W_i']
    #     list_dict_[r'$\Vert W_i^TW_i - I\Vert _F$'] = list_dict['W_i^TW_i=I']
    #     list_dict_[r'$objective$'] = list_dict['dual']
    #     plot_multiple_lists(list_dict_, f"./out/result_{idx}/loss.eps")

    # plot_bar_charts('dummy', {'Frank':[12.7, 0.4, 4.4, 5.3, 7.1, 3.2], 'Guido':[6.3, 10.3, 10, 0.3, 5.3, 2.9]}, ['RR', 'Lasso', 'SVR', 'CNN', 'SVR', 'LR'], ytitle="RMSE of Prediction of TRIAILB-A")

    # item_length_dict = {0: 3, 1: 4, 4: 2}
    # pick_keep_probs_dict = {-1: 0.3, 0: 0.5}
    # keep_non_prob_item = True
    # print(random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item))
    


