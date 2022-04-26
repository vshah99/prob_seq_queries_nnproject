#################################################################################
#
#             Project Title:  Visualizations
#             Author:         Sam Showalter
#             Date:           2022-02-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import read_pkl, write_pkl

#################################################################################
#   Function-Class Declaration
#################################################################################

#######################################################################
# Visualizations for Search v. Sample Scatter plots
#######################################################################

def plot_search_vs_sample(gt_data_path, samp_data_path_imp, samp_data_path_rand,hist_len,seq_len,
                   sample_sizes, shuffle = False,plot_cols=2, search_type="Ground truth"):
    assert len(sample_sizes)%plot_cols == 0,"Must have number of samples ({}) divisible by plot_cols ({})"\
        .format(len(sample_sizes),plot_cols)
    title = "Number of samples: {}"
    fig, axs = plt.subplots(len(sample_sizes)//plot_cols,plot_cols, figsize = (len(sample_sizes)*3,plot_cols*6))
    gt_data = read_pkl(gt_data_path)
    samp_data_imp = read_pkl(samp_data_path_imp)
    samp_data_rand = read_pkl(samp_data_path_rand)
    data_dict = {"gt_data":gt_data,
                 "imp_samp":samp_data_imp,
                 "rand_samp":samp_data_rand,}
    # Check if it is a dictionary
    for key,data in data_dict.items():
        if isinstance(data,dict):
            data_dict[key] = data['sample_estimates']

    for i in range(len(sample_sizes)):
        # data = read_pkl(paths[i])
        ref = np.arange(0,1,0.01)
        axs[i//plot_cols][i%plot_cols].scatter(data_dict['gt_data'], data_dict['rand_samp'][:,:sample_sizes[i]].mean(axis = -1), color = "green", label = "random")
        axs[i//plot_cols][i%plot_cols].scatter(data_dict['gt_data'], data_dict['imp_samp'][:,:sample_sizes[i]].mean(axis = -1), color = "blue", label = "importance")
        axs[i//plot_cols][i%plot_cols].set_title(title.format(sample_sizes[i]))
        axs[i//plot_cols][i%plot_cols].plot(list(ref),list(ref),linestyle="dashed", color = "red",linewidth=2)
        axs[i//plot_cols][i%plot_cols].legend()
        axs[i//plot_cols][i%plot_cols].set_ylim((0,1))

    fig.supxlabel(f"{search_type} query probability: {hist_len}h-{seq_len}s")
    fig.supylabel("query probability from sampling")
    plt.tight_layout()
    return axs

@torch.no_grad()
def plot_entropy_variance(
    data_path,
    temperatures=[0.1,0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10,50,100],
    std = True,
 ):
    data_dict = read_pkl(data_path)
    logits = data_dict['logits']
    ref_probs = F.softmax(logits,dim=-1)
    sammples = defaultdict(dict)
    for t in temperatures:
        temp_logits = logits/t # (seqs x samples x vocab)
        temp_probs = F.softmax(temp_logits,dim=-1)
        # (vocab)
        entropy_est_vocab = -((temp_probs/ref_probs)*torch.log(temp_probs)).sum(dim=1)
        # (seqs, 1)
        entropy_est = torch.gather(entropy_est_vocab, dim=1, data_dict['excluded_tokens'].unsqueeze(0)).flatten()
        variance = torch.gather(torch.var(temp_probs,dim=1),dim=1,data_dict['excluded_tokens'].unsqueeze(0)).flatten()
        samples[t] = {"entropy":entropy_est, "variance": variance_est}




def plot_search_vs_sample_mae(
    gt_data_path, samp_data_path_imp,
    samp_data_path_rand, hist_len, seq_len,
 ):
    title = "Error vs. Sample Sizes"
    fig, axs = plt.subplots(1, 2, figsize = (12, 5))
    fig.suptitle(title)
    gt_data = read_pkl(gt_data_path)
    samp_data_imp = read_pkl(samp_data_path_imp)
    samp_data_rand = read_pkl(samp_data_path_rand)

    ax = axs[0]
    total_samples = samp_data_imp.shape[1]
    imp_ests = np.cumsum(samp_data_imp, axis=-1) / np.arange(1, total_samples+1)[np.newaxis, :]
    imp_ests = abs(gt_data[:, np.newaxis] - imp_ests)
    mc_ests = np.cumsum(samp_data_rand, axis=-1) / np.arange(1, total_samples+1)[np.newaxis, :]
    mc_ests = abs(gt_data[:, np.newaxis] - mc_ests)

    ax.fill_between(np.arange(1, total_samples+1), y1=imp_ests.mean(axis=0)-imp_ests.std(axis=0), y2=imp_ests.mean(axis=0)+imp_ests.std(axis=0), color="blue", alpha=0.3)
#    ax.fill_between(np.arange(1, total_samples+1), y1=mc_ests.mean(axis=0)-mc_ests.std(axis=0), y2=mc_ests.mean(axis=0)+mc_ests.std(axis=0), color="green", alpha=0.3)

    ax.plot(np.arange(1, total_samples+1), imp_ests.mean(axis=0), color = "blue", label="importance")
    ax.plot(np.arange(1, total_samples+1), mc_ests.mean(axis=0), color = "green", label="random")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Number of Samples per Estimate")
    ax.set_xscale("log")
    ax.legend()

    ax = axs[1]
    imp_ests = np.cumsum(samp_data_imp, axis=-1) / np.arange(1, total_samples+1)[np.newaxis, :]
    imp_ests = abs(gt_data[:, np.newaxis] - imp_ests) / gt_data[:, np.newaxis]
    mc_ests = np.cumsum(samp_data_rand, axis=-1) / np.arange(1, total_samples+1)[np.newaxis, :]
    mc_ests = abs(gt_data[:, np.newaxis] - mc_ests) / gt_data[:, np.newaxis]

    ax.fill_between(np.arange(1, total_samples+1), y1=imp_ests.mean(axis=0)-imp_ests.std(axis=0), y2=imp_ests.mean(axis=0)+imp_ests.std(axis=0), color="blue", alpha=0.3)
#    ax.fill_between(np.arange(1, total_samples+1), y1=mc_ests.mean(axis=0)-mc_ests.std(axis=0), y2=mc_ests.mean(axis=0)+mc_ests.std(axis=0), color="green", alpha=0.3)


    ax.plot(np.arange(1, total_samples+1), imp_ests.mean(axis=0), color = "blue", label="importance")
    ax.plot(np.arange(1, total_samples+1), mc_ests.mean(axis=0), color = "green", label="random")
    ax.set_ylabel("Mean Relative Absolute Error")
    ax.set_xlabel("Number of Samples per Estimate")
    ax.set_xscale("log")
    ax.legend()

    plt.tight_layout()
    return axs


#################################################################################
#   Main Method
#################################################################################



