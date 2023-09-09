#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/04/26 19:03
# @Author : ZYC
# @Site : 15207269894@163.com
# @File : test_plot.py
# @Software: PyCharm

import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def smoothed(initial_data, weight=0.95):
    size = initial_data.shape[1]
    last = initial_data[:, 0]
    smoothed_data = []
    for i in range(size):
        sm = last * weight + (1 - weight) * initial_data[:, i]
        smoothed_data.append(sm)
        last = sm
    smoothed_data = np.stack(smoothed_data, axis=1)
    return smoothed_data


def read_csv(current_file_dir, weight):
    win_rates = []
    steps = []
    for file_csv in list(current_file_dir.glob('*.csv')):
        tmp = np.loadtxt(str(file_csv), dtype=np.str, delimiter=",")
        data = tmp[1:, 1:].astype(np.float)
        label = tmp[1:, 0].astype(np.float)
        win_rates.append(data[:, 1])
        steps.append(label)
    if len(steps) != 0:
        max_step = min([data.shape[0] for data in win_rates])
        win_rates_ = [data[:max_step] for data in win_rates]
        steps_rates_ = [data[:max_step] for data in steps]
        win_rates_sm = smoothed(np.array(win_rates_), weight)
        time = np.array(steps_rates_)
        return time[0], win_rates_sm
    else:
        return None, None


def read_csv_2(current_file_dir, weight):
    win_rates = []
    steps = []
    for file_csv in list(current_file_dir.glob('*.csv')):
        tmp = np.loadtxt(str(file_csv), dtype=np.str, delimiter=",")
        data = tmp[1:, 1:].astype(np.float)
        win_rates.append(data[:, 1])
        steps.append(data[:, 0])
    if len(steps) != 0:
        max_step = min([data.shape[0] for data in win_rates])
        win_rates_ = [data[:max_step] for data in win_rates]
        steps_rates_ = [data[:max_step] for data in steps]
        win_rates_sm = smoothed(np.array(win_rates_), weight)
        time = np.stack(steps_rates_)
        return time[0], win_rates_sm
    else:
        return None, None


def fun_sc2_plot(env_name, weight, plt_row, plt_col, num, grid):

    algo_name = ['EXPODE', 'CDS', 'QMIX', 'QPLEX', 'EMC', 'EMC-wo-em', 'VDN', 'CW_QMIX', 'OW_QMIX']
    algo_paths = ['EXPODE/sc2', 'cds/sc2', 'qmix/sc2', 'qplex/sc2', 'emc/sc2', 'emc/sc2_noem', 'vdn/sc2', 'wqmix/sc2/cw_qmix', 'wqmix/sc2/ow_qmix']

    times = []
    win_rates = []
    for i, algo_path in enumerate(algo_paths):
        current_file_dir = Path(__file__).resolve().parent / algo_path / env_name
        if 'cds' in algo_path:
            time, win_rate = read_csv(current_file_dir, weight)
        else:
            time, win_rate = read_csv_2(current_file_dir, weight)
        times.append(time)
        win_rates.append(win_rate)

    plt.subplot(grid[num//(plt_col+1), num%plt_col])
    color = ['r', 'g', 'm', 'b', 'y', 'orange', 'deepskyblue', 'darkcyan', 'slategray']
    for (j, algo), time, win_rate in zip(enumerate(algo_name), times, win_rates):
        if time is not None:
            if num == 2:
                ax = sns.tsplot(time=time, data=win_rate, color=color[j], condition=algo, legend=True)
            else:
                sns.tsplot(time=time, data=win_rate, color=color[j], condition=algo, legend=False)
            # sns.tsplot(time=time, data=win_rate, color=color[j], condition=algo)

            plt.xlabel("Steps")
            plt.ylabel("Test_win_rate %")
            plt.title(env_name)
            plt.xlim(0, 2e6)

            plt.yticks(np.arange(0, 1.2, 0.2))
    if num == 2:
        plt.legend(loc='upper center', bbox_to_anchor=(-0.89, 1.3), ncol=len(algo_name), frameon=False,
                   columnspacing=1.0, handletextpad=0.5)


if __name__ == '__main__':
    ''' SC2 '''

    env_name = "3s_vs_5z"
    rolling_intv = 20
    weight = 0.9

    sns.set(style="darkgrid", font_scale=1.5)
    env_names = ['corridor', 'MMM2', '3s5z_vs_3s6z', '6h_vs_8z', '3s_vs_5z', '5m_vs_6m']
    algo_name = ['EXPODE', 'CDS', 'QMIX', 'QPLEX', 'EMC', 'EMC-wo-em', 'VDN', 'CW_QMIX', 'OW_QMIX']
    plt_row = 2
    plt_col = 3

    fig, ax = plt.subplots(plt_row, plt_col, figsize=(18, 8))

    grid = plt.GridSpec(plt_row, plt_col, hspace=0.4, wspace=0.3)

    for i, env_name in enumerate(env_names):
        fun_sc2_plot(env_name, weight, plt_row, plt_col, i+1, grid)

    plt.savefig('SC2.svg')
    plt.show()



