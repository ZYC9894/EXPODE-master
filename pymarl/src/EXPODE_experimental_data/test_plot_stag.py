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


def fun_sc2_plot(env_name, weight):
    sns.set(style="darkgrid", font_scale=1.5)

    algo_name = ['EXPODE-VDN', 'EXPODE-QMIX','VDN', 'QMIX', 'EMC', 'QPLEX', 'CW_QMIX', 'OW_QMIX']
    algo_paths = ['EXPODE/stag_hunt', 'EXPODE/stag_hunt/QMIX', 'vdn/stag_hunt', 'qmix/stag_hunt', 'emc/stag_hunt', 'qplex/stag_hunt', 'wqmix/stag_hunt/cw_qmix', 'wqmix/stag_hunt/ow_qmix']

    times = []
    win_rates = []
    algo_names = []
    for i, algo_path in enumerate(algo_paths):
        current_file_dir = Path(__file__).resolve().parent / algo_path / env_name
        time, win_rate = read_csv_2(current_file_dir, weight)
        if time is not None:
            times.append(time)
            win_rates.append(win_rate)
            algo_names.append(algo_name[i])

    color = ['r', 'g', 'm', 'b', 'y', 'orange', 'deepskyblue', 'darkcyan']
    for (j, algo), time, win_rate in zip(enumerate(algo_names), times, win_rates):
        if time is not None:
            sns.tsplot(time=time, data=win_rate, color=color[j], condition=algo)

    plt.xlabel("Steps")
    plt.ylabel("Test_reward")
    plt.title("Predator Prey")
    plt.xlim(0, 1e6)

    plt.tight_layout()
    plt.savefig('PP.svg')
    plt.show()


if __name__ == '__main__':
    # stag
    env_name = "origin"
    rolling_intv = 20
    weight = 0.9
    fun_sc2_plot(env_name, weight)
