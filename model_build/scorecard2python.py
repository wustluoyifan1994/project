# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/9 14:16
# @Author  : pineapple
# @File    : scorecard2python
# @Software: PyCharm


import numpy as np
import pandas as pd


def scorecard_if_print(data, special_value=-999):
    data = data.copy().reset_index()
    variable = data.loc[0, "variable"]
    data_len = data.shape[0]
    for i in data.index:
        x = data.loc[i, ""]
        bin_min = data.loc[i, "bin_min"]
        bin_max = data.loc[i, "bin_max"]
        # 检测有无特殊值
        if bin_min == bin_max:
            print("\tif {} ")