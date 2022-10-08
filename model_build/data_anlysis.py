# !/usr/bin/env python
# coding=utf-8
import string

# @Time    : 2022/10/8 15:28
# @Author  : pineapple
# @File    : data_analysis
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame


def n0(x): return sum(x == 0)


def n1(x): return sum(x == 1)


def calculate_overdue_rate_by_month(
        data: DataFrame, loan_date: str, y: str) -> DataFrame:
    """
    按月计算逾期率分布
    :param data:
    :param loan_date:
    :param y:
    :return:
    """
    data = data.copy()
    data[loan_date] = data[loan_date].apply(lambda x: str(x)[:7])
    gb = data.groupby(by=loan_date)[y].agg([
        ("计数", np.size), ("坏", np.sum), ("坏样本率", np.mean)
    ])
    return gb


def calculate_missing_rate(
        data: DataFrame, y: str, features: list = None, threshold: float = 0.95) -> object:
    """
    计算特征的缺失率
    :param data: 数据
    :param y: label
    :param features: 需要计算缺失率的特征集合
    :param threshold: 剔除特征缺失率>=N的阈值
    :return:
    """
    if features is None:
        features = data.drop(y, axis=1).columns.tolist()
    missing_data = (
        pd.DataFrame(
            {
                "variable": features,
                "missing_rate": data[features].isnull().sum() / data.shape[0],
            }
        )
        .sort_values(by="missing_rate", ascending=False)
        .reset_index(drop=True)
    )
    del_features_by_missing = missing_data.loc[
        missing_data["missing_rate"] >= threshold, "feature"
    ].tolist()
    return {
        "missing_data": missing_data,
        "del_features_by_missing": del_features_by_missing,
    }

