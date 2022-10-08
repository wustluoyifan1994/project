# !/usr/bin/env python
# coding=utf-8
# @Time    : 2022/10/8 15:28
# @Author  : pineapple
# @File    : data_analysis
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame


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
        data: DataFrame, y: str, features: list = None, threshold: float = 0.95) -> dict:
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
        missing_data["missing_rate"] >= threshold, "variable"
    ].tolist()
    return {
        "missing_data": missing_data,
        "del_features_by_missing": del_features_by_missing,
    }


def calculate_mode_rate(data: DataFrame, y: str,
                        features: list = None, threshold: float = 0.95) -> dict:

    if features is None:
        features = data.drop(y, axis=1).columns.tolist()
    mode_rate_list = []
    for var in features:
        mode_rate = data[var].value_counts().values[0] / data.shape[0]
        mode_rate_list.append(mode_rate)
    mode_data = (
        pd.DataFrame({"variable": features, "mode_rate": mode_rate_list})
        .sort_values(by="mode_rate", ascending=False)
        .reset_index(drop=True)
    )
    del_features_by_mode = mode_data.loc[
        mode_data["mode_rate"] >= threshold, "variable"
    ].tolist()
    return {
        "mode_data": mode_data,
        "del_features_by_mode": del_features_by_mode}


if __name__ == '__main__':
    data = pd.DataFrame({
        "x1": list(np.random.randint(1, 100, 1000)) + [np.nan] * 1000,
        "y": np.random.randint(0, 2, 2000)
    })

    missing_data, del_var_by_missing_rate = calculate_missing_rate(data, "y").values()
    mode_data, del_var_by_mode_rate = calculate_mode_rate(data, "y").values()
