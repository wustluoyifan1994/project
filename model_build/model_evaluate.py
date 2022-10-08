# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/8 15:32
# @Author  : pineapple
# @File    : model_evaluate
# @Software: PyCharm


import numpy as np
import pandas as pd
from utils import n0, n1
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.metrics import roc_auc_score
from pandas import DataFrame


warnings.filterwarnings("ignore")
matplotlib.rc("font", **{"family": "Heiti TC"})
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100


def calculate_iv_ks(data: DataFrame, bin: str, y: str,
                    default_bad_ratio: float, is_score: bool = True) -> DataFrame:
    data = data.copy()
    gb = data.groupby(by=bin, as_index=False)[
        y].agg({"good": n0, "bad": n1})
    if not is_score:
        gb = gb.sort_index(ascending=False)
    gb = (
        gb.assign(
            group=lambda x: x.good +
            x.bad).assign(
            group_ratio=lambda x: x.group /
            x.group.sum(),
            bad_ratio=lambda x: x.bad /
            x.group,
            cum_bad_rate=lambda x: x.bad.cumsum() /
            x.bad.sum(),
            cum_good_rate=lambda x: x.good.cumsum() /
            x.good.sum(),
        ).assign(
            lift=lambda x: x.bad_ratio /
            default_bad_ratio,
            cum_lift=lambda x: (
                x.bad.cumsum() /
                x.group.cumsum()) /
            default_bad_ratio,
            ks=lambda x: np.abs(
                x.cum_bad_rate -
                x.cum_good_rate),
            iv=lambda x: (
                x.bad /
                x.bad.sum() -
                x.good /
                x.good.sum()) *
            np.log(
                ((x.bad + 0.001) /
                 x.bad.sum()) /
                (
                    (x.good + 0.001) /
                    x.good.sum())),
        ).assign(
            max_ks=lambda x: np.abs(
                x.ks).max(),
            total_iv=lambda x: x.iv.sum()))

    return gb


def plot_auc_ks(gb: DataFrame, auc: float) -> None:
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 110
    x1 = [0] + gb["cum_good_rate"].tolist()
    y1 = [0] + gb["cum_bad_rate"].tolist()
    plt.plot(x1, y1)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.text(0.85, 0.05, f"auc:{round(auc, 4)}")
    plt.fill_between(x1, y1, color="grey", alpha=0.3)
    plt.title("ROC曲线")
    plt.xlabel("累计好样本占比")
    plt.ylabel("累计坏样本占比")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.show()

    # ks
    x2 = [0] + gb["group_ratio"].cumsum().tolist()
    y21 = [0] + gb["cum_bad_rate"].tolist()
    y22 = [0] + gb["cum_good_rate"].tolist()
    y23 = [0] + (gb["cum_bad_rate"] - gb["cum_good_rate"]).tolist()
    plt.plot(x2, y21, label="fpr")
    plt.plot(x2, y22, label="tpr")
    plt.plot(x2, y23, label="ks")
    max_ks = gb.ks.max()
    max_ks_idx = gb.ks.idxmax()
    max_ks_group_ratio = gb.loc[0:max_ks_idx, "group_ratio"].sum()
    plt.text(
        max_ks_group_ratio + 0.02,
        max_ks + 0.02,
        f"ks:{round(max_ks, 4)}")
    plt.plot([max_ks_group_ratio, max_ks_group_ratio],
             [0, max_ks], "--", color="grey")
    plt.title("K-S曲线")
    plt.xlabel("累计样本占比")
    plt.ylabel("累计好/坏样本占比")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.01, 0.1))
    # plt.savefig("K-S曲线.png")
    plt.legend()
    plt.show()


def calculate_cut_off_data(
        data: DataFrame, y: str, score: str, split_points: list, is_score: bool = True, is_plot: bool = True) -> DataFrame:
    split_points = [float("-inf")] + split_points + [float("inf")]
    data = data.copy()
    default_bad_ratio = data[y].mean()
    data["bin"] = pd.cut(data[score], bins=split_points, right=False)
    gb1 = calculate_iv_ks(
        data=data,
        bin="bin",
        y=y,
        default_bad_ratio=default_bad_ratio,
        is_score=is_score)
    if is_plot:
        gb2 = calculate_iv_ks(
            data=data,
            bin=score,
            y=y,
            default_bad_ratio=default_bad_ratio,
            is_score=is_score)
        auc = roc_auc_score(data[y], -data[score])
        plot_auc_ks(gb2, auc)
    return gb1


if __name__ == '__main__':
    import joblib
    res = joblib.load(
        "/Users/luoyifan/DailyAnalysis/api2shop/modelFile/thirdSourceModel/all_feature/res.pkl")
    split_points = list(range(400, 910, 40))
    gb = calculate_cut_off_data(res, 'label', 'score', split_points)
