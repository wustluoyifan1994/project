# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/8 15:31
# @Author  : pineapple
# @File    : utils
# @Software: PyCharm


import numpy as np
import pandas as pd


def n0(x): return sum(x == 0)


def n1(x): return sum(x == 1)


def get_label(x, bad, good):
    if x >= bad:
        return 1
    elif x <= good:
        return 0
    else:
        return -1


class Proba2Score:

    def __init__(self, point=600, odds=0.04, pdo=100):
        self.point = point
        self.odds = odds
        self.pdo = pdo

    def ab(self):
        b = self.pdo / np.log(2)
        a = self.point + b * np.log(self.odds)
        return {"a": a, "b": b}

    def proba2score(self, y_proba):
        a, b = self.ab().values()
        return a - b * np.log(y_proba / (1 - y_proba))

    def scorecard(self, params, bins):
        a, b = self.ab().values()
        bins_data = pd.concat(bins, ignore_index=True)
        variable = []
        bin_list = []
        score = []
        for i in params.index:
            feature = params.loc[i, "variable"]
            coef = params.loc[i, "coef"]
            if feature == "const":
                base = a - b * coef
                variable.append("Intercept")
                bin_list.append("")
                score.append(base)
            else:
                bins_tmp = bins_data.loc[bins_data["variable"]
                                         == feature.replace("_woe", "")]
                bins_tmp["score"] = bins_tmp["woe"].apply(
                    lambda x: -b * coef * x)
                variable = variable + bins_tmp["variable"].tolist()
                bin_list = bin_list + bins_tmp["bin"].tolist()
                score = score + bins_tmp["score"].tolist()

        scorecard_data = pd.DataFrame({
            "variable": variable,
            "bin": bin_list,
            "score": score
        })
        scorecard_data["score"] = scorecard_data["score"].astype(int)
        x_list = [f'x{i}' for i in range(scorecard_data["variable"].nunique())]
        mapper = dict(zip(list(scorecard_data["variable"].unique()), x_list))
        scorecard_data["variable_no"] = scorecard_data["variable"].apply(lambda x: mapper[x])
        return scorecard_data[[
            "variable_no", "variable", "bin", "score"
        ]]
#
#
# if __name__ == '__main__':
#     y_proba = 0.01
#     y_score = Proba2Score().proba2score(y_proba)
#
#     import joblib
#     file_path = "/Users/luoyifan/DailyAnalysis/api2shop/"
#     model = joblib.load(file_path+"modelFile/thirdSourceModel/all_feature/model.pkl")
#     bins = joblib.load(file_path+"modelFile/thirdSourceModel/all_feature/bins.pkl")
#     model_features = joblib.load(file_path+"modelFile/thirdSourceModel/all_feature/model_features.pkl")
#
#     params = model.params.reset_index().rename(
#         columns={"index": "variable", 0: "coef"})
#     scorecard_data = Proba2Score().scorecard(params, bins)
#     print("over")
