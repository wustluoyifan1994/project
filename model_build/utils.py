# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/8 15:31
# @Author  : pineapple
# @File    : utils
# @Software: PyCharm


import numpy as np


def n0(x): return sum(x == 0)


def n1(x): return sum(x == 1)


def get_label(x, bad, good):
    if x >= bad:
        return 1
    elif x <= good:
        return 0
    else:
        return -1


def ab(points=600, odds=0.04, pdo=100):
    b = pdo / np.log(2)
    a = points + b * np.log(odds)
    return {"a": a, "b": b}


# 概率转分数
def proba2score(y_proba, points=600, pdo=100, odds=0.04):
    a, b = ab(points, odds, pdo).values()
    return a - b * np.log(y_proba / (1 - y_proba))
