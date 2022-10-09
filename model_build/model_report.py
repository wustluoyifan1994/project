# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/8 17:45
# @Author  : pineapple
# @File    : model_report
# @Software: PyCharm


from data_anlysis import calculate_overdue_rate_by_month


class ModelReport:

    def __init__(self, data, loan_date, y):
        self.data = data
        self.loan_date = loan_date
        self.y = y
        self.res = {}

    def _calculate_overdue_rate_by_month(self):
        gb = calculate_overdue_rate_by_month(self.data, self.loan_date, self.y)
        self.res["overdue_rate_by_month"] = gb

    def _calculate_overdue_rate_by_samples(self):
        pass
