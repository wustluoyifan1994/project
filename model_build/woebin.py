# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/8 15:38
# @Author  : pineapple
# @File    : woebin
# @Software: PyCharm


from optbinning import OptimalBinning
import pandas as pd
import re
import numpy as np
import scorecardpy as sc


class WOE:
    def __init__(self, data,
                 y,
                 x=None,
                 breaks_list=None,
                 special_values=None,
                 count_distr_limit=0.05,
                 bin_num_limit=8,
                 method="tree",
                 **kwargs):
        self.data = data
        self.y = y
        if x is None:
            self.x = self.data.drop(self.y, axis=1).columns.tolist()
        else:
            self.x = x
        self.breaks_list = breaks_list
        self.special_values = special_values
        self.count_distr_limit = count_distr_limit
        self.bin_num_limit = bin_num_limit
        self.method = method
        self.print_info = kwargs.get('print_info', False)

    @property
    def get_woebin(self):
        bins = sc.woebin(
            dt=self.data,
            y=self.y,
            x=self.x,
            breaks_list=self.breaks_list,
            special_values=self.special_values,
            count_distr_limit=self.count_distr_limit,
            bin_num_limit=self.bin_num_limit,
            print_info=self.print_info
        )
        for key in bins.keys():
            bins[key] = (
                bins[key].assign(
                    default_bad_rate=lambda x: x.bad.sum() /
                    x['count'].sum(),
                    lift=lambda x: x.badprob /
                    (
                        x.bad.sum() /
                        x['count'].sum()),
                    ks=lambda x: np.abs(
                        x.bad.cumsum() /
                        x.bad.sum() -
                        x.good.cumsum() /
                        x.good.sum()),
                ).assign(
                    max_ks=lambda x: x.ks.max()))
        return bins

    @property
    def get_woebin_auto_asc_desc(self):
        self.breaks_list = {}
        for key in self.x:
            optb = OptimalBinning(name='',
                                  dtype="numerical",
                                  solver="cp",
                                  divergence='iv',
                                  max_n_bins=self.bin_num_limit,
                                  min_bin_size=self.count_distr_limit,
                                  min_bin_n_event=1,  # 必须为正值
                                  monotonic_trend='auto_asc_desc', special_codes=self.special_values)

            X = self.data[key]
            y = self.data["y"]
            optb.fit(X, y)
            df_bin_ = optb.binning_table.build()
            self.breaks_list[key] = df_bin_.loc[~df_bin_["Bin"].isin(["Special", "Missing", ""])]["Bin"].apply(
                lambda x: re.findall(", (.+?)\\)", str(x))[0]).tolist()[:-1]
            print(self.breaks_list[key])
        bins = self.get_woebin

        return bins


if __name__ == '__main__':
    data2 = pd.DataFrame({
        "x1": np.random.randint(1, 100, 10000),
        "x2": np.random.randint(100, 1000, 10000),
        "y": np.random.randint(0, 2, 10000)
    })
    woe = WOE(data2, "y", x=["x1", "x2"])
    bins2 = woe.get_woebin_auto_asc_desc

    print("over!")
