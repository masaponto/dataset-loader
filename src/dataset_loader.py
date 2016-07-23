#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import csv
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.datasets import load_svmlight_file
from dexter.dexter_parser import Dexter


class Loader:

    def __init__(self, path=os.path.expanduser('~/Data/')):
        self.path = path

    def load_qsar(self):
        try:
            with open(self.path + 'qsar/biodeg.csv') as file:
                reader = csv.reader(file, delimiter=';',)
                xs, ys = [], []
                for row in reader:
                    y = -1 if row[-1] == 'NRB' else 1
                    ys.append(y)
                    xs.append([float(x) for x in row[:-1]])

            return Bunch(data=np.array(xs),
                         target=np.array(ys),
                         feature_names='qsar',
                         DESCR='no')

        except EnvironmentError:
            print('Data file not found in ' + self.path)

    def load_madelon(self):
        try:
            madelon = load_svmlight_file(self.path + 'madelon/madelon')
            madelon_test = load_svmlight_file(self.path + 'madelon/madelon.t')

            xs = np.r_[madelon[0].todense(), madelon_test[0].todense()]
            ys = np.r_[madelon[1], madelon_test[1]]

            return Bunch(data=xs,
                         target=ys,
                         feature_names='madelon',
                         DESCR='no')

        except EnvironmentError:
            print('Data file not found in ' + self.path)

    def load_dexter(self):
        try:
            return Dexter().get_datas()
        except EnvironmentError:
            print('Data file not found in ' + self.path)

    def load_farmads(self):
        try:
            farm_ads = load_svmlight_file(
                self.path + 'Farm Ads/farm-ads-vect', n_features=54877)
            xs = np.array(farm_ads[0].todense())
            ys = farm_ads[1]

            return Bunch(data=xs,
                         target=ys,
                         feature_names='madelon',
                         DESCR='no')

        except EnvironmentError:
            print('Data file not found in ' + self.path)


def main():
    # data = Loader().load_dexter()
    data = Loader().load_farmads()

if __name__ == "__main__":
    main()
