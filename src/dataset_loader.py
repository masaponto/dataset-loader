#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import csv
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.datasets import load_svmlight_file
from dexter.dexter_parser import Dexter
from sklearn.datasets import fetch_mldata


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

    def load_gisette(self):
        try:
            with open(self.path + 'gisette/gisette_train.data') as file:
                reader = csv.reader(file, delimiter=' ')
                xs = [[float(e) for e in row[:-1]] for row in reader]

            with open(self.path + 'gisette/gisette_valid.data') as file:
                reader = csv.reader(file, delimiter=' ')
                xs = xs + [[float(e) for e in row[:-1]] for row in reader]

            train_path = self.path + 'gisette/gisette_train.labels'
            valid_path = self.path + 'gisette/gisette_valid.labels'

            train_ys = np.loadtxt(train_path)
            valid_ys = np.loadtxt(valid_path)

            ys = np.r_[train_ys, valid_ys]

            return Bunch(data=np.array(xs),
                         target=ys,
                         feature_names='gisette',
                         DESCR='no')

        except EnvironmentError:
            print('Data file not found in ' + self.path)

    def load_mnist(self):
        data_set = fetch_mldata('MNIST original')
        data_set.target = data_set.target + 1
        data_set.target = np.array([int(y) for y in data_set.target])
        return Bunch(data=data_set.data,
                     target=data_set.target,
                     feature_names='mnist',
                     DESCR='no')


def main():
    # data = Loader().load_dexter()
    #data = Loader().load_farmads()
    #data = Loader().load_gisette()
    data = Loader().load_mnist()


if __name__ == "__main__":
    main()
