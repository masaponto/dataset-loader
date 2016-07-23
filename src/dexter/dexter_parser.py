#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets.base import Bunch
import numpy as np
import os.path


class Dexter:

    def __init__(self, db_name="dexter", path=os.path.expanduser('~/Data/Dexter/'), f_num=2000):
        self.db_name = db_name
        self.f_num = f_num

        self.path = path + self.db_name

        train = '_train'
        valid = '_valid'

        self.X_train = np.array(self.__set_xdata(train))
        self.y_train = np.array(self.__parce_label(train))

        self.X_valid = np.array(self.__set_xdata(valid))
        self.y_valid = np.array(self.__parce_label(valid))

        self.data = np.r_[self.X_train, self.X_valid]
        self.target = np.r_[self.y_train, self.y_valid]
        # print(len(self.data))
        # print(len(self.target))

    def __set_xdata(self, fs):
        """
        Exmples:
        >>> d = Dexter('test', './test/', 10)

        >>> d._Dexter__set_xdata('_train')
        [[10, 20, 0, 0, 0, 66, 0, 0, 0, 1], [0, 10, 20, 66, 0, 0, 0, 1, 3, 0]]

        """

        f_dics = self.__parce_data(fs)
        # print(fs)
        #print(sorted(f_dics[0].items(), key=lambda x : x[0]))
        X_train = []
        xdata = [0 for i in range(self.f_num)]

        for i in range(len(f_dics)):
            xdata = [0 for i in range(self.f_num)]
            for j in range(1, self.f_num + 1):
                if j in f_dics[i]:
                    xdata[j - 1] = f_dics[i][j]
            X_train.append(xdata)

        return X_train

    def __parce_data(self, fs):
        """
        Exmples:
        >>> d = Dexter('test', './test/', 10)

        >>> dics = d._Dexter__parce_data('_train')

        >>> [sorted(dic.items(), key=lambda x : x[0]) for dic in dics]
        [[(1, 10), (2, 20), (6, 66), (10, 1)], [(2, 10), (3, 20), (4, 66), (8, 1), (9, 3)]]

        """

        path = self.path + fs + '.data'
        with open(path) as d_file:
            str_rows = d_file.readlines()
            str_rows = [row.rstrip(' \n') for row in str_rows]
            rows = [row.split(' ') for row in str_rows]
            str_dics = [dict(r.split(':') for r in row) for row in rows]
            f_dics = [{int(k): int(v) for k, v in dic.items()}
                      for dic in str_dics]
        return f_dics

    def __parce_label(self, fs):
        """
        Exmples:
        >>> d = Dexter('test', './test/', 10)

        >>> d._Dexter__parce_label('_train')
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        """

        path = self.path + fs + '.labels'
        with open(path) as l_file:
            str_row = l_file.readlines()
            str_row = [row.rstrip('\n') for row in str_row]
            str_row = [row.rstrip(' ') for row in str_row]
            y_train = list(map(int, str_row))
        return y_train

    def get_divided_datas(self):
        return [self.X_train, self.X_valid, self.y_train, self.y_valid]

    def get_datas(self):
        data = np.r_[self.X_train, self.X_valid]
        target = np.r_[self.y_train, self.y_valid]

        return Bunch(data=data,
                     target=target,
                     feature_names='dexter',
                     DESCR='no')


def main():
    datas = Dexter().get_datas()
    print('done')

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
