# -*- coding: utf-8 -*-
import os
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


class UpSampling(object):
    def __init__(self, **kwargs):
        self._model = RandomOverSampler(**kwargs, random_state=0)
        self._name = 'UpSampling'

    def run(self, dataframe, store_path=''):
        data = np.array(dataframe.values[:, 1:])
        label = np.array(dataframe['label'].tolist())
        feature_name = dataframe.columns.tolist()

        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = ['Balance' + str(index) for index in range(data_resampled.shape[0])]
        new_data = np.concatenate((label_resampled[..., np.newaxis], data_resampled), axis=1)
        # np.newaxis插入新的维度方便两个拼接
        new_dataframe = pd.DataFrame(data=new_data, index=new_case_name, columns=feature_name)

        if store_path != '':
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            new_dataframe.to_csv(os.path.join(store_path, '01_{}_features.csv'.format(self._name)))
        return new_dataframe


class DownSampling(object):
    def __init__(self, seed,**kwargs):
        self._model = RandomUnderSampler(**kwargs, random_state=seed)
        self._name = 'DownSampling'

    def run(self, dataframe, store_path=''):
        data = np.array(dataframe.values[:, 1:])
        label = np.array(dataframe['label'].tolist())
        feature_name = dataframe.columns.tolist()

        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = ['Balance' + str(index) for index in range(data_resampled.shape[0])]
        new_data = np.concatenate((label_resampled[..., np.newaxis], data_resampled), axis=1)
        # np.newaxis插入新的维度方便两个拼接
        new_dataframe = pd.DataFrame(data=new_data, index=new_case_name, columns=feature_name)

        if store_path != '':
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            new_dataframe.to_csv(os.path.join(store_path, '01_{}_features.csv'.format(self._name)))
        return new_dataframe


class SmoteSamplingDataBalance(object):
    def __init__(self, **kwargs):
        self._model = SMOTE(**kwargs, random_state=0)
        self._name = 'SMOTE'

    def run(self, dataframe, store_path=''):
        data = np.array(dataframe.values[:, 1:])
        label = np.array(dataframe['label'].tolist())
        feature_name = dataframe.columns.tolist()
        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = ['Generate' + str(index) for index in range(data_resampled.shape[0])]
        new_data = np.concatenate((label_resampled[..., np.newaxis], data_resampled), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=new_case_name, columns=feature_name)

        if store_path != '':
            if os.path.isdir(store_path):
                new_dataframe.to_csv(os.path.join(store_path, '01_{}_features.csv'.format(self._name)))
            else:
                new_dataframe.to_csv(store_path)
        return new_dataframe


if __name__ == '__main__':
    data_path = r'E:\study\EC\subtype\20240403 单独复旦\chaoyang154MRI.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=1, how='any')
    # smote = SmoteSamplingDataBalance()
    up = UpSampling()
    # save_path = r'C:\Users\HJ Wang\Desktop'
    output_df = up.run(df)
    print(output_df.shape)
