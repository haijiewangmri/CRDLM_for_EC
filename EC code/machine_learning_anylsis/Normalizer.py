# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from abc import abstractmethod


class Normalizer:
    """
    z-score only
    """
    def __init__(self, train_df, minmax=False):
        self.minmax = minmax
        self.std = np.array([])
        self.mean = np.array([])

        self._init_std_mean(train_df)

    def _init_std_mean(self, dataframe):
        data = np.array(dataframe.values[:, 1:])
        if self.minmax:
            self.std = np.max(data, axis=0) - np.min(data, axis=0)
            self.mean = np.min(data, axis=0)
        else:
            self.std = np.std(data, axis=0)
            self.mean = np.mean(data, axis=0)

    @abstractmethod
    def run(self, dataframe):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()

        data = data.astype(np.float32)
        data -= self.mean
        data /= self.std
        new_data = np.concatenate((label[..., np.newaxis], data), axis=1)

        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=feature_name)
        return new_dataframe


if __name__ == '__main__':
    data_path = r'D:\data\EC 20231101\train 217.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=1, how='any')
    z_score = Normalizer(df)
    z_score_dataframe = z_score.run(df)
    z_score_dataframe.to_csv(r'D:\data\EC 20231101\train 217 zscore.csv')
