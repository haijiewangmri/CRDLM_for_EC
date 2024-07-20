# -*- coding: utf-8 -*-
# 皮尔逊相关系数，这是对高维特征的一种降维方法，由于几百个样本却有上千个特征，因此防止过拟合需要排除许多价值较低的特征
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


class DimensionReductionByPCC(object):
    def __init__(self, name='PCC', model=None, number=0, is_transform=False, threshold=0.9):
        self._name = name
        self.__model = model
        self.__remained_number = number
        self.__is_transform = is_transform
        self.__threshold = threshold
        self.__selected_index = []

    @staticmethod
    def pcc_similarity(data1, data2):
        return np.abs(pearsonr(data1, data2)[0])

    @staticmethod
    def description():
        text = "Since the dimension of feature space was high, we compared the similarity of each feature pair. " \
               "If the PCC value of the feature pair was larger than 0.99, we removed one of them. After this " \
               "process, the dimension of the feature space was reduced and each feature was independent to each other "
        return text

    def get_selected_feature_by_pcc(self, data, label):
        data = data.astype(np.float32)
        data /= np.linalg.norm(data, ord=2, axis=0)
        for feature_index in range(data.shape[1]):
            is_similar = False
            assert(feature_index not in self.__selected_index)
            for save_index in self.__selected_index:
                if self.pcc_similarity(data[:, save_index], data[:, feature_index]) > self.__threshold:
                    if self.pcc_similarity(data[:, save_index], label) <\
                            self.pcc_similarity(data[:, feature_index], label):
                        self.__selected_index[self.__selected_index.index(save_index)] = feature_index
                    is_similar = True
                    break
            if not is_similar:
                self.__selected_index.append(feature_index)
        self.__selected_index = sorted(self.__selected_index)

    def run(self, dataframe, store_folder=''):
        origin_data = dataframe.values[:, 1:]
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()[1:]
        self.get_selected_feature_by_pcc(data, label)
        # self.__selected_index = [4,5,6]
        new_data = origin_data[:, self.__selected_index]
        new_feature_name = [feature_name[t] for t in self.__selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)

        if store_folder and os.path.isdir(store_folder):
            new_dataframe.to_csv(os.path.join(store_folder, '{}_features.csv'.format(self._name)))
        return new_dataframe


class DimensionReductionByPCA(object):
    def __init__(self, name='PCA', number=0, is_transform=True):
        self._name = name
        self.__model = PCA(n_components=0)
        self.__remained_number = number
        self.__is_transform = is_transform

    def run(self, dataframe, store_folder='', store_key=''):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        remained = np.min(data.shape)
        self.model = PCA(n_components=remained)
        self.model.fit(data)

        sub_data = self.model.transform(data)

        sub_feature_name = ['PCA_feature_' + str(index) for index in range(1, remained+1)]

        sub_feature_name.insert(0, 'label')
        sub_data = np.concatenate((label[..., np.newaxis], sub_data), axis=1)
        new_dataframe = pd.DataFrame(data=sub_data, index=dataframe.index, columns=sub_feature_name)

        if store_folder and os.path.isdir(store_folder):
            new_dataframe.to_csv(os.path.join(store_folder, '{}_features.csv'.format(self._name)))
        return new_dataframe

if __name__ == '__main__':
    data_path = r'D:\data\EC\feature\train_单变量.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=1, how='any')
    pcc = DimensionReductionByPCC()
    save_path = r'D:\data\EC\feature\012_3单变量'
    output_df = pcc.run(df, save_path)
    print(output_df)
