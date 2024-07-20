# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression


class FeatureSelectByRFE(object):
    # def __init__(self, n_features_to_select=20, classifier=SVC(kernel='linear')):
    def __init__(self, n_features_to_select=20, classifier=LogisticRegression(penalty='none', solver='saga', tol=0.01, random_state=170)):
        self.n_features_to_select = n_features_to_select
        self.__classifier = classifier
        self._rank = None
        pass

    def get_selected_feature_index(self, dataframe):
        data = np.array(dataframe.values[:, 1:])
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = np.array(dataframe['label'].tolist())

        if data.shape[1] < self.n_features_to_select:
            print('RFE: The number of features {:d} in dataframe is smaller than the required number {:d}'.format(
                data.shape[1], self.n_features_to_select))
            self.n_features_to_select = data.shape[1]

        if data.shape[1] == 1 and self.n_features_to_select == 1:
            return [0]

        fs = RFE(self.__classifier, n_features_to_select=self.n_features_to_select, step=0.05)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        self._rank = fs.ranking_

        return feature_index.tolist()

    def run(self, dataframe, store_folder=''):
        data = np.array(dataframe.values[:, 1:])
        label = np.array(dataframe['label'].tolist())
        feature_name = dataframe.columns.tolist()[1:]
        selected_index = self.get_selected_feature_index(dataframe)

        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'RFE_selected_features.csv'))

        return new_dataframe


class FeatureSelectByANOVA(object):
    def __init__(self, n_features_to_select=20):
        self.n_features_to_select = n_features_to_select
        self._f_value = np.array([])
        self._p_value = np.array([])

    def GetSelectedFeatureIndex(self, data, label):
        if data.shape[1] < self.n_features_to_select:
            print(
                'ANOVA: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                    data.shape[1], self.n_features_to_select))
            self.n_features_to_select = data.shape[1]
        fs = SelectKBest(f_classif, k=self.n_features_to_select)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        f_value, p_value = f_classif(data, label)
        return feature_index.tolist(), f_value, p_value

    def run(self, dataframe, store_folder=''):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()[1:]

        selected_index, self._f_value, self._p_value = self.GetSelectedFeatureIndex(data, label)
        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'ANOVA_selected_features.csv'))
        return new_dataframe



if __name__ == '__main__':
    data_path = r'D:\data\EC\feature\01_23\t1 train.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=1, how='any')
    rfe = FeatureSelectByRFE(n_features_to_select=5)
    # save_path = r'.\output'
    output_df = rfe.run(df)
    print(output_df)
