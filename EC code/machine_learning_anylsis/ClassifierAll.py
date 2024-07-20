import numpy as np
import pandas as pd
import os
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

class Classifier:
    def __init__(self):
        self.__model = None
        self._x = np.array([])
        self._y = np.array([])

    def set_model(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def predict(self, x, is_probability=True):
        if is_probability:
            return self.__model.predict_proba(x)[:, 1]
        else:
            return self.__model.predict(x)

    def fit(self):
        self.__model.fit(self._x, self._y)

    def save(self, store_folder):
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        if not os.path.isdir(store_folder):
            print('The store function must be a folder path')
            return
        # store_path = os.path.join(store_folder, 'model.pickle')
        # with open(store_path, 'wb') as f:
        #     pickle.dump(self.get_model(), f)
        try:
            coef_path = os.path.join(store_folder, 'coef.csv')
            df = pd.DataFrame(data=np.transpose(self.get_model().coef_),
                              index=self.dataframe.columns.tolist()[1:], columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'model can not load coef: '
            print('{} \n{}'.format(content, e.__str__()))

        # Save the intercept_
        try:
            intercept_path = os.path.join(store_folder, 'intercept.csv')
            intercept_df = pd.DataFrame(data=self.get_model().intercept_.reshape(1, 1),
                                        index=['intercept'], columns=['value'])
            intercept_df.to_csv(intercept_path)
        except Exception as e:
            content = 'model can not load intercept: '
            print('{} \n{}'.format(content, e.__str__()))


class SVM(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(SVM, self).__init__()
        kwargs.setdefault('kernel', 'linear')
        kwargs.setdefault('C', 1.0)
        kwargs.setdefault('probability', True)
        self.set_model(SVC(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class LR(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(LR, self).__init__()
        if 'solver' in kwargs:
            self.set_model(LogisticRegression(penalty='none', **kwargs))
        else:
            self.set_model(LogisticRegression(penalty='none', solver='saga', tol=0.01, random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class LDA(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(LDA, self).__init__()
        self.set_model(LinearDiscriminantAnalysis(**kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class AdaBoost(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(AdaBoost, self).__init__()
        self.set_model(AdaBoostClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class ANN(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(ANN, self).__init__()
        self.set_model(MLPClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class DT(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(DT, self).__init__()
        self.set_model(DecisionTreeClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class ET(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(ET, self).__init__()
        self.set_model(ExtraTreesClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class GBM(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(GBM, self).__init__()
        self.set_model(GradientBoostingClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class KNN(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(KNN, self).__init__()
        self.set_model(KNeighborsClassifier(**kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class LightGBM(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(LightGBM, self).__init__()
        self.set_model(lgb.LGBMClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class RF(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(RF, self).__init__()
        self.set_model(RandomForestClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


class XGBoost(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(XGBoost, self).__init__()
        self.set_model(xgb.XGBClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.iloc[:, 1:].values)
        self._y = np.array(dataframe['label'].values)
        self.dataframe = dataframe
        self.fit()


if __name__ == '__main__':
    data_path = r'E:\study\EC\subtype\20240521 DL模型\moco_feature\fudan292_DL pcc task1.csv'
    train_df = pd.read_csv(data_path, index_col=0)

    # model = LR(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)
    #
    # model = SVM(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)
    #
    # model = LDA(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = AdaBoost(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = ANN(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = DT(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = ET(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = GBM(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # # print('coe:', model.get_model().coef_)
    # # print('intercept:', model.get_model().intercept_)
    # # # print('predict:', predict)
    #
    # model = KNN(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # print('coe:', model.get_model().coef_)
    # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)

    # model = LightGBM(train_df, num_leaves=50, max_depth=10, min_child_samples=30, learning_rate=0.05, n_estimators=200)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # print('coe:', model.get_model().coef_)
    # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)

    # model = RF(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # print('coe:', model.get_model().coef_)
    # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)

    # model = XGBoost(train_df)
    # predict = model.predict(np.array(train_df.values[:, 1:]))
    # print('coe:', model.get_model().coef_)
    # print('intercept:', model.get_model().intercept_)
    # # print('predict:', predict)

