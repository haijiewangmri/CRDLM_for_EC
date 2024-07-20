import pandas as pd
import numpy as np
import os
from ClassifierAll import LR, SVM, LDA, AdaBoost, ANN, DT, ET, GBM, KNN, LightGBM, RF, XGBoost
from DataBalance import UpSampling
from DataSplit import DataSplit
from FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA
from Normalizer import Normalizer
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Evaluate import Evaluate


class AutoRadiomics:
    """
    输入多个序列特征，随机种子，临床特征，保存路径
    输出全部组学结果

    """
    def __init__(self, Train_dataframe_list: list, Test_dataframe_list1: list, Test_dataframe_list2: list, Test_dataframe_list3: list,
                 output_file: str, random_seed=0, max_feature_num=10, dataframe_name_list=None, mode=0):
        if not isinstance(dataframe_name_list, list):
            dataframe_name_list = [dataframe_name_list]
        self.output_file = output_file
        self.random_seed = random_seed
        self.mode = mode  # 0计算和比较AUC，1计算和比较PR AUC

        self.radiomics_train_df = {}
        for i, path in enumerate(Train_dataframe_list):
            temp_df = Train_dataframe_list[i]
            temp_name = f'data{i + 1}'
            if dataframe_name_list is not None and len(Train_dataframe_list) == len(dataframe_name_list):
                temp_name = str(dataframe_name_list[i])
            self.radiomics_train_df[temp_name] = temp_df

        self.radiomics_test_df1 = {}
        for i, path in enumerate(Test_dataframe_list1):
            temp_df = Test_dataframe_list1[i]
            temp_name = f'data{i + 1}'
            if dataframe_name_list is not None and len(Test_dataframe_list1) == len(dataframe_name_list):
                temp_name = str(dataframe_name_list[i])
            self.radiomics_test_df1[temp_name] = temp_df

        self.radiomics_test_df2 = {}
        for i, path in enumerate(Test_dataframe_list2):
            temp_df = Test_dataframe_list2[i]
            temp_name = f'data{i + 1}'
            if dataframe_name_list is not None and len(Test_dataframe_list2) == len(dataframe_name_list):
                temp_name = str(dataframe_name_list[i])
            self.radiomics_test_df2[temp_name] = temp_df

        self.radiomics_test_df3 = {}
        for i, path in enumerate(Test_dataframe_list3):
            temp_df = Test_dataframe_list3[i]
            temp_name = f'data{i + 1}'
            if dataframe_name_list is not None and len(Test_dataframe_list3) == len(dataframe_name_list):
                temp_name = str(dataframe_name_list[i])
            self.radiomics_test_df3[temp_name] = temp_df

        index = None  # 病例名称放在index上，以此判断样本顺序是否一样
        for key in self.radiomics_train_df.keys():
            if index is None:
                index = self.radiomics_train_df[key].index
            else:
                assert list(index) == list(
                    self.radiomics_train_df[key].index), f'The train index of {key} is not consistent to the data'

        index = None  # 病例名称放在index上，以此判断样本顺序是否一样
        for key in self.radiomics_test_df1.keys():
            if index is None:
                index = self.radiomics_test_df1[key].index
            else:
                assert list(index) == list(
                    self.radiomics_test_df1[key].index), f'The test 1 index of {key} is not consistent to the data'

        index = None  # 病例名称放在index上，以此判断样本顺序是否一样
        for key in self.radiomics_test_df2.keys():
            if index is None:
                index = self.radiomics_test_df2[key].index
            else:
                assert list(index) == list(
                    self.radiomics_test_df2[key].index), f'The test 1 index of {key} is not consistent to the data'

        index = None  # 病例名称放在index上，以此判断样本顺序是否一样
        for key in self.radiomics_test_df3.keys():
            if index is None:
                index = self.radiomics_test_df3[key].index
            else:
                assert list(index) == list(
                    self.radiomics_test_df3[key].index), f'The test 1 index of {key} is not consistent to the data'

        self.modelings = [LR, SVM, LDA, AdaBoost, ANN, DT, ET, GBM, KNN, LightGBM, RF, XGBoost]
        self.modelings_name = ['LR', 'SVM', 'LDA', 'AdaBoost', 'ANN', 'DT', 'ET', 'GBM', 'KNN', 'LightGBM', 'RF', 'XGBoost']
        self.selection = FeatureSelectByRFE
        # self.selection = FeatureSelectByANOVA

        self.random_seed = random_seed
        self._up_sampling = UpSampling()
        self._max_feature_num = max_feature_num
        self.run()

    def run(self):
        for i, modeling in enumerate(self.modelings):
            modeling_name = self.modelings_name[i]

            print('model: ', modeling_name)
            combine_train_df = None
            combine_test_df1 = None
            combine_test_df2 = None
            combine_test_df3 = None
            store_path = os.path.join(self.output_file, modeling_name)
            cv = StratifiedKFold(shuffle=True, random_state=self.random_seed)
            # 每个序列单独
            for key in self.radiomics_train_df.keys():
                data_result_dict = {'feature num': [],
                                    'train AUC': [], 'train 95%CI': [], 'train PR AUC': [], 'train PR 95%CI': [],
                                    'train positive num': [], 'train negative num': [],
                                    'train ACC': [], 'train SEN': [], 'train SPE': [], 'train PPV': [], 'train NPV': [],
                                    'cv train AUC': [], 'cv val AUC': [], 'cv val 95%CI': [], 'cv val PR AUC': [], 'cv val PR 95%CI': [],
                                    'cv val ACC': [], 'cv val SEN': [], 'cv val SPE': [], 'cv val PPV': [],
                                    'cv val NPV': [],
                                    'test1 AUC': [], 'test1 95%CI': [], 'test1 PR AUC': [], 'test1 PR 95%CI': [],
                                    'test1 positive num': [], 'test1 negative num': [],
                                    'test1 ACC': [], 'test1 SEN': [], 'test1 SPE': [], 'test1 PPV': [], 'test1 NPV': [],
                                    'test2 AUC': [], 'test2 95%CI': [], 'test2 PR AUC': [], 'test2 PR 95%CI': [],
                                    'test2 positive num': [], 'test2 negative num': [],
                                    'test2 ACC': [], 'test2 SEN': [], 'test2 SPE': [], 'test2 PPV': [], 'test2 NPV': [],
                                    'test3 AUC': [], 'test3 95%CI': [], 'test3 PR AUC': [], 'test3 PR 95%CI': [],
                                    'test3 positive num': [], 'test3 negative num': [],
                                    'test3 ACC': [], 'test3 SEN': [], 'test3 SPE': [], 'test3 PPV': [], 'test3 NPV': [],
                                    'features': []
                                    }

                print(f'processing {key}')
                data_store_path = os.path.join(store_path, key)  # 对应到每个序列
                if not os.path.exists(data_store_path):
                    os.makedirs(data_store_path)

                train_df = self.radiomics_train_df[key]
                test_df1 = self.radiomics_test_df1[key]
                test_df2 = self.radiomics_test_df2[key]
                test_df3 = self.radiomics_test_df3[key]

                normalizer = Normalizer(train_df, minmax=True)
                train_df = normalizer.run(train_df)
                test_df1 = normalizer.run(test_df1)
                test_df2 = normalizer.run(test_df2)
                test_df3 = normalizer.run(test_df3)

                # ------------------------------------------------------------------------------------

                max_train_AUC = 0
                max_test_AUC1 = 0
                max_test_AUC2 = 0
                max_test_AUC3 = 0
                max_val_AUC = 0

                selected_features = []
                for k in range(self._max_feature_num):
                    rfe_feature_store_path = os.path.join(data_store_path, 'RFE_' + str(k + 1))
                    if not os.path.exists(rfe_feature_store_path):
                        os.makedirs(rfe_feature_store_path)

                    if k + 1 > (len(train_df.columns.tolist()) - 1):
                        break
                    rfe = self.selection(n_features_to_select=k + 1)
                    temp_up_train = self._up_sampling.run(train_df)
                    selected_up_train_df = rfe.run(temp_up_train)
                    selected_train_df = train_df[selected_up_train_df.columns.tolist()]

                    fold5_val_auc = []
                    fold5_train_auc = []
                    fold5_val_pr_auc = []
                    fold5_train_pr_auc = []

                    fold5_name = []
                    fold5_group = []
                    fold5_val_predict = []
                    fold5_val_label = []
                    for l, (train_index, val_index) in enumerate(
                            cv.split(train_df.values[:, 1:], train_df['label'].values)):
                        real_index = selected_train_df.index
                        cv_train_df = selected_train_df.loc[real_index[train_index], :]
                        cv_val_df = selected_train_df.loc[real_index[val_index], :]
                        up_cv_train_df = self._up_sampling.run(cv_train_df)
                        # cv_val_df = self._up_sampling.run(cv_val_df)

                        model = modeling(up_cv_train_df)
                        cv_train_predict = model.predict(cv_train_df.values[:, 1:])
                        cv_val_predict = model.predict(cv_val_df.values[:, 1:])

                        cv_train_label = cv_train_df['label'].tolist()
                        cv_val_label = cv_val_df['label'].tolist()
                        metric = Evaluate(cv_train_predict, cv_train_label, cv_val_predict, cv_val_label)

                        fold5_name.extend(cv_val_df.index.tolist())
                        fold5_group.extend([l] * len(cv_val_predict))
                        fold5_val_predict.extend(cv_val_predict)
                        fold5_val_label.extend(cv_val_label)
                        fold5_val_auc.append(metric['test_AUC'])
                        fold5_val_pr_auc.append(metric['test_PR_AUC'])
                        fold5_train_auc.append(metric['AUC'])
                        fold5_train_pr_auc.append(metric['PR_AUC'])

                    temp_test1 = test_df1.loc[:, selected_train_df.columns.tolist()]
                    temp_test2 = test_df2.loc[:, selected_train_df.columns.tolist()]
                    temp_test3 = test_df3.loc[:, selected_train_df.columns.tolist()]
                    model = modeling(selected_up_train_df)
                    model.save(rfe_feature_store_path)
                    train_predict = model.predict(selected_train_df.values[:, 1:])
                    test_predict1 = model.predict(temp_test1.values[:, 1:])
                    test_predict2 = model.predict(temp_test2.values[:, 1:])
                    test_predict3 = model.predict(temp_test3.values[:, 1:])
                    train_label = selected_train_df['label'].values
                    test_label1 = temp_test1['label'].values
                    test_label2 = temp_test2['label'].values
                    test_label3 = temp_test3['label'].values

                    metric1 = Evaluate(train_predict, train_label, test_predict1, test_label1)
                    metric2 = Evaluate(fold5_val_predict, fold5_val_label, test_predict2, test_label2)
                    metric3 = Evaluate(test_predict3, test_label3, test_predict3, test_label3)

                    selected_train_df.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_train.csv'))
                    temp_test1.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test1.csv'))
                    temp_test2.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test2.csv'))
                    temp_test3.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test3.csv'))

                    predict_columns = ['label', 'Pred']
                    new_train_data = np.concatenate((train_label[:, np.newaxis], train_predict[:, np.newaxis]), axis=1)
                    new_test_data1 = np.concatenate((test_label1[:, np.newaxis], test_predict1[:, np.newaxis]), axis=1)
                    new_test_data2 = np.concatenate((test_label2[:, np.newaxis], test_predict2[:, np.newaxis]), axis=1)
                    new_test_data3 = np.concatenate((test_label3[:, np.newaxis], test_predict3[:, np.newaxis]), axis=1)
                    train_predict_df = pd.DataFrame(data=new_train_data, index=train_df.index, columns=predict_columns)
                    test_predict_df1 = pd.DataFrame(data=new_test_data1, index=test_df1.index, columns=predict_columns)
                    test_predict_df2 = pd.DataFrame(data=new_test_data2, index=test_df2.index, columns=predict_columns)
                    test_predict_df3 = pd.DataFrame(data=new_test_data3, index=test_df3.index, columns=predict_columns)
                    train_predict_df.to_csv(os.path.join(rfe_feature_store_path, 'train_prediction.csv'))
                    test_predict_df1.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction1.csv'))
                    test_predict_df2.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction2.csv'))
                    test_predict_df3.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction3.csv'))

                    fold5_name = np.array(fold5_name)
                    fold5_group = np.array(fold5_group)
                    fold5_val_predict = np.array(fold5_val_predict)
                    fold5_val_label = np.array(fold5_val_label)
                    new_val_data = np.concatenate(
                        (fold5_group[:, np.newaxis], fold5_val_label[:, np.newaxis], fold5_val_predict[:, np.newaxis]),
                        axis=1)
                    val_predict_df = pd.DataFrame(data=new_val_data, index=fold5_name, columns=['Group', 'label', 'Pred'])
                    val_predict_df.to_csv(os.path.join(rfe_feature_store_path, 'val_prediction.csv'))

                    mean_cv_val_auc = np.array(fold5_val_auc).mean()
                    mean_cv_train_auc = np.array(fold5_train_auc).mean()

                    mean_cv_val_pr_auc = np.array(fold5_val_pr_auc).mean()

                    if self.mode == 0:
                        if mean_cv_val_auc > max_val_AUC and mean_cv_val_auc > 0.5:
                            max_val_AUC = mean_cv_val_auc
                            max_train_AUC = metric1['AUC']
                            max_test_AUC1 = metric1['test_AUC']
                            max_test_AUC2 = metric2['test_AUC']
                            max_test_AUC3 = metric3['test_AUC']
                            selected_features = selected_train_df.columns.tolist()[1:]
                    else:
                        if mean_cv_val_pr_auc > max_val_AUC and mean_cv_val_pr_auc > 0.5:
                            max_val_AUC = mean_cv_val_pr_auc
                            max_train_AUC = metric1['PR_AUC']
                            max_test_AUC1 = metric1['test_PR_AUC']
                            max_test_AUC2 = metric2['test_PR_AUC']
                            max_test_AUC3 = metric3['test_PR_AUC']
                            selected_features = selected_train_df.columns.tolist()[1:]

                    data_result_dict['feature num'].append(k + 1)
                    data_result_dict['train AUC'].append(metric1['AUC'])
                    data_result_dict['train 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['AUC_DOWN'], metric1['AUC_UP']))
                    data_result_dict['train PR AUC'].append(metric1['PR_AUC'])
                    data_result_dict['train PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['PR_AUC_DOWN'], metric1['PR_AUC_UP']))
                    data_result_dict['train positive num'].append(metric1['positive_number'])
                    data_result_dict['train negative num'].append(metric1['negative_number'])
                    data_result_dict['train ACC'].append(metric1['ACC'])
                    data_result_dict['train SEN'].append(metric1['SEN'])
                    data_result_dict['train SPE'].append(metric1['SPE'])
                    data_result_dict['train PPV'].append(metric1['PPV'])
                    data_result_dict['train NPV'].append(metric1['NPV'])
                    data_result_dict['cv train AUC'].append(mean_cv_train_auc)
                    data_result_dict['cv val AUC'].append(metric2['AUC'])
                    data_result_dict['cv val 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['AUC_DOWN'], metric2['AUC_UP']))
                    data_result_dict['cv val PR AUC'].append(metric2['PR_AUC'])
                    data_result_dict['cv val PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['PR_AUC_DOWN'], metric2['PR_AUC_UP']))
                    data_result_dict['cv val ACC'].append(metric2['ACC'])
                    data_result_dict['cv val SEN'].append(metric2['SEN'])
                    data_result_dict['cv val SPE'].append(metric2['SPE'])
                    data_result_dict['cv val PPV'].append(metric2['PPV'])
                    data_result_dict['cv val NPV'].append(metric2['NPV'])

                    data_result_dict['test1 AUC'].append(metric1['test_AUC'])
                    data_result_dict['test1 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['test_AUC_DOWN'], metric1['test_AUC_UP']))
                    data_result_dict['test1 PR AUC'].append(metric1['test_PR_AUC'])
                    data_result_dict['test1 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['test_PR_AUC_DOWN'], metric1['test_PR_AUC_UP']))
                    data_result_dict['test1 positive num'].append(metric1['test positive_number'])
                    data_result_dict['test1 negative num'].append(metric1['test negative_number'])
                    data_result_dict['test1 ACC'].append(metric1['test_ACC'])
                    data_result_dict['test1 SEN'].append(metric1['test_SEN'])
                    data_result_dict['test1 SPE'].append(metric1['test_SPE'])
                    data_result_dict['test1 PPV'].append(metric1['test_PPV'])
                    data_result_dict['test1 NPV'].append(metric1['test_NPV'])

                    data_result_dict['test2 AUC'].append(metric2['test_AUC'])
                    data_result_dict['test2 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['test_AUC_DOWN'], metric2['test_AUC_UP']))
                    data_result_dict['test2 PR AUC'].append(metric2['test_PR_AUC'])
                    data_result_dict['test2 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['test_PR_AUC_DOWN'], metric2['test_PR_AUC_UP']))
                    data_result_dict['test2 positive num'].append(metric2['test positive_number'])
                    data_result_dict['test2 negative num'].append(metric2['test negative_number'])
                    data_result_dict['test2 ACC'].append(metric2['test_ACC'])
                    data_result_dict['test2 SEN'].append(metric2['test_SEN'])
                    data_result_dict['test2 SPE'].append(metric2['test_SPE'])
                    data_result_dict['test2 PPV'].append(metric2['test_PPV'])
                    data_result_dict['test2 NPV'].append(metric2['test_NPV'])
                    data_result_dict['test3 AUC'].append(metric3['test_AUC'])

                    data_result_dict['test3 95%CI'].append('{:.4f}-{:.4f}'.format(metric3['test_AUC_DOWN'], metric3['test_AUC_UP']))
                    data_result_dict['test3 PR AUC'].append(metric3['test_PR_AUC'])
                    data_result_dict['test3 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric3['test_PR_AUC_DOWN'], metric3['test_PR_AUC_UP']))
                    data_result_dict['test3 positive num'].append(metric3['test positive_number'])
                    data_result_dict['test3 negative num'].append(metric3['test negative_number'])
                    data_result_dict['test3 ACC'].append(metric3['test_ACC'])
                    data_result_dict['test3 SEN'].append(metric3['test_SEN'])
                    data_result_dict['test3 SPE'].append(metric3['test_SPE'])
                    data_result_dict['test3 PPV'].append(metric3['test_PPV'])
                    data_result_dict['test3 NPV'].append(metric3['test_NPV'])

                    data_result_dict['features'].append(selected_train_df.columns.tolist()[1:])

                metric_df = pd.DataFrame(data_result_dict)
                metric_df.to_csv(os.path.join(data_store_path, 'total result.csv'))

                print(f'best {key} model  feature num: {len(selected_features)}')
                if self.mode == 0:
                    print(f'     train AUC {max_train_AUC}  val AUC {max_val_AUC}  test1 AUC {max_test_AUC1}  test2 AUC {max_test_AUC2}  test3 AUC {max_test_AUC3}')
                else:
                    print(f'     train PR AUC {max_train_AUC}  val PR AUC {max_val_AUC}  test1 PR AUC {max_test_AUC1} test2 PR AUC {max_test_AUC2} test3 PR AUC {max_test_AUC3}')
                if combine_train_df is None:
                    selected_features.insert(0, 'label')
                    combine_train_df = train_df[selected_features]
                    combine_test_df1 = test_df1[selected_features]
                    combine_test_df2 = test_df2[selected_features]
                    combine_test_df3 = test_df3[selected_features]
                else:
                    combine_train_df = pd.concat((combine_train_df, train_df[selected_features]), axis=1)
                    combine_test_df1 = pd.concat((combine_test_df1, test_df1[selected_features]), axis=1)
                    combine_test_df2 = pd.concat((combine_test_df2, test_df2[selected_features]), axis=1)
                    combine_test_df3 = pd.concat((combine_test_df3, test_df3[selected_features]), axis=1)
            # ------------------------------------------------------------------------------------
            # 多个序列的特征合并了，接下来对汇总的特征，建立一个多序列的模型
            if len(self.radiomics_train_df) == 1:
                continue
            if combine_train_df is None:
                print('without any model AUC > 0.5')
                continue
            combine_result_dict = {'feature num': [],
                                   'train AUC': [], 'train 95%CI': [], 'train PR AUC': [], 'train PR 95%CI': [],
                                   'train positive num': [], 'train negative num': [],
                                   'train ACC': [], 'train SEN': [], 'train SPE': [], 'train PPV': [], 'train NPV': [],
                                   'cv train AUC': [],
                                   'cv val AUC': [], 'cv val 95%CI': [], 'cv val PR AUC': [], 'cv val PR 95%CI': [],
                                   'cv val ACC': [], 'cv val SEN': [], 'cv val SPE': [], 'cv val PPV': [], 'cv val NPV': [],
                                   'test1 AUC': [], 'test1 95%CI': [], 'test1 PR AUC': [], 'test1 PR 95%CI': [],
                                   'test1 positive num': [], 'test1 negative num': [],
                                   'test1 ACC': [], 'test1 SEN': [], 'test1 SPE': [], 'test1 PPV': [], 'test1 NPV': [],
                                   'test2 AUC': [], 'test2 95%CI': [], 'test2 PR AUC': [], 'test2 PR 95%CI': [],
                                   'test2 positive num': [], 'test2 negative num': [],
                                   'test2 ACC': [], 'test2 SEN': [], 'test2 SPE': [], 'test2 PPV': [], 'test2 NPV': [],
                                   'test3 AUC': [], 'test3 95%CI': [], 'test3 PR AUC': [], 'test3 PR 95%CI': [],
                                   'test3 positive num': [], 'test3 negative num': [],
                                   'test3 ACC': [], 'test3 SEN': [], 'test3 SPE': [], 'test3 PPV': [], 'test3 NPV': [],
                                   'features': [],
                                   }
            print(f'processing combined model')
            combine_store_path = os.path.join(store_path, 'combined')  # 对应到每个序列
            # 单个序列的组学，交叉验证-test，保存
            max_train_AUC = 0
            max_test_AUC1 = 0
            max_test_AUC2 = 0
            max_test_AUC3 = 0
            max_val_AUC = 0
            selected_features = []
            for k in range(self._max_feature_num):
                rfe_feature_store_path = os.path.join(combine_store_path, 'RFE_' + str(k + 1))
                if not os.path.exists(rfe_feature_store_path):
                    os.makedirs(rfe_feature_store_path)

                if k + 1 > (len(combine_train_df.columns.tolist()) - 1):
                    break
                rfe = self.selection(n_features_to_select=k + 1)
                temp_up_train = self._up_sampling.run(combine_train_df)
                selected_up_train_df = rfe.run(temp_up_train)
                selected_train_df = combine_train_df[selected_up_train_df.columns.tolist()]

                fold5_val_auc = []
                fold5_train_auc = []
                fold5_val_pr_auc = []
                fold5_train_pr_auc = []

                fold5_name = []
                fold5_group = []
                fold5_val_predict = []
                fold5_val_label = []
                for l, (train_index, val_index) in enumerate(
                        cv.split(combine_train_df.values[:, 1:], combine_train_df['label'].values)):
                    real_index = selected_train_df.index
                    cv_train_df = selected_train_df.loc[real_index[train_index], :]
                    cv_val_df = selected_train_df.loc[real_index[val_index], :]
                    up_cv_train_df = self._up_sampling.run(cv_train_df)
                    # cv_val_df = self._up_sampling.run(cv_val_df)

                    model = modeling(up_cv_train_df)
                    cv_train_predict = model.predict(cv_train_df.values[:, 1:])
                    cv_val_predict = model.predict(cv_val_df.values[:, 1:])

                    cv_train_label = cv_train_df['label'].tolist()
                    cv_val_label = cv_val_df['label'].tolist()
                    metric = Evaluate(cv_train_predict, cv_train_label, cv_val_predict, cv_val_label)

                    fold5_name.extend(cv_val_df.index.tolist())
                    fold5_group.extend([l] * len(cv_val_predict))
                    fold5_val_predict.extend(cv_val_predict)
                    fold5_val_label.extend(cv_val_label)
                    fold5_val_auc.append(metric['test_AUC'])
                    fold5_val_pr_auc.append(metric['test_PR_AUC'])
                    fold5_train_auc.append(metric['AUC'])
                    fold5_train_pr_auc.append(metric['PR_AUC'])

                temp_test1 = combine_test_df1.loc[:, selected_train_df.columns.tolist()]
                temp_test2 = combine_test_df2.loc[:, selected_train_df.columns.tolist()]
                temp_test3 = combine_test_df3.loc[:, selected_train_df.columns.tolist()]
                model = modeling(selected_up_train_df)
                model.save(rfe_feature_store_path)
                train_predict = model.predict(selected_train_df.values[:, 1:])
                test_predict1 = model.predict(temp_test1.values[:, 1:])
                test_predict2 = model.predict(temp_test2.values[:, 1:])
                test_predict3 = model.predict(temp_test3.values[:, 1:])
                train_label = selected_train_df['label'].values
                test_label1 = temp_test1['label'].values
                test_label2 = temp_test2['label'].values
                test_label3 = temp_test3['label'].values

                metric1 = Evaluate(train_predict, train_label, test_predict1, test_label1)
                metric2 = Evaluate(fold5_val_predict, fold5_val_label, test_predict2, test_label2)
                metric3 = Evaluate(test_predict3, test_label3, test_predict3, test_label3)

                selected_train_df.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_train.csv'))
                temp_test1.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test1.csv'))
                temp_test2.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test2.csv'))
                temp_test3.to_csv(os.path.join(rfe_feature_store_path, 'RFE_selected_test3.csv'))
                predict_columns = ['label', 'Pred']
                new_train_data = np.concatenate((train_label[:, np.newaxis], train_predict[:, np.newaxis]), axis=1)
                new_test_data1 = np.concatenate((test_label1[:, np.newaxis], test_predict1[:, np.newaxis]), axis=1)
                new_test_data2 = np.concatenate((test_label2[:, np.newaxis], test_predict2[:, np.newaxis]), axis=1)
                new_test_data3 = np.concatenate((test_label3[:, np.newaxis], test_predict3[:, np.newaxis]), axis=1)
                train_predict_df = pd.DataFrame(data=new_train_data, index=combine_train_df.index, columns=predict_columns)
                test_predict_df1 = pd.DataFrame(data=new_test_data1, index=combine_test_df1.index, columns=predict_columns)
                test_predict_df2 = pd.DataFrame(data=new_test_data2, index=combine_test_df2.index, columns=predict_columns)
                test_predict_df3 = pd.DataFrame(data=new_test_data3, index=combine_test_df3.index, columns=predict_columns)
                train_predict_df.to_csv(os.path.join(rfe_feature_store_path, 'train_prediction.csv'))
                test_predict_df1.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction1.csv'))
                test_predict_df2.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction2.csv'))
                test_predict_df3.to_csv(os.path.join(rfe_feature_store_path, 'test_prediction3.csv'))

                fold5_name = np.array(fold5_name)
                fold5_group = np.array(fold5_group)
                fold5_val_predict = np.array(fold5_val_predict)
                fold5_val_label = np.array(fold5_val_label)
                new_val_data = np.concatenate(
                    (fold5_group[:, np.newaxis], fold5_val_label[:, np.newaxis], fold5_val_predict[:, np.newaxis]), axis=1)
                val_predict_df = pd.DataFrame(data=new_val_data, index=fold5_name, columns=['Group', 'label', 'Pred'])
                val_predict_df.to_csv(os.path.join(rfe_feature_store_path, 'val_prediction.csv'))

                mean_cv_val_auc = np.array(fold5_val_auc).mean()
                mean_cv_train_auc = np.array(fold5_train_auc).mean()
                mean_cv_val_pr_auc = np.array(fold5_val_pr_auc).mean()

                if self.mode == 0:
                    if mean_cv_val_auc > max_val_AUC and mean_cv_val_auc > 0.5:
                        max_val_AUC = mean_cv_val_auc
                        max_train_AUC = metric1['AUC']
                        max_test_AUC1 = metric1['test_AUC']
                        max_test_AUC2 = metric2['test_AUC']
                        max_test_AUC3 = metric3['test_AUC']
                        selected_features = selected_train_df.columns.tolist()[1:]
                else:
                    if mean_cv_val_pr_auc > max_val_AUC and mean_cv_val_pr_auc > 0.5:
                        max_val_AUC = mean_cv_val_pr_auc
                        max_train_AUC = metric1['PR_AUC']
                        max_test_AUC1 = metric1['test_PR_AUC']
                        max_test_AUC2 = metric2['test_PR_AUC']
                        max_test_AUC3 = metric3['test_PR_AUC']
                        selected_features = selected_train_df.columns.tolist()[1:]

                combine_result_dict['feature num'].append(k + 1)
                combine_result_dict['train AUC'].append(metric1['AUC'])
                combine_result_dict['train 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['AUC_DOWN'], metric1['AUC_UP']))
                combine_result_dict['train PR AUC'].append(metric1['PR_AUC'])
                combine_result_dict['train PR 95%CI'].append(
                    '{:.4f}-{:.4f}'.format(metric1['PR_AUC_DOWN'], metric1['PR_AUC_UP']))
                combine_result_dict['train positive num'].append(metric1['positive_number'])
                combine_result_dict['train negative num'].append(metric1['negative_number'])
                combine_result_dict['train ACC'].append(metric1['ACC'])
                combine_result_dict['train SEN'].append(metric1['SEN'])
                combine_result_dict['train SPE'].append(metric1['SPE'])
                combine_result_dict['train PPV'].append(metric1['PPV'])
                combine_result_dict['train NPV'].append(metric1['NPV'])
                combine_result_dict['cv train AUC'].append(mean_cv_train_auc)
                combine_result_dict['cv val AUC'].append(metric2['AUC'])
                combine_result_dict['cv val 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['AUC_DOWN'], metric2['AUC_UP']))
                combine_result_dict['cv val PR AUC'].append(metric2['PR_AUC'])
                combine_result_dict['cv val PR 95%CI'].append(
                    '{:.4f}-{:.4f}'.format(metric2['PR_AUC_DOWN'], metric2['PR_AUC_UP']))
                combine_result_dict['cv val ACC'].append(metric2['ACC'])
                combine_result_dict['cv val SEN'].append(metric2['SEN'])
                combine_result_dict['cv val SPE'].append(metric2['SPE'])
                combine_result_dict['cv val PPV'].append(metric2['PPV'])
                combine_result_dict['cv val NPV'].append(metric2['NPV'])

                combine_result_dict['test1 AUC'].append(metric1['test_AUC'])
                combine_result_dict['test1 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['test_AUC_DOWN'], metric1['test_AUC_UP']))
                combine_result_dict['test1 PR AUC'].append(metric1['test_PR_AUC'])
                combine_result_dict['test1 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric1['test_PR_AUC_DOWN'], metric1['test_PR_AUC_UP']))
                combine_result_dict['test1 positive num'].append(metric1['test positive_number'])
                combine_result_dict['test1 negative num'].append(metric1['test negative_number'])
                combine_result_dict['test1 ACC'].append(metric1['test_ACC'])
                combine_result_dict['test1 SEN'].append(metric1['test_SEN'])
                combine_result_dict['test1 SPE'].append(metric1['test_SPE'])
                combine_result_dict['test1 PPV'].append(metric1['test_PPV'])
                combine_result_dict['test1 NPV'].append(metric1['test_NPV'])

                combine_result_dict['test2 AUC'].append(metric2['test_AUC'])
                combine_result_dict['test2 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['test_AUC_DOWN'], metric2['test_AUC_UP']))
                combine_result_dict['test2 PR AUC'].append(metric2['test_PR_AUC'])
                combine_result_dict['test2 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric2['test_PR_AUC_DOWN'], metric2['test_PR_AUC_UP']))
                combine_result_dict['test2 positive num'].append(metric2['test positive_number'])
                combine_result_dict['test2 negative num'].append(metric2['test negative_number'])
                combine_result_dict['test2 ACC'].append(metric2['test_ACC'])
                combine_result_dict['test2 SEN'].append(metric2['test_SEN'])
                combine_result_dict['test2 SPE'].append(metric2['test_SPE'])
                combine_result_dict['test2 PPV'].append(metric2['test_PPV'])
                combine_result_dict['test2 NPV'].append(metric2['test_NPV'])

                combine_result_dict['test3 AUC'].append(metric3['test_AUC'])
                combine_result_dict['test3 95%CI'].append('{:.4f}-{:.4f}'.format(metric3['test_AUC_DOWN'], metric3['test_AUC_UP']))
                combine_result_dict['test3 PR AUC'].append(metric3['test_PR_AUC'])
                combine_result_dict['test3 PR 95%CI'].append('{:.4f}-{:.4f}'.format(metric3['test_PR_AUC_DOWN'], metric3['test_PR_AUC_UP']))
                combine_result_dict['test3 positive num'].append(metric3['test positive_number'])
                combine_result_dict['test3 negative num'].append(metric3['test negative_number'])
                combine_result_dict['test3 ACC'].append(metric3['test_ACC'])
                combine_result_dict['test3 SEN'].append(metric3['test_SEN'])
                combine_result_dict['test3 SPE'].append(metric3['test_SPE'])
                combine_result_dict['test3 PPV'].append(metric3['test_PPV'])
                combine_result_dict['test3 NPV'].append(metric3['test_NPV'])

                combine_result_dict['features'].append(selected_train_df.columns.tolist()[1:])

            metric_df = pd.DataFrame(combine_result_dict)
            metric_df.to_csv(os.path.join(combine_store_path, 'total result.csv'))

        print(f'best combined model  feature num: {len(selected_features)}')
        if self.mode == 0:
            print(f'     train AUC {max_train_AUC}  val AUC {max_val_AUC}  test1 AUC {max_test_AUC1}  test2 AUC {max_test_AUC2}  test3 AUC {max_test_AUC3}')
        else:
            print(f'     train PR AUC {max_train_AUC}  val PR AUC {max_val_AUC}  test1 PR AUC {max_test_AUC1} test2 PR AUC {max_test_AUC2} test3 PR AUC {max_test_AUC3}')


def split_sequence(split_df):
    col_name = split_df.columns.tolist()
    t1_feature = [col_name[0]]
    ce_feature = [col_name[0]]
    t2_feature = [col_name[0]]
    dwi_feature = [col_name[0]]
    clinical_feature = [col_name[0]]

    for feature in col_name[1:]:
        if 'dce' in feature:
            ce_feature.append(feature)
        elif 't2' in feature:
            t2_feature.append(feature)
        elif 't1' in feature:
            t1_feature.append(feature)
        elif 'dwi' in feature:
            dwi_feature.append(feature)
        else:
            clinical_feature.append(feature)

    t1_df = split_df[t1_feature].copy()
    t2_df = split_df[t2_feature].copy()
    ce_df = split_df[ce_feature].copy()
    dwi_df = split_df[dwi_feature].copy()
    cl_df = split_df[clinical_feature].copy()
    return t1_df, t2_df, ce_df, dwi_df, cl_df

if __name__ == '__main__':
    csv_path1 = r'.\internal_data.csv'
    csv_path2 = r'.\external_data1.csv'
    csv_path3 = r'.\external_data2.csv'

    output = r'.\output'
    df1 = pd.read_csv(csv_path1, index_col=0, header=0)
    df2 = pd.read_csv(csv_path2, index_col=0, header=0)
    df3 = pd.read_csv(csv_path3, index_col=0, header=0)

    max_feature = 30
    seed = 170
    print('random seed', seed)

    data_split = DataSplit(test_data_percentage=0.3)

    train_df, test_df, _ = data_split.run(df1, random_state=seed, store_folder=os.path.join(output, str(seed)))

    output_folder = os.path.join(output, str(seed))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    AutoRadiomics([train_df],
                  [test_df],
                  [df2],
                  [df3],
                  output_folder, mode=0, random_seed=seed, max_feature_num=max_feature)