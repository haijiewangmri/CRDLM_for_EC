# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import pandas as pd
from scipy import stats


# 给总数据df和拆分的索引列表，把新的df提取出来
# 这里的index就是病人的case_name，就和建模的index_col=0对应上了
def set_new_dataframe(dataframe, case_index):
    col_name = dataframe.columns.tolist()
    new_dataframe = pd.DataFrame()
    new_dataframe = pd.concat([new_dataframe, dataframe.loc[case_index, :]], ignore_index=True)
    new_dataframe = new_dataframe[col_name]  # 不知道为什么遇到过顺序变乱的情况，这样重新调整顺序
    return new_dataframe


def count_list(input):
    if not isinstance(input, list):
        input = list(input)
    dict = {}
    for i in set(input):
        dict[i] = input.count(i)
    return dict


def p_test_categories(train_data_arr, test_data_arr):  # 用于对非连续值的卡方检验
    count1 = count_list(train_data_arr)
    count2 = count_list(test_data_arr)  # dict, 每个类别为key，统计了每个类别的次数
    categories = set(list(count1.keys()) + list(count2.keys()))
    contingency_dict = {}
    for category in categories:
        contingency_dict[category] = [count1[category] if category in count1.keys() else 0,
                                      count2[category] if category in count2.keys() else 0]

    contingency_pd = pd.DataFrame(contingency_dict)
    contingency_array = np.array(contingency_pd)
    _, p_value, _, _ = stats.chi2_contingency(contingency_array)
    return p_value


# 输入两组数据df，计算所有P-value
def p_test(train_df, test_df, alpha=1e-3, feature_select=None):
    assert train_df.columns.tolist() == test_df.columns.tolist(), 'train and test feature mismatch'
    label_idx = train_df.columns.tolist().index('label')
    if feature_select:
        if isinstance(feature_select, list):
            features = feature_select
        else:
            features = list(feature_select)
    else:
        features = train_df.columns.tolist()[label_idx + 1:]
    p_list = []
    distribute = []
    for feature in features:
        train_data_arr = train_df[feature].values
        test_data_arr = test_df[feature].values

        _, normal_p = stats.normaltest(np.concatenate((train_data_arr, test_data_arr), axis=0))
        if len(set(train_data_arr)) < 10:  # 少于5个数认为是离散值，用卡方检验
            p_value = p_test_categories(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('categories')
        elif normal_p > alpha:  # 正态分布用T检验
            _, p_value = stats.ttest_ind(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('normal')
        else:  # P很小，拒绝假设，假设是来自正态分布，非正态分布用u检验
            _, p_value = stats.mannwhitneyu(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('non-normal')
    return features, p_list, distribute


# 利用index随机重排
def data_separate_random(total_data, test_data_percentage=0.3, random_state=None):
    if total_data.columns.isin(['label', 'Label']).any():
        label_column_name = 'label' if total_data.columns.isin(['label']).any() else 'Label'
    else:
        label_column_name = total_data.columns.to_list()[0]
    label_list = np.array(total_data[label_column_name].tolist())

    real_index = total_data.index
    train_index_list, test_index_list = [], []
    for group in range(int(np.max(label_list)) + 1):  # label=0,1时就分成两组
        index = np.where(label_list == group)[0]

        random.seed(random_state)
        random.shuffle(index)

        train_index = real_index[index[round(len(index) * test_data_percentage):]]
        test_index = real_index[index[:round(len(index) * test_data_percentage)]]
        train_index_list.extend(train_index)
        test_index_list.extend(test_index)

    train_dataframe = set_new_dataframe(total_data, train_index_list)
    test_dataframe = set_new_dataframe(total_data, test_index_list)

    return train_dataframe, test_dataframe


class DataSplit(object):
    def __init__(self, test_data_percentage=0.3):
        self.test_data_percentage = test_data_percentage

    def run(self, df, repeat_times=1, store_folder='', random_state=None, feature_select=None):
        max_mean_p_value = 0
        output_train_df = None
        output_test_df = None
        output_p_value_list = []
        output_total_feature_list = []
        output_distribution = []

        if repeat_times == 1:
            train_split_df, test_split_df = data_separate_random(df, test_data_percentage=self.test_data_percentage,
                                                                 random_state=random_state)
            output_train_df = train_split_df
            output_test_df = test_split_df
        elif repeat_times == 2:
            train_split_df, test_split_df = data_separate_random(df, test_data_percentage=self.test_data_percentage,
                                                                 random_state=random_state)

            feature_list, p_value_list, distribution = p_test(train_split_df, test_split_df,
                                                              feature_select=feature_select)
            p_value_arr = np.array(p_value_list)
            mean_p_value = p_value_arr.mean()
            max_mean_p_value = mean_p_value
            output_train_df = train_split_df
            output_test_df = test_split_df
            output_total_feature_list = feature_list
            output_p_value_list = p_value_list
            output_distribution = distribution

            output_total_feature_list.append('average P-value')
            output_p_value_list.append(max_mean_p_value)
            output_distribution.append('')

            output_total_feature_list.append('random seed')
            output_p_value_list.append(random_state)
            output_distribution.append('')
        else:
            try:
                for idx in range(repeat_times):
                    train_split_df, test_split_df = data_separate_random(df,
                                                                         test_data_percentage=0.3,
                                                                         random_state=None)
                    feature_list, p_value_list, distribution = p_test(train_split_df, test_split_df,
                                                                      feature_select=feature_select)
                    # for i in range(len(feature_list)):
                    #     print('%s p-value is %.5f' % (feature_list[i], p_value_list[i]))
                    p_value_arr = np.array(p_value_list)
                    mean_p_value = p_value_arr.mean()
                    # print('p-value average is', mean_p_value)
                    # print('times', idx)
                    if mean_p_value > max_mean_p_value and p_value_arr.min(initial=None) > 0.05:
                        max_mean_p_value = mean_p_value
                        output_train_df = train_split_df
                        output_test_df = test_split_df
                        output_total_feature_list = feature_list
                        output_p_value_list = p_value_list
                        output_distribution = distribution
                    # print('max p-value average is', max_mean_p_value)
                    # print('--------------------------------------------------------------------------------------')

                output_total_feature_list.append('average P-value')
                output_p_value_list.append(max_mean_p_value)
                output_distribution.append('')
                output_total_feature_list.append('random seed')
                output_p_value_list.append(random_state)
                output_distribution.append('')
                if output_train_df is None:
                    raise ValueError
            except ValueError as e:
                print('-------------------------------------------')
                print('do not get suitable split, please repeat or add repeat_times')
                print('-------------------------------------------')
                print('经过 {} 次，未找到满足P值的拆分'.format(repeat_times), e)

        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            output_train_df.to_csv(os.path.join(store_folder, 'train_numeric_feature.csv'), index=False,
                                   encoding="utf_8_sig")
            output_test_df.to_csv(os.path.join(store_folder, 'test_numeric_feature.csv'), index=False,
                                  encoding="utf_8_sig")
            if output_p_value_list:
                output_p_test_dict = {'feature': output_total_feature_list,
                                      'p-value': output_p_value_list,
                                      'distribution': output_distribution}
                output_p_test_df = pd.DataFrame(output_p_test_dict)
                output_p_test_df.to_csv(os.path.join(store_folder, 'p_test_split_statistics.csv'), index=False,
                                        encoding="utf_8_sig")

        return output_train_df, output_test_df, output_p_value_list[:-2]


if __name__ == '__main__':
    data_path = r'D:\workcode\zhongshan_brain\最终结果\多脑区-多参数图\正常人-病人\多参数图-多脑区特征.csv'
    data = pd.read_csv(data_path, index_col=0)
    data = data.replace(np.inf, np.nan)
    data = data.dropna(axis=1, how='any')

    data_split = DataSplit()
    train, test, _ = data_split.run(data, random_state=51,
                                    store_folder=r'C:\Users\HJ Wang\Desktop')

    # train2, test2, _ = data_split.run(data, repeat_times=100, random_state=None,
    #                                   store_folder=r'C:\Users\HJ Wang\Desktop\model')
