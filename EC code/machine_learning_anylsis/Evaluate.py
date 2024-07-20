# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix


def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    '''
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.

    Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    :return: The AUC value, a list of the confidence interval, the boot strap result.
    '''

    single_auc = roc_auc_score(y_true, y_pred)

    bootstrapped_scores = []

    np.random.seed(42) # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc = np.std(sorted_scores)
    mean_auc = np.mean(sorted_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]
    # final_auc = (confidence_lower+confidence_upper)/2
    # print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return single_auc, mean_auc, CI, sorted_scores, std_auc


def AP_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    '''
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.

    Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    :return: The AUC value, a list of the confidence interval, the boot strap result.
    '''

    # precision, recall, thersholds = precision_recall_curve(y_true, y_pred, pos_label=1)
    # pos_label指定哪个标签为正样本
    single_auc = average_precision_score(y_true, y_pred, pos_label=1)  # 计算PR曲线下面积

    bootstrapped_scores = []

    np.random.seed(42) # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = average_precision_score(label_one_sample, pred_one_sample, pos_label=1)  # 计算PR曲线下面积
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc = np.std(sorted_scores)
    mean_auc = np.mean(sorted_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]
    # final_auc = (confidence_lower+confidence_upper)/2
    # print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return single_auc, mean_auc, CI, sorted_scores, std_auc


def Evaluate(prediction, label, test_prediction, test_label):
    prediction = np.array(prediction)
    label = np.array(label)
    test_prediction = np.array(test_prediction)
    test_label = np.array(test_label)

    metric = {}

    AUC, mean_AUC, CI, _, _ = AUC_Confidence_Interval(label, prediction)
    AP, mean_AP, CI_AP, _, _ = AP_Confidence_Interval(label, prediction)

    metric['AUC'] = round(AUC, 4)
    metric['AUC_UP'] = round(CI[1], 4)
    metric['AUC_DOWN'] = round(CI[0], 4)

    metric['PR_AUC'] = round(AP, 4)
    metric['PR_AUC_UP'] = round(CI_AP[1], 4)
    metric['PR_AUC_DOWN'] = round(CI_AP[0], 4)

    metric['sample_number'] = len(label)
    metric['positive_number'] = np.sum(label)
    metric['negative_number'] = len(label) - np.sum(label)

    metric['test positive_number'] = np.sum(test_label)
    metric['test negative_number'] = len(test_label) - np.sum(test_label)

    fpr, tpr, threshold = roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    pred = np.zeros_like(label)
    metric['cutoff'] = threshold[index]
    pred[prediction >= threshold[index]] = 1
    C = confusion_matrix(label, pred, labels=[1, 0])

    metric['ACC'] = round(np.where(pred == label)[0].size / label.size, 4)
    if np.sum(C[0, :]) < 1e-6:
        metric['SEN'] = 0
    else:
        metric['SEN'] = round(C[0, 0] / np.sum(C[0, :]), 4)
    if np.sum(C[1, :]) < 1e-6:
        metric['SPE'] = 0
    else:
        metric['SPE'] = round(C[1, 1] / np.sum(C[1, :]), 4)
    if np.sum(C[:, 0]) < 1e-6:
        metric['PPV'] = 0
    else:
        metric['PPV'] = round(C[0, 0] / np.sum(C[:, 0]), 4)
    if np.sum(C[:, 1]) < 1e-6:
        metric['NPV'] = 0
    else:
        metric['NPV'] = round(C[1, 1] / np.sum(C[:, 1]), 4)

    test_AUC, test_mean_AUC, test_CI, _, _ = AUC_Confidence_Interval(test_label, test_prediction)
    test_AP, test_mean_AP, test_CI_AP, _, _ = AP_Confidence_Interval(test_label, test_prediction)

    metric['test_AUC'] = round(test_AUC, 4)
    metric['test_AUC_UP'] = round(test_CI[1], 4)
    metric['test_AUC_DOWN'] = round(test_CI[0], 4)

    metric['test_PR_AUC'] = round(test_AP, 4)
    metric['test_PR_AUC_UP'] = round(test_CI_AP[1], 4)
    metric['test_PR_AUC_DOWN'] = round(test_CI_AP[0], 4)

    test_pred = np.zeros_like(test_label)
    test_pred[test_prediction >= threshold[index]] = 1
    test_C = confusion_matrix(test_label, test_pred, labels=[1, 0])
    metric['test_ACC'] = round(np.where(test_pred == test_label)[0].size / test_label.size, 4)
    if np.sum(test_C[0, :]) < 1e-6:
        metric['test_SEN'] = 0
    else:
        metric['test_SEN'] = round(test_C[0, 0] / np.sum(test_C[0, :]), 4)
    if np.sum(test_C[1, :]) < 1e-6:
        metric['test_SPE'] = 0
    else:
        metric['test_SPE'] = round(test_C[1, 1] / np.sum(test_C[1, :]), 4)
    if np.sum(test_C[:, 0]) < 1e-6:
        metric['test_PPV'] = 0
    else:
        metric['test_PPV'] = round(test_C[0, 0] / np.sum(test_C[:, 0]), 4)
    if np.sum(test_C[:, 1]) < 1e-6:
        metric['test_NPV'] = 0
    else:
        metric['test_NPV'] = round(test_C[1, 1] / np.sum(test_C[:, 1]), 4)
    return metric


def plot_cv_AUC(df):
    fig = plt.figure()
    x = df['feature num']
    # train = df['train AUC']
    # cv_train = df['cv train AUC']
    cv_val = df['cv val AUC']
    test = df['test AUC']
    # plt.plot(x, train)
    # plt.plot(x, cv_train, c='black')
    plt.plot(x, cv_val, c='blue')
    plt.plot(x, test, c='orange')
    plt.show()



if __name__ == '__main__':
    # for i in range(1, 8):
    df = pd.read_csv(r'E:\study\placenta\20240614最终结果\深度学习 多头\2\reg_cls_praevia\acc\1\合并双头\2\LR\combine\RFE_2\test_prediction2.csv')
        # print(i, Evaluate(df['Pre_blood'].values, df['label'].values, df['Pre_blood'].values, df['label'].values))
    print(Evaluate(df['Pred'].values, df['label'].values, df['Pred'].values, df['label'].values))