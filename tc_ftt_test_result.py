# coding=utf-8
import csv
import sys, os, time
import cmath
import math
from datetime import datetime, timedelta
from sklearn import metrics

import numpy as np
import pandas as pd
from pandas import Series

import warnings
import paddlets
import matplotlib.pyplot as plt
from paddlets.datasets.repository import get_dataset

warnings.filterwarnings('ignore', category=DeprecationWarning)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def rmse(y_test, y):
    return math.sqrt(sum((y_test - y) ** 2) / len(y))

def max_vote():
    # df201 = pd.read_csv("./result/20230803_M5_y18_m01_y23_m7_result_data_30.csv",header=0, sep=',',encoding='gbk')
    # df202 = pd.read_csv("./result/20230803_M5_y22_m07_y23_m7_result_data_30.csv",header=0, sep=',',encoding='gbk')
    # df203 = pd.read_csv("./result/20230803_M5_y22_m10_y23_m7_result_data_30.csv",header=0, sep=',',encoding='gbk')
    # df204 = pd.read_csv("./result/20230803_M5_y23_m01_y23_m7_result_data_30.csv",header=0, sep=',',encoding='gbk')
    # df205 = pd.read_csv("./result/20230803_M6_y18_m01_y23_m7_result_data_30.csv",header=0, sep=',',encoding='gbk')

    df201 = pd.read_csv("./result/20230810_occur_M6_y16_m01_y23_m7_result_30.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/20230810_occur_M6_y16_m01_y23_m7_result_60.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/20230810_occur_M6_y16_m01_y23_m7_result_90.csv", header=0, sep=',', encoding='gbk')

    df201 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_14.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_21.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_28.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_35.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/20230810_treat_M6_y16_m01_y23_m7_result_42.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    # df_all = pd.concat([df201, df202, df203])
    df_all = pd.concat([df201, df202, df203, df204, df205, df206])

    # 选择前几列数据
    num_columns = 4  # 前三列
    selected_data = df_all.iloc[:, :num_columns]  # 使用索引选择前几列

    print("all.shape:", selected_data.shape)

    col = df_all.columns.tolist()
    print(col)
    # col.remove('customerid')
    # col.remove('Y')

    filtered_df = selected_data[selected_data['Y'] != 1]
    print("filtered_df.shape", filtered_df.shape)
    result = filtered_df.groupby('customerid').apply(lambda x: (x > 0.5).sum().sum()).sort_values(ascending=False)
    print(result)

def weight_vote_occur_static():
    type = 'occur'
    ks_occur_30 = 0.4621 # 0.544  # 0.4621
    ks_occur_60 = 0.4516 # 0.432  # 0.4516
    ks_occur_90 = 0.45916  # 0.496  #  0.45916
    ks_occur_120 = 0.3525 # 0.6   # 0.3525   0.007985
    ks_occur_150 = 0.3333  # 0.6479999999   # 0.3333   0.0083
    ks_occur_180 = 0.464
    ks_occur_210 = 0.608
    df201 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_30_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_60_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_90_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    #df204 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_120_fl_20000.csv", header=0, sep=',', encoding='gbk')
    #df205 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_150_fl_20000.csv", header=0, sep=',', encoding='gbk')
    #df206 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_180.csv", header=0, sep=',', encoding='gbk')
    #df207 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_210.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    #df_all = pd.concat([df201, df202, df203, df204, df205, df206, df207])
    #col = df_all.columns.tolist()
    #print(col)
    #weight_df = (df201['prob'] * ks_occur_30 + df202['prob'] * ks_occur_60 + df203['prob'] * ks_occur_90 + df204['prob'] * ks_occur_120 + \
    #            df205['prob'] * ks_occur_150 + df206['prob'] * ks_occur_180 + df207['prob'] * ks_occur_210)/ \
    #            (ks_occur_30 + ks_occur_60 + ks_occur_90 + ks_occur_120 + ks_occur_150 + ks_occur_180 + ks_occur_210)
    weight_df = (df201['prob'] * ks_occur_30 + df202['prob'] * ks_occur_60 + df203['prob'] * ks_occur_90) / \
                (ks_occur_30 + ks_occur_60 + ks_occur_90)
    #print(weight_df.tolist())
    # 使用value_counts()函数计算值的频次
    value_counts = df201['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    fpr, tpr, thresholds = metrics.roc_curve(df201['Y'], weight_df, pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i],(tpr[i]*count_1 + fpr[i]*count_0)/(count_0 + count_1),tpr[i],tpr[i]*count_1, fpr[i],fpr[i]*count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def weight_vote_treat_static():
    type = 'treat'
    ks_treat_7 = 0.5650  # 0.6650
    ks_treat_14 = 0.5196  # 0.6196
    ks_treat_21 = 0.5796 # 0.6796
    ks_treat_28 = 0.6499999  # 0.6317  0.00868
    ks_treat_35 = 0.6916666  # 0.5133  0.0225
    ks_treat_42 = 0.583333
    ks_treat_49 = 0.60
    df201 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_7_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_14_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_21_fl.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_28_fl.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_35_fl.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_42.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_49.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    #df_all = pd.concat([df201, df202, df203, df204, df205, df206])
    #df_all = pd.concat([df201, df202, df203])
    #col = df_all.columns.tolist()
    #print(col)
    #weight_df = (df201['prob'] * ks_treat_7 + df202['prob'] * ks_treat_14 + df203['prob'] * ks_treat_21 + df204['prob'] * ks_treat_28 + \
    #            df205['prob'] * ks_treat_35 + df206['prob'] * ks_treat_42 + df207['prob'] * ks_treat_49 )/ \
    #            (ks_treat_7 + ks_treat_14 + ks_treat_21 + ks_treat_35 + ks_treat_42 + ks_treat_28 + ks_treat_49)
    weight_df = (df201['prob'] * ks_treat_7 + df202['prob'] * ks_treat_14 + df203['prob'] * ks_treat_21 + df204['prob'] * ks_treat_28 +df205['prob'] * ks_treat_35)/ \
                (ks_treat_7 + ks_treat_14 + ks_treat_21 + + ks_treat_28 + + ks_treat_35)
    #weight_df = (df201['prob'] * ks_treat_7 + df202['prob'] * ks_treat_14 + df203['prob'] * ks_treat_21) / \
    #            (ks_treat_7 + ks_treat_14 + ks_treat_21)
    #print(weight_df.tolist())
    df = pd.DataFrame()
    df['Y'] = df201['Y']
    df['customerid'] = df201['customerid']  # .str.replace('_.*', '', regex=True)
    df['prob'] = weight_df  # df201[column_key]
    top_df = df[df['prob']>0.3]
    print(top_df.head(50))
    exit(0)
    # 使用value_counts()函数计算值的频次
    value_counts = df201['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    fpr, tpr, thresholds = metrics.roc_curve(df201['Y'], weight_df, pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
        #          fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def weight_vote_occur_treat_static():
    type_a = 'occur'
    ks_occur_30 = 0.544
    ks_occur_60 = 0.432
    ks_occur_90 = 0.496
    ks_occur_120 = 0.6
    ks_occur_150 = 0.6479999999
    ks_occur_180 = 0.464
    ks_occur_210 = 0.608
    df201 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_30.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_60.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_90.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_120.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_150.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_180.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_210.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    df_all = pd.concat([df201, df202, df203, df204, df205, df206, df207])
    col = df_all.columns.tolist()
    print(col)
    weight_occur_df = (df201['prob'] * ks_occur_30 + df202['prob'] * ks_occur_60 + df203['prob'] * ks_occur_90 + df204['prob'] * ks_occur_120 + \
                 df205['prob'] * ks_occur_150 + df206['prob'] * ks_occur_180 + df207['prob'] * ks_occur_210) / \
                (ks_occur_30 + ks_occur_60 + ks_occur_90 + ks_occur_120 + ks_occur_150 + ks_occur_180 + ks_occur_210)

    df_occur = pd.DataFrame()
    df_occur['customerid'] = df201['customerid']
    df_occur['probwo'] = weight_occur_df

    type_b = 'treat'
    ks_treat_7 = 0.49167
    ks_treat_14 = 0.51666
    ks_treat_21 = 0.5666666
    ks_treat_28 = 0.5499999
    ks_treat_35 = 0.5916666
    ks_treat_42 = 0.583333
    ks_treat_49 = 0.60
    df201 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_14.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_21.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_28.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_35.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_42.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0815/20230815_treat_M6_y16_m1_y23_m1_result_test_49.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    df_all = pd.concat([df201, df202, df203, df204, df205, df206])
    col = df_all.columns.tolist()
    print(col)
    weight_treat_df = (df201['prob'] * ks_treat_7 + df202['prob'] * ks_treat_14 + df203['prob'] * ks_treat_21 + df204['prob'] * ks_treat_28 + \
                 df205['prob'] * ks_treat_35 + df206['prob'] * ks_treat_42 + df207['prob'] * ks_treat_49) / \
                (ks_treat_7 + ks_treat_14 + ks_treat_21 + ks_treat_35 + ks_treat_42 + ks_treat_28 + ks_treat_49)

    df_treat = pd.DataFrame()
    df_treat['Y'] = df201['Y']
    df_treat['customerid'] = df201['customerid']
    df_treat['probwt'] = weight_treat_df

    # 按照列 'customerid' 进行交集操作
    df_intersection = pd.merge(df_occur, df_treat, on='customerid')
    print('df_intersection.shape:',df_intersection.shape)
    # 按照列 'probwo' 的值进行过滤
    threshold = 0.5  # adjust
    df_filtered = df_intersection[df_intersection['probwo'] > threshold]
    print(df_filtered)
    # 使用value_counts()函数计算值的频次
    value_counts = df_filtered['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    #exit(0)
    fpr, tpr, thresholds = metrics.roc_curve(df_filtered['Y'], df_filtered['probwt'], pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
        #          fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    plt.savefig("./result/20230826_" + type_a + "_"+ str(threshold) + "_" + type_b +"_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/20230826_" + type_a + "_"+ str(threshold) + "_" + type_b + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def weight_vote_occur_dynamic_for_report():
    type = 'occur'
    ks_occur_30 = 0.4812
    ks_occur_60 = 0.45875
    ks_occur_90 = 0.549583
    df201 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_30_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_60_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_90_fl.csv", header=0, sep=',', encoding='gbk')

    col = df201.columns.tolist()
    print(col)
    start_date = datetime(2023, 6, 4)

    df_all = pd.DataFrame()
    for i in np.arange(0, 24, 1):
        print('=' * 16, i)
        column_key = start_date.strftime("%Y-%m-%d")
        weight_df = (df201[column_key] * ks_occur_30 + df202[column_key] * ks_occur_60 + df203[column_key] * ks_occur_90 ) / \
                    (ks_occur_30 + ks_occur_60 + ks_occur_90 )
        df = pd.DataFrame()
        df['Y'] = df201['Y']
        df['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
        df['prob'] = weight_df #df201[column_key]
        #print(df.head())
        # 按照列 'A' 的值进行过滤
        filtered_df = df[df['prob'] > 0.5]
        print(filtered_df)
        df_all = pd.concat([df_all, filtered_df])
        df_all.drop_duplicates(subset=['customerid'],keep='first',inplace=True)
        value_counts = df_all['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print(column_key, count_1, count_0)
        start_date -= timedelta(days=7)

def weight_vote_treat_dynamic_for_report():
    type = 'treat'
    ks_treat_7 = 0.49167
    ks_treat_14 = 0.51666
    ks_treat_21 = 0.5666666
    ks_treat_28 = 0.5499999
    ks_treat_35 = 0.5916666
    df201 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m01_y23_m01_y23_m07_result_dynamic_7_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m01_y23_m01_y23_m07_result_dynamic_14_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m01_y23_m01_y23_m07_result_dynamic_21_fl.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m01_y23_m01_y23_m07_result_dynamic_28_fl.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m01_y23_m01_y23_m07_result_dynamic_35_fl.csv", header=0, sep=',', encoding='gbk')

    col = df201.columns.tolist()
    print(col)
    start_date = datetime(2023, 8, 20)

    df_all = pd.DataFrame()
    for i in np.arange(0, 24, 1):
        print('='*16,i)
        column_key = start_date.strftime("%Y-%m-%d")
        weight_df = (df201[column_key] * ks_treat_7 + df202[column_key] * ks_treat_14 + df203[column_key] * ks_treat_21 + df204[column_key] * ks_treat_28 + \
                     df205[column_key] * ks_treat_35) / \
                    (ks_treat_7 + ks_treat_14 + ks_treat_21 + ks_treat_28 + ks_treat_35 )
        df = pd.DataFrame()
        df['Y'] = df201['Y']
        df['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
        df['prob'] = weight_df   #weight_df #df201[column_key]
        #print(column_key + ' ='*20)
        # 按照列 'A' 的值进行过滤
        filtered_df = df[df['prob'] > 0.5]
        #print(filtered_df)
        df_all = pd.concat([df_all, filtered_df])
        df_all.drop_duplicates(subset=['customerid'],keep='first',inplace=True)
        value_counts = df_all['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print(column_key, count_1, count_0)
        start_date -= timedelta(days=1)

def weight_vote_occur_predict_weekly():
    type = 'occur'
    ks_occur_30 = 0.544 # 0.472 focalloss
    ks_occur_60 = 0.432 # 0.512
    ks_occur_90 = 0.496 # 0.528
    ks_occur_120 = 0.6 # 0.559
    ks_occur_150 = 0.6479999999 # 0.472
    ks_occur_180 = 0.464  # 0.576
    ks_occur_210 = 0.608  # 0.6639
    df201 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_30.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_60.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_90.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_120.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_150.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_180.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0817/20230817_occur_M6_y16_m01_y23_m7_result_210.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    df_all = pd.concat([df201, df202, df203, df204, df205, df206, df207])
    col = df_all.columns.tolist()
    print(col)
    start_date = datetime(2023, 8, 16)
    column_key = start_date.strftime("%Y-%m-%d")
    weight_df = (df201[column_key] * ks_occur_30 + df202[column_key] * ks_occur_60 + df203[column_key] * ks_occur_90 + df204[column_key] * ks_occur_120 + \
                 df205[column_key] * ks_occur_150 + df206[column_key] * ks_occur_180 + df207[column_key] * ks_occur_210) / \
                (ks_occur_30 + ks_occur_60 + ks_occur_90 + ks_occur_120 + ks_occur_150 + ks_occur_180 + ks_occur_210)
    #print(weight_df.tolist())
    df = pd.DataFrame()
    df['Y'] = df201['Y']
    df['customerid'] = df201['customerid']
    df['prob'] = weight_df  # df201[column_key]
    # print(column_key + ' ='*20)
    # 按照列 'A' 的值进行过滤
    #filtered_df = df[df['prob'] > 0.5]
    filtered_df = df[df['Y'] == 0]
    df_sort = filtered_df.sort_values(by='prob', ascending=False).head(10)
    print(df_sort)

def weight_vote_treat_predict_weekly():
    type = 'treat'
    ks_treat_7 = 0.49167 # 0.60
    ks_treat_14 = 0.51666 # 0.65
    ks_treat_21 = 0.5666666 # 0.75
    ks_treat_28 = 0.5499999 # 0.666
    ks_treat_35 = 0.5916666 # 0.675
    ks_treat_42 = 0.583333 # 0.566
    ks_treat_49 = 0.60   # 0.625
    df201 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_14.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_21.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_28.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_35.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_42.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0817/20230810_treat_M6_y16_m01_y23_m7_result_49.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    df_all = pd.concat([df201, df202, df203, df204, df205, df206, df207])
    col = df_all.columns.tolist()
    print(col)
    start_date = datetime(2023, 7, 31)
    column_key = start_date.strftime("%Y-%m-%d")
    weight_df = (df201[column_key] * ks_treat_7 + df202[column_key] * ks_treat_14 + df203[column_key] * ks_treat_21 + df204[column_key] * ks_treat_28 + \
                 df205[column_key] * ks_treat_35 + df206[column_key] * ks_treat_42 + df207[column_key] * ks_treat_49) / \
                (ks_treat_7 + ks_treat_14 + ks_treat_21 + ks_treat_28 + ks_treat_35 + ks_treat_42 + ks_treat_49)
    #print(weight_df.tolist())
    df = pd.DataFrame()
    df['Y'] = df201['Y']
    df['customerid'] = df201['customerid']
    df['prob'] = weight_df  # df201[column_key]
    # print(column_key + ' ='*20)
    # 按照列 'A' 的值进行过滤
    #filtered_df = df[df['prob'] > 0.5]
    filtered_df = df[df['Y'] == 0]
    df_sort = filtered_df.sort_values(by='prob', ascending=False).head(10)
    print(df_sort)

def max_vote_occur_static():
    type = 'occur'
    ks_occur_30 = 0.4621 # 0.544  # 0.4621->30   0.4812->7  0.00448
    ks_occur_60 = 0.4516 # 0.432  # 0.4516 ->30 0.005017   0.45875->7  0.006962
    ks_occur_90 = 0.45916 # 0.496  # 0.45916 0.003088  0.549583 ->7  0.00178
    ks_occur_120 = 0.3525  # 0.6
    ks_occur_150 = 0.3333 # 0.6479999999
    ks_occur_180 = 0.464
    ks_occur_210 = 0.608
    df201 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_30_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_60_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_90_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_120_fl_20000.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_150_fl_20000.csv", header=0, sep=',', encoding='gbk')
    df206 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_180.csv", header=0, sep=',', encoding='gbk')
    df207 = pd.read_csv("./result/0815/20230815_occur_M6_y16_m1_y23_m1_result_test_210.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    merged_df = pd.merge(df201, df202, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df203, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df204, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df205, on=['Y', 'customerid'])
    print(merged_df.head())
    #merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob_x', 'prob_y', 'prob']].max(axis=1)
    merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob']].max(axis=1)
    print(merged_df.head())
    #print(weight_df.tolist())
    # 使用value_counts()函数计算值的频次
    value_counts = df201['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    fpr, tpr, thresholds = metrics.roc_curve(df201['Y'], merged_df['max_prob'], pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i],(tpr[i]*count_1 + fpr[i]*count_0)/(count_0 + count_1),tpr[i],tpr[i]*count_1, fpr[i],fpr[i]*count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def max_vote_treat_static():
    type = 'treat'
    df201 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_7_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_14_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_21_fl.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_28_fl.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_35_fl.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    merged_df = pd.merge(df201, df202, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df203, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df204, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df205, on=['Y', 'customerid'])
    print(merged_df.head())
    merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob_x', 'prob_y','prob']].max(axis=1)
    print(merged_df.head())
    #print(weight_df.tolist())
    # 使用value_counts()函数计算值的频次
    value_counts = df201['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    fpr, tpr, thresholds = metrics.roc_curve(merged_df['Y'], merged_df['max_prob'], pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
        #          fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def max_vote_occur_treat_static():
    type_a = 'occur'
    df201 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_30_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_60_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m1_y23_m1_result_test_90_fl_35000_7.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_120_fl_20000.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_occur_M6_y16_m1_y23_m1_result_test_150_fl_20000.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    merged_df = pd.merge(df201, df202, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df203, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df204, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df205, on=['Y', 'customerid'])
    print(merged_df.head())
    #merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob_x', 'prob_y', 'prob']].max(axis=1)
    merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob']].max(axis=1)

    df_occur = pd.DataFrame()
    df_occur['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
    df_occur['probwo'] = merged_df['max_prob']

    type_b = 'treat'
    df201 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_7_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_14_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_21_fl.csv", header=0, sep=',', encoding='gbk')
    df204 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_28_fl.csv", header=0, sep=',', encoding='gbk')
    df205 = pd.read_csv("./result/0826/20230826_treat_M6_y16_m1_y23_m1_result_test_35_fl.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    print(df202.shape)
    print(df203.shape)
    merged_df = pd.merge(df201, df202, on=['Y', 'customerid'])
    merged_df = pd.merge(merged_df, df203, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df204, on=['Y', 'customerid'])
    #merged_df = pd.merge(merged_df, df205, on=['Y', 'customerid'])
    print(merged_df.head())
    #merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob_x', 'prob_y','prob']].max(axis=1)
    merged_df['max_prob'] = merged_df[['prob_x', 'prob_y', 'prob']].max(axis=1)
    print(merged_df.head())

    df_treat = pd.DataFrame()
    df_treat['Y'] = df201['Y']
    df_treat['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
    df_treat['probwt'] = merged_df['max_prob']

    # 按照列 'customerid' 进行交集操作
    df_intersection = pd.merge(df_occur, df_treat, on='customerid')
    print('df_intersection.shape:',df_intersection.shape)
    print(df_intersection.head())
    # 按照列 'probwo' 的值进行过滤
    threshold = 0.05  # adjust
    df_filtered = df_intersection[df_intersection['probwo'] > threshold]
    print(df_filtered)
    # 使用value_counts()函数计算值的频次
    value_counts = df_filtered['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    #exit(0)
    fpr, tpr, thresholds = metrics.roc_curve(df_filtered['Y'], df_filtered['probwt'], pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.3:
        #    print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
        #          fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type_a + "_"+ str(threshold) + "_" + type_b +"_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/20230826_" + type_a + "_"+ str(threshold) + "_" + type_b + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_fl.png")
    plt.show()

def max_vote_occur_dynamic_for_report():
    type = 'occur'
    df201 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_30_fl.csv", header=0, sep=',', encoding='gbk')
    df202 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_60_fl.csv", header=0, sep=',', encoding='gbk')
    df203 = pd.read_csv("./result/0826/20230901_occur_M6_y16_m01_y23_m01_y23_m07_result_dynamic_90_fl.csv", header=0, sep=',', encoding='gbk')

    col = df201.columns.tolist()
    print(col)
    start_date = datetime(2023, 6, 4)

    df_all = pd.DataFrame()
    for i in np.arange(0, 24, 1):
        print('='*16,i)
        column_key = start_date.strftime("%Y-%m-%d")
        max_values = np.maximum(np.maximum(df201[column_key], df202[column_key]), df203[column_key])
        df = pd.DataFrame()
        df['Y'] = df201['Y']
        df['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
        df['prob'] = max_values #df201[column_key]
        #print(column_key + ' ='*20)
        # 按照列 'A' 的值进行过滤
        filtered_df = df[df['prob'] > 0.9]
        print(filtered_df)
        df_all = pd.concat([df_all, filtered_df])
        df_all.drop_duplicates(subset=['customerid'],keep='first',inplace=True)
        value_counts = df_all['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print(column_key, count_1, count_0)
        start_date -= timedelta(days=7)

def treat_ks():
    type = 'treat'
    df201 = pd.read_csv("./result/0825/20230825_treat_M6_y16_m1_y23_m1_result_val_7_fl.csv", header=0, sep=',', encoding='gbk')

    print(df201.shape)
    df_1 = df201[df201['Y'] == 1]
    df_0_all = df201[df201['Y'] == 0]
    selected_groups_0 = df_0_all['customerid'].drop_duplicates().sample(n=1000, random_state=1000)
    df_0 = df_0_all.groupby('customerid').apply(lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_0 = df_0.dropna(subset=['Y'])
    print(df_1.shape)
    #print(df_0)
    df201 = pd.concat([df_0_all, df_1])
    print(df201.shape)
    weight_df = df201['prob']
    #print(weight_df.tolist())
    # 使用value_counts()函数计算值的频次
    value_counts = df201['Y'].value_counts()
    # 获取值为1的频次
    count_1 = value_counts.get(1, 0)
    count_0 = value_counts.get(0, 0)
    print(count_0,count_1)
    fpr, tpr, thresholds = metrics.roc_curve(df201['Y'], weight_df, pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(thresholds[i], (tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1), tpr[i], tpr[i] * count_1,
              fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
        #if tpr[i] > 0.01:
        #    print(thresholds[i],(tpr[i] * count_1 + fpr[i] * count_0) / (count_0 + count_1),tpr[i], tpr[i] * count_1,
        #          fpr[i], fpr[i] * count_0, tpr[i] - fpr[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    #plt.savefig("0815_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_ensemble_e50.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("0815_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_ensemble_e50.png")
    plt.show()

def augment_data_occur_for_report():
    #df201 = pd.read_csv("./result/0920/20230920_occur_step5_reclass_less200_100_20230101_20_10_16_ftr_17_t30_fl_test_aug_2_2.csv", header=0,sep=',', encoding='gbk')
    df201 = pd.read_csv("./result/0926/20230926_occur_2017_addcredit_step5_reclass_less200_100_20230101_20_10_16_ftr_91_t30_fl_test_aug_1_1.csv",
        header=0, sep=',', encoding='gbk')
    df201 = pd.read_csv("./result/0926/20230926_occur_2017_addcredit_step5_reclass_less200_100_20230101_20_10_16_ftr_91_t30_fl_test_aug_0_0.csv",
        header=0, sep=',', encoding='gbk')
    df201 = pd.read_csv(
        "./result/20230926_occur_2016_addcredit_step5_reclass_less200_100_20230101_20_10_16_ftr_91_t30_fl_test_aug_1_1.csv",
        header=0, sep=',', encoding='gbk')
    df_all = pd.DataFrame()
    #for i in np.arange(0, 1, ):
    for i in [0,1]:
        print('=' * 16, i)
        df = pd.DataFrame()
        df['Y'] = df201['Y']
        df['customerid'] = df201['customerid'].str.replace('_.*', '', regex=True)
        #df['customerid'] = df201['customerid']
        df['prob'] = df201['prob']  # df201[column_key]
        # print(column_key + ' ='*20)
        # 按照列 'A' 的值进行过滤
        #filtered_df = df[df['prob'] >= (1-i)]
        #filtered_df = df[df['prob'] >= 0.974]
        filtered_df = df[df['prob'] >= 0.925]  # 0.925  0.946
        #print(filtered_df)
        df_all = pd.concat([df_all, filtered_df])
        df_all.drop_duplicates(subset=['customerid'], keep='first', inplace=True)
        value_counts = df_all['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print(1-i, count_1, count_0)
        print(df_all[df_all['Y']==1])

def test_file():
    step = 5
    date_str = datetime(2023, 9, 20).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '17'
    filter_num_ratio = 1 / 8
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16
    cluster_model_path = './model/cluster_step'+str(step) +'/'
    cluster_model_file = date_str+'-repr-cluster-partial-train-6.pkl'
    file_path = cluster_model_path + cluster_model_file
    if not os.path.exists(file_path):
        print(file_path)
        print('file not exsit')
    else:
        print('file exist')

if __name__ == '__main__':
    # test_file()
    #weight_vote_occur_treat()
    #weight_vote_occur_static()
    #weight_vote_treat_static()
    #weight_vote_occur_dynamic_for_report()
    #weight_vote_treat_dynamic_for_report()
    #weight_vote_treat_static()
    #weight_vote_occur_predict_weekly()
    #weight_vote_treat_predict_weekly()
    #treat_ks()
    #max_vote_occur_static()
    #max_vote_treat_static()
    #max_vote_occur_treat_static()
    #max_vote_occur_dynamic_for_report()
    augment_data_occur_for_report()