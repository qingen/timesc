# coding=utf-8
import sys, os, time
import cmath
import math
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime, timedelta

import warnings
import paddlets
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.models.classify.dl.cnn import CNNClassifier
from paddlets.models.classify.dl.inception_time import InceptionTimeClassifier
from paddlets.datasets.repository import get_dataset

warnings.filterwarnings('ignore',category=DeprecationWarning)

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def rmse(y_test, y):
    return math.sqrt(sum((y_test - y) ** 2) / len(y))



def train_treat_for_report():
    df23_2 = pd.read_csv("./data/0825_train/treat/2023_4_202308252055.csv", header=0, sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0825_train/treat/2023_1_4_202308252051.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/treat/2022_10_12_202308251700.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/treat/2022_7_10_202308251655.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/treat/2022_4_7_202308251652.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/treat/2022_1_4_202308251648.csv", header=0, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/treat/2021_10_12_202308251644.csv", header=0, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/treat/2021_7_10_202308251637.csv", header=0, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/treat/2021_4_7_202308251627.csv", header=0, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/treat/2021_1_4_202308251610.csv", header=0, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/treat/2020_10_12_202308251606.csv", header=0, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/treat/2020_7_10_202308251745.csv", header=0, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/treat/2020_4_7_202308251558.csv", header=0, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/treat/2020_1_4_202308251554.csv", header=0, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/treat/2019_10_12_202308251548.csv", header=0, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/treat/2019_7_10_202308251543.csv", header=0, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/treat/2019_4_7_202308251537.csv", header=0, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/treat/2019_1_4_202308251532.csv", header=0, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/treat/2018_10_12_202308251527.csv", header=0, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/treat/2018_7_10_202308251522.csv", header=0, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/treat/2018_4_7_202308251519.csv", header=0, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/treat/2018_1_4_202308251511.csv", header=0, sep=',', encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/treat/2017_10_12_202308251507.csv", header=0, sep=',', encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/treat/2017_7_10_202308251503.csv", header=0, sep=',', encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/treat/2017_4_7_202308251451.csv", header=0, sep=',', encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/treat/2017_1_4_202308251449.csv", header=0, sep=',', encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/treat/2016_7_12_202308251444.csv", header=0, sep=',', encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/treat/2016_1_7_202308251439.csv", header=0, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    #col = df_all.columns.tolist()
    #col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    col = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
           'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90',
           'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90',
           'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR',
           'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180',
           'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60',
           'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO',
           'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180',
           'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90',
           'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15',
           'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7',
           'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365',
           'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60',
           'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30',
           'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7',
           'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180',
           'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30',
           'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365',
           'JH_30_CNT', 'JH_60_CNT', 'JH_90_CNT', 'JH_180_CNT', 'JH_HEGE', 'JH_WANSHAN', 'JH_XIANYI', 'JH_XIANYI_R',
           'JH_WAIFANG', 'JH_WAIFANG_R', 'JH_YIDONGCL', 'JH_YIDONGCL_R', 'JH_CCC', 'JH_SC_R', 'JH_SALE_R', 'JH_ZT_R',
           'JH_WT_R', 'JH_XFEW_R', 'JH_CZ_R', 'JH_WGWF_R', 'JH_HGZ', 'JH_HGZ_R', 'JH_JTS', 'JH_3YCHK_R', 'JH_3SZYD_R',
           'JH_3HGZWF_R', 'JH_5YCHK_R', 'JH_5SZYD_R', 'JH_5HGZWF_R', 'JH_10YCHK_R', 'JH_10SZYD_R', 'JH_10HGZWF_R',
           'JH_3YCHK10_R', 'JH_3SZYD10_R', 'JH_3HGZWF10_R', 'JH_6YCHK_R', 'JH_6SZYD_R', 'JH_6HGZWF_R', 'PES_30HUIDIZHI',
           'PES_30HCL', 'PES_30MAHCL', 'PES_30MAHTS', 'PES_30MIHTS', 'PES_30AVGHTS', 'PES_30AVGHCL', 'PES_30MAHCL_R',
           'PES_30CHUDIZHI', 'PES_30CCL', 'PES_30MACCL', 'PES_30AVGCCL', 'PES_30MACCL_R', 'GRP_CNT', 'GRP_AVAILAMT_SUM',
           'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'LOAN_GHD_30D_IND',
           'LOAN_GHD_30D_CNT', 'LOAN_AJ_30D_IND', 'LOAN_AJ_30D_CNT', 'LOAN_GHDAJ_30D_IND', 'LOAN_GHDAJ_30D_CNT',
           'LOAN_GHD_90D_IND', 'LOAN_GHD_90D_CNT', 'LOAN_AJ_90D_IND', 'LOAN_AJ_90D_CNT', 'LOAN_GHDAJ_90D_IND',
           'LOAN_GHDAJ_90D_CNT', 'SN_XFDQ_180D_CNT_2', 'SNEX_30D_HKKDDZ_CNT', 'SNEX_30D_HKCL_CNT',
           'SNEX_30D_DKDHKCL_MAX', 'SNEX_30D_HKTS_MAX', 'SNEX_30D_HKTS_MIN', 'SNEX_30D_HKTS_AVG',
           'SNEX_30D_SYKDHKCL_AVG', 'SNEX_30D_DKDHKCL_MAX_R', 'SNEX_30D_CKKDDZ_CNT', 'SNEX_30D_CKCL_CNT',
           'SNEX_30D_DKDCKCL_MAX', 'SNEX_30D_SYKDCKCL_AVG', 'SNEX_30D_DKDCKCL_MAX_R', 'SNEX_CKRJSQ_30D_CNT',
           'SNEX_CKSQKDDZ_30D_R', 'SNEX_CKRJSQ_90D_CNT', 'SNEX_CKSQKDDZ_90D_R', 'SNEX_CKSQKDDZ_180D_R',
           'SNEX_ONLINE90D_R', 'SNEX_XFDQ_30D_CNT', 'SNEX_XFDQ_90D_CNT', 'SNEX_XFDQ_180D_CNT', 'XSZQ30D_DIFF',
           'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R', 'FREESPANRP_30D_R', 'FREESPANRP_90D_R',
           'FREESPANRP_180D_R', 'FREESPANRP_360D_R', 'REPAYCNT3_90D', 'REPAYCNT7_90D', 'REPAYCNT3_180D',
           'REPAYCNT7_180D', 'INV_RATIO_90', 'STOCK_OVER_91_RATIO.1', 'RPCNT3_90_90AGE_R', 'RPCNT7_90_90AGE_R',
           'RPCNT3_180_90AGE_R', 'RPCNT7_180_90AGE_R', 'RPCNT3_90_90INV_R', 'RPCNT7_90_90INV_R', 'RPCNT3_180_90INV_R',
           'RPCNT7_180_90INV_R', 'AUDIT_1YCHK_IND', 'AUDIT_5YCHKSZYD_R', 'AUDIT_10YCHKSZYD_R', 'AUDIT_5YCHKSZYDHGWF_R',
           'AUDIT_10YCHKSZYDHGWF_R', 'AUDIT_1YCHKWGWF_IND', 'AUDIT_1YCHKPCT25_IND', 'EXT_12M_R']

    n_line_tail = 35  # (1-5)* 7
    n_line_back = 7  # back 7
    n_line_head = 35
    type = 'treat'
    date_str = datetime(2023, 8, 26).strftime("%Y%m%d")

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test 14411 24
    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train valid 193997 318
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: min(x["RDATE"]) >= 20160101)  # for train valid

    del df_all
    ###################### for test 1:100
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part1[df_part1['Y'] == 1]  # 24
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part1_1.shape:',df_part1_1.shape)
    # 从 0 中 筛选出 2400 个
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=2400, random_state=2400)
    # 获取每个选中组的所有样本
    df_part1_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    df_part1 = pd.concat([df_part1_0_selected, df_part1_1])
    print('df_part1.shape: ', df_part1.shape)
    del  df_part1_0,df_part1_1,df_part1_0_selected
    ###################### for train/valid  1:100
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    # 随机选择若干个组
    selected_groups = df_part2['CUSTOMER_ID'].drop_duplicates().sample(n=15000, random_state=15000) # 14975 25
    # 获取每个选中组的所有样本
    df_val = df_part2.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_val = df_val.dropna(subset=['Y'])
    df_val_1 = df_val[df_val['Y'] == 1]
    df_val_1 = df_val_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_val_1.shape:',df_val_1.shape)
    df_val_0_all = df_val[df_val['Y'] == 0]
    selected_groups_0 = df_val_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=2500, random_state=2500)
    print('selected_groups_0  length:',len(selected_groups_0))
    df_val_0 = df_val_0_all.groupby('CUSTOMER_ID').apply(
                lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_val_0 = df_val_0.dropna(subset=['Y'])
    print('df_val_0.shape:',df_val_0.shape)
    df_val = pd.concat([df_val_0, df_val_1])
    print('df_val.shape: ', df_val.shape)
    del df_val_0, df_val_1, df_val_0_all

    # 获取剩余的组
    df_train = df_part2[~df_part2['CUSTOMER_ID'].isin(selected_groups)]
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)

    df_train_1 = df_train[df_train['Y'] == 1]  # 318
    df_train_1 = df_train_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_train_1.shape:', df_train_1.shape)
    df_train_0_all = df_train[df_train['Y'] == 0]
    selected_groups_0 = df_train_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=30000, random_state=30000) # 7 14->35000 21->30000 28->30000
    df_train_0 = df_train_0_all.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_train_0 = df_train_0.dropna(subset=['Y'])
    df_train = pd.concat([df_train_0, df_train_1])
    print('df_train.shape: ', df_train.shape)
    del df_train_0, df_train_1, df_train_0_all

    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)  # .\
    # reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    # .filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

    df_test = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_test.shape: ', df_test.shape)
    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)  # .\
    # reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    # .filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    ###################### del
    del df_part1, df_part2
    ######################

    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT
    tsdatasets_train = TSDataset.load_from_dataframe(
        df=df_train,
        group_id='CUSTOMER_ID',
        # time_col='date',
        # target_cols='TARGET',
        # target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
        # target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
        target_cols=col,
        # known_cov_cols='CUSTOMER_ID',
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )
    tsdatasets_val = TSDataset.load_from_dataframe(
        df=df_val,
        group_id='CUSTOMER_ID',
        target_cols=col,
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )
    tsdatasets_test = TSDataset.load_from_dataframe(
        df=df_test,
        group_id='CUSTOMER_ID',
        target_cols=col,
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )

    fft = FFT(fs=1, half=False)  # _amplitude  half
    # cwt = CWT(scales=n_line_tail/2)
    for data in tsdatasets_train:
        resfft = fft(data)
        # rescwt = cwt(data)  # coefs 63*24 complex128 x+yj
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # --------------- cwt
    #        for i in range(len(rescwt[x+"_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x+"_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x+"_coefs_"+str(i),value=coefs,type='target')
    #            data.set_column(column=x + "_phase_cwt_"+str(i), value=phase, type='target')

    for data in tsdatasets_val:
        resfft = fft(data)
        # rescwt = cwt(data)
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # ----------------- cwt
    #        for i in range(len(rescwt[x + "_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x + "_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x + "_coefs_" + str(i), value=coefs, type='target')
    #            data.set_column(column=x + "_phase_cwt_" + str(i), value=phase, type='target')

    for data in tsdatasets_test:
        resfft = fft(data)
        # rescwt = cwt(data)
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # ----------------- cwt
    #        for i in range(len(rescwt[x + "_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x + "_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x + "_coefs_" + str(i), value=coefs, type='target')
    #            data.set_column(column=x + "_phase_cwt_" + str(i), value=phase, type='target')

    y_train = []
    y_val = []
    y_test = []
    y_train_customerid = []
    y_val_customerid = []
    y_test_customerid = []
    for dataset in tsdatasets_train:
        y_train.append(dataset.static_cov['Y'])
        y_train_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_train = np.array(y_train)
    for dataset in tsdatasets_val:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_val = np.array(y_val)
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
    tsdatasets_val = min_max_scaler.fit_transform(tsdatasets_val)
    tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)

    network = InceptionTimeClassifier(max_epochs=50, patience=20, kernel_size=16)
    network.fit(tsdatasets_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score
    # network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
    # network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network.save('./model/0808_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y22_m10_v1.itc')
    #network.save('./model/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    #network.save('./model/0815_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_v1.itc')
    network.save('./model/'+date_str+'_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m01_fl.itc')

    ############################### val
    filename = './result/'+date_str+'_' + type + '_M6_y16_m1_y23_m1_result_val_' + str(n_line_tail) + '_fl.csv'
    pred_val = network.predict(tsdatasets_val)
    pred_val_prob = network.predict_proba(tsdatasets_val)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_val
    df['customerid'] = y_val_customerid
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    preds_prob = pred_val_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(filename, index=False)

    print(metrics.accuracy_score(y_val, pred_val, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_val, pred_val, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_val, pred_val, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_val, pred_val_prob, pos_label=1, )  # drop_intermediate=True
    print('val_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        #if tpr[i] > 0.5:
            #print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(val)')
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m01_y23_m7_ROC_val_" + str(n_line_tail) + "_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m01_y23_m7_KS_val_" + str(n_line_tail) + "_fl.png")
    plt.show()

    ############################### test
    filename = './result/'+date_str+'_' + type + '_M6_y16_m1_y23_m1_result_test_' + str(n_line_tail) + '_fl.csv'
    pred_test = network.predict(tsdatasets_test)
    pred_test_prob = network.predict_proba(tsdatasets_test)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_test
    df['customerid'] = y_test_customerid
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    preds_prob = pred_test_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(filename, index=False)

    print(metrics.accuracy_score(y_test, pred_test, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_test, pred_test, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_test, pred_test, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test_prob, pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        #if tpr[i] > 0.5:
            #print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
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
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_" + str(n_line_tail) + "_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_" + str(n_line_tail) + "_fl.png")
    plt.show()

    ############################### psi
    ytr_prob_psi = pred_val_prob
    yte_prob_psi = pred_test_prob

    # 计算 psi
    num = 10
    part = 1 / (num + 0.0)
    intervals_1 = {'{}-{}'.format(part * x, part * (x + 1)): 0 for x in range(num)}
    for pp in list(ytr_prob_psi):
        for interval in intervals_1:
            start, end = tuple(interval.split('-'))
            if float(start) <= pp <= float(end):
                intervals_1[interval] += 1
                break
    intervals_2 = {'{}-{}'.format(part * x, part * (x + 1)): 0 for x in range(num)}
    for pp in yte_prob_psi:
        for interval in intervals_2:
            start, end = tuple(interval.split('-'))
            if float(start) <= pp <= float(end):
                intervals_2[interval] += 1
                break
    print(intervals_1)
    print(intervals_2)
    psi_list = []
    len1 = len(ytr_prob_psi)
    len2 = len(yte_prob_psi)
    for interval in intervals_1:
        pct_1 = intervals_1[interval] / (len1 + 0.0)
        pct_2 = intervals_2[interval] / (len2 + 0.0)
        psi_list.append((pct_2 - pct_1) * math.log((pct_2 + 0.0001) / (pct_1 + 0.0001), math.e))
    print(psi_list)
    print('psi = ', sum(psi_list))

    ############################### train
    pred_train = network.predict(tsdatasets_train)
    pred_train_prob = network.predict_proba(tsdatasets_train)[:, 1]

    print(metrics.accuracy_score(y_train, pred_train, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_train, pred_train, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_train, pred_train, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_train_prob, pos_label=1, )  # drop_intermediate=True
    print('train_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        if tpr[i] > 0.5:
            print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(train)')
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_ROC_train_" + str(n_line_tail) + "_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_KS_train_" + str(n_line_tail) + "_fl.png")
    plt.show()



def train_treat_for_predict():
    df23_2 = pd.read_csv("./data/0825_train/treat/2023_4_202308252055.csv", header=0, sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0825_train/treat/2023_1_4_202308252051.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/treat/2022_10_12_202308251700.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/treat/2022_7_10_202308251655.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/treat/2022_4_7_202308251652.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/treat/2022_1_4_202308251648.csv", header=0, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/treat/2021_10_12_202308251644.csv", header=0, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/treat/2021_7_10_202308251637.csv", header=0, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/treat/2021_4_7_202308251627.csv", header=0, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/treat/2021_1_4_202308251610.csv", header=0, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/treat/2020_10_12_202308251606.csv", header=0, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/treat/2020_7_10_202308251745.csv", header=0, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/treat/2020_4_7_202308251558.csv", header=0, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/treat/2020_1_4_202308251554.csv", header=0, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/treat/2019_10_12_202308251548.csv", header=0, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/treat/2019_7_10_202308251543.csv", header=0, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/treat/2019_4_7_202308251537.csv", header=0, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/treat/2019_1_4_202308251532.csv", header=0, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/treat/2018_10_12_202308251527.csv", header=0, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/treat/2018_7_10_202308251522.csv", header=0, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/treat/2018_4_7_202308251519.csv", header=0, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/treat/2018_1_4_202308251511.csv", header=0, sep=',', encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/treat/2017_10_12_202308251507.csv", header=0, sep=',', encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/treat/2017_7_10_202308251503.csv", header=0, sep=',', encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/treat/2017_4_7_202308251451.csv", header=0, sep=',', encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/treat/2017_1_4_202308251449.csv", header=0, sep=',', encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/treat/2016_7_12_202308251444.csv", header=0, sep=',', encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/treat/2016_1_7_202308251439.csv", header=0, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    # col = df_all.columns.tolist()
    # col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    col = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
           'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90',
           'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90',
           'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR',
           'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180',
           'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60',
           'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO',
           'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180',
           'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90',
           'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15',
           'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7',
           'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365',
           'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60',
           'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30',
           'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7',
           'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180',
           'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30',
           'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365',
           'JH_30_CNT', 'JH_60_CNT', 'JH_90_CNT', 'JH_180_CNT', 'JH_HEGE', 'JH_WANSHAN', 'JH_XIANYI', 'JH_XIANYI_R',
           'JH_WAIFANG', 'JH_WAIFANG_R', 'JH_YIDONGCL', 'JH_YIDONGCL_R', 'JH_CCC', 'JH_SC_R', 'JH_SALE_R', 'JH_ZT_R',
           'JH_WT_R', 'JH_XFEW_R', 'JH_CZ_R', 'JH_WGWF_R', 'JH_HGZ', 'JH_HGZ_R', 'JH_JTS', 'JH_3YCHK_R', 'JH_3SZYD_R',
           'JH_3HGZWF_R', 'JH_5YCHK_R', 'JH_5SZYD_R', 'JH_5HGZWF_R', 'JH_10YCHK_R', 'JH_10SZYD_R', 'JH_10HGZWF_R',
           'JH_3YCHK10_R', 'JH_3SZYD10_R', 'JH_3HGZWF10_R', 'JH_6YCHK_R', 'JH_6SZYD_R', 'JH_6HGZWF_R', 'PES_30HUIDIZHI',
           'PES_30HCL', 'PES_30MAHCL', 'PES_30MAHTS', 'PES_30MIHTS', 'PES_30AVGHTS', 'PES_30AVGHCL', 'PES_30MAHCL_R',
           'PES_30CHUDIZHI', 'PES_30CCL', 'PES_30MACCL', 'PES_30AVGCCL', 'PES_30MACCL_R', 'GRP_CNT', 'GRP_AVAILAMT_SUM',
           'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'LOAN_GHD_30D_IND',
           'LOAN_GHD_30D_CNT', 'LOAN_AJ_30D_IND', 'LOAN_AJ_30D_CNT', 'LOAN_GHDAJ_30D_IND', 'LOAN_GHDAJ_30D_CNT',
           'LOAN_GHD_90D_IND', 'LOAN_GHD_90D_CNT', 'LOAN_AJ_90D_IND', 'LOAN_AJ_90D_CNT', 'LOAN_GHDAJ_90D_IND',
           'LOAN_GHDAJ_90D_CNT', 'SN_XFDQ_180D_CNT_2', 'SNEX_30D_HKKDDZ_CNT', 'SNEX_30D_HKCL_CNT',
           'SNEX_30D_DKDHKCL_MAX', 'SNEX_30D_HKTS_MAX', 'SNEX_30D_HKTS_MIN', 'SNEX_30D_HKTS_AVG',
           'SNEX_30D_SYKDHKCL_AVG', 'SNEX_30D_DKDHKCL_MAX_R', 'SNEX_30D_CKKDDZ_CNT', 'SNEX_30D_CKCL_CNT',
           'SNEX_30D_DKDCKCL_MAX', 'SNEX_30D_SYKDCKCL_AVG', 'SNEX_30D_DKDCKCL_MAX_R', 'SNEX_CKRJSQ_30D_CNT',
           'SNEX_CKSQKDDZ_30D_R', 'SNEX_CKRJSQ_90D_CNT', 'SNEX_CKSQKDDZ_90D_R', 'SNEX_CKSQKDDZ_180D_R',
           'SNEX_ONLINE90D_R', 'SNEX_XFDQ_30D_CNT', 'SNEX_XFDQ_90D_CNT', 'SNEX_XFDQ_180D_CNT', 'XSZQ30D_DIFF',
           'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R', 'FREESPANRP_30D_R', 'FREESPANRP_90D_R',
           'FREESPANRP_180D_R', 'FREESPANRP_360D_R', 'REPAYCNT3_90D', 'REPAYCNT7_90D', 'REPAYCNT3_180D',
           'REPAYCNT7_180D', 'INV_RATIO_90', 'STOCK_OVER_91_RATIO.1', 'RPCNT3_90_90AGE_R', 'RPCNT7_90_90AGE_R',
           'RPCNT3_180_90AGE_R', 'RPCNT7_180_90AGE_R', 'RPCNT3_90_90INV_R', 'RPCNT7_90_90INV_R', 'RPCNT3_180_90INV_R',
           'RPCNT7_180_90INV_R', 'AUDIT_1YCHK_IND', 'AUDIT_5YCHKSZYD_R', 'AUDIT_10YCHKSZYD_R', 'AUDIT_5YCHKSZYDHGWF_R',
           'AUDIT_10YCHKSZYDHGWF_R', 'AUDIT_1YCHKWGWF_IND', 'AUDIT_1YCHKPCT25_IND', 'EXT_12M_R']

    n_line_tail = 35  # (1-5)* 7
    n_line_back = 7  # back 7
    n_line_head = 35
    train_0_sample = 34200   # 7 14->35000 21->30000 28->30000
    type = 'treat'
    date_str = datetime(2023, 9, 6).strftime("%Y%m%d")

    #df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test 14411 24
    #df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train valid 193997 318
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: min(x["RDATE"]) >= 20160101)  # for train valid
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train 1:100
    df_all_0 = df_all[df_all['Y'] == 0]
    df_all_1 = df_all[df_all['Y'] == 1]  # 24 + 318
    df_all_1 = df_all_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_all_1.shape:', df_all_1.shape)
    # 从 0 中 筛选出 34200 个
    selected_groups = df_all_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_sample, random_state=train_0_sample - n_line_head ) ## change everyone
    # 获取每个选中组的所有样本
    df_all_0_selected = df_all_0.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_all_0_selected = df_all_0_selected.dropna(subset=['Y'])
    df_all_0_selected = df_all_0_selected.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_all_0_selected.shape:', df_all_0_selected.shape)
    df_train = pd.concat([df_all_0_selected, df_all_1])
    print('df_train.shape: ', df_train.shape)
    del  df_all,df_all_0,df_all_0_selected,df_all_1

    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT
    tsdatasets_train = TSDataset.load_from_dataframe(
        df=df_train,
        group_id='CUSTOMER_ID',
        # time_col='date',
        # target_cols='TARGET',
        # target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
        # target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
        target_cols=col,
        # known_cov_cols='CUSTOMER_ID',
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )

    fft = FFT(fs=1, half=False)  # _amplitude  half
    # cwt = CWT(scales=n_line_tail/2)
    for data in tsdatasets_train:
        resfft = fft(data)
        # rescwt = cwt(data)  # coefs 63*24 complex128 x+yj
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # --------------- cwt
    #        for i in range(len(rescwt[x+"_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x+"_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x+"_coefs_"+str(i),value=coefs,type='target')
    #            data.set_column(column=x + "_phase_cwt_"+str(i), value=phase, type='target')

    y_train = []
    y_train_customerid = []
    for dataset in tsdatasets_train:
        y_train.append(dataset.static_cov['Y'])
        y_train_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_train = np.array(y_train)

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)

    network = InceptionTimeClassifier(max_epochs=50, patience=20, kernel_size=16)
    network.fit(tsdatasets_train, y_train)

    #network.save('./model/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    network.save('./model/' + date_str + '_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_fl.itc')

def clean_data_train_treat_continue_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE','XSZQ30D_DIFF','XSZQ90D_DIFF','UAR_AVG_365','UAR_AVG_180','UAR_AVG_90','UAR_AVG_7',
               'UAR_AVG_15','UAR_AVG_30','UAR_AVG_60','GRP_AVAILAMT_SUM','USEAMOUNT_RATIO','UAR_CHA_365','UAR_CHA_15','UAR_CHA_30',
               'UAR_CHA_60','UAR_CHA_90','UAR_CHA_180','UAR_CHA_7','STOCK_AGE_AVG_365','SDV_REPAY_365','INV_AVG_365',
               'GRP_REPAYCARS180_SUM','JH_CCC','JH_HGZ','JH_JTS','LRR_AVG_365','LSR_91_AVG_365','STOCK_AGE_AVG_180',
               'FREESPANRP_360D_R','SDV_REPAY_180','XSZQ180D_R','JH_SC_R','INV_AVG_180','GRP_REPAYCARS90_SUM','GRP_CNT',
               'JH_HGZ_R','GRP_USEAMT_SUM','GRP_REPAYCARS30_SUM','STOCK_AGE_AVG_90','LSR_91_AVG_180']   # 40 cols
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 20 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df23_2 = pd.read_csv("./data/0825_train/treat/2023_4_202308252055.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0825_train/treat/2023_1_4_202308252051.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/treat/2022_10_12_202308251700.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/treat/2022_7_10_202308251655.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/treat/2022_4_7_202308251652.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/treat/2022_1_4_202308251648.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/treat/2021_10_12_202308251644.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/treat/2021_7_10_202308251637.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/treat/2021_4_7_202308251627.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/treat/2021_1_4_202308251610.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/treat/2020_10_12_202308251606.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/treat/2020_7_10_202308251745.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/treat/2020_4_7_202308251558.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/treat/2020_1_4_202308251554.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/treat/2019_10_12_202308251548.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/treat/2019_7_10_202308251543.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/treat/2019_4_7_202308251537.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/treat/2019_1_4_202308251532.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/treat/2018_10_12_202308251527.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/treat/2018_7_10_202308251522.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/treat/2018_4_7_202308251519.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/treat/2018_1_4_202308251511.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/treat/2017_10_12_202308251507.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/treat/2017_7_10_202308251503.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/treat/2017_4_7_202308251451.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/treat/2017_1_4_202308251449.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/treat/2016_7_12_202308251444.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/treat/2016_1_7_202308251439.csv", header=0, usecols=usecols, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23_1, df23_2

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    #col = df_all.columns.tolist()
    #col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    col = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
           'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90',
           'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90',
           'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR',
           'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180',
           'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60',
           'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO',
           'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180',
           'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90',
           'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15',
           'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7',
           'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365',
           'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60',
           'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30',
           'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7',
           'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180',
           'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30',
           'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365',
           'JH_30_CNT', 'JH_60_CNT', 'JH_90_CNT', 'JH_180_CNT', 'JH_HEGE', 'JH_WANSHAN', 'JH_XIANYI', 'JH_XIANYI_R',
           'JH_WAIFANG', 'JH_WAIFANG_R', 'JH_YIDONGCL', 'JH_YIDONGCL_R', 'JH_CCC', 'JH_SC_R', 'JH_SALE_R', 'JH_ZT_R',
           'JH_WT_R', 'JH_XFEW_R', 'JH_CZ_R', 'JH_WGWF_R', 'JH_HGZ', 'JH_HGZ_R', 'JH_JTS', 'JH_3YCHK_R', 'JH_3SZYD_R',
           'JH_3HGZWF_R', 'JH_5YCHK_R', 'JH_5SZYD_R', 'JH_5HGZWF_R', 'JH_10YCHK_R', 'JH_10SZYD_R', 'JH_10HGZWF_R',
           'JH_3YCHK10_R', 'JH_3SZYD10_R', 'JH_3HGZWF10_R', 'JH_6YCHK_R', 'JH_6SZYD_R', 'JH_6HGZWF_R', 'PES_30HUIDIZHI',
           'PES_30HCL', 'PES_30MAHCL', 'PES_30MAHTS', 'PES_30MIHTS', 'PES_30AVGHTS', 'PES_30AVGHCL', 'PES_30MAHCL_R',
           'PES_30CHUDIZHI', 'PES_30CCL', 'PES_30MACCL', 'PES_30AVGCCL', 'PES_30MACCL_R', 'GRP_CNT', 'GRP_AVAILAMT_SUM',
           'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'LOAN_GHD_30D_IND',
           'LOAN_GHD_30D_CNT', 'LOAN_AJ_30D_IND', 'LOAN_AJ_30D_CNT', 'LOAN_GHDAJ_30D_IND', 'LOAN_GHDAJ_30D_CNT',
           'LOAN_GHD_90D_IND', 'LOAN_GHD_90D_CNT', 'LOAN_AJ_90D_IND', 'LOAN_AJ_90D_CNT', 'LOAN_GHDAJ_90D_IND',
           'LOAN_GHDAJ_90D_CNT', 'SN_XFDQ_180D_CNT_2', 'SNEX_30D_HKKDDZ_CNT', 'SNEX_30D_HKCL_CNT',
           'SNEX_30D_DKDHKCL_MAX', 'SNEX_30D_HKTS_MAX', 'SNEX_30D_HKTS_MIN', 'SNEX_30D_HKTS_AVG',
           'SNEX_30D_SYKDHKCL_AVG', 'SNEX_30D_DKDHKCL_MAX_R', 'SNEX_30D_CKKDDZ_CNT', 'SNEX_30D_CKCL_CNT',
           'SNEX_30D_DKDCKCL_MAX', 'SNEX_30D_SYKDCKCL_AVG', 'SNEX_30D_DKDCKCL_MAX_R', 'SNEX_CKRJSQ_30D_CNT',
           'SNEX_CKSQKDDZ_30D_R', 'SNEX_CKRJSQ_90D_CNT', 'SNEX_CKSQKDDZ_90D_R', 'SNEX_CKSQKDDZ_180D_R',
           'SNEX_ONLINE90D_R', 'SNEX_XFDQ_30D_CNT', 'SNEX_XFDQ_90D_CNT', 'SNEX_XFDQ_180D_CNT', 'XSZQ30D_DIFF',
           'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R', 'FREESPANRP_30D_R', 'FREESPANRP_90D_R',
           'FREESPANRP_180D_R', 'FREESPANRP_360D_R', 'REPAYCNT3_90D', 'REPAYCNT7_90D', 'REPAYCNT3_180D',
           'REPAYCNT7_180D', 'INV_RATIO_90', 'STOCK_OVER_91_RATIO.1', 'RPCNT3_90_90AGE_R', 'RPCNT7_90_90AGE_R',
           'RPCNT3_180_90AGE_R', 'RPCNT7_180_90AGE_R', 'RPCNT3_90_90INV_R', 'RPCNT7_90_90INV_R', 'RPCNT3_180_90INV_R',
           'RPCNT7_180_90INV_R', 'AUDIT_1YCHK_IND', 'AUDIT_5YCHKSZYD_R', 'AUDIT_10YCHKSZYD_R', 'AUDIT_5YCHKSZYDHGWF_R',
           'AUDIT_10YCHKSZYDHGWF_R', 'AUDIT_1YCHKWGWF_IND', 'AUDIT_1YCHKPCT25_IND', 'EXT_12M_R']

    col = ['XSZQ30D_DIFF','XSZQ90D_DIFF','UAR_AVG_365','UAR_AVG_180','UAR_AVG_90','UAR_AVG_7',
               'UAR_AVG_15','UAR_AVG_30','UAR_AVG_60','GRP_AVAILAMT_SUM','USEAMOUNT_RATIO','UAR_CHA_365','UAR_CHA_15','UAR_CHA_30',
               'UAR_CHA_60','UAR_CHA_90','UAR_CHA_180','UAR_CHA_7','STOCK_AGE_AVG_365','SDV_REPAY_365','INV_AVG_365',
               'GRP_REPAYCARS180_SUM','JH_CCC','JH_HGZ','JH_JTS','LRR_AVG_365','LSR_91_AVG_365','STOCK_AGE_AVG_180',
               'FREESPANRP_360D_R','SDV_REPAY_180','XSZQ180D_R','JH_SC_R','INV_AVG_180','GRP_REPAYCARS90_SUM','GRP_CNT',
               'JH_HGZ_R','GRP_USEAMT_SUM','GRP_REPAYCARS30_SUM','STOCK_AGE_AVG_90','LSR_91_AVG_180']   # 40

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 7  # back 7
    n_line_head = 30  # = tail
    type = 'occur'
    date_str = datetime(2023, 9, 12).strftime("%Y%m%d")
    split_date_str = '20230201'
    ftr_num_str = '18'
    ########## model
    epochs = 5
    patiences = 5  # 10
    kernelsize = 16

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220801)
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230201)  # for train

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230201)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230301)  # for test
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2 , good:bad 100:1
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part1[df_part1['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
            reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail+n_line_back). \
            reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part1_1.shape:',df_part1_1.shape)
    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:',train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    train_0_num_sample = int(train_1_selected.shape[0]/n_line_head * 100)
    print('train_0_num_sample:',train_0_num_sample)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    train_0_selected = train_0_selected.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del  train_0_selected,train_1_selected

    valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 100)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    valid_0_selected = valid_0_selected.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del  df_part1_0,df_part1_1,valid_0_remain,valid_0_selected,valid_1_selected


    ###################### for test good:bad 100:1
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part2_1.shape:', df_part2_1.shape)
    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100)
    print('test_0_num_sample:', train_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    df_part2_0_selected = df_part2_0_selected.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2
    ######################
    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT
    tsdatasets_train = TSDataset.load_from_dataframe(
        df=df_train,
        group_id='CUSTOMER_ID',
        # time_col='date',
        # target_cols='TARGET',
        # target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
        # target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
        target_cols=col,
        # known_cov_cols='CUSTOMER_ID',
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )
    tsdatasets_val = TSDataset.load_from_dataframe(
        df=df_val,
        group_id='CUSTOMER_ID',
        target_cols=col,
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )
    tsdatasets_test = TSDataset.load_from_dataframe(
        df=df_test,
        group_id='CUSTOMER_ID',
        target_cols=col,
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )

    fft = FFT(fs=1, half=False)  # _amplitude  half
    # cwt = CWT(scales=n_line_tail/2)
    for data in tsdatasets_train:
        resfft = fft(data)
        # rescwt = cwt(data)  # coefs 63*24 complex128 x+yj
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # --------------- cwt
    #        for i in range(len(rescwt[x+"_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x+"_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x+"_coefs_"+str(i),value=coefs,type='target')
    #            data.set_column(column=x + "_phase_cwt_"+str(i), value=phase, type='target')

    for data in tsdatasets_val:
        resfft = fft(data)
        # rescwt = cwt(data)
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # ----------------- cwt
    #        for i in range(len(rescwt[x + "_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x + "_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x + "_coefs_" + str(i), value=coefs, type='target')
    #            data.set_column(column=x + "_phase_cwt_" + str(i), value=phase, type='target')

    for data in tsdatasets_test:
        resfft = fft(data)
        # rescwt = cwt(data)
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
            # ----------------- cwt
    #        for i in range(len(rescwt[x + "_coefs"])):
    #            coefs = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            phase = [n for n in range(len(rescwt[x + "_coefs"][-1]))]
    #            for j in range(len(rescwt[x + "_coefs"][i])):
    #                phase[j] = cmath.phase(rescwt[x + "_coefs"][i][j])
    #                coefs[j] = abs(rescwt[x + "_coefs"][i][j])
    #            coefs = Series(coefs)
    #            phase = Series(phase)
    #            coefs.index = data[x].index
    #            phase.index = data[x].index
    #            data.set_column(column=x + "_coefs_" + str(i), value=coefs, type='target')
    #            data.set_column(column=x + "_phase_cwt_" + str(i), value=phase, type='target')

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('3 extract ftr:', formatted_time)

    y_train = []
    y_val = []
    y_test = []
    y_train_customerid = []
    y_val_customerid = []
    y_test_customerid = []
    for dataset in tsdatasets_train:
        y_train.append(dataset.static_cov['Y'])
        y_train_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_train = np.array(y_train)
    for dataset in tsdatasets_val:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_val = np.array(y_val)
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
    tsdatasets_val = min_max_scaler.fit_transform(tsdatasets_val)
    tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)


    network = InceptionTimeClassifier(max_epochs=epochs, patience=patiences, kernel_size=kernelsize)
    network.fit(tsdatasets_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score
    # network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
    # network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network.save('./model/0808_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y22_m10_v1.itc')
    #network.save('./model/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    #network.save('./model/0815_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_v1.itc')
    network.save('./model/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'
                 +ftr_num_str+'_t' + str(n_line_tail) + '_fl.itc')

    ############################### val
    filename = './result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_val.csv'
    pred_val = network.predict(tsdatasets_val)
    pred_val_prob = network.predict_proba(tsdatasets_val)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_val
    df['customerid'] = y_val_customerid
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    preds_prob = pred_val_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(filename, index=False)

    print(metrics.accuracy_score(y_val, pred_val, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_val, pred_val, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_val, pred_val, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_val, pred_val_prob, pos_label=1, )  # drop_intermediate=True
    print('val_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        #if tpr[i] > 0.5:
        #    print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(val)')
    plt.legend(loc="lower right")
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_roc_val.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_ks_val.png')
    plt.show()

    ############################### test
    filename = './result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_test.csv'
    pred_test = network.predict(tsdatasets_test)
    pred_test_prob = network.predict_proba(tsdatasets_test)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_test
    df['customerid'] = y_test_customerid
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    preds_prob = pred_test_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(filename, index=False)

    print(metrics.accuracy_score(y_test, pred_test, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_test, pred_test, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_test, pred_test, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test_prob, pos_label=1, )  # drop_intermediate=True
    print('test_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        #if tpr[i] > 0.5:
        #    print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
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
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_roc_test.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_ks_test.png')
    plt.show()

    ################################# psi
    ytr_prob_psi = pred_val_prob
    yte_prob_psi = pred_test_prob

    # 计算 psi
    num = 10
    part = 1 / (num + 0.0)
    intervals_1 = {'{}-{}'.format(part * x, part * (x + 1)): 0 for x in range(num)}
    for pp in list(ytr_prob_psi):
        for interval in intervals_1:
            start, end = tuple(interval.split('-'))
            if float(start) <= pp <= float(end):
                intervals_1[interval] += 1
                break
    intervals_2 = {'{}-{}'.format(part * x, part * (x + 1)): 0 for x in range(num)}
    for pp in yte_prob_psi:
        for interval in intervals_2:
            start, end = tuple(interval.split('-'))
            if float(start) <= pp <= float(end):
                intervals_2[interval] += 1
                break
    print(intervals_1)
    print(intervals_2)
    psi_list = []
    len1 = len(ytr_prob_psi)
    len2 = len(yte_prob_psi)
    for interval in intervals_1:
        pct_1 = intervals_1[interval] / (len1 + 0.0)
        pct_2 = intervals_2[interval] / (len2 + 0.0)
        psi_list.append((pct_2 - pct_1) * math.log((pct_2 + 0.0001) / (pct_1 + 0.0001), math.e))
    print(psi_list)
    print('psi = ', sum(psi_list))

    ############################### train
    pred_train = network.predict(tsdatasets_train)
    pred_train_prob = network.predict_proba(tsdatasets_train)[:, 1]

    print(metrics.accuracy_score(y_train, pred_train, normalize=True, sample_weight=None))

    cfm = metrics.confusion_matrix(y_train, pred_train, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_train, pred_train, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_train_prob, pos_label=1, )  # drop_intermediate=True
    print('train_ks = ', max(tpr - fpr))
    for i in range(tpr.shape[0]):
        print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        #if tpr[i] > 0.5:
        #    print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
            # break
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(train)')
    plt.legend(loc="lower right")
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_roc_train.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'+ftr_num_str+'_t'+str(n_line_tail)+'_fl_ks_train.png')
    plt.show()


if __name__ == '__main__':
    #train_treat_for_report()
    train_treat_for_predict()