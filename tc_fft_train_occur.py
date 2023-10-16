# coding=utf-8
import sys, os, time
import cmath
import math
import pickle
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
from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier
from paddlets.datasets.repository import get_dataset

warnings.filterwarnings('ignore', category=DeprecationWarning)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# "BLAS : Program is Terminated. Because you tried to allocate too many memory regions."
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"

def rmse(y_test, y):
    return math.sqrt(sum((y_test - y) ** 2) / len(y))


def train_occur_for_report():
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, sep=',', encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, sep=',', encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, sep=',', encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, sep=',', encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, sep=',', encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, sep=',', encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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

    n_line_tail = 60  # (1-5) * 30
    n_line_back = 7  # back 30/7
    n_line_head = 60  # = tail
    train_0_sample = 35000  # 30`60`90 -> 35000 120 150 ->20000
    type = 'occur'
    date_str = datetime(2023, 9, 1).strftime("%Y%m%d")

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test  #3272 24
    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(
        lambda x: max(x["RDATE"]) < 20230101)  # for train valid  #43815 315
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: min(x["RDATE"]) >= 20160101)  # for train valid
    del df_all
    ###################### for test 1:100
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part1[df_part1['Y'] == 1]  # 24
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part1_1.shape:', df_part1_1.shape)
    # 从 0 中 筛选出 2400 个
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=2400, random_state=2400)
    # 获取每个选中组的所有样本
    df_part1_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    df_part1 = pd.concat([df_part1_0_selected, df_part1_1])
    print('df_part1.shape: ', df_part1.shape)
    del df_part1_0, df_part1_1, df_part1_0_selected
    ###################### for train/valid  1:100
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    # 随机选择若干个组
    selected_groups = df_part2['CUSTOMER_ID'].drop_duplicates().sample(n=3500, random_state=3500)  # 3472 28
    # 获取每个选中组的所有样本
    df_val = df_part2.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_val = df_val.dropna(subset=['Y'])
    df_val_1 = df_val[df_val['Y'] == 1]
    df_val_1 = df_val_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_val_1.shape:', df_val_1.shape)
    df_val_0_all = df_val[df_val['Y'] == 0]
    selected_groups_0 = df_val_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=2800, random_state=2800)
    print('selected_groups_0  length:', len(selected_groups_0))
    df_val_0 = df_val_0_all.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_val_0 = df_val_0.dropna(subset=['Y'])
    print('df_val_0.shape:', df_val_0.shape)
    df_val = pd.concat([df_val_0, df_val_1])
    print('df_val.shape: ', df_val.shape)
    del df_val_0, df_val_1, df_val_0_all

    # 获取剩余的组
    df_train = df_part2[~df_part2['CUSTOMER_ID'].isin(selected_groups)]
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)

    df_train_1 = df_train[df_train['Y'] == 1]  # 287
    df_train_1 = df_train_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_train_1.shape:', df_train_1.shape)
    df_train_0_all = df_train[df_train['Y'] == 0]
    selected_groups_0 = df_train_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_sample,
                                                                               random_state=train_0_sample)  # 3w 1w 5k -> mem leak
    df_train_0 = df_train_0_all.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_train_0 = df_train_0.dropna(subset=['Y'])
    df_train = pd.concat([df_train_0, df_train_1])
    print('df_train.shape: ', df_train.shape)
    del df_train_0, df_train_1, df_train_0_all

    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)

    df_test = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_test.shape: ', df_test.shape)
    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)  # .\
    # reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    # .filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

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

    network = InceptionTimeClassifier(max_epochs=50, patience=20, kernel_size=16)
    network.fit(tsdatasets_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score
    # network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
    # network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network.save('./model/0808_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y22_m10_v1.itc')
    # network.save('./model/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    # network.save('./model/0815_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_v1.itc')
    network.save('./model/' + date_str + '_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(
        n_line_tail) + '_y16_m01_y23_m01_fl_' + str(train_0_sample) + '_' + str(n_line_back) + '.itc')

    ############################### val
    filename = './result/' + date_str + '_' + type + '_M6_y16_m1_y23_m1_result_val_' + str(n_line_tail) + '_fl_' + str(
        train_0_sample) + '_' + str(n_line_back) + '.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_ROC_val_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_KS_val_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()

    ############################### test
    filename = './result/' + date_str + '_' + type + '_M6_y16_m1_y23_m1_result_test_' + str(n_line_tail) + '_fl_' + str(
        train_0_sample) + '_' + str(n_line_back) + '.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_ROC_train_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_KS_train_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()


def train_occur_for_predict():
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, sep=',', encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, sep=',', encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, sep=',', encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, sep=',', encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, sep=',', encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, sep=',', encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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

    n_line_tail = 120  # (1-5) * 30
    n_line_back = 7  # back 30/7
    n_line_head = 120  # = tail
    train_0_sample = 25000  # 30`60`90 -> 35000 120 150 ->20000     33900   25000
    type = 'occur'
    date_str = datetime(2023, 9, 6).strftime("%Y%m%d")

    # df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test  #3272 24
    # df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train valid  #43815 315
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: min(x["RDATE"]) >= 20160101)  # for train valid
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train 1:100
    df_all_0 = df_all[df_all['Y'] == 0]
    df_all_1 = df_all[df_all['Y'] == 1]  # 24 + 315
    df_all_1 = df_all_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_all_1.shape:', df_all_1.shape)
    # 从 0 中 筛选出 33900 个
    selected_groups = df_all_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_sample,
                                                                       random_state=train_0_sample - n_line_head)  ## change each
    # 获取每个选中组的所有样本
    df_all_0_selected = df_all_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_all_0_selected = df_all_0_selected.dropna(subset=['Y'])
    df_all_0_selected = df_all_0_selected.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_all_0_selected.shape:', df_all_0_selected.shape)
    df_train = pd.concat([df_all_0_selected, df_all_1])
    print('df_train.shape: ', df_train.shape)
    del df_all, df_all_0, df_all_0_selected, df_all_1

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
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

    from sklearn.metrics import accuracy_score, f1_score
    # network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
    # network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network.save('./model/0808_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y22_m10_v1.itc')
    # network.save('./model/0815_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_v1.itc')
    network.save(
        './model/' + date_str + '_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_fl.itc')


# 定义判断全为 0 或为空值的函数
def all_zero_or_empty(x):
    return (x == 0).all() or x.isnull().all()


# 定义函数用于去除每个组的最后一行
def remove_last_row(group):
    return group.iloc[:-1]


def clean_data_train_occur_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7',
               'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30',
               'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365',
               'INV_AVG_365',
               'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180',
               'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
               'GRP_CNT',
               'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40 cols
    usecolss = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
                'UAR_AVG_7',
                'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
                'UAR_CHA_15', 'UAR_CHA_30',
                'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 20 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    print(df_16_18.shape)
    print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365', 'UAR_CHA_15',
           'UAR_CHA_30',
           'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365', 'INV_AVG_365',
           'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
           'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
           'GRP_CNT',
           'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
            'UAR_AVG_7',
            'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
            'UAR_CHA_15', 'UAR_CHA_30',
            'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 7  # back 30/7
    n_line_head = 30  # = tail
    train_0_sample = 35000  # 30`60`90 -> 35000 120 150 ->20000
    type = 'occur'
    date_str = datetime(2023, 9, 11).strftime("%Y%m%d")

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test  #3272 24
    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(
        lambda x: max(x["RDATE"]) < 20230101)  # for train valid  #43815 315
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: min(x["RDATE"]) >= 20160101)  # for train valid

    #    selected_groups = df_all['CUSTOMER_ID'].drop_duplicates().sample(n=1000, random_state=1000)

    # 获取每个选中组的所有样本
    #    df_part1_0_selected = df_all.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    #    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    #    del df_all

    #    df_part1 = df_part1_0_selected.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
    #            reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail+n_line_back). \
    #            reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)

    #    zero_or_empty_columns = df_part1.groupby('CUSTOMER_ID').apply(lambda x: x.loc[:, ~x.columns.isin(['CUSTOMER_ID'])].apply(all_zero_or_empty))
    #    print(zero_or_empty_columns)
    #    true_counts = zero_or_empty_columns.sum().sort_values(ascending=False)
    #    print(true_counts)
    #    exit(0)
    del df_all
    ###################### for test 1:100
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part1[df_part1['Y'] == 1]  # 24
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part1_1.shape:', df_part1_1.shape)
    # 从 0 中 筛选出 2400 个
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=2400, random_state=2400)
    # 获取每个选中组的所有样本
    df_part1_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    df_part1 = pd.concat([df_part1_0_selected, df_part1_1])
    print('df_part1.shape: ', df_part1.shape)
    del df_part1_0, df_part1_1, df_part1_0_selected
    ###################### for train/valid  1:100
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    # 随机选择若干个组
    selected_groups = df_part2['CUSTOMER_ID'].drop_duplicates().sample(n=3500, random_state=3500)  # 3472 28
    # 获取每个选中组的所有样本
    df_val = df_part2.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_val = df_val.dropna(subset=['Y'])
    df_val_1 = df_val[df_val['Y'] == 1]
    df_val_1 = df_val_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_val_1.shape:', df_val_1.shape)
    df_val_0_all = df_val[df_val['Y'] == 0]
    selected_groups_0 = df_val_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=2800, random_state=2800)
    print('selected_groups_0  length:', len(selected_groups_0))
    df_val_0 = df_val_0_all.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_val_0 = df_val_0.dropna(subset=['Y'])
    print('df_val_0.shape:', df_val_0.shape)
    df_val = pd.concat([df_val_0, df_val_1])
    print('df_val.shape: ', df_val.shape)
    del df_val_0, df_val_1, df_val_0_all

    # 获取剩余的组
    df_train = df_part2[~df_part2['CUSTOMER_ID'].isin(selected_groups)]
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)

    df_train_1 = df_train[df_train['Y'] == 1]  # 287
    df_train_1 = df_train_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_train_1.shape:', df_train_1.shape)
    df_train_0_all = df_train[df_train['Y'] == 0]
    selected_groups_0 = df_train_0_all['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_sample,
                                                                               random_state=train_0_sample)  # 3w 1w 5k -> mem leak
    df_train_0 = df_train_0_all.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups_0.values else None).reset_index(drop=True)
    df_train_0 = df_train_0.dropna(subset=['Y'])
    df_train = pd.concat([df_train_0, df_train_1])
    print('df_train.shape: ', df_train.shape)
    del df_train_0, df_train_1, df_train_0_all

    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)

    df_test = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_test.shape: ', df_test.shape)
    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)  # .\
    # reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    # .filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

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

    network = InceptionTimeClassifier(max_epochs=10, patience=10, kernel_size=16)
    network.fit(tsdatasets_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score
    # network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
    # network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network.save('./model/0808_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y22_m10_v1.itc')
    # network.save('./model/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    # network.save('./model/0815_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_v1.itc')
    network.save('./model/' + date_str + '_' + type + '_10_10_16_244_fft_p_t_SS_t' + str(
        n_line_tail) + '_y16_m01_y23_m01_fl_' + str(train_0_sample) + '_' + str(n_line_back) + '.itc')

    ############################### val
    filename = './result/' + date_str + '_' + type + '_M6_y16_m1_y23_m1_result_val_' + str(n_line_tail) + '_fl_' + str(
        train_0_sample) + '_' + str(n_line_back) + '.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_ROC_val_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_KS_val_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()

    ############################### test
    filename = './result/' + date_str + '_' + type + '_M6_y16_m1_y23_m1_result_test_' + str(n_line_tail) + '_fl_' + str(
        train_0_sample) + '_' + str(n_line_back) + '.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
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
        # if tpr[i] > 0.5:
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
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_ROC_train_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig(
        "./result/" + date_str + "_" + type + "_y16_m1_y23_m1_y23_m7_KS_train_" + str(n_line_tail) + "_fl_" + str(
            train_0_sample) + "_" + str(n_line_back) + ".png")
    plt.show()


def clean_data_train_occur_continue_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 88 cols
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
              'UAR_AVG_7',
              'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
              'UAR_CHA_15', 'UAR_CHA_30',
              'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365',
              'INV_AVG_365',
              'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365',
              'STOCK_AGE_AVG_180',
              'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
              'GRP_CNT',
              'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40 cols
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
              'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
              'UAR_CHA_365',
              'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    # col = df_all.columns.tolist()
    # col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    cols = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
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
            'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180',
            'LSR_121_CHA_365',
            'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60',
            'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30',
            'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG',
            'STOCK_AGE_AVG_7',
            'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180',
            'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30',
            'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365',
            'JH_30_CNT', 'JH_60_CNT', 'JH_90_CNT', 'JH_180_CNT', 'JH_HEGE', 'JH_WANSHAN', 'JH_XIANYI', 'JH_XIANYI_R',
            'JH_WAIFANG', 'JH_WAIFANG_R', 'JH_YIDONGCL', 'JH_YIDONGCL_R', 'JH_CCC', 'JH_SC_R', 'JH_SALE_R', 'JH_ZT_R',
            'JH_WT_R', 'JH_XFEW_R', 'JH_CZ_R', 'JH_WGWF_R', 'JH_HGZ', 'JH_HGZ_R', 'JH_JTS', 'JH_3YCHK_R', 'JH_3SZYD_R',
            'JH_3HGZWF_R', 'JH_5YCHK_R', 'JH_5SZYD_R', 'JH_5HGZWF_R', 'JH_10YCHK_R', 'JH_10SZYD_R', 'JH_10HGZWF_R',
            'JH_3YCHK10_R', 'JH_3SZYD10_R', 'JH_3HGZWF10_R', 'JH_6YCHK_R', 'JH_6SZYD_R', 'JH_6HGZWF_R',
            'PES_30HUIDIZHI',
            'PES_30HCL', 'PES_30MAHCL', 'PES_30MAHTS', 'PES_30MIHTS', 'PES_30AVGHTS', 'PES_30AVGHCL', 'PES_30MAHCL_R',
            'PES_30CHUDIZHI', 'PES_30CCL', 'PES_30MACCL', 'PES_30AVGCCL', 'PES_30MACCL_R', 'GRP_CNT',
            'GRP_AVAILAMT_SUM',
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
            'AUDIT_10YCHKSZYDHGWF_R', 'AUDIT_1YCHKWGWF_IND', 'AUDIT_1YCHKPCT25_IND', 'EXT_12M_R']  # 244

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30']  # 88

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
            'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
            'UAR_CHA_15', 'UAR_CHA_30',
            'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365', 'INV_AVG_365',
            'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
            'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
            'GRP_CNT',
            'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
            'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
            'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail
    type = 'occur'
    date_str = datetime(2023, 9, 13).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '88'
    filter_num_ratio = 1 / 8
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220101)  # 7 8 9 10 11 12
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2 , good:bad 100:1
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part1_1.shape:', df_part1_1.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    print('df_part2_1.shape:', df_part2_1.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    # count_df = df_part2_1.groupby('CUSTOMER_ID').apply(lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    # filtered_groups = count_df[count_df.gt(K)].index
    # print(filtered_groups)
    # df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    # print('after filter df_part2_1.shape:', df_part2_1.shape)

    # test_0_num_sample = (int(df_part2_1.shape[0]/n_line_head*100) < 2000) ? 2000 : int(df_part2_1.shape[0] / n_line_head * 100)
    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100)
    print('test_0_num_sample:', train_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
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

    # network.save('./model/'+date_str+'_'+type+'_'+split_date_str+'_'+str(epochs)+'_'+str(patiences)+'_'+str(kernelsize)+'_ftr_'
    #             +ftr_num_str+'_t' + str(n_line_tail) + '_fl.itc')

    ############################### val
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_val.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_val.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_val.png')
    plt.show()

    ############################### test
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_test.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_test.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_test.png')
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_train.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_train.png')
    plt.show()


def analysis_error_sample():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    # df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',', encoding='gbk')

    # df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    # df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    # df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    df_22_23 = pd.concat([df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_22_23.shape)

    del df22_4, df23

    df_all = df_22_23
    print('df_all.shape:', df_all.shape)

    del df_22_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365', 'UAR_CHA_15',
           'UAR_CHA_30',
           'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365', 'INV_AVG_365',
           'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
           'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
           'GRP_CNT',
           'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    # 按照 CUSTOMER_ID 列的值进行分组
    grouped = df_part2_0.groupby(['CUSTOMER_ID'])
    group_F = grouped.get_group('SMCRWMQ530R_12')
    # ’YJ4107008_3‘SMCRWMQ2706_27,SMCRWSQ200501_2,SVWVW2212112_49 0->1
    print('SMCRWMQ530R_12:', group_F)

    # SMCRWSQ2313_1  一半以上都是 0


def augment_bad_data_train_occur_continue_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 90 cols
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
              'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
              'UAR_CHA_365',
              'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365',
              'SDV_REPAY_365',
              'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365',
              'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
              'GRP_REPAYCARS90_SUM',
              'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90',
              'LSR_91_AVG_180']  # 40 cols
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
              'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
              'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
              'UAR_CHA_7']  # 18 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    # col = df_all.columns.tolist()
    # col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    cols = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
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
            'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180',
            'LSR_121_CHA_365',
            'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60',
            'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30',
            'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG',
            'STOCK_AGE_AVG_7',
            'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180',
            'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30',
            'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365',
            'JH_30_CNT', 'JH_60_CNT', 'JH_90_CNT', 'JH_180_CNT', 'JH_HEGE', 'JH_WANSHAN', 'JH_XIANYI', 'JH_XIANYI_R',
            'JH_WAIFANG', 'JH_WAIFANG_R', 'JH_YIDONGCL', 'JH_YIDONGCL_R', 'JH_CCC', 'JH_SC_R', 'JH_SALE_R', 'JH_ZT_R',
            'JH_WT_R', 'JH_XFEW_R', 'JH_CZ_R', 'JH_WGWF_R', 'JH_HGZ', 'JH_HGZ_R', 'JH_JTS', 'JH_3YCHK_R', 'JH_3SZYD_R',
            'JH_3HGZWF_R', 'JH_5YCHK_R', 'JH_5SZYD_R', 'JH_5HGZWF_R', 'JH_10YCHK_R', 'JH_10SZYD_R', 'JH_10HGZWF_R',
            'JH_3YCHK10_R', 'JH_3SZYD10_R', 'JH_3HGZWF10_R', 'JH_6YCHK_R', 'JH_6SZYD_R', 'JH_6HGZWF_R',
            'PES_30HUIDIZHI',
            'PES_30HCL', 'PES_30MAHCL', 'PES_30MAHTS', 'PES_30MIHTS', 'PES_30AVGHTS', 'PES_30AVGHCL', 'PES_30MAHCL_R',
            'PES_30CHUDIZHI', 'PES_30CCL', 'PES_30MACCL', 'PES_30AVGCCL', 'PES_30MACCL_R', 'GRP_CNT',
            'GRP_AVAILAMT_SUM',
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
            'AUDIT_10YCHKSZYDHGWF_R', 'AUDIT_1YCHKWGWF_IND', 'AUDIT_1YCHKPCT25_IND', 'EXT_12M_R']  # 240

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30']  # 90

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
            'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
            'UAR_CHA_15',
            'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365',
            'INV_AVG_365',
            'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
            'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
            'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
            'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
            'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail
    type = 'occur_step1'
    step = 1
    date_str = datetime(2023, 9, 13).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '90'
    filter_num_ratio = 1 / 8
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220101)  # 7 8 9 10 11 12
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000  bad augment * 180
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)
    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part2_1.shape:', df_part2_1.shape)

    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part2_1.shape:', df_part2_1.shape)

    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100) if int(
        df_part2_1.shape[0] / n_line_head * 100) < 2400 else 2400
    print('test_0_num_sample:', test_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    print('test_0_num_sample:', test_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
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

    network.save('./model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_'
                 + ftr_num_str + '_t' + str(n_line_tail) + '_fl_aug.itc')

    ############################### val
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_val_aug.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_val_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_val_aug.png')
    plt.show()

    ############################### test
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_test_aug.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_test_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_test_aug.png')
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_train_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_train_aug.png')
    plt.show()


def ts2vec_test():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    # df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    # df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',', encoding='gbk')

    # df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    # df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    # df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    df_22_23 = pd.concat([df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_22_23.shape)

    del df22_4, df23

    df_all = df_22_23
    print('df_all.shape:', df_all.shape)

    del df_22_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365', 'UAR_CHA_15',
           'UAR_CHA_30',
           'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365', 'INV_AVG_365',
           'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
           'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
           'GRP_CNT',
           'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 17
    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    df_train = pd.concat([df_part2_0, df_part2_1])

    from paddlets import TSDataset
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
    tsdatasets_test = TSDataset.load_from_dataframe(
        df=df_part2_1,
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
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
    tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)

    ts2vec_params = {"segment_size": 30,
                     "repr_dims": 320,  # 320
                     "batch_size": 128,
                     # "sampling_stride": 200,
                     "max_epochs": 10,
                     "verbose": 1,
                     "hidden_dims": 256, }
    cost_params = {"segment_size": 30,
                   "repr_dims": 320,
                   "batch_size": 128,
                   # "sampling_stride": 200,
                   "max_epochs": 5,
                   "hidden_dims": 1024, }
    from paddlets.models.representation import ReprClassifier, ReprCluster
    from paddlets.models.representation import TS2Vec, CoST
    # model = ReprCluster(in_chunk_len=30,
    #                        out_chunk_len=24,
    #                        sampling_stride=1,
    #                        repr_model=TS2Vec,
    #                        repr_model_params=ts2vec_params)
    # model = ReprCluster(repr_model=CoST, repr_model_params=cost_params)
    model = ReprCluster(repr_model=TS2Vec, repr_model_params=ts2vec_params)
    model.fit(tsdatasets_train)
    y_pred = model.predict(tsdatasets_train)
    print(y_pred, y_pred.sum(), len(y_pred))

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_train
    df['customerid'] = y_train_customerid
    # df.to_csv(filename, index=False)
    # df = pd.read_csv(filename)
    preds_prob = y_pred.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    for i in range(4):
        print('==' * 16)
        print(df[df['prob'] == i])
    print('==' * 16)
    print(df[df['Y'] == 1])


from paddlets import TSDataset
from typing import List, Dict, Any, Callable, Optional, Tuple


def ts2vec_relabel():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
               'UAR_CHA_7']  # 17 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 17

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail
    type = 'occur_step1'
    step = 10
    date_str = datetime(2023, 9, 13).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '17'
    filter_num_ratio = 1 / 8

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220101)  # 7 8 9 10 11 12
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)

    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
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

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
    # tsdatasets_val = min_max_scaler.fit_transform(tsdatasets_val)
    # tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)

    ts2vec_params = {"segment_size": 30,
                     "repr_dims": 320,  # 320
                     "batch_size": 128,
                     # "sampling_stride": 200,
                     "max_epochs": 5,
                     "verbose": 1,
                     "hidden_dims": 64, }
    from paddlets.models.representation import ReprCluster
    from paddlets.models.representation import TS2Vec
    model = ReprCluster(repr_model=TS2Vec, repr_model_params=ts2vec_params)
    model.fit(tsdatasets_train)
    y_pred = model.predict(tsdatasets_train)
    print(y_pred, y_pred.sum(), len(y_pred))

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_train
    df['customerid'] = y_train_customerid
    # df.to_csv(filename, index=False)
    # df = pd.read_csv(filename)
    preds_prob = y_pred.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    for i in range(8):
        print(i, '==' * 16)
        df_tmp = df[df['prob'] == i]
        value_counts = df_tmp['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print('class:', i, '\t1:', count_1, '\t0:', count_0)
    print('==' * 16)
    # print(df[df['Y'] == 1])


def ts2vec_relabel(tsdatasets: List[TSDataset], y_labels: np.ndarray, y_cutomersid: np.ndarray, ):
    ts2vec_params = {"segment_size": 30,
                     "repr_dims": 320,  # 320
                     "batch_size": 128,
                     # "sampling_stride": 200,
                     "max_epochs": 5,
                     "verbose": 1,
                     "hidden_dims": 64, }
    from paddlets.models.representation import ReprCluster
    from paddlets.models.representation import TS2Vec
    model = ReprCluster(repr_model=TS2Vec, repr_model_params=ts2vec_params)
    model.fit(tsdatasets)
    model.save('./model/cluster/', 'repr-cluster-partial-20230919-6.pkl')
    # model.load('./model/cluster/','repr-cluster-partial-20230919-6.pkl')
    y_pred = model.predict(tsdatasets)
    print(y_pred, y_pred.sum(), len(y_pred))

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_labels
    df['customerid'] = y_cutomersid
    # df.to_csv(filename, index=False)
    # df = pd.read_csv(filename)
    preds_prob = y_pred.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    for i in range(6):
        print(i, '==' * 16)
        df_tmp = df[df['prob'] == i]
        value_counts = df_tmp['Y'].value_counts()
        # 获取值为1的频次
        count_1 = value_counts.get(1, 0)
        count_0 = value_counts.get(0, 0)
        print('class:', i, '\t1:', count_1, '\t0:', count_0)
        if (count_1 < 600):
            # 按条件修改 Value 列的值
            df.loc[(df['prob'] == i) & (df['Y'] == 1), 'Y'] = 1
            df.loc[(df['prob'] == i) & (df['Y'] == 0), 'Y'] = 0
            print('use class:', i, '\t1:', count_1, '\t0:', count_0)
        else:
            df.loc[df['prob'] == i, 'Y'] = 0
    y_labels[:] = np.array(df['Y'].tolist())


def ts2vec_cluster_datagroup_model(tsdatasets: List[TSDataset], y_labels: np.ndarray, y_cutomersid: np.ndarray,
                                   model_path: str, repr_cluster_file_name: str = "repr-cluster-partial.pkl",
                                   del_num:int = 200):
    ts2vec_params = {"segment_size": 30,
                     "repr_dims": 320,  # 320
                     "batch_size": 128,
                     # "sampling_stride": 200,
                     "max_epochs": 5,
                     "verbose": 1,
                     "hidden_dims": 64, }
    from paddlets.models.representation import ReprCluster
    from paddlets.models.representation import TS2Vec
    model = ReprCluster(repr_model=TS2Vec, repr_model_params=ts2vec_params)
    file_path = model_path + repr_cluster_file_name
    print(file_path)
    if not os.path.exists(file_path):
        model.fit(tsdatasets)
        model.save(model_path, repr_cluster_file_name)
    else:
        model = ReprCluster.load(model_path, repr_cluster_file_name)
    y_pred = model.predict(tsdatasets)
    print(y_pred, y_pred.sum(), len(y_pred))
    n_class = max(y_pred) + 1
    tsdataset_list = [[] for _ in range(n_class)]
    label_list = [[] for _ in range(n_class)]
    customersid_list = [[] for _ in range(n_class)]

    for y, data, label, id in zip(y_pred, tsdatasets, y_labels, y_cutomersid):
        tsdataset_list[y].append(data)
        label_list[y].append(label)
        customersid_list[y].append(id)

    i = 0
    while i < len(label_list):
        if len(label_list[i]) < del_num:
            print('warning del class ', i, ' less 100 elements:',len(label_list[i]),sum(label_list[i]))
            label_list.pop(i)
            tsdataset_list.pop(i)
            customersid_list.pop(i)
        else:
            i += 1

    print('length each class')
    for i in range(len(label_list)):
        print(i, '=' * 16)
        print('tsdataset length:',len(tsdataset_list[i]))
        print('label length:', len(label_list[i]), ' sum: ', sum(label_list[i]))
        print('customerid length:',len(customersid_list[i]))
        label_list[i][:] = np.array(label_list[i])
        customersid_list[i][:] = np.array(customersid_list[i])

    return tsdataset_list, label_list, customersid_list


def dl_model_forward_ks_roc(model_file_path: str, result_file_path: str, tsdatasets: List[TSDataset], y_labels: np.ndarray,
                         y_cutomersid: np.ndarray, ):
    network = PaddleBaseClassifier.load(model_file_path)
    #pred_val = network.predict(tsdatasets)
    pred_val_prob = network.predict_proba(tsdatasets)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_labels
    df['customerid'] = y_cutomersid
    df.to_csv(result_file_path, index=False)
    df = pd.read_csv(result_file_path)
    preds_prob = pred_val_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(result_file_path, index=False)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, pred_val_prob, pos_label=1, )  # drop_intermediate=True
    print('ks = ', max(tpr - fpr))
    #for i in range(tpr.shape[0]):
        #print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        # if tpr[i] > 0.5:
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig(roc_file_path)
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f' % max(tpr - fpr))
    plt.legend(loc="lower right")
    # plt.savefig(ks_file_path)
    plt.show()


def augment_bad_data_relabel_train_occur_continue_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
               'UAR_CHA_7']  # 17 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 17

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail
    type = 'occur_step10_relabel_less300_01'
    step = 10
    date_str = datetime(2023, 9, 18).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '17'
    filter_num_ratio = 1 / 8
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220101)  # 7 8 9 10 11 12
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000  bad augment * 180
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)
    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part2_1.shape:', df_part2_1.shape)

    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part2_1.shape:', df_part2_1.shape)

    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100) if int(
        df_part2_1.shape[0] / n_line_head * 100) < 2400 else 2400
    print('test_0_num_sample:', test_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    print('test_0_num_sample:', test_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
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

    print('y_train.sum: ', y_train.sum(), len(y_train))
    ts2vec_relabel(tsdatasets_train, y_train, y_train_customerid)
    print('after relabel y_train.sum: ', y_train.sum(), len(y_train))

    network = InceptionTimeClassifier(max_epochs=epochs, patience=patiences, kernel_size=kernelsize)
    network.fit(tsdatasets_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score

    network.save('./model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_'
                 + ftr_num_str + '_t' + str(n_line_tail) + '_fl_aug.itc')

    ############################### val
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_val_aug.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_val_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_val_aug.png')
    plt.show()

    ############################### test
    filename = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_test_aug.csv'
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_test_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_test_aug.png')
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
        # if tpr[i] > 0.5:
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
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_roc_train_aug.png')
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig('./result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(
        patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_ks_train_aug.png')
    plt.show()


def augment_bad_data_relabel_multiclass_train_occur_continue_for_report():
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
               'UAR_CHA_7']  # 17 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    del df_16_18, df_19_20, df_21_23
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 17

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail

    step = 5
    date_str = datetime(2023, 9, 21).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '17'
    filter_num_ratio = 1 / 8
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16
    cluster_model_path = './model/cluster_step'+str(step) + '/'
    cluster_model_file = date_str+'-repr-cluster-partial-train-6.pkl'
    cluster_less_train_num = 200
    cluster_less_val_num = 200
    cluster_less_test_num = 100
    type = 'occur_step'+str(step)+'_reclass_less' + str(cluster_less_train_num) +'_'+str(cluster_less_test_num)

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220101)  # 7 8 9 10 11 12
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000  bad augment * 180
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)
    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part2_1.shape:', df_part2_1.shape)

    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part2_1.shape:', df_part2_1.shape)

    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100) if int(
        df_part2_1.shape[0] / n_line_head * 100) < 2400 else 2400
    print('test_0_num_sample:', test_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    print('test_0_num_sample:', test_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
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
    y_train_customerid = np.array(y_train_customerid)
    for dataset in tsdatasets_val:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_val = np.array(y_val)
    y_val_customerid = np.array(y_val_customerid)
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)
    y_test_customerid = np.array(y_test_customerid)

    from paddlets.transform import StandardScaler
    min_max_scaler = StandardScaler()
    tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
    tsdatasets_val = min_max_scaler.fit_transform(tsdatasets_val)
    tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)

    tsdataset_list_train, label_list_train, customersid_list_train = ts2vec_cluster_datagroup_model(tsdatasets_train,
                                                                                                    y_train,
                                                                                                    y_train_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                 cluster_less_train_num)
    for i in range(len(label_list_train)):
        network = InceptionTimeClassifier(max_epochs=epochs, patience=patiences, kernel_size=kernelsize,seed=0)
        model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                          str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) +\
                          '_fl_aug_' + str(i) + '.itc'
        if not os.path.exists(model_file_path):
            network.fit(tsdataset_list_train[i], label_list_train[i])
            network.save(model_file_path)

    tsdataset_list_val, label_list_val, customersid_list_val = ts2vec_cluster_datagroup_model(tsdatasets_val,
                                                                                                    y_val,
                                                                                                    y_val_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                   cluster_less_val_num)
    for i in range(len(label_list_val)):
        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                                  str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_fl_aug_' + str(0) + '.itc'  # default 0
                j = 0
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) +\
                       '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_val_aug_'+str(j)+'_'+str(i)+'.csv'
            print(result_file_path)
            dl_model_forward_ks_roc(model_file_path,result_file_path,tsdataset_list_val[i],label_list_val[i],customersid_list_val[i])

    tsdataset_list_test, label_list_test, customersid_list_test = ts2vec_cluster_datagroup_model(tsdatasets_test,
                                                                                                    y_test,
                                                                                                    y_test_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                 cluster_less_test_num)
    for i in range(len(label_list_test)):
        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                                  str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_fl_aug_' + str(0) + '.itc'  # default 0
                j = 0
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) +\
                       '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_test_aug_'+str(j)+'_'+str(i)+'.csv'
            print(result_file_path)
            dl_model_forward_ks_roc(model_file_path,result_file_path,tsdataset_list_test[i],label_list_test[i],customersid_list_test[i])


def augment_bad_data_add_credit_relabel_multiclass_train_occur_continue_for_report():
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30',
               'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
               'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
               'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
               'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
               'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30','LSR_121_CHA_15',
               'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60']  # 128 cols 1/5
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 90 cols  1/8
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60','GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
               'UAR_CHA_7']  # 18 cols 1/8
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    usecols = ['CUSTOMER_ID', 'RDATE', 'ICA_30',]  # ICA_30,PCA_30,ZCA_30  'PCA_30', 'ZCA_30'
    df_credit = pd.read_csv("./data/0825_train/credit/202309221506.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)
    # merge credit
    df_all = pd.merge(df_all, df_credit, on=['CUSTOMER_ID', 'RDATE'], how='left')
    print('after merge df_all.shape:', df_all.shape)
    #df_all = df_all.astype(float)

    del df_16_18, df_19_20, df_21_23, df_credit
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30',
           'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
           'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
           'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
           'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
           'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30',
           'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60', 'ICA_30']  # 127 + 1
    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30','ICA_30']  # 90 + 1
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60','GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'ICA_30', ]  # 18  add ICA_30 PCA_30 ZCA_30 add 3 will be filtered

    df_all[col] = df_all[col].astype(float)

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail

    step = 5
    date_str = datetime(2023, 10, 10).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '91'
    filter_num_ratio = 1 / 8  # 1/5
    ftr_good_year_split = 2017
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16
    cluster_model_path = './model/cluster_step' + str(step) + '_credit1_90_'+str(ftr_good_year_split)+ '_'+date_str +'/'
    cluster_model_file = date_str + '-repr-cluster-partial-train-6.pkl'
    cluster_less_train_num = 200
    cluster_less_val_num = 200
    cluster_less_test_num = 100
    type = 'occur_'+str(ftr_good_year_split)+'_addcredit_step' + str(step) + '_reclass_less' + str(cluster_less_train_num) + '_' + str(cluster_less_test_num)

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20170101)  #
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000  bad augment * 180
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)
    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part2_1.shape:', df_part2_1.shape)

    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part2_1.shape:', df_part2_1.shape)

    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100) if int(
        df_part2_1.shape[0] / n_line_head * 100) < 2400 else 2400
    print('test_0_num_sample:', test_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    print('test_0_num_sample:', test_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 normal data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
    ######################
    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT
    tsdatasets_train = TSDataset.load_from_dataframe(
        df=df_train,
        group_id='CUSTOMER_ID',
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
    y_train_customerid = np.array(y_train_customerid)
    for dataset in tsdatasets_val:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_val = np.array(y_val)
    y_val_customerid = np.array(y_val_customerid)
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)
    y_test_customerid = np.array(y_test_customerid)

    from paddlets.transform import StandardScaler
    ss_scaler = StandardScaler()
    tsdatasets_train = ss_scaler.fit_transform(tsdatasets_train)
    tsdatasets_val = ss_scaler.fit_transform(tsdatasets_val)
    tsdatasets_test = ss_scaler.fit_transform(tsdatasets_test)

    tsdataset_list_train, label_list_train, customersid_list_train = ts2vec_cluster_datagroup_model(tsdatasets_train,
                                                                                                    y_train,
                                                                                                    y_train_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                    cluster_less_train_num)
    for i in range(len(label_list_train)):
        network = InceptionTimeClassifier(max_epochs=epochs, patience=patiences, kernel_size=kernelsize, seed=0)
        model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                          str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                          '_fl_aug_' + str(i) + '.itc'
        if not os.path.exists(model_file_path):
            network.fit(tsdataset_list_train[i], label_list_train[i])
            network.save(model_file_path)
    # get train dataset score for train ensemble model
    for i in range(len(label_list_train)):
        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                                  str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_fl_aug_' + str(0) + '.itc'  # default 0
                j = 0
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) + \
                               '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_train_aug_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            dl_model_forward_ks_roc(model_file_path, result_file_path, tsdataset_list_train[i], label_list_train[i], customersid_list_train[i])

    tsdataset_list_val, label_list_val, customersid_list_val = ts2vec_cluster_datagroup_model(tsdatasets_val,
                                                                                              y_val,
                                                                                              y_val_customerid,
                                                                                              cluster_model_path,
                                                                                              cluster_model_file,
                                                                                              cluster_less_val_num)
    for i in range(len(label_list_val)):
        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                                  str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(
                    n_line_tail) + \
                                  '_fl_aug_' + str(0) + '.itc'  # default 0
                j = 0
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(
                epochs) + '_' + str(patiences) + \
                               '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(
                n_line_tail) + '_fl_val_aug_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            dl_model_forward_ks_roc(model_file_path, result_file_path, tsdataset_list_val[i], label_list_val[i], customersid_list_val[i])

    tsdataset_list_test, label_list_test, customersid_list_test = ts2vec_cluster_datagroup_model(tsdatasets_test,
                                                                                                 y_test,
                                                                                                 y_test_customerid,
                                                                                                 cluster_model_path,
                                                                                                 cluster_model_file,
                                                                                                 cluster_less_test_num)
    for i in range(len(label_list_test)):
        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                                  str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(
                    n_line_tail) + \
                                  '_fl_aug_' + str(0) + '.itc'  # default 0
                j = 0
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(
                epochs) + '_' + str(patiences) + \
                               '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(
                n_line_tail) + '_fl_test_aug_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            dl_model_forward_ks_roc(model_file_path, result_file_path, tsdataset_list_test[i], label_list_test[i], customersid_list_test[i])

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from lightgbm import plot_importance
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
import joblib
import json

def tsfresh_test():
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30',
               'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
               'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
               'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
               'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
               'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30','LSR_121_CHA_15',
               'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60']  # 128 cols 1/5
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 90 cols  1/8
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60','GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
               'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18 cols
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',', encoding='gbk')
    #df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',', encoding='gbk')

    #df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_16_18 = pd.concat([df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    #df_22_23 = pd.concat([df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    #print(df_22_23.shape)

    #del df22_4, df23

    #df_all = df_22_23
    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)

    #del df_22_23
    del df_16_18,df_19_20, df_21_23
    del df18_1, df18_2, df18_3, df18_4, df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4,
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

    # col = df_all.columns.tolist()
    # col.remove('CUSTOMER_ID').remove('RDATE').remove('Y')
    cols = ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180',
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
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30',
           'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
           'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
           'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
           'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
           'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30',
           'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60',]  # 127 + 1
    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30',]  # 90
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90', 'UAR_AVG_7',
           'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365', 'UAR_CHA_15',
           'UAR_CHA_30',
           'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7', 'STOCK_AGE_AVG_365', 'SDV_REPAY_365', 'INV_AVG_365',
           'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365', 'LSR_91_AVG_365', 'STOCK_AGE_AVG_180',
           'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180', 'GRP_REPAYCARS90_SUM',
           'GRP_CNT',
           'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'STOCK_AGE_AVG_90', 'LSR_91_AVG_180']  # 40

    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7']  # 18

    n_line_tail = 30  # (1-5) * 30
    n_line_back = 1  # back 7
    n_line_head = 30  # = tail
    # fill nan with 0
    df_all.fillna(0, inplace=True)

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20180101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail + n_line_back). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    df_train = pd.concat([df_part2_0, df_part2_1])
    print('df_train.shape:', df_train.shape)


    extraction_settings = ComprehensiveFCParameters()
    col.append('CUSTOMER_ID')
    col.append('RDATE')

    # 按照 group 列进行分组
    grouped_data = df_train.groupby('CUSTOMER_ID')
    if(len(grouped_data) > 500):
        num_groups_per_data = 500
    else:
        num_groups_per_data = len(grouped_data) / 2
    # 每份数据包含的相同组数
    # 划分数据
    splitted_data = [group for _, group in grouped_data]
    split_indices = np.array_split(np.arange(len(splitted_data)), len(splitted_data) // num_groups_per_data)
    X = pd.DataFrame()
    # 打印划分结果
    for indices in split_indices:
        df_part = pd.concat([splitted_data[i] for i in indices])
        X_part = extract_features(df_part[col], column_id='CUSTOMER_ID', column_sort='RDATE',
                             default_fc_parameters=extraction_settings, impute_function=impute)  # chunksize=10,n_jobs=8,
        X = pd.concat([X, X_part])

    print(X.head(2))
    print(X.columns.tolist())
    impute(X)
    y = df_train.loc[:, ['CUSTOMER_ID','Y']].drop_duplicates().reset_index(drop=True)
    y_train = np.array(y['Y'])
    print(len(X),y.head(),len(y))
    # Tsfresh将对每一个特征进行假设检验，以检查它是否与给定的目标相关
    X_filtered = select_features(X, y_train,)  # chunksize=10,n_jobs=8,
    print(X_filtered.columns)
    select_cols = X_filtered.columns.tolist()
    merged = pd.merge(X_filtered, y, left_index=True, right_index=True)
    print(merged.head())
    # 第二个数值是有多少个特征(列)，第一个数值是有多少行
    print("原始数据：", len(df_train[col]), len(df_train[col].columns))
    print("特征提取之后：", len(X), len(X.columns))
    print("特征选择之后:", len(X_filtered), len(X_filtered.columns))
    print("特征选择之后,合并customerid，y列之后:", len(merged), len(merged.columns))

    if 0:
        ts_x_trains = TSDataset.load_from_dataframe(
            df=merged,
            group_id='CUSTOMER_ID',
            target_cols=select_cols,
            fill_missing_dates=True,
            fillna_method="zero",
            static_cov_cols=['Y', 'CUSTOMER_ID'],
        )

        network = CNNClassifier(max_epochs=100, patience=50)
        network.fit(ts_x_trains, y_train)
        from sklearn.metrics import accuracy_score, f1_score
        preds = network.predict(ts_x_trains)
        score = accuracy_score(y_train, preds)
        f1 = f1_score(y_train, preds, average="macro")
        preds = network.predict_proba(ts_x_trains)
        print('network.predict_proba')
        print(preds)
        network.save('./model/cnn_test')
        load_network = PaddleBaseClassifier.load('./model/cnn_test')
        preds = load_network.predict_proba(ts_x_trains)
        print('load_network.predict_proba')
        print(preds)

    if 1:
        X_full_train, X_full_test, y_train, y_test = train_test_split(X, y_train, test_size=.4)
        # 进行特征选择（也可以直接使用特征选择后的数据而不用到这里再选择）
        X_filtered_train, X_filtered_test = X_full_train[X_filtered.columns], X_full_test[X_filtered.columns]
        lc = LGBMClassifier(max_depth=2, num_leaves=3, n_estimators=50, reg_lambda=1, reg_alpha=1,
                                objective='binary', seed=3)
        # lr = lgb.LGBMClassifier(objective='binary')
        # 决策树&随机森林
        # lr = tree.DecisionTreeClassifier(criterion="entropy", min_impurity_decrease=0.000001, class_weight={0:0.3, 1:0.7})
        # lr = RandomForestClassifier(n_estimators=100, criterion="entropy", min_impurity_decrease=0.00005, class_weight={0:0.2, 1:0.8})

        model = lc.fit(X_filtered_train, y_train)
        # 保存
        joblib.dump(model, "lgbm_model.pkl")
        # 显示重要特征
        plot_importance(model,max_num_features=10,xlim=(0,20))
        plt.show()
        importance = model.booster_.feature_importance(importance_type='gain')
        ftr_name = model.booster_.feature_name()
        print('len importance:',len(importance))
        importance_dict = {}
        x_list = []
        for x_n, im in zip(list(ftr_name), importance):
            importance_dict[x_n] = im
        target = 'tsfresh'
        if importance_dict:
            with open('%s_importance.json' % target, 'w', encoding='utf-8') as ff:
                json.dump(importance_dict, ff, ensure_ascii=False)
            im_df = pd.DataFrame([[x[0], x[1]] for x in importance_dict.items()])
            im_df.to_csv('%s_importance.csv' % target, index=False)
        # importances = lr.feature_importances_
        # feat_labels = ['x1','x2','x3','x4','x5','x6','x7','x8','x9']
        # indices = np.argsort(importances)[::-1]
        # for f in range(x_train.shape[1]):
        #     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

        # 模型效果获取
        # print('系数为：', lr.coef_)
        # print('截距为：', lr.intercept_)

        # 预测
        #y_predict = lr.predict(X_filtered_test)  # 预测
        y_prob = lc.predict_proba(X_filtered_test)[:, 1]
        for i in range(len(y_prob)):
            print(y_test[i],y_prob[i])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob, pos_label=1)
        print('test_ks = ',max(tpr - fpr))

        for i in range(tpr.shape[0]):
            if tpr[i] > 0.5:
                print(tpr[i], 1 - fpr[i], thresholds[i])
                break
        # test
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example(test)')
        plt.legend(loc="lower right")
        # plt.savefig("ROC（test）.png")
        plt.show()
        plt.plot(tpr, lw=2, label='tpr')
        plt.plot(fpr, lw=2, label='fpr')
        plt.plot(tpr - fpr, label='ks')
        plt.title('KS = %0.2f(test)' % max(tpr - fpr))
        plt.legend(loc="lower right")
        # plt.savefig("KS（test）.png")
        plt.show()
        # 加载
        my_model = joblib.load("lgbm_model.pkl")
        y_prob = my_model.predict_proba(X_filtered_test)[:, 1]
        for i in range(len(y_prob)):
            print(y_test[i], y_prob[i])

# origin_cols , include customer,rdate,y,ftr
def tsfresh_ftr_augment_select(df: pd.DataFrame,origin_cols:List[str],select_cols:List[str]):
    extraction_settings = ComprehensiveFCParameters()
    print('head df :',df[origin_cols].head())
    data_cols = origin_cols[:]
    data_cols.remove('Y')
    df.fillna(0, inplace=True)
    grouped_data = df.groupby('CUSTOMER_ID')
    if (len(grouped_data) > 500):
        num_groups_per_data = 500
    else:
        num_groups_per_data = len(grouped_data) / 2
    splitted_data = [group for _, group in grouped_data]
    split_indices = np.array_split(np.arange(len(splitted_data)), len(splitted_data) // num_groups_per_data) # num < 500 -> 0
    X = pd.DataFrame()
    for indices in split_indices:
        df_part = pd.concat([splitted_data[i] for i in indices])
        X_part = extract_features(df_part[data_cols], column_id='CUSTOMER_ID', column_sort='RDATE', chunksize=10,
                                  default_fc_parameters=extraction_settings, impute_function=impute)  # chunksize=10,n_jobs=8,
        X = pd.concat([X, X_part])
    impute(X)
    print('head X:',X.iloc[:2, :5])
    print('columns X:',X.columns,'\n length X:',len(X))
    #print(df.loc[:, ['CUSTOMER_ID','Y']].head(31))
    y = df.loc[:, ['CUSTOMER_ID','Y']].drop_duplicates().reset_index(drop=True)
    #print(y.head(2))
    #y = np.array(y['Y'])
    print('y:',y.iloc[:2,:],len(y))
    # Tsfresh将对每一个特征进行假设检验，以检查它是否与给定的目标相关
    if len(select_cols) == 0:
        print('train: select_cols is empty')
        X_filtered = select_features(X, np.array(y['Y']), chunksize=10,) # n_jobs=8,
        select_cols[:] = X_filtered.columns.tolist().copy()
    else:
        print('val & test: select_cols directly because it is not empty')
        X_filtered = X.loc[:,select_cols]
    print('select cols:',select_cols[:4])
    print('head X_filtered:', X_filtered.iloc[:2, :5])
    X_filtered.reset_index(inplace=True)
    X_filtered.rename(columns={'index': 'CUSTOMER_ID'}, inplace=True)
    print('head X_filtered after  rename:', X_filtered.iloc[:2, :5])
    merged = pd.merge(X_filtered, y, on=['CUSTOMER_ID'])
    #merged = pd.concat([X_filtered, y], axis=1) # 按列进行连接
    print('merge head:',merged.iloc[:1, :5])
    print('merge tail:', merged.iloc[:1, -5:])
    # 第二个数值是有多少个特征(列)，第一个数值是有多少行
    print("原始数据：", len(df[data_cols]), len(df[data_cols].columns))
    print("特征提取之后：", len(X), len(X.columns))
    print("特征选择之后:", len(X_filtered), len(X_filtered.columns))
    print("特征选择之后,合并 CUSTOMER_ID，Y 列之后:", len(merged), len(merged.columns))
    return  merged

def ml_model_forward_ks_roc(model_file_path: str, result_file_path: str, datasets: pd.DataFrame, y_labels: np.ndarray,
                         y_cutomersid: np.ndarray, ):
    model = joblib.load(model_file_path)
    #pred_val = network.predict(tsdatasets)
    pred_val_prob = model.predict_proba(datasets)[:, 1]

    ## prob to csv
    df = pd.DataFrame()
    df['Y'] = y_labels
    df['customerid'] = y_cutomersid
    df.to_csv(result_file_path, index=False)
    df = pd.read_csv(result_file_path)
    preds_prob = pred_val_prob.tolist()
    new_data = {"prob": preds_prob, }
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(result_file_path, index=False)
    sorted_df = df.sort_values(by='prob', ascending=False)
    print(sorted_df.head(40))

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, pred_val_prob, pos_label=1, )  # drop_intermediate=True
    print('ks = ', max(tpr - fpr))
    #for i in range(tpr.shape[0]):
        #print(tpr[i], fpr[i], tpr[i] - fpr[i], thresholds[i])
        # if tpr[i] > 0.5:
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig(roc_file_path)
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f' % max(tpr - fpr))
    plt.legend(loc="lower right")
    # plt.savefig(ks_file_path)
    plt.show()

def augment_bad_data_add_credit_relabel_multiclass_augment_ftr_select_train_occur_continue_for_report():
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30',
               'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
               'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
               'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
               'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
               'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30','LSR_121_CHA_15',
               'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60']  # 128 cols 1/5
    usecols = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
               'STOCK_AGE_AVG_365',
               'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
               'LSR_91_AVG_365',
               'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
               'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
               'STOCK_AGE_AVG_90',
               'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
               'FREESPANRP_180D_R', 'SDV_REPAY_60',
               'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
               'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
               'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
               'STOCK_AGE_CHA_RATIO_180',
               'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
               'LSR_91_AVG_60',
               'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
               'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
               'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
               'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
               'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 90 cols  1/8
    usecol = ['CUSTOMER_ID', 'Y', 'RDATE', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
               'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60','GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
               'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180',
               'UAR_CHA_7']  # 18 cols 1/8
    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, usecols=usecols, sep=',',
                       encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_4 = pd.read_csv("./data/0825_train/occur/2021_10_12_202308250937.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_3 = pd.read_csv("./data/0825_train/occur/2021_7_10_202308251006.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_2 = pd.read_csv("./data/0825_train/occur/2021_4_7_202308251012.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df21_1 = pd.read_csv("./data/0825_train/occur/2021_1_4_202308251017.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_4 = pd.read_csv("./data/0825_train/occur/2020_10_12_202308251023.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_3 = pd.read_csv("./data/0825_train/occur/2020_7_10_202308251033.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_2 = pd.read_csv("./data/0825_train/occur/2020_4_7_202308251037.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df20_1 = pd.read_csv("./data/0825_train/occur/2020_1_4_202308251042.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_4 = pd.read_csv("./data/0825_train/occur/2019_10_12_202308251047.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_3 = pd.read_csv("./data/0825_train/occur/2019_7_10_202308251052.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_2 = pd.read_csv("./data/0825_train/occur/2019_4_7_202308251057.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df19_1 = pd.read_csv("./data/0825_train/occur/2019_1_4_202308251238.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_4 = pd.read_csv("./data/0825_train/occur/2018_10_12_202308251253.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_3 = pd.read_csv("./data/0825_train/occur/2018_7_10_202308251257.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_2 = pd.read_csv("./data/0825_train/occur/2018_4_7_202308251301.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df18_1 = pd.read_csv("./data/0825_train/occur/2018_1_4_202308251306.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_4 = pd.read_csv("./data/0825_train/occur/2017_10_12_202308251310.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_3 = pd.read_csv("./data/0825_train/occur/2017_7_10_202308251313.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_2 = pd.read_csv("./data/0825_train/occur/2017_4_7_202308251316.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df17_1 = pd.read_csv("./data/0825_train/occur/2017_1_4_202308251320.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_2 = pd.read_csv("./data/0825_train/occur/2016_7_12_202308251325.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    df16_1 = pd.read_csv("./data/0825_train/occur/2016_1_7_202308251331.csv", header=0, usecols=usecols, sep=',',
                         encoding='gbk')
    credit_usecols = ['CUSTOMER_ID', 'RDATE', 'ICA_30',]  # ICA_30,PCA_30,ZCA_30  'PCA_30', 'ZCA_30'
    df_credit = pd.read_csv("./data/0825_train/credit/202309221506.csv", header=0, usecols=credit_usecols, sep=',', encoding='gbk')

    df_16_18 = pd.concat([df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4])
    df_19_20 = pd.concat([df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4])
    df_21_23 = pd.concat([df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23])
    # print(df_16_18.shape)
    # print(df_19_20.shape)
    print(df_21_23.shape)

    del df16_1, df16_2, df17_1, df17_2, df17_3, df17_4, df18_1, df18_2, df18_3, df18_4
    del df19_1, df19_2, df19_3, df19_4, df20_1, df20_2, df20_3, df20_4
    del df21_1, df21_2, df21_3, df21_4, df22_1, df22_2, df22_3, df22_4, df23

    df_all = pd.concat([df_16_18, df_19_20, df_21_23])
    # df_all = pd.concat([df_19_20, df_21_23])
    print('df_all.shape:', df_all.shape)
    # merge credit
    df_all = pd.merge(df_all, df_credit, on=['CUSTOMER_ID', 'RDATE'], how='left')
    print('after merge credit df_all.shape:', df_all.shape)
    #df_all = df_all.astype(float)

    del df_16_18, df_19_20, df_21_23, df_credit
    # del df_19_20, df_21_23

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30',
           'LRR_AVG_60','LSR_91_AVG_15','LRR_AVG_30','LSR_91_AVG_7','STOCK_OVER_91_RATIO',
           'LSR_121_AVG_90','FREESPANRP_30D_R','JH_60_CNT','LSR_91_CHA_30','LSR_91_CHA_7','LSR_91_CHA_15','LSR_91_CHA_60',
           'LSR_91_CHA_180','LRR_AVG_15','LSR_91_CHA_365','LSR_91_CHA_90','LRR_AVG_7','LSR_121_AVG_60','LRR_CHA_365','LRR_CHA_180',
           'LRR_CHA_30','LRR_CHA_60','LRR_CHA_7','LRR_CHA_90','LOAN_REPAY_RATIO','LRR_CHA_15','LSR_121_AVG_30','LSR_121_AVG_15',
           'LSR_121_AVG_7','STOCK_OVER_121_RATIO','LSR_121_CHA_180','LSR_121_CHA_90','LSR_121_CHA_30',
           'LSR_121_CHA_15','LSR_121_CHA_7','LSR_121_CHA_60', 'ICA_30']  # 127 + 1
    col = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO',
           'UAR_CHA_365', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'STOCK_AGE_AVG_365',
           'SDV_REPAY_365', 'INV_AVG_365', 'GRP_REPAYCARS180_SUM', 'JH_CCC', 'JH_HGZ', 'JH_JTS', 'LRR_AVG_365',
           'LSR_91_AVG_365',
           'STOCK_AGE_AVG_180', 'FREESPANRP_360D_R', 'SDV_REPAY_180', 'XSZQ180D_R', 'JH_SC_R', 'INV_AVG_180',
           'GRP_REPAYCARS90_SUM', 'GRP_CNT', 'JH_HGZ_R', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM',
           'STOCK_AGE_AVG_90',
           'LSR_91_AVG_180', 'STOCK_AGE_AVG_60', 'XSZQ90D_R', 'SDV_REPAY_90', 'INV_AVG_90', 'LSR_121_AVG_365',
           'FREESPANRP_180D_R', 'SDV_REPAY_60',
           'LRR_AVG_180', 'INV_AVG_60', 'STOCK_AGE_AVG_30', 'JH_180_CNT', 'INV_AVG_30', 'STOCK_AGE_AVG_15',
           'XSZQ30D_R', 'STOCK_AGE_AVG_7', 'SDV_REPAY_30',
           'LSR_91_AVG_90', 'STOCK_AGE_CHA_RATIO_7', 'INV_RATIO_90', 'STOCK_AGE_AVG', 'STOCK_AGE_CHA_RATIO_365',
           'STOCK_AGE_CHA_RATIO_180',
           'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_15',
           'LSR_91_AVG_60',
           'INV_AVG_15', 'JH_90_CNT', 'INV_AVG_7', 'SDV_REPAY_15', 'INV_RATIO', 'INV_CHA_15', 'INV_CHA_30',
           'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180',
           'INV_CHA_365', 'INV_CHA_7', 'LSR_121_AVG_180', 'FREESPANRP_90D_R', 'REPAY_STD_RATIO_7_180',
           'SDV_REPAY_7', 'REPAY_STD_RATIO_7_15',
           'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_365',
           'LRR_AVG_90', 'LSR_91_AVG_30','ICA_30']  # 90 + 1
    cols = ['XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'UAR_AVG_365', 'UAR_AVG_180', 'UAR_AVG_90',
           'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60','GRP_AVAILAMT_SUM', 'USEAMOUNT_RATIO', 'UAR_CHA_365',
           'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_7',
           'ICA_30']  # 18 + ICA_30

    df_all[col] = df_all[col].astype(float)

    n_line_tail = 30  # (1-5) * 30
    n_line_head = 30  # = tail

    step = 5
    date_str = datetime(2023, 10, 10).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '91'
    filter_num_ratio = 1 / 8
    ftr_good_year_split = 2017   #  quick start 2022, at last 2016/2017
    ########## model
    max_depth = 2 # 2 3 4 5
    num_leaves = 3 # 3 7 15 31
    n_estimators = 50 # 100
    class_weight =  None # 'balanced'  None
    cluster_model_path = './model/cluster_step' + str(step) + '_credit1_90_'+str(ftr_good_year_split)+ '_'+date_str +'/'
    cluster_model_file = date_str + '-repr-cluster-partial-train-6.pkl'
    cluster_less_train_num = 200
    cluster_less_val_num = 200
    cluster_less_test_num = 100
    type = 'occur_'+str(ftr_good_year_split)+'_addcredit_augmentftr_step' + str(step) + '_reclass_less' + str(cluster_less_train_num) + '_' + str(cluster_less_test_num)

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20170101)  # 20170101 20221201
    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train good

    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)  # for test

    df_part3 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20160101)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train bad
    del df_all

    df_part1 = df_part1.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    df_part3 = df_part3.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    ###################### for train valid 8:2  bad augment * 180
    df_part1_0 = df_part1[df_part1['Y'] == 0]
    df_part1_1 = df_part3[df_part3['Y'] == 1]
    df_part1_1 = df_part1_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)
    # 使用 groupby 方法按照 CUSTOMER_ID 列的值分组，并应用函数去除最后一行
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part1_1.shape:', df_part1_1.shape)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = i
            end_position = i + batch_size
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i + 1}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_part1_1 = df_part1_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_part1_1.head:', df_part1_1.head(32))
    print('df_part1_1.shape:', df_part1_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_1 = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_1.shape:', df_part1_1.shape)

    train_1_num_sample = int(df_part1_1.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_1['CUSTOMER_ID'].drop_duplicates().sample(n=train_1_num_sample, random_state=int(
        train_1_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_1_selected = df_part1_1.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_1_selected = train_1_selected.dropna(subset=['Y'])
    print('train_1_selected.shape:', train_1_selected.shape)
    # 获取剩余的组
    valid_1_selected = df_part1_1[~df_part1_1['CUSTOMER_ID'].isin(selected_groups)]
    print('valid_1_selected.shape:', valid_1_selected.shape)

    df_part1_0 = df_part1_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part1_0.shape:', df_part1_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part1_0 = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part1_0.shape:', df_part1_0.shape)

    # train_0_num_sample = train_1_num_sample * 100 if train_1_num_sample * 100 < df_part1_0.shape[0]/n_line_head else df_part1_0.shape[0]/n_line_head
    train_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.8)
    selected_groups = df_part1_0['CUSTOMER_ID'].drop_duplicates().sample(n=train_0_num_sample, random_state=int(
        train_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    train_0_selected = df_part1_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    train_0_selected = train_0_selected.dropna(subset=['Y'])
    print('train_0_selected.shape:', train_0_selected.shape)
    df_train = pd.concat([train_0_selected, train_1_selected])
    print('df_train.shape: ', df_train.shape)

    del train_0_selected, train_1_selected

    # valid_0_num_sample = int(valid_1_selected.shape[0] / n_line_head * 10)  # down to 10
    valid_0_num_sample = int(df_part1_0.shape[0] / n_line_head * 0.2)
    # 获取剩余的组
    valid_0_remain = df_part1_0[~df_part1_0['CUSTOMER_ID'].isin(selected_groups)]
    selected_groups = valid_0_remain['CUSTOMER_ID'].drop_duplicates().sample(n=valid_0_num_sample, random_state=int(
        valid_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    valid_0_selected = valid_0_remain.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    valid_0_selected = valid_0_selected.dropna(subset=['Y'])
    print('valid_0_selected.shape:', valid_0_selected.shape)
    df_val = pd.concat([valid_0_selected, valid_1_selected])

    del df_part1_0, df_part1_1, valid_0_remain, valid_0_selected, valid_1_selected

    ###################### for test good:bad 100:1, good >= 2000  bad augment * 180
    df_part2_0 = df_part2[df_part2['Y'] == 0]
    df_part2_1 = df_part2[df_part2['Y'] == 1]
    df_part2_1 = df_part2_1.groupby(['CUSTOMER_ID']).apply(
        lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)
    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(remove_last_row).reset_index(drop=True)
    print('after del last row df_part2_1.shape:', df_part2_1.shape)

    df_part2_1 = df_part2_1.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    print('df_part2_1.head:', df_part2_1.head(32))
    print('df_part2_1.shape:', df_part2_1.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_1.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_1 = df_part2_1[~df_part2_1['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_part2_1.shape:', df_part2_1.shape)

    test_0_num_sample = int(df_part2_1.shape[0] / n_line_head * 100) if int(
        df_part2_1.shape[0] / n_line_head * 100) < 2400 else 2400
    print('test_0_num_sample:', test_0_num_sample)

    df_part2_0 = df_part2_0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_part2_0.shape:', df_part2_0.shape)
    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和
    count_df = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_part2_0 = df_part2_0[~df_part2_0['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter df_part2_0.shape:', df_part2_0.shape)
    test_0_num_sample = test_0_num_sample if ((df_part2_0.shape[0] / n_line_head) > test_0_num_sample) else int(
        df_part2_0.shape[0] / n_line_head)
    print('test_0_num_sample:', test_0_num_sample)
    selected_groups = df_part2_0['CUSTOMER_ID'].drop_duplicates().sample(n=test_0_num_sample, random_state=int(
        test_0_num_sample + n_line_head))
    # 获取每个选中组的所有样本
    df_part2_0_selected = df_part2_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part2_0_selected = df_part2_0_selected.dropna(subset=['Y'])
    print('df_part2_0_selected.shape:', df_part2_0_selected.shape)
    df_test = pd.concat([df_part2_0_selected, df_part2_1])
    print('df_test.shape: ', df_test.shape)
    del df_part2_0, df_part2_1, df_part2_0_selected

    df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_test.shape: ', df_test.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_train.shape: ', df_train.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 load data:', formatted_time)
    ###################### del
    del df_part1, df_part2, df_part3
    ######################
    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT
    tsdatasets_train = TSDataset.load_from_dataframe(
        df=df_train,
        group_id='CUSTOMER_ID',
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

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('3 transform data:', formatted_time)

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
    y_train_customerid = np.array(y_train_customerid)
    for dataset in tsdatasets_val:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_val = np.array(y_val)
    y_val_customerid = np.array(y_val_customerid)
    for dataset in tsdatasets_test:
        y_test.append(dataset.static_cov['Y'])
        y_test_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_test = np.array(y_test)
    y_test_customerid = np.array(y_test_customerid)

    from paddlets.transform import StandardScaler
    ss_scaler = StandardScaler()
    tsdatasets_train = ss_scaler.fit_transform(tsdatasets_train)
    tsdatasets_val = ss_scaler.fit_transform(tsdatasets_val)
    tsdatasets_test = ss_scaler.fit_transform(tsdatasets_test)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('4 group data:', formatted_time)

    tsdataset_list_train, label_list_train, customersid_list_train = ts2vec_cluster_datagroup_model(tsdatasets_train,
                                                                                                    y_train,
                                                                                                    y_train_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                    cluster_less_train_num)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('5 classifier train:', formatted_time)

    for i in range(len(label_list_train)):
        select_cols = []
        model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                          str(num_leaves) + '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                          '_ftr_select_' + str(i) + '.pkl'
        if os.path.exists(model_file_path):
            print('{} already exists, so no more train.'.format(model_file_path))
            break
        df_train_part = df_train[df_train['CUSTOMER_ID'].isin(customersid_list_train[i])]
        df_train_ftr_select_notime = tsfresh_ftr_augment_select(df_train_part, usecols, select_cols)
        lc = LGBMClassifier(max_depth=max_depth, num_leaves=num_leaves, n_estimators=n_estimators, reg_lambda=1,
                            reg_alpha=1,objective='binary', class_weight=class_weight, seed=0)
        # lr = lgb.LGBMClassifier(objective='binary')
        # 决策树&随机森林
        # lr = tree.DecisionTreeClassifier(criterion="entropy", min_impurity_decrease=0.000001, class_weight={0:0.3, 1:0.7})
        # lr = RandomForestClassifier(n_estimators=100, criterion="entropy", min_impurity_decrease=0.00005, class_weight={0:0.2, 1:0.8})

        ftr_list_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + 'ftr_list_'+str(i) + '.pkl'
        model = lc.fit(df_train_ftr_select_notime.loc[:,select_cols], np.array(df_train_ftr_select_notime.loc[:,'Y']))
        joblib.dump(model, model_file_path)
        if not os.path.exists(ftr_list_file_path):
            with open(ftr_list_file_path, 'wb') as f:
                pickle.dump(select_cols, f)

    for i in range(len(label_list_train)):
        df_train_part = df_train[df_train['CUSTOMER_ID'].isin(customersid_list_train[i])]

        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                              str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_ftr_select_' + str(j) + '.pkl'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                                  str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_ftr_select_' + str(0) + '.pkl'  # default 0
                j = 0
            ftr_list_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + 'ftr_list_' + str(j) + '.pkl'
            print(ftr_list_file_path)
            with open(ftr_list_file_path, 'rb') as f:
                select_cols = pickle.load(f)
            print('len select cols:',len(select_cols))
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                               '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_ftr_select_train_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            if os.path.exists(result_file_path):
                print('{} already exists, so no more infer.'.format(result_file_path))
                break
            df_train_ftr_select_notime = tsfresh_ftr_augment_select(df_train_part, usecols, select_cols)
            ml_model_forward_ks_roc(model_file_path, result_file_path, df_train_ftr_select_notime.loc[:,select_cols], np.array(df_train_ftr_select_notime.loc[:,'Y']),
                                 np.array(df_train_ftr_select_notime.loc[:,'CUSTOMER_ID']))

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('6 group data:', formatted_time)

    tsdataset_list_val, label_list_val, customersid_list_val = ts2vec_cluster_datagroup_model(tsdatasets_val,
                                                                                              y_val,
                                                                                              y_val_customerid,
                                                                                              cluster_model_path,
                                                                                              cluster_model_file,
                                                                                              cluster_less_val_num)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('7 classifier test:', formatted_time)

    for i in range(len(label_list_val)):
        df_val_part = df_val[df_val['CUSTOMER_ID'].isin(customersid_list_val[i])]

        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                              str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_ftr_select_' + str(j) + '.pkl'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                                  str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_ftr_select_' + str(0) + '.pkl'  # default 0
                j = 0
            ftr_list_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + 'ftr_list_' + str(j) + '.pkl'
            print(ftr_list_file_path)
            with open(ftr_list_file_path, 'rb') as f:
                select_cols = pickle.load(f)
            print('len select cols:',len(select_cols))
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                               '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_ftr_select_val_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            if os.path.exists(result_file_path):
                print('{} already exists, so no more infer.'.format(result_file_path))
                break
            df_val_ftr_select_notime = tsfresh_ftr_augment_select(df_val_part, usecols, select_cols)
            ml_model_forward_ks_roc(model_file_path, result_file_path, df_val_ftr_select_notime.loc[:,select_cols], np.array(df_val_ftr_select_notime.loc[:,'Y']),
                                 np.array(df_val_ftr_select_notime.loc[:,'CUSTOMER_ID']))

    tsdataset_list_test, label_list_test, customersid_list_test = ts2vec_cluster_datagroup_model(tsdatasets_test,
                                                                                                 y_test,
                                                                                                 y_test_customerid,
                                                                                                 cluster_model_path,
                                                                                                 cluster_model_file,
                                                                                                 cluster_less_test_num)
    for i in range(len(label_list_test)):
        df_test_part = df_test[df_test['CUSTOMER_ID'].isin(customersid_list_test[i])]

        for j in range(len(label_list_train)):
            model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                              str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_ftr_select_' + str(j) + '.pkl'
            if not os.path.exists(model_file_path):
                model_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                                  str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_ftr_select_' + str(0) + '.pkl'  # default 0
                j = 0
            ftr_list_file_path = './model/' + date_str + '_' + type + '_' + split_date_str + '_' + 'ftr_list_' + str(j) + '.pkl'
            print(ftr_list_file_path)
            with open(ftr_list_file_path, 'rb') as f:
                select_cols = pickle.load(f)
            print('len select cols:', len(select_cols))
            result_file_path = './result/' + date_str + '_' + type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                               '_' + str(n_estimators)+'_' +str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_ftr_select_test_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            if os.path.exists(result_file_path):
                print('{} already exists, so no more infer.'.format(result_file_path))
                break
            df_test_ftr_select_notime = tsfresh_ftr_augment_select(df_test_part, usecols, select_cols)
            ml_model_forward_ks_roc(model_file_path, result_file_path, df_test_ftr_select_notime.loc[:,select_cols], np.array(df_test_ftr_select_notime.loc[:,'Y']),
                                 np.array(df_test_ftr_select_notime.loc[:,'CUSTOMER_ID']))

def ensemble_dl_ml_base_score_train(dl_result_file_path:str, ml_result_file_path:str, ensemble_model_file_path:str, lc_c:float=0.02):
    usecols = ['customerid', 'Y', 'prob', ]
    if not os.path.exists(dl_result_file_path):
        print('dl result file not exists:',dl_result_file_path)
        return
    df_train_dl = pd.read_csv(dl_result_file_path, header=0, usecols=usecols, sep=',',encoding='gbk')
    df_train_ml = pd.read_csv(ml_result_file_path, header=0, usecols=usecols, sep=',',encoding='gbk')
    #print(df_train_ml.head(2))
    #print(df_train_dl.head(2))
    df_train_dl.rename(columns={'prob': 'prob_dl'}, inplace=True)
    df_train_ml.rename(columns={'prob': 'prob_ml'}, inplace=True)
    #print(df_train_ml.head(2))
    #print(df_train_dl.head(2),len(df_train_dl))
    df_train = pd.merge(df_train_dl,df_train_ml,on=['customerid', 'Y'])
    #print(df_train.head(2),len(df_train))

    ########## model
    max_depth = 2
    num_leaves = 3
    n_estimators = 50
    class_weight = 'balanced'   # None  'balanced'
    select_cols = ['prob_dl','prob_ml']
    #lc = LGBMClassifier(max_depth=max_depth, num_leaves=num_leaves, n_estimators=n_estimators, reg_lambda=1,
    #                    reg_alpha=1, objective='binary', class_weight=class_weight, seed=0)
    lc = LogisticRegression(penalty="l2",C=lc_c,random_state=0) # 0.02 -> 1  0.2-> 0
    #lc = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10, random_state=42)
    print('train========ing')
    model = lc.fit(df_train.loc[:, select_cols], np.array(df_train.loc[:, 'Y']))
    joblib.dump(model, ensemble_model_file_path)
    print('train========done')

def ensemble_dl_ml_base_score_test(dl_result_file_path:str, ml_result_file_path:str, ensemble_model_file_path:str,ensemble_result_file_path:str):
    usecols = ['customerid', 'Y', 'prob', ]
    if not os.path.exists(dl_result_file_path):
        print('dl result file not exists:',dl_result_file_path)
        return
    df_train_dl = pd.read_csv(dl_result_file_path, header=0, usecols=usecols, sep=',',encoding='gbk')
    df_train_ml = pd.read_csv(ml_result_file_path, header=0, usecols=usecols, sep=',',encoding='gbk')
    #print(df_train_ml.head(2))
    #print(df_train_dl.head(2))
    df_train_dl.rename(columns={'prob': 'prob_dl'}, inplace=True)
    df_train_ml.rename(columns={'prob': 'prob_ml'}, inplace=True)
    #print(df_train_ml.head(2))
    #print(df_train_dl.head(2),len(df_train_dl))
    df_train = pd.merge(df_train_dl,df_train_ml,on=['customerid', 'Y'])
    #print(df_train.head(2),len(df_train))
    select_cols = ['prob_dl', 'prob_ml']
    ml_model_forward_ks_roc(ensemble_model_file_path,ensemble_result_file_path,df_train.loc[:,select_cols],np.array(df_train.loc[:,'Y']),np.array(df_train.loc[:,'customerid']))

def ensemble_data_augment_group_ts_dl_ftr_select_nts_ml_base_score():

    n_line_tail = 30  # (1-5) * 30
    step = 5
    date_str = datetime(2023, 10, 10).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '91'
    ftr_good_year_split = 2017
    ########## model
    epochs = 20
    patiences = 10  # 10
    kernelsize = 16
    max_depth = 3 # 2 3  -1
    num_leaves = 7 # 3 7  31
    n_estimators = 100 #  50 100 100
    class_weight =  'balanced' # None 'balanced' None
    lc_c = [0.2, 0.02, 0.02] # 0.02 -> 1  0.2-> 0
    cluster_less_train_num = 200
    cluster_less_test_num = 100

    dl_type = 'occur_' + str(ftr_good_year_split) + '_addcredit_step' + str(step) + '_reclass_less' + str(cluster_less_train_num) + '_' + str(
        cluster_less_test_num)
    ml_type = 'occur_' + str(ftr_good_year_split) + '_addcredit_augmentftr_step' + str(step) + '_reclass_less' + str(cluster_less_train_num) + '_' + str(
        cluster_less_test_num)
    ensemble_type = 'occur_ensemble'

    # model train
    for i in range(3):
        dl_result_file_path = './result/' + date_str + '_' + dl_type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) + \
                              '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_train_aug_' + str(i) + '_' + str(i) + '.csv'
        ml_result_file_path = './result/' + date_str + '_' + ml_type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                              '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_ftr_select_train_' + \
                              str(i) + '_' + str(i) + '.csv'
        ensemble_model_file_path = './model/' + date_str + '_' + ensemble_type + '_' +str(lc_c[i]) + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                          str(num_leaves) + '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                          '_' + str(i) + '_lr.pkl'
        if os.path.exists(ensemble_model_file_path):
            print('{} already exists, so no more train.'.format(ensemble_model_file_path))
            break
        ensemble_dl_ml_base_score_train(dl_result_file_path,ml_result_file_path,ensemble_model_file_path,lc_c[i])
    # model infer
    for i in range(2):
        dl_result_file_path = './result/' + date_str + '_' + dl_type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) + \
                              '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_test_aug_' + str(i) + '_' + str(i) + '.csv'
        ml_result_file_path = './result/' + date_str + '_' + ml_type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                              '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_ftr_select_test_' + \
                              str(i) + '_' + str(i) + '.csv'
        for j in range(3):
            ensemble_model_file_path = './model/' + date_str + '_' + ensemble_type + '_' +str(lc_c[i]) + '_'+ split_date_str + '_' + str(max_depth) + '_' + \
                          str(num_leaves) + '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                          '_' + str(j) + '_lr.pkl'
            ensemble_result_file_path = './result/' + date_str + '_' + ensemble_type + '_'+str(lc_c[i]) + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                          str(num_leaves) + '_' + str(n_estimators) + '_' + str(class_weight) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                          '_' + str(j) + '_' + str(i) + '.csv'
            if os.path.exists(ensemble_result_file_path) or (i != j):
                print('{} already exists, so no more infer.'.format(ensemble_result_file_path))
                continue
            print(ensemble_result_file_path)
            ensemble_dl_ml_base_score_test(dl_result_file_path,ml_result_file_path,ensemble_model_file_path,ensemble_result_file_path)

if __name__ == '__main__':
    # train_occur_for_report()
    # train_occur_for_predict()
    # clean_data_train_occur_for_report()
    # clean_data_train_occur_continue_for_report()
    # analysis_error_sample()
    # augment_bad_data_train_occur_continue_for_report()
    # ts2vec_test()
    # ts2vec_relabel()
    # augment_bad_data_relabel_train_occur_continue_for_report()
    # augment_bad_data_relabel_multiclass_train_occur_continue_for_report()
    # augment_bad_data_add_credit_relabel_multiclass_train_occur_continue_for_report()
    # tsfresh_test()
    # augment_bad_data_add_credit_relabel_multiclass_augment_ftr_select_train_occur_continue_for_report()
    ensemble_data_augment_group_ts_dl_ftr_select_nts_ml_base_score()
