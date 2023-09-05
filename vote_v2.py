# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
from datetime import datetime

import pandas as pd
import numpy as np
from numba import NumbaTypeSafetyWarning
from numpy import testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import warnings
import math
import matplotlib.pyplot as plt
from sklearn import metrics

from sktime.classification.hybrid import HIVECOTEV2,HIVECOTEV1
from sktime.classification.sklearn import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.datatypes import  convert

warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=NumbaTypeSafetyWarning)



def train_occur_for_tmp():
    # dfAlt = pd.read_csv("./data/0617_3746.csv",header=0, nrows=86697,sep=',',encoding='gbk')
    # dfAlt = pd.read_csv("./data/0614_150_s_3.csv",header=0,nrows=12689, sep=',',encoding='gbk')
    dfAlt = pd.read_csv("./data/0606_train/2023_202307111247.csv", header=0, sep=',', encoding='gbk')
    # dfAlt = pd.read_csv("./data/2023_202307120942.csv",header=0, sep=',',encoding='gbk')
    # dfAlt0 = dfAlt.iloc[:69606,:] # 69606  10017
    # dfAlt1 = dfAlt.iloc[69606:,:]

    col = dfAlt.columns.tolist()
    print(col)
    col.remove('CUSTOMER_ID')
    col.remove('RDATE')
    col.remove('Y')
    print(col)

    # dfAlt = shuffle(dfAlt,random_state=0)

    # print(dfAlt.shape)
    # print(dfAlt.tail(48))

    dfAlt0 = dfAlt.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    dfAlt0 = dfAlt0.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230401)
    n_line_tail = 150
    n_line_head = 0
    dfAlt0 = dfAlt0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    # reset_index(drop=True).groupby(['CUSTOMERID'],sort=False).head(24)

    dfAlt1 = dfAlt.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230401)
    dfAlt1 = dfAlt1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)

    dfAlt1 = dfAlt1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    # reset_index(drop=True).groupby(['CUSTOMERID'],sort=False).head(24)

    print(dfAlt0.shape)
    print(dfAlt1.shape)

    from paddlets import TSDataset
    tsdatasets0 = TSDataset.load_from_dataframe(
        df=dfAlt0,
        group_id='CUSTOMER_ID',
        # time_col='date',
        # target_cols='TARGET',
        # target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
        # target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
        # target_cols=col,
        target_cols=['GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'GRP_USEAMT_SUM', 'SDV_REPAY_90', 'EXT_12M_R',
                     'LSR_181_AVG_180', 'GRP_CNT', 'INV_AVG_180', 'XSZQ180D_R', 'LRR_AVG_180'],
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID']
    )
    tsdatasets1 = TSDataset.load_from_dataframe(
        df=dfAlt1,
        group_id='CUSTOMER_ID',
        # time_col='date',
        # target_cols='TARGET',
        # target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
        # target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
        # target_cols=col,
        target_cols=['GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'GRP_USEAMT_SUM', 'SDV_REPAY_90', 'EXT_12M_R',
                     'LSR_181_AVG_180', 'GRP_CNT', 'INV_AVG_180', 'XSZQ180D_R', 'LRR_AVG_180'],
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID']
    )

    from paddlets.transform import MinMaxScaler, StandardScaler
    # min_max_scaler = MinMaxScaler()
    # min_max_scaler = StandardScaler()
    # tsdatasets0 = min_max_scaler.fit_transform(tsdatasets0)
    # tsdatasets1 = min_max_scaler.fit_transform(tsdatasets1)

    y_label0 = [0, ]
    y_label1 = [0, ]
    y_label0_customerid = ['null', ]
    y_label1_customerid = ['null', ]
    for dataset in tsdatasets0:
        y_label0.append(dataset.static_cov['Y'])
        y_label0_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_label0 = np.array(y_label0)
    for dataset in tsdatasets1:
        y_label1.append(dataset.static_cov['Y'])
        y_label1_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_label1 = np.array(y_label1)
    X_0 = tsdatasets0[0].to_numpy()
    X_0 = X_0.transpose()
    X_1 = tsdatasets1[0].to_numpy()
    X_1 = X_1.transpose()
    aa = X_0
    dim = aa.shape
    for dataset in tsdatasets0:
        data = dataset.to_numpy()
        data = data.transpose()
        X_0 = np.append(X_0, data)
    X_0 = X_0.reshape(int(len(X_0) / (dim[0] * dim[1])), dim[0], dim[1])
    for dataset in tsdatasets1:
        data = dataset.to_numpy()
        data = data.transpose()
        X_1 = np.append(X_1, data)
    X_1 = X_1.reshape(int(len(X_1) / (dim[0] * dim[1])), dim[0], dim[1])

    # print(X_0)
    # print(X_1)
    print(y_label0, sum(y_label0), len(y_label0), y_label1, sum(y_label1), len(y_label1))
    # train HIVE-COTE v2

    hc2 = HIVECOTEV2(
        random_state=0,
        n_jobs=2,  # 48  -1
        verbose=1,
        time_limit_in_minutes=0,
        stc_params={
            "n_shapelet_samples": 10000,  # 10000
            "max_shapelets": None,  # None
            "max_shapelet_length": None,  # None
            "estimator": RotationForest(n_estimators=200,  # 200
                                        min_group=3,
                                        max_group=30,  # 3
                                        remove_proportion=0.1,  # 0.5
                                        base_estimator=None,
                                        time_limit_in_minutes=0,
                                        contract_max_n_estimators=500),  #
            "transform_limit_in_minutes": 0,
            "time_limit_in_minutes": 0,  # 0
            "contract_max_n_shapelet_samples": np.inf,  # np.inf
            "batch_size": 5,  # 5
        },
        drcif_params={
            "n_estimators": 200,
            "n_intervals": None,  # None
            "att_subsample_size": 10,  # 10
            "min_interval": 4,
            "max_interval": None,
            "base_estimator": "CIT",
            "time_limit_in_minutes": 0,
            "contract_max_n_estimators": 500,  #
        },
        arsenal_params={
            "num_kernels": 2000,
            "n_estimators": 250,  # 25
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": 32,
            "n_features_per_kernel": 40,  # 4
            "time_limit_in_minutes": 0,
            "contract_max_n_estimators": 100,
        },
        tde_params={
            "n_parameter_samples": 250,
            "max_ensemble_size": 150,  # 50
            "max_win_len_prop": 1,
            "min_window": 10,
            "randomly_selected_params": 50,
            "bigrams": False,
            "dim_threshold": 0.85,  # 0.85
            "max_dims": 20,  # 20
            "time_limit_in_minutes": 0,
            "contract_max_n_parameter_samples": np.inf,
            "typed_dict": True,  # T
        },
    )
    hc3 = HIVECOTEV2(
        stc_params={
            "estimator": RotationForest(contract_max_n_estimators=1),
            "contract_max_n_shapelet_samples": 5,
            "max_shapelets": 5,
            "batch_size": 5,
        },
        drcif_params={
            "contract_max_n_estimators": 1,
            "n_intervals": 2,
            "att_subsample_size": 2,
        },
        arsenal_params={"num_kernels": 5, "contract_max_n_estimators": 1},
        tde_params={
            "contract_max_n_parameter_samples": 1,
            "max_ensemble_size": 1,
            "randomly_selected_params": 1,
        },
        time_limit_in_minutes=0,
        random_state=0,
        n_jobs=48,
        verbose=1,
    )

    # X_train, y_train = load_basic_motions(split="train")
    # X_test, y_test = load_basic_motions(split="test")
    # indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)
    hc4 = HIVECOTEV2(
        random_state=0,
        n_jobs=48,
        verbose=1,
        stc_params={
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
        },
        drcif_params={"n_estimators": 10},
        arsenal_params={"num_kernels": 100, "n_estimators": 5},
        tde_params={
            "n_parameter_samples": 25,
            "max_ensemble_size": 5,
            "randomly_selected_params": 10,
        },
    )
    # X_train0 = X_train.iloc[indices]
    # y_train0 = y_train[indices]
    # hc2.fit(X_train0, y_train0)

    # assert probabilities are the same
    # probas = hc2.predict_proba(X_test.iloc[indices[:10]])
    # print(probas)

    hc2.fit(X_0, y_label0)

    # hc2.save("./model/0711_244_t"+str(line)+"_t20_r0_j1_y23_m147_v1")   #
    # hc2.save("./model/0705_motion_v1")   #
    # probas = hc2.predict_proba(X_1)
    # print(probas)
    preds = hc2.predict(X_1)
    print("pred:", preds)
    for i in range(len(preds)):
        print(y_label1_customerid[i], y_label1[i], '->', preds[i])
    print(len(preds), sum(preds), sum(preds) / len(preds))
    print("test:", y_label1)
    print("train:", y_label0)
    corr0 = 0
    corr1 = 0
    err0 = 0
    err1 = 0
    for i in range(len(preds)):
        if (preds[i] == y_label1[i] and y_label1[i] == 0):
            corr0 += 1
        elif (preds[i] == y_label1[i] and y_label1[i] == 1):
            corr1 += 1
        elif (preds[i] != y_label1[i] and y_label1[i] == 0):
            err0 += 1
        else:
            err1 += 1
    acc = (corr0 + corr1) / len(preds) * 1.0
    recall = corr1 / sum(y_label1) * 1.0  # bad sample
    precison = corr1 / (corr1 + err0) * 1.0  # bad sample
    print("acc:", acc)
    print("recall:", recall)
    print("precison:", precison)
    print(corr0, corr1, err0, err1, sum(y_label1), len(preds), sum(y_label0), len(y_label0))
    f1 = 2 * recall * precison / (recall + precison) * 1.0
    print("f1:", f1)

    # hc2 = HIVECOTEV2.load_from_path("./model/0711_244_t90_t09_r0_j48_y23_m147_v1.zip")
    # preds = hc2.predict(X_1)
    # print(preds)

def train_occur_for_report():
    df23 = pd.read_csv("./data/0808_train/occur/2023_202308081713.csv", header=0, sep=',', encoding='gbk')
    df22 = pd.read_csv("./data/0808_train/occur/2022_202308081710.csv", header=0, sep=',', encoding='gbk')
    df21 = pd.read_csv("./data/0808_train/occur/2021_202308081707.csv", header=0, sep=',', encoding='gbk')
    df20 = pd.read_csv("./data/0808_train/occur/2020_202308081617.csv", header=0, sep=',', encoding='gbk')
    df19 = pd.read_csv("./data/0808_train/occur/2019_202308081614.csv", header=0, sep=',', encoding='gbk')
    df18 = pd.read_csv("./data/0808_train/occur/2018_202308081610.csv", header=0, sep=',', encoding='gbk')
    df17 = pd.read_csv("./data/0808_train/occur/2017_202308081606.csv", header=0, sep=',', encoding='gbk')
    df16 = pd.read_csv("./data/0808_train/occur/2016_202308081603.csv", header=0, sep=',', encoding='gbk')

    #df_all = pd.concat([df16, df17, df18, df19, df20, df21, df22, df23])
    df_all = pd.concat([df21, df22, df23])
    print(df16.shape)
    print(df17.shape)
    print(df18.shape)
    print(df19.shape)
    print(df20.shape)
    print(df21.shape)
    print(df22.shape)
    print(df23.shape)
    print('df_all.shape:', df_all.shape)

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
    #col = ['GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'GRP_USEAMT_SUM', 'SDV_REPAY_90', 'EXT_12M_R', 'LSR_181_AVG_180', 'GRP_CNT', 'INV_AVG_180', 'XSZQ180D_R', 'LRR_AVG_180']

    n_line_tail = 30  # (1-7) * 30
    n_line_head = 0
    type = 'occur'
    date_str = datetime(2023, 8, 23).strftime("%Y%m%d")

    df_part1 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)  # for test
    df_part2 = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230101)  # for train valid

    df_part2 = df_part2.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    # 随机选择若干个组
    selected_groups = df_part2['CUSTOMER_ID'].drop_duplicates().sample(n=150, random_state=150)
    # 获取每个选中组的所有样本
    df_val = df_part2.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    print('df_val.shape: ', df_val.shape)
    # 获取剩余的组
    df_train = df_part2[~df_part2['CUSTOMER_ID'].isin(selected_groups)]
    df_train = df_train.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_train.shape: ', df_train.shape)

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

    y_train = [0,]
    y_val = [0,]
    y_test = [0,]
    y_train_customerid = ['null',]
    y_val_customerid = ['null',]
    y_test_customerid = ['null',]
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

    x_train = tsdatasets_train[0].to_numpy()
    x_train = x_train.transpose()
    x_val = tsdatasets_val[0].to_numpy()
    x_val = x_val.transpose()
    x_test = tsdatasets_test[0].to_numpy()
    x_test = x_test.transpose()
    dim = x_train.shape
    for dataset in tsdatasets_train:
        data = dataset.to_numpy()
        data = data.transpose()
        x_train = np.append(x_train, data)
    x_train = x_train.reshape(int(len(x_train) / (dim[0] * dim[1])), dim[0], dim[1])
    for dataset in tsdatasets_val:
        data = dataset.to_numpy()
        data = data.transpose()
        x_val = np.append(x_val, data)
    x_val = x_val.reshape(int(len(x_val) / (dim[0] * dim[1])), dim[0], dim[1])
    for dataset in tsdatasets_test:
        data = dataset.to_numpy()
        data = data.transpose()
        x_test = np.append(x_test, data)
    x_test = x_test.reshape(int(len(x_test) / (dim[0] * dim[1])), dim[0], dim[1])

    hc2 = HIVECOTEV2(
        random_state=0,
        n_jobs=2,  # 48  -1
        verbose=1,
        time_limit_in_minutes=0,
        stc_params={
            "n_shapelet_samples": 10000,  # 10000
            "max_shapelets": None,  # None
            "max_shapelet_length": 30,  # None
            "estimator": RotationForest(n_estimators=400,  # 200
                                        min_group=3,
                                        max_group=30,  # 3
                                        remove_proportion=0.01,  # 0.5
                                        base_estimator=None,
                                        time_limit_in_minutes=0,
                                        contract_max_n_estimators=500),  #
            "transform_limit_in_minutes": 0,
            "time_limit_in_minutes": 0,  # 0
            "contract_max_n_shapelet_samples": np.inf,  # np.inf
            "batch_size": 100,  # 100
        },
        drcif_params={
            "n_estimators": 200,
            "n_intervals": None,  # None
            "att_subsample_size": 10,  # 10
            "min_interval": 4,
            "max_interval": None,
            "base_estimator": "CIT",
            "time_limit_in_minutes": 0,
            "contract_max_n_estimators": 500,  #
        },
        arsenal_params={
            "num_kernels": 2000,
            "n_estimators": 250,  # 25
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": 32,
            "n_features_per_kernel": 40,  # 4
            "time_limit_in_minutes": 0,
            "contract_max_n_estimators": 100,
        },
        tde_params={
            "n_parameter_samples": 250,
            "max_ensemble_size": 150,  # 50
            "max_win_len_prop": 1,
            "min_window": 10,
            "randomly_selected_params": 50,
            "bigrams": False,
            "dim_threshold": 0.85,  # 0.85
            "max_dims": 20,  # 20
            "time_limit_in_minutes": 0,
            "contract_max_n_parameter_samples": np.inf,
            "typed_dict": True,  # T
        },
    )

    hc2.fit(x_train, y_train)

    hc2.save('./model/'+date_str+'_' + type + '_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m01.hc2')

    ############################### val
    filename = './result/'+date_str+ '_' + type + '_M6_y16_m1_y23_m1_result_val_' + str(n_line_tail) + '_hc2.csv'
    pred_val = hc2.predict(x_val)
    pred_val_prob = hc2.predict_proba(x_val)[:, 1]

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
    plt.title('Receiver operating characteristic example(val)')
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m01_y23_m7_ROC_val_" + str(n_line_tail) + "_hc2.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str +"_" + type + "_y16_m1_y23_m01_y23_m7_KS_val_" + str(n_line_tail) + "_hc2.png")
    plt.show()

    ############################### test
    filename = './result/'+date_str +'_' + type + '_M6_y16_m1_y23_m1_result_test_' + str(n_line_tail) + '_hc2.csv'
    pred_test = hc2.predict(x_test)
    pred_test_prob = hc2.predict_proba(x_test)[:, 1]

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
    plt.title('Receiver operating characteristic example(test)')
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_ROC_test_" + str(n_line_tail) + "_hc2.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_KS_test_" + str(n_line_tail) + "_hc2.png")
    plt.show()

    ############################### train
    pred_train = hc2.predict(x_train)
    pred_train_prob = hc2.predict_proba(x_train)[:, 1]

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
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_ROC_train_" + str(n_line_tail) + "_hc2.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(train)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("./result/"+date_str+"_" + type + "_y16_m1_y23_m1_y23_m7_KS_train_" + str(n_line_tail) + "_hc2.png")
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

if __name__ == '__main__':
    train_occur_for_report()
    #train_occur_for_predict()