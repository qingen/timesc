# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""

import pytest
import pandas as pd
import numpy as np
from numba import NumbaTypeSafetyWarning
from numpy import testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import warnings

from sktime.classification.hybrid import HIVECOTEV2,HIVECOTEV1
from sktime.classification.sklearn import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.datatypes import  convert

warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=NumbaTypeSafetyWarning)

#dfAlt = pd.read_csv("./data/0617_3746.csv",header=0, nrows=86697,sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0614_150_s_3.csv",header=0,nrows=12689, sep=',',encoding='gbk')
dfAlt = pd.read_csv("./data/2023_202307111247.csv",header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/2023_202307120942.csv",header=0, sep=',',encoding='gbk')
#dfAlt0 = dfAlt.iloc[:69606,:] # 69606  10017
#dfAlt1 = dfAlt.iloc[69606:,:]

col = dfAlt.columns.tolist()
print(col)
col.remove('CUSTOMER_ID')
col.remove('RDATE')
col.remove('Y')
print(col)

#dfAlt = shuffle(dfAlt,random_state=0)

#print(dfAlt.shape)
#print(dfAlt.tail(48))

dfAlt0 = dfAlt.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
dfAlt0 = dfAlt0.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230401)
n_line_tail = 60
n_line_head = 0
dfAlt0 = dfAlt0.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending = True)).\
    reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    #reset_index(drop=True).groupby(['CUSTOMERID'],sort=False).head(24)

dfAlt1 = dfAlt.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230401)
dfAlt1 = dfAlt1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)

dfAlt1 = dfAlt1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending = True)).\
    reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    #reset_index(drop=True).groupby(['CUSTOMERID'],sort=False).head(24)

print(dfAlt0.shape)
print(dfAlt1.shape)

from paddlets import TSDataset
tsdatasets0 = TSDataset.load_from_dataframe(
    df=dfAlt0,
    group_id='CUSTOMER_ID',
    #time_col='date',
    #target_cols='TARGET',
    #target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
    #target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
    #target_cols=col,
    target_cols= ['GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','LRR_AVG_180'],
    fill_missing_dates=True,
    fillna_method="zero",
    static_cov_cols=['Y','CUSTOMER_ID']
)
tsdatasets1 = TSDataset.load_from_dataframe(
    df=dfAlt1,
    group_id='CUSTOMER_ID',
    #time_col='date',
    #target_cols='TARGET',
    #target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
    #target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
    #target_cols=col,
    target_cols= ['GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','LRR_AVG_180'],
    fill_missing_dates=True,
    fillna_method="zero",
    static_cov_cols=['Y','CUSTOMER_ID']
)

from paddlets.transform import MinMaxScaler, StandardScaler
#min_max_scaler = MinMaxScaler()
min_max_scaler = StandardScaler()
tsdatasets0 = min_max_scaler.fit_transform(tsdatasets0)
tsdatasets1 = min_max_scaler.fit_transform(tsdatasets1)

y_label0 = [0,]
y_label1 = [0,]
y_label0_customerid = ['null',]
y_label1_customerid = ['null',]
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
    X_0 = np.append(X_0,data)
X_0 = X_0.reshape(int(len(X_0)/(dim[0]*dim[1])),dim[0],dim[1])
for dataset in tsdatasets1:
    data = dataset.to_numpy()
    data = data.transpose()
    X_1 = np.append(X_1,data)
X_1 = X_1.reshape(int(len(X_1)/(dim[0]*dim[1])),dim[0],dim[1])

#print(X_0)
#print(X_1)
print(y_label0,sum(y_label0),len(y_label0),y_label1,sum(y_label1),len(y_label1))
    # train HIVE-COTE v2

hc2 = HIVECOTEV2(
        random_state=0,
        n_jobs=2,  # 48  -1
        verbose=1,
        time_limit_in_minutes=0,
        stc_params={
            "n_shapelet_samples": 10000, # 10000
            "max_shapelets": None, # None
            "max_shapelet_length":None, # None
            "estimator": RotationForest(n_estimators=200, # 200
                                        min_group=3,
                                        max_group=30, # 3
                                        remove_proportion=0.1, # 0.5
                                        base_estimator=None,
                                        time_limit_in_minutes=0,
                                        contract_max_n_estimators=500), #
            "transform_limit_in_minutes":0,
            "time_limit_in_minutes":0, # 0
            "contract_max_n_shapelet_samples": np.inf, # np.inf
            "batch_size": 5,  # 5
        },
        drcif_params={
            "n_estimators":200,
            "n_intervals": None, # None
            "att_subsample_size": 10,  # 10
            "min_interval":4,
            "max_interval":None,
            "base_estimator":"CIT",
            "time_limit_in_minutes":0,
            "contract_max_n_estimators": 500, #
        },
        arsenal_params={
            "num_kernels": 2000,
            "n_estimators": 250, # 25
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": 32,
            "n_features_per_kernel": 40, # 4
            "time_limit_in_minutes": 0,
            "contract_max_n_estimators": 100,
        },
        tde_params={
            "n_parameter_samples":250,
            "max_ensemble_size": 150, # 50
            "max_win_len_prop": 1,
            "min_window": 10,
            "randomly_selected_params": 50,
            "bigrams": False,
            "dim_threshold": 0.85, # 0.85
            "max_dims": 20, # 20
            "time_limit_in_minutes": 0,
            "contract_max_n_parameter_samples":np.inf,
            "typed_dict": True, # T
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

#X_train, y_train = load_basic_motions(split="train")
#X_test, y_test = load_basic_motions(split="test")
#indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)
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
#X_train0 = X_train.iloc[indices]
#y_train0 = y_train[indices]
#hc2.fit(X_train0, y_train0)

    # assert probabilities are the same
#probas = hc2.predict_proba(X_test.iloc[indices[:10]])
#print(probas)

hc2.fit(X_0, y_label0)

#hc2.save("./model/0711_244_t"+str(line)+"_t20_r0_j1_y23_m147_v1")   #
#hc2.save("./model/0705_motion_v1")   #
#probas = hc2.predict_proba(X_1)
#print(probas)
preds = hc2.predict(X_1)
print("pred:",preds)
for i in range(len(preds)):
    print(y_label1_customerid[i],y_label1[i],'->',preds[i])
print(len(preds),sum(preds),sum(preds)/len(preds))
print("test:",y_label1)
print("train:",y_label0)
corr0 = 0
corr1 = 0
err0 = 0
err1 = 0
for i in range(len(preds)):
    if(preds[i]==y_label1[i] and y_label1[i]==0):
        corr0 += 1
    elif (preds[i] == y_label1[i] and y_label1[i] == 1):
        corr1 += 1
    elif(preds[i] != y_label1[i] and y_label1[i]==0):
        err0 += 1
    else:
        err1 += 1
acc = (corr0+corr1) / len(preds) * 1.0
recall = corr1/sum(y_label1) * 1.0 # bad sample
precison = corr1/(corr1 + err0) * 1.0 # bad sample
print("acc:",acc)
print("recall:",recall)
print("precison:",precison)
print(corr0,corr1,err0,err1,sum(y_label1),len(preds),sum(y_label0),len(y_label0))
f1=2*recall*precison/(recall+precison)*1.0
print("f1:",f1)

#hc2 = HIVECOTEV2.load_from_path("./model/0711_244_t90_t09_r0_j48_y23_m147_v1.zip")
#preds = hc2.predict(X_1)
#print(preds)