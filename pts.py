# coding=utf-8
import sys, os, time
import subprocess
import numpy as np
import pandas as pd

import warnings
import paddlets
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.models.anomaly import AutoEncoder
from paddlets.models.ml_model_wrapper import make_ml_model
from pyod.models.knn import KNN
#from pyod.models.vae import VAE
from pyod.models.gmm import GMM
from pyod.models.mcd import MCD
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore',category=DeprecationWarning)
from paddlets.datasets.repository import get_dataset

ts_x_trains, y_train = get_dataset("BasicMotions_Train")
ts_x_tests, y_test = get_dataset("BasicMotions_Test")
print(ts_x_trains[0],y_train[0])
print("=="*50)
print(ts_x_trains[1],y_train[1])
print("=="*50)
print(ts_x_tests[0],y_test[0])
print("=="*50)
print(ts_x_tests[1],y_test[1])
print("=="*50)
from paddlets.models.classify.dl.cnn import CNNClassifier

network = CNNClassifier(max_epochs=100, patience=50)
network.fit(ts_x_trains, y_train)

from sklearn.metrics import accuracy_score, f1_score

preds = network.predict(ts_x_tests)
print(preds)
score = accuracy_score(y_test, preds)
print(score)
f1 = f1_score(y_test, preds, average="macro")
print(f1)
preds = network.predict_proba(ts_x_tests)
print(preds)
exit(0)

#dfAlt = pd.read_csv("./data/0614_150.csv", nrows=3280,header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0614_10.csv",usecols=[0,1,2,3,4,5,6,7,8,9],nrows=500, header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0614_10.csv", header=0, sep=',',encoding='gbk')
dfAlt = pd.read_csv("./data/0614_150_s.csv",header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0617_3746.csv",header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0614_2.csv",header=0, sep=',',encoding='gbk')
#print(dfAlt.shape)
#print(dfAlt.head(5))
#print(list(dfAlt.columns))

#dfAlt['date'] = pd.to_datetime(dfAlt['RDATE'])
dfAlt0 = dfAlt.iloc[:2898,:]
dfAlt1 = dfAlt.iloc[2898:,:]
print(dfAlt0.shape)
print(dfAlt1.shape)
print(dfAlt1.head())

#Load TSDatasets by group_id
from paddlets import TSDataset
from paddlets.analysis import FFT
tsdatasets0 = TSDataset.load_from_dataframe(
    df=dfAlt0,
    #group_id='CUSTOMERID',
    #time_col='date',
    #target_cols='TARGET',
    label_col='TARGET',
    #observed_cov_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180'],
    #feature_cols= ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180'],
    feature_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM',
                   'SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT',
                   'AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
    #feature_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
    fill_missing_dates=True,
    fillna_method="zero",
    #freq = '1M',
    #drop_tail_nan=True,
    #static_cov_cols='id'
)


tsdatasets1 = TSDataset.load_from_dataframe(
    df=dfAlt1,
    #group_id='CUSTOMERID',
    #time_col='date',
    #target_cols='TARGET',
    label_col='TARGET',
    #observed_cov_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180'],
    #feature_cols= ['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180'],
    feature_cols=['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R',
                  'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R', 'LSR_181_AVG_180', 'GRP_CNT', 'INV_AVG_180',
                  'XSZQ180D_R', 'DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
    #feature_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
    fill_missing_dates=True,
    fillna_method="zero",
    #drop_tail_nan=True,
    #static_cov_cols='id'
)
from paddlets.transform import MinMaxScaler, StandardScaler

min_max_scaler = MinMaxScaler()
#min_max_scaler = StandardScaler()
tsdatasets0 = min_max_scaler.fit_transform(tsdatasets0)
tsdatasets1 = min_max_scaler.fit_transform(tsdatasets1)

tsdatasets0['TARGET'] = tsdatasets0['TARGET'].astype('int64')
tsdatasets1['TARGET'] = tsdatasets1['TARGET'].astype('int64')

from paddlets.models.forecasting import MLPRegressor, InformerModel, NBEATSModel, DeepARModel, NHiTSModel, \
    RNNBlockRegressor, SCINetModel, TCNRegressor, TFTModel, TransformerModel, LSTNetRegressor

#  deepAR  error
#mlp = DeepARModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10)
#mlp = InformerModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10)
#mlp = LSTNetRegressor(in_chunk_len=1, out_chunk_len=1,max_epochs=10,kernel_size=1) # bd
#mlp = MLPRegressor(in_chunk_len=4, out_chunk_len=3,max_epochs=100)
#mlp = NBEATSModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10)
#mlp = NHiTSModel(in_chunk_len=5, out_chunk_len=1,max_epochs=10) # gd
#mlp = RNNBlockRegressor(in_chunk_len=1, out_chunk_len=1,max_epochs=10) #bd
#mlp = SCINetModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10) #bd
#mlp = TCNRegressor(in_chunk_len=1, out_chunk_len=1,max_epochs=10) #bd
#mlp = TFTModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10) # Known covariates are necessary
#mlp = TransformerModel(in_chunk_len=1, out_chunk_len=1,max_epochs=10) # low gd
#mlp = AutoEncoder(in_chunk_len=4,max_epochs=10)
#mlp = make_ml_model(in_chunk_len=1,model_class=KNN)
mlp = make_ml_model(in_chunk_len=12,model_class=MCD) #gd
#mlp = make_ml_model(in_chunk_len=12,model_class=IForest) #

mlp.fit(tsdatasets0)
#os.remove("./model/0614_150.*")
#mlp.save("./model/0617_3746.nhi") #SVWVW2333322  SVWVW2510266
#mlp = load("./model/0617_3746.nhi") #gd
data = mlp.predict(tsdatasets1)
data.plot()
plt.show()
exit(0)
for tsdataset in tsdatasets1:
    print(tsdataset)
    print(mlp.predict(tsdataset))
exit(0)
print("=="*50)
from paddlets.models.model_loader import load
#mlp = load("./model/0614_150.mlp")
#mlp = load("./model/0614_150.infor")
#mlp = load("./model/0614_150.nbeats")
#mlp = load("./model/0614_150.nhi") #gd
#mlp = load("./model/0614_150.rnn") #bd
#mlp = load("./model/0614_150.sci") #bd
#mlp = load("./model/0614_150.tcn") #bd
#mlp = load("./model/0614_150.lst") #
#mlp = load("./model/0617_3746.nhi") #gd
#mlp = load("./model/0617_10.mlp") #gd
mlp = load("./model/0617_10.nhi") #gd
for tsdataset in tsdatasets1:
    print(tsdataset)
    #print(mlp.predict(tsdataset))

for tsdataset in tsdatasets1:
    print(tsdataset)
    print(mlp.predict(tsdataset))
exit(0)
for tsdataset in tsdatasets1:
    print(tsdataset)
    tsdatasetA,tsdatasetB = tsdataset.split(0.99)
    print("==" * 50)
    #print(tsdatasetA)
    tsdatasetC, tsdatasetD = tsdatasetA.split(0.99)
    tsdatasetE, tsdatasetF = tsdatasetC.split(0.99)
    tsdatasetG, tsdatasetH = tsdatasetE.split(0.99)
    #tsdatasetI, tsdatasetJ = tsdatasetG.split(0.5)
    #tsdatasetK, tsdatasetL = tsdatasetI.split(0.5)
    print(mlp.predict(tsdatasetB))
    print(mlp.predict(tsdatasetD))
    print(mlp.predict(tsdatasetF))
    print(mlp.predict(tsdatasetH))
    #print(mlp.predict(tsdatasetJ))
    #print(mlp.predict(tsdatasetL))

#tsdatasets0[0].plot(columns=['INV_RATIO','INV_AVG_7','INV_AVG_60'])
#print(tsdatasets0[0])
#print(tsdatasets[0].summary())
#print(f"built-in datasets: {dataset_list()}")
#print(paddlets.__version__)


from paddlets.analysis import FFT
#fft = FFT()
#res = fft(tsdatasets0[0], columns=['INV_AVG_7','INV_AVG_60'])
#fft.plot()


#from paddlets.xai.post_hoc.shap_explainer import

plt.show()