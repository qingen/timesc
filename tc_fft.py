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

#dfAlt = pd.read_csv("./data/0614_150_s_2.csv",header=0, sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0617_3746.csv",header=0, nrows=86697,sep=',',encoding='gbk')
#dfAlt = pd.read_csv("./data/0614_150_s_3.csv",header=0,nrows=12689, sep=',',encoding='gbk')
df23 = pd.read_csv("./data/0606_train/2023_202307111247.csv",header=0, sep=',',encoding='gbk')
df22 = pd.read_csv("./data/0606_train/2022_202307111254.csv",header=0, sep=',',encoding='gbk')
df21 = pd.read_csv("./data/0606_train/2021_202307111301.csv",header=0, sep=',',encoding='gbk')
df20 = pd.read_csv("./data/0606_train/2020_202307111314.csv",header=0, sep=',',encoding='gbk')
df19 = pd.read_csv("./data/0606_train/2019_202307111456.csv",header=0, sep=',',encoding='gbk')
df18 = pd.read_csv("./data/0606_train/2018_202307111502.csv",header=0, sep=',',encoding='gbk')
#df23_2 = pd.read_csv("./data/0802_train/2023_4_train_202308021803.csv",header=0, sep=',',encoding='gbk')
#df23_1 = pd.read_csv("./data/0802_train/2023_1_4_train_202308021758.csv",header=0, sep=',',encoding='gbk')
#df22_3 = pd.read_csv("./data/0802_train/2022_10_12_train_202308021750.csv",header=0, sep=',',encoding='gbk')
#df22_2 = pd.read_csv("./data/0802_train/2022_6_10_train_202308021746.csv",header=0, sep=',',encoding='gbk')
#df22_1 = pd.read_csv("./data/0802_train/2022_1_6_train_202308021630.csv",header=0, sep=',',encoding='gbk')
#df21 = pd.read_csv("./data/0802_train/2021_train_202308021559.csv",header=0, sep=',',encoding='gbk')
#df20 = pd.read_csv("./data/0802_train/2020_train_202308021521.csv",header=0, sep=',',encoding='gbk')
#df19 = pd.read_csv("./data/0802_train/2019_train_202308021516.csv",header=0, sep=',',encoding='gbk')
#df18 = pd.read_csv("./data/0802_train/2018_train_202308021513.csv",header=0, sep=',',encoding='gbk')
#df23 = pd.read_csv("./data/2023_202307120942.csv",header=0, sep=',',encoding='gbk')
#dfAlt0 = dfAlt.iloc[:69606,:] # 69606  2904  6002
#dfAlt1 = dfAlt.iloc[69606:,:]

#dfAlt = shuffle(dfAlt,random_state=0)

df22_train = df22.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20220801) # 20220101 20211001 20210701 20210401 20210101
df22_test = df22.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20220801)  # 20220401 20220301 20220201 20220101 20211201
#df22_test = df22.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20221001)  # 20220401 20220301 20220201 20220101 20211201
print(df22_train.shape)
print(df22_test.shape)
print('df22.shape: ',df22.shape)

df_all = pd.concat([df18, df19, df20, df21, df21, df22_train])
print(df18.shape)
print(df19.shape)
print(df20.shape)
print(df21.shape)
print(df22_train.shape)
print('df_all.shape:',df_all.shape)

col = df_all.columns.tolist()
print(col)
col.remove('CUSTOMER_ID')
col.remove('RDATE')
col.remove('Y')
print(col)


n_line_tail = 30
n_line_head = 0

# 随机选择若干个组
selected_groups = df_all['CUSTOMER_ID'].drop_duplicates().sample(n=240, random_state=240)
# 获取每个选中组的所有样本
df_val = df_all.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
print('df_val.shape: ',df_val.shape)
# 获取剩余的组
df_train = df_all[~df_all['CUSTOMER_ID'].isin(selected_groups)]
#df_train = pd.concat([df23,df22_test])   # just for test
print('df_train.shape: ',df_train.shape)

df_train = df_train.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending = True)).\
    reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
df_val = df_val.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending = True)).\
    reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail) #.\
    #reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    #.filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

#df_test = df23.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101) # 20230401 11/66
#df_test = df_test.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)
df_test = pd.concat([df22_test,df23])
print('df_test.shape: ',df_test.shape)
df_test = df_test.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending = True)).\
    reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail) #.\
    #reset_index(drop=True).groupby(['CUSTOMERID']).head(24)
    #.filter(lambda x: len(x["RDATE"]) >= 2)  len(x["C"]) > 2

print('df_test.shape: ',df_test.shape)
print('df_val.shape: ',df_val.shape)
print('df_train.shape: ',df_train.shape)

from paddlets import TSDataset
from paddlets.analysis import FFT,CWT
tsdatasets_train = TSDataset.load_from_dataframe(
    df=df_train,
    group_id='CUSTOMER_ID',
    #time_col='date',
    #target_cols='TARGET',
    #target_cols= ['GRP_REPAYCARS90_SUM', 'AVG_DBDKTC1W_180', 'GRP_REPAYCARS180_SUM', 'PUSHREPAY15W90_R', 'GRP_USEAMT_SUM','SDV_REPAY_90', 'EXT_12M_R','LSR_181_AVG_180','GRP_CNT','INV_AVG_180','XSZQ180D_R','DBFJZC_AMT','AVG_HKBZJKY_SQHKKY_180','LRR_AVG_180'],
    #target_cols=['INV_RATIO', 'INV_AVG_7', 'INV_AVG_15', 'INV_AVG_30', 'INV_AVG_60', 'INV_AVG_90', 'INV_AVG_180', 'INV_AVG_365', 'INV_CHA_7', 'INV_CHA_15', 'INV_CHA_30', 'INV_CHA_60', 'INV_CHA_90', 'INV_CHA_180', 'INV_CHA_365', 'LOAN_REPAY_RATIO', 'LRR_AVG_7', 'LRR_AVG_15', 'LRR_AVG_30', 'LRR_AVG_60', 'LRR_AVG_90', 'LRR_AVG_180', 'LRR_AVG_365', 'LRR_CHA_7', 'LRR_CHA_15', 'LRR_CHA_30', 'LRR_CHA_60', 'LRR_CHA_90', 'LRR_CHA_180', 'LRR_CHA_365', 'AMOUNT_CHANGE_SIGNAL', 'USEAMOUNT_RATIO', 'UAR_LAG_YEAR', 'UAR_AVG_7', 'UAR_AVG_15', 'UAR_AVG_30', 'UAR_AVG_60', 'UAR_AVG_90', 'UAR_AVG_180', 'UAR_AVG_365', 'UAR_CHA_7', 'UAR_CHA_15', 'UAR_CHA_30', 'UAR_CHA_60', 'UAR_CHA_90', 'UAR_CHA_180', 'UAR_CHA_365', 'UAR_CHA_YEAR', 'SDV_REPAY_7', 'SDV_REPAY_15', 'SDV_REPAY_30', 'SDV_REPAY_60', 'SDV_REPAY_90', 'SDV_REPAY_180', 'SDV_REPAY_365', 'REPAY_STD_RATIO_7_15', 'REPAY_STD_RATIO_7_30', 'REPAY_STD_RATIO_7_60', 'REPAY_STD_RATIO_7_90', 'REPAY_STD_RATIO_7_180', 'REPAY_STD_RATIO_7_365', 'STOCK_OVER_91_RATIO', 'LSR_91_AVG_7', 'LSR_91_AVG_15', 'LSR_91_AVG_30', 'LSR_91_AVG_60', 'LSR_91_AVG_90', 'LSR_91_AVG_180', 'LSR_91_AVG_365', 'LSR_91_CHA_7', 'LSR_91_CHA_15', 'LSR_91_CHA_30', 'LSR_91_CHA_60', 'LSR_91_CHA_90', 'LSR_91_CHA_180', 'LSR_91_CHA_365', 'STOCK_OVER_121_RATIO', 'LSR_121_AVG_7', 'LSR_121_AVG_15', 'LSR_121_AVG_30', 'LSR_121_AVG_60', 'LSR_121_AVG_90', 'LSR_121_AVG_180', 'LSR_121_AVG_365', 'LSR_121_CHA_7', 'LSR_121_CHA_15', 'LSR_121_CHA_30', 'LSR_121_CHA_60', 'LSR_121_CHA_90', 'LSR_121_CHA_180', 'LSR_121_CHA_365', 'STOCK_OVER_181_RATIO', 'LSR_181_AVG_7', 'LSR_181_AVG_15', 'LSR_181_AVG_30', 'LSR_181_AVG_60', 'LSR_181_AVG_90', 'LSR_181_AVG_180', 'LSR_181_AVG_365', 'LSR_181_CHA_7', 'LSR_181_CHA_15', 'LSR_181_CHA_30', 'LSR_181_CHA_60', 'LSR_181_CHA_90', 'LSR_181_CHA_180', 'LSR_181_CHA_365', 'STOCK_AGE_AVG', 'STOCK_AGE_AVG_7', 'STOCK_AGE_AVG_15', 'STOCK_AGE_AVG_30', 'STOCK_AGE_AVG_60', 'STOCK_AGE_AVG_90', 'STOCK_AGE_AVG_180', 'STOCK_AGE_AVG_365', 'STOCK_AGE_CHA_RATIO_7', 'STOCK_AGE_CHA_RATIO_15', 'STOCK_AGE_CHA_RATIO_30', 'STOCK_AGE_CHA_RATIO_60', 'STOCK_AGE_CHA_RATIO_90', 'STOCK_AGE_CHA_RATIO_180', 'STOCK_AGE_CHA_RATIO_365', 'JXDG_NUM_RATIO_7', 'JXDG_NUM_RATIO_15', 'JXDG_NUM_RATIO_30', 'JXDG_NUM_RATIO_60', 'JXDG_NUM_RATIO_90', 'JXDG_NUM_RATIO_180', 'JXDG_NUM_RATIO_365', 'JXDG_NUM_CHA_7_15', 'JXDG_NUM_CHA_7_30', 'JXDG_NUM_CHA_7_60', 'JXDG_NUM_CHA_7_90', 'JXDG_NUM_CHA_7_180', 'JXDG_NUM_CHA_7_365', 'JXDG_PRICE_RATIO_7', 'JXDG_PRICE_RATIO_15', 'JXDG_PRICE_RATIO_30', 'JXDG_PRICE_RATIO_60', 'JXDG_PRICE_RATIO_90', 'JXDG_PRICE_RATIO_180', 'JXDG_PRICE_RATIO_365', 'JXDG_PRICE_CHA_7_15', 'JXDG_PRICE_CHA_7_30', 'JXDG_PRICE_CHA_7_60', 'JXDG_PRICE_CHA_7_90', 'JXDG_PRICE_CHA_7_180', 'JXDG_PRICE_CHA_7_365', 'PUSHREPAY30_R', 'PUSHREPAY90_R', 'PUSHREPAY180_R', 'PUSHREPAY15W30_R', 'PUSHREPAY15W90_R', 'PUSHREPAY15W180_R', 'PUSHJFWBWL90_R', 'PUSHGWJFWBWL90_R', 'PUSHJFWBWL180_R', 'PUSHGWJFWBWL180_R', 'PUSHJFWBWL15W_90_R', 'PUSHGWJFWBWL15W_90_R', 'PUSHJFWBWL15W_180_R', 'PUSHGWJFWBWL15W_180_R', 'AVG_DBDKTC1W_30', 'AVG_DBDKTC1W_90', 'AVG_DBDKTC1W_180', 'AVG_YZQCLDQDKHOUR_30', 'AVG_YZQCLDQHKHOUR_30', 'AVG_KXRKXHOUR_30', 'AVG_HBFXRZHOUR_30', 'FJXSZSDK_30', 'FJXSZSDKJE30_R', 'FJXSZSDK_90', 'FJXSZSDKJE90_R', 'FJXSZSDKJE30_90_R', 'AVG_ACCTHKBZJ_30', 'AVG_ACCTHKBZJ_90', 'AVG_ACCTHKBZJ_180', 'AVG_ACCTHKBZJKY_30', 'AVG_ACCTHKBZJKY_90', 'AVG_ACCTHKBZJKY_180', 'AVG_ACCTHKBZJKY_30_180', 'AVG_ACCTSQHK_30', 'AVG_ACCTSQHK_90', 'AVG_ACCTSQHK_180', 'AVG_ACCTSQHKKY_30', 'AVG_ACCTSQHKKY_90', 'AVG_ACCTSQHKKY_180', 'AVG_HKBZJKY_SQHKKY_30', 'AVG_HKBZJKY_SQHKKY_90', 'AVG_HKBZJKY_SQHKKY_180', 'GRP_CNT', 'GRP_AVAILAMT_SUM', 'GRP_USEAMT_SUM', 'GRP_REPAYCARS30_SUM', 'GRP_REPAYCARS90_SUM', 'GRP_REPAYCARS180_SUM', 'FR_CNT', 'MORT_CNT', 'BZJ_AMT', 'DBFZZC_AMT', 'DBFJZC_AMT', 'EXT_12M_R', 'XSZQ30D_DIFF', 'XSZQ90D_DIFF', 'XSZQ30D_R', 'XSZQ90D_R', 'XSZQ180D_R'],
    target_cols=col,
    #known_cov_cols='CUSTOMER_ID',
    fill_missing_dates=True,
    fillna_method="zero",
    static_cov_cols=['Y','CUSTOMER_ID'],
)
tsdatasets_val = TSDataset.load_from_dataframe(
    df=df_val,
    group_id='CUSTOMER_ID',
    target_cols=col,
    fill_missing_dates=True,
    fillna_method="zero",
    static_cov_cols=['Y','CUSTOMER_ID'],
)
tsdatasets_test = TSDataset.load_from_dataframe(
    df=df_test,
    group_id='CUSTOMER_ID',
    target_cols=col,
    fill_missing_dates=True,
    fillna_method="zero",
    static_cov_cols=['Y','CUSTOMER_ID'],
)

fft = FFT(fs=1,half=False) #_amplitude  half
#cwt = CWT(scales=n_line_tail/2)
for data in tsdatasets_train:
    resfft = fft(data)
    #rescwt = cwt(data)  # coefs 63*24 complex128 x+yj
    for x in data.columns:
        #----------------- fft
        resfft[x+"_amplitude"].index = data[x].index
        resfft[x + "_phase"].index = data[x].index
        data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
        data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')
        #--------------- cwt
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
    #rescwt = cwt(data)
    for x in data.columns:
        # ----------------- fft
        resfft[x+"_amplitude"].index = data[x].index
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
    #rescwt = cwt(data)
    for x in data.columns:
        # ----------------- fft
        resfft[x+"_amplitude"].index = data[x].index
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

from paddlets.transform import MinMaxScaler, StandardScaler
#min_max_scaler = MinMaxScaler()
min_max_scaler = StandardScaler()
tsdatasets_train = min_max_scaler.fit_transform(tsdatasets_train)
tsdatasets_val = min_max_scaler.fit_transform(tsdatasets_val)
tsdatasets_test = min_max_scaler.fit_transform(tsdatasets_test)

#network = CNNClassifier(max_epochs=1500,patience=50,kernel_size=8)
network = InceptionTimeClassifier(max_epochs=5,patience=5,kernel_size=16)
network.fit(tsdatasets_train, y_train)

from sklearn.metrics import accuracy_score, f1_score
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t22_y21_m1036_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y21_m726_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t21_y21_m416_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t21_y21_m1126_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y20_m1126_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y19_m1126_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y18_m1126_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y17_m1126_v1.itc')
#network.save('./model/0705_50_20_16_209_fft_p_t_SS_t24_y16_m1126_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t30_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t60_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t90_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t120_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t150_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t180_y23_m147_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t30_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t60_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t90_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t120_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t150_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t180_y22_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t30_y21_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t60_y21_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t90_y21_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t180_y21_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t180_y20_m11012_v1.itc')
#network.save('./model/0711_50_20_16_244_fft_p_t_SS_t180_y18_m11012_v1.itc')
#network.save('./model/0712_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
#network.save('./model/0712_100_50_16_244_fft_p_t_cwt_p_t_SS_t'+str(n_line_tail)+'_y23_m147_v1.itc')
#network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
#network.save('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
network.save('./model/0606_5_5_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y18_m01_y22_m10_v1.itc')

from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier
#network=PaddleBaseClassifier.load('./model/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')


############################### val
pred_val = network.predict(tsdatasets_val)
pred_val_prob = network.predict_proba(tsdatasets_val)[:,1]

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
        print(tpr[i], fpr[i], tpr[i]-fpr[i], thresholds[i])
        # break

        # test
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(val)')
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_ROC_val_" + str(n_line_tail) + "_e5.png")
plt.show()
plt.plot(tpr, lw=2, label='tpr')
plt.plot(fpr, lw=2, label='fpr')
plt.plot(tpr - fpr, label='ks')
plt.title('KS = %0.2f(val)' % max(tpr - fpr))
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_KS_val_" + str(n_line_tail) + "_e5.png")
plt.show()

############################### test
pred_test = network.predict(tsdatasets_test)
pred_test_prob = network.predict_proba(tsdatasets_test)[:,1]

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
        print(tpr[i], fpr[i], tpr[i]-fpr[i], thresholds[i])
        # break

        # test
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(test)')
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_ROC_test_" + str(n_line_tail) + "_e5.png")
plt.show()
plt.plot(tpr, lw=2, label='tpr')
plt.plot(fpr, lw=2, label='fpr')
plt.plot(tpr - fpr, label='ks')
plt.title('KS = %0.2f(test)' % max(tpr - fpr))
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_KS_test_" + str(n_line_tail) + "_e5.png")
plt.show()

############################### train AVG_DBDKTC1W_180
pred_train = network.predict(tsdatasets_train)
pred_train_prob = network.predict_proba(tsdatasets_train)[:,1]

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
        print(tpr[i], fpr[i], tpr[i]-fpr[i], thresholds[i])
        # break

        # test
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(train)')
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_ROC_train_" + str(n_line_tail) + "_e5.png")
plt.show()
plt.plot(tpr, lw=2, label='tpr')
plt.plot(fpr, lw=2, label='fpr')
plt.plot(tpr - fpr, label='ks')
plt.title('KS = %0.2f(train)' % max(tpr - fpr))
plt.legend(loc="lower right")
plt.savefig("0606_y18_m1_y22_m8_y23_m7_KS_train_" + str(n_line_tail) + "_e5.png")
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



def test_plt():
    import matplotlib.pyplot as plt
    from sklearn import metrics

    print(metrics.accuracy_score(y_label1, preds, normalize=True, sample_weight=None))
    cfm = metrics.confusion_matrix(y_label1, preds, labels=None, sample_weight=None)
    print(cfm)
    print(metrics.classification_report(y_label1, preds, labels=None, sample_weight=None))
    row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
    error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
    # 将矩阵中对角线上的元素都定位0
    # np.fill_diagonal(error_matrix, 0)
    plt.matshow(error_matrix, cmap=plt.cm.gray)
    plt.show()
    print("RMSE = ", rmse(y_label1, preds_prob))

    fpr, tpr, thresholds = metrics.roc_curve(y_label1, preds_prob, pos_label=1, )  # drop_intermediate=True

    print('test_ks = ', max(tpr - fpr))

    for i in range(tpr.shape[0]):
        if tpr[i] > 0.5:
            print(tpr[i], 1 - fpr[i], thresholds[i])
            # break

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
    plt.savefig("y18_m01_y23_m4-22-4-ROC-test-" + str(n_line_tail) + ".png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("y18_m01_y23_m4-22-4-KS-test-" + str(n_line_tail) + ".png")
    plt.show()