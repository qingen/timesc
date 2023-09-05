# coding=utf-8
import csv
import sys, os, time
import cmath
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.utils import shuffle

import warnings
import paddlets
from paddlets.datasets.repository import get_dataset, dataset_list
import matplotlib.pyplot as plt
from paddlets.models.classify.dl.cnn import CNNClassifier
from paddlets.models.classify.dl.inception_time import InceptionTimeClassifier
from paddlets.datasets.repository import get_dataset

warnings.filterwarnings('ignore', category=DeprecationWarning)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def rmse(y_test, y):
    return math.sqrt(sum((y_test - y) ** 2) / len(y))


from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier

#network = PaddleBaseClassifier.load('./model/0711_50_20_16_244_fft_p_t_SS_t30_y23_m147_v1.itc')
# network=PaddleBaseClassifier.load('./model/0711_50_20_16_244_fft_p_t_SS_t30_y21_m11012_v1.itc')
#network=PaddleBaseClassifier.load('./model/0713_50_20_16_244_fft_p_t_SS_t60_y20_m10_y23_m4_v1.itc')
#network=PaddleBaseClassifier.load('./model/0718_50_20_16_244_fft_p_t_SS_t30_y18_m01_y23_m4_v1.itc')



# dfAlt = pd.read_csv("./data/0614_150_s_2.csv",header=0, sep=',',encoding='gbk')
# dfAlt = pd.read_csv("./data/0617_3746.csv",header=0, nrows=86697,sep=',',encoding='gbk')
# dfAlt = pd.read_csv("./data/0614_150_s_3.csv",header=0,nrows=12689, sep=',',encoding='gbk')
# dfAlt = pd.read_csv("./data/2023_202307111247.csv",header=0, sep=',',encoding='gbk')
# df22 = pd.read_csv("./data/2022_202307111254.csv",header=0, sep=',',encoding='gbk')
# df21 = pd.read_csv("./data/2021_202307111301.csv",header=0, sep=',',encoding='gbk')
# df20 = pd.read_csv("./data/2020_202307111314.csv",header=0, sep=',',encoding='gbk')
# dfAlt = pd.read_csv("./data/2019_202307111456.csv",header=0, sep=',',encoding='gbk')
# dfAlt = pd.read_csv("./data/2018_202307111502.csv",header=0, sep=',',encoding='gbk')
# df23 = pd.read_csv("./data/2023_202307120942.csv",header=0, sep=',',encoding='gbk')
#df23 = pd.read_csv("./data/2023_202307111247.csv", header=0, sep=',', encoding='gbk')

#df207 = pd.read_csv("./data/0720_2639/20_7_202307201701.csv",header=0, sep=',',encoding='gbk')
#df211 = pd.read_csv("./data/0720_2639/21_1_202307201651.csv",header=0, sep=',',encoding='gbk')
#df217 = pd.read_csv("./data/0720_2639/21_7_202307201639.csv",header=0, sep=',',encoding='gbk')
df221 = pd.read_csv("./data/0720_2639/22_1_202307201615.csv",header=0, sep=',',encoding='gbk')
df227 = pd.read_csv("./data/0720_2639/22_7_202307201610.csv",header=0, sep=',',encoding='gbk')
df23_1 = pd.read_csv("./data/0720_2639/2023_1_4_202308070924.csv",header=0, sep=',',encoding='gbk')
df23_2 = pd.read_csv("./data/0720_2639/2023_4_7_202308070930.csv",header=0, sep=',',encoding='gbk')

#print(df207.shape)
#print(df211.shape)
#print(df217.shape)
#print(df221.shape)
#print(df227.shape)
#print(df23.shape)
#df_all = pd.concat([df207, df211, df217, df221, df227, df23])
df_all = pd.concat([df221, df227, df23_1, df23_2])
#df_all = pd.concat([df227,df23])
#df_all = df23
print(df_all.shape)

col = df_all.columns.tolist()
print(col)
col.remove('CUSTOMER_ID')
col.remove('RDATE')
col.remove('Y')
print(col)
# dfAlt = shuffle(dfAlt,random_state=0)

n_line_tail = 30
n_line_head = 30
n_max_time = 450  # 365 + 90
n_step_time = 7   # occur 7, treat 1
type = 'occur'  # occur  treat

#network=PaddleBaseClassifier.load('./model/0718_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m4_v1.itc')
#network=PaddleBaseClassifier.load('./model/0baseM5/0719_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m7_v1.itc')
#network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
#network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
#network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m07_y23_m07_v1.itc')
#network=PaddleBaseClassifier.load('./model/2baseM6/0802_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y18_m01_y23_m7_t0.itc')
network=PaddleBaseClassifier.load('./model/3multiM6/0809_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m7_v1.itc')

df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
#df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230530)
#dfAlt2 = dfAlt1.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) < 20230701)
print('df_all.shape:', df_all.shape)

# 指定要追加的文件名
#filename = './result/20230803_M5_y18_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
#filename = './result/20230803_M5_y23_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
#filename = './result/20230803_M5_y22_m10_y23_m7_result_data_'+str(n_line_head)+'.csv'
#filename = './result/20230803_M5_y22_m07_y23_m7_result_data_'+str(n_line_head)+'.csv'
filename = './result/20230803_M6_y18_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'

start_date = datetime(2023, 7, 31)

for i in np.arange(n_line_tail, n_max_time, n_step_time):
    dfAlt1 = df_all
    dfAlt3 = dfAlt1.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(i)
    dfAlt4 = dfAlt3.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).head(n_line_head)
    # print('='*100)
    # print('i:',i)
    print('dfAlt3.shape:', dfAlt3.shape)
    print('dfAlt4.shape:', dfAlt4.shape)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('current time1:',formatted_time)

    from paddlets import TSDataset
    from paddlets.analysis import FFT, CWT

    tsdatasets1 = TSDataset.load_from_dataframe(
        df=dfAlt4,
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

    for data in tsdatasets1:
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
    print('current time2:',formatted_time)

    y_label1 = []
    y_label1_customerid = []
    for dataset in tsdatasets1:
        y_label1.append(dataset.static_cov['Y'])
        y_label1_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_label1 = np.array(y_label1)

    from paddlets.transform import MinMaxScaler, StandardScaler

    min_max_scaler = StandardScaler()
    tsdatasets1 = min_max_scaler.fit_transform(tsdatasets1)

    #preds = network.predict(tsdatasets1)
    # print("pred:", preds)
    # for i in range(len(preds)):
    #    print(y_label1_customerid[i], y_label1[i], '->', preds[i])
    # print(len(preds), sum(preds), sum(preds) / len(preds))
    # print("test:", y_label1)
    preds_prob = network.predict_proba(tsdatasets1)[:, 1]
    print(preds_prob)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('current time3:',formatted_time)

    if i == n_line_tail:
        df = pd.DataFrame()
        df['Y'] = y_label1
        df['customerid'] = y_label1_customerid
        df.to_csv(filename, index=False)

    df = pd.read_csv(filename)
    preds_prob = preds_prob.tolist()
    new_data = {start_date.strftime("%Y-%m-%d"):preds_prob,}
    # 追加新数据到DataFrame中
    for column, values in new_data.items():
        df[column] = values
    # 将DataFrame写回CSV文件
    df.to_csv(filename, index=False)
    start_date -= timedelta(days=10)


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
    plt.savefig("ROC（test）.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(test)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    plt.savefig("KS（test）.png")
    plt.show()
