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
from sklearn import metrics

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



def predict_weekly():
    # network = PaddleBaseClassifier.load('./model/0711_50_20_16_244_fft_p_t_SS_t30_y23_m147_v1.itc')
    # network=PaddleBaseClassifier.load('./model/0711_50_20_16_244_fft_p_t_SS_t30_y21_m11012_v1.itc')
    # network=PaddleBaseClassifier.load('./model/0713_50_20_16_244_fft_p_t_SS_t60_y20_m10_y23_m4_v1.itc')
    # network=PaddleBaseClassifier.load('./model/0718_50_20_16_244_fft_p_t_SS_t30_y18_m01_y23_m4_v1.itc')

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
    # df23 = pd.read_csv("./data/2023_202307111247.csv", header=0, sep=',', encoding='gbk')

    # df207 = pd.read_csv("./data/0720_2639/20_7_202307201701.csv",header=0, sep=',',encoding='gbk')
    # df211 = pd.read_csv("./data/0720_2639/21_1_202307201651.csv",header=0, sep=',',encoding='gbk')
    # df217 = pd.read_csv("./data/0720_2639/21_7_202307201639.csv",header=0, sep=',',encoding='gbk')
    df221 = pd.read_csv("./data/0720_2639/22_1_202307201615.csv", header=0, sep=',', encoding='gbk')
    df227 = pd.read_csv("./data/0720_2639/22_7_202307201610.csv", header=0, sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0720_2639/2023_1_5_202308171425.csv", header=0, sep=',', encoding='gbk')
    #df23_2 = pd.read_csv("./data/0720_2639/2023_5_8_202308171416.csv", header=0, sep=',', encoding='gbk')
    df23_2 = pd.read_csv("./data/0720_2639/2023_5_6_202309081528.csv", header=0, sep=',', encoding='gbk')
    df23_3 = pd.read_csv("./data/0720_2639/2023_6_7_202309081530.csv", header=0, sep=',', encoding='gbk')
    df23_4 = pd.read_csv("./data/0720_2639/2023_7_8_202309081532.csv", header=0, sep=',', encoding='gbk')
    df23_5 = pd.read_csv("./data/0720_2639/2023_8_202309081535.csv", header=0, sep=',', encoding='gbk')

    # print(df207.shape)
    # print(df211.shape)
    # print(df217.shape)
    # print(df221.shape)
    # print(df227.shape)
    # print(df23.shape)
    # df_all = pd.concat([df207, df211, df217, df221, df227, df23])
    df_all = pd.concat([df221, df227, df23_1, df23_2, df23_3, df23_4, df23_5])
    print(df_all.shape)

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
    # dfAlt = shuffle(dfAlt,random_state=0)

    n_line_tail = 14  # (1-7) * 7
    n_line_head = 14  #
    n_max_time = 28*3 + n_line_tail  # 1*7 + n_line_tail
    n_step_time = 1  # occur 7, treat 1
    type = 'treat'  # occur  treat

    # network=PaddleBaseClassifier.load('./model/0718_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m4_v1.itc')
    # network=PaddleBaseClassifier.load('./model/0baseM5/0719_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m7_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m07_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/2baseM6/0802_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y18_m01_y23_m7_t0.itc')
    #network = PaddleBaseClassifier.load('./model/3multiM6/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    network = PaddleBaseClassifier.load('./model/4multiM6/20230906_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m7_fl.itc')

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230901)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_all.shape:', df_all.shape)

    # 指定要追加的文件名
    # filename = './result/20230803_M5_y18_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y23_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y22_m10_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y22_m07_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M6_y18_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230810_' + type + '_M6_y16_m01_y23_m7_result_' + str(n_line_head) + '.csv'
    filename = './result/20230905_' + type + '_M6_y16_m01_y23_m7_result_' + str(n_line_head) + '.csv'

    start_date = datetime(2023, 9, 5)

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
        print('current time1:', formatted_time)

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
        print('current time2:', formatted_time)

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

        # preds = network.predict(tsdatasets1)
        # print("pred:", preds)
        # for i in range(len(preds)):
        #    print(y_label1_customerid[i], y_label1[i], '->', preds[i])
        # print(len(preds), sum(preds), sum(preds) / len(preds))
        # print("test:", y_label1)
        preds_prob = network.predict_proba(tsdatasets1)[:, 1]
        print(preds_prob)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print('current time3:', formatted_time)

        if i == n_line_tail:
            df = pd.DataFrame()
            df['Y'] = y_label1
            df['customerid'] = y_label1_customerid
            df.to_csv(filename, index=False)

        df = pd.read_csv(filename)
        preds_prob = preds_prob.tolist()
        new_data = {start_date.strftime("%Y-%m-%d"): preds_prob, }
        # 追加新数据到DataFrame中
        for column, values in new_data.items():
            df[column] = values
        # 将DataFrame写回CSV文件
        df.to_csv(filename, index=False)
        start_date -= timedelta(days=n_step_time)

def test_for_report():
    df23_2 = pd.read_csv("./data/0825_train/treat/2023_4_202308252055.csv", header=0, sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0825_train/treat/2023_1_4_202308252051.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/treat/2022_10_12_202308251700.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/treat/2022_7_10_202308251655.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/treat/2022_4_7_202308251652.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/treat/2022_1_4_202308251648.csv", header=0, sep=',', encoding='gbk')
    # print(df207.shape)
    df_all = pd.concat([df22_1,df22_2,df22_3,df22_4,df23_1,df23_2])
    print(df_all.shape)
    del df22_1,df22_2,df22_3,df22_4,df23_1,df23_2

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

    n_line_tail = 35  # (1~5) * 7
    n_line_head = 35  # = tail
    n_max_time = 1*4*6 + n_line_tail  # 24 + n_line_tail
    n_step_time = 1  # occur 7, treat 1
    type = 'treat'  # occur  treat
    date_str = datetime(2023, 8, 26).strftime("%Y%m%d")

    network = PaddleBaseClassifier.load('./model/'+date_str+'_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m01_fl.itc')

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_all.shape:', df_all.shape)

    ###################### for test 1:100
    df_all_0 = df_all[df_all['Y'] == 0]
    df_all_1 = df_all[df_all['Y'] == 1]  # 24
    print('df_all_1.shape:',df_all_1.shape)
    # 从 0 中 筛选出 2400 个
    selected_groups = df_all_0['CUSTOMER_ID'].drop_duplicates().sample(n=2400, random_state=2400)
    # 获取每个选中组的所有样本
    df_part1_0_selected = df_all_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    df_all = pd.concat([df_part1_0_selected, df_all_1])
    print('df_all.shape: ', df_all.shape)
    del  df_all_0,df_all_1,df_part1_0_selected


    # 指定要追加的文件名
    filename = './result/'+date_str+'_' + type + '_M6_y16_m01_y23_m01_y23_m07_result_dynamic_' + str(n_line_head) + '_fl.csv'

    start_date = datetime(2023, 8, 20)

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
        print('current time1:', formatted_time)

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
        print('current time2:', formatted_time)

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

        # preds = network.predict(tsdatasets1)
        # print("pred:", preds)
        # for i in range(len(preds)):
        #    print(y_label1_customerid[i], y_label1[i], '->', preds[i])
        # print(len(preds), sum(preds), sum(preds) / len(preds))
        # print("test:", y_label1)
        preds_prob = network.predict_proba(tsdatasets1)[:, 1]
        print(preds_prob)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print('current time3:', formatted_time)

        if i == n_line_tail:
            df = pd.DataFrame()
            df['Y'] = y_label1
            df['customerid'] = y_label1_customerid
            df.to_csv(filename, index=False)

        df = pd.read_csv(filename)
        preds_prob = preds_prob.tolist()
        new_data = {start_date.strftime("%Y-%m-%d"): preds_prob, }
        # 追加新数据到DataFrame中
        for column, values in new_data.items():
            df[column] = values
        # 将DataFrame写回CSV文件
        df.to_csv(filename, index=False)
        start_date -= timedelta(days=n_step_time)

def test_for_ks():
    df23 = pd.read_csv("./data/0808_train/treat/2023_202308081720.csv", header=0, sep=',', encoding='gbk')
    df22 = pd.read_csv("./data/0808_train/treat/2022_202308081722.csv", header=0, sep=',', encoding='gbk')

    # print(df207.shape)
    df_all = pd.concat([df22,df23])
    print(df_all.shape)

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

    n_line_tail = 7  # (1-7) * 7
    n_line_head = 7  #
    n_max_time = 7*4*3 + n_line_tail  # 7*4*3 + n_line_tail
    n_step_time = 1  # occur 7, treat 1
    type = 'treat'  # occur  treat
    date_str = datetime(2023, 8, 21).strftime("%Y%m%d")

    network = PaddleBaseClassifier.load('./model/'+date_str+'_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m01_fl.itc')

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230601)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_all.shape:', df_all.shape)

    start_date = datetime(2023, 7, 31)

    dfAlt4 = df_all.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_tail)
    # print('='*100)
    # print('i:',i)
    print('dfAlt4.shape:', dfAlt4.shape)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('current time1:', formatted_time)

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
    print('current time2:', formatted_time)

    y_val = []
    y_val_customerid = []
    for dataset in tsdatasets1:
        y_val.append(dataset.static_cov['Y'])
        y_val_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_label1 = np.array(y_val)

    from paddlets.transform import MinMaxScaler, StandardScaler

    min_max_scaler = StandardScaler()
    tsdatasets1 = min_max_scaler.fit_transform(tsdatasets1)

    pred_val = network.predict(tsdatasets1)
    pred_val_prob = network.predict_proba(tsdatasets1)[:, 1]
    print(pred_val)
    print(pred_val_prob)

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
    #plt.savefig("./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_ROC_val_" + str(n_line_tail) + "_fl.png")
    plt.show()
    plt.plot(tpr, lw=2, label='tpr')
    plt.plot(fpr, lw=2, label='fpr')
    plt.plot(tpr - fpr, label='ks')
    plt.title('KS = %0.2f(val)' % max(tpr - fpr))
    plt.legend(loc="lower right")
    #plt.savefig("./result/" + date_str + "_" + type + "_y16_m1_y23_m01_y23_m7_KS_val_" + str(n_line_tail) + "_fl.png")
    plt.show()



if __name__ == '__main__':
    #test_for_ks()
    #test_for_report()
    predict_weekly()

