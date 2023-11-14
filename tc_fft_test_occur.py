# coding=utf-8
import csv
import sys, os
import pickle
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.utils import shuffle

import warnings
from paddlets.datasets.repository import get_dataset, dataset_list
import matplotlib.pyplot as plt
from typing import List

from tc_fft_train_occur import ts2vec_cluster_datagroup_model,dl_model_forward_ks_roc
from tc_fft_train_occur import tsfresh_ftr_augment_select,ml_model_forward_ks_roc
from tc_fft_train_occur import ensemble_dl_ml_base_score_test
warnings.filterwarnings('ignore', category=DeprecationWarning)

from paddlets import TSDataset
from paddlets.analysis import FFT
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

    n_line_tail = 60  # (1-7) * 30
    n_line_head = 60
    n_max_time = 28*3 + n_line_tail  # 7*4*2 + n_line_tail    2 month
    n_step_time = 7  # occur 7, treat 1
    type = 'occur'  # occur  treat

    # network=PaddleBaseClassifier.load('./model/0718_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m4_v1.itc')
    # network=PaddleBaseClassifier.load('./model/0baseM5/0719_50_20_16_244_fft_p_t_SS_t' +str(n_line_head) +'_y18_m01_y23_m7_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y23_m01_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m10_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/1multiM5/0726_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y22_m07_y23_m07_v1.itc')
    # network=PaddleBaseClassifier.load('./model/2baseM6/0802_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y18_m01_y23_m7_t0.itc')
    #network = PaddleBaseClassifier.load('./model/3multiM6/0809_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_v1.itc')
    #network = PaddleBaseClassifier.load('./model/3multiM6/20230821_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_fl.itc')
    network = PaddleBaseClassifier.load('./model/4multiM6/20230906_' + type + '_50_20_16_244_fft_p_t_SS_t' + str(n_line_tail) + '_y16_m01_y23_m7_fl.itc')

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230901)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_all.shape:', df_all.shape)

    # 指定要追加的文件名
    # filename = './result/20230803_M5_y18_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y23_m01_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y22_m10_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230803_M5_y22_m07_y23_m7_result_data_'+str(n_line_head)+'.csv'
    # filename = './result/20230817_' + type + '_M6_y16_m01_y23_m7_result_' + str(n_line_head) + '.csv'
    filename = './result/20230905_' + type + '_M6_y16_m01_y23_m7_result_' + str(n_line_head) + '_fl.csv'

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

    # df23 = pd.read_csv("./data/0808_train/occur/2023_202308081713.csv", header=0, sep=',', encoding='gbk')
    # df22 = pd.read_csv("./data/0808_train/occur/2022_202308081710.csv", header=0, sep=',', encoding='gbk')

    df23 = pd.read_csv("./data/0825_train/occur/2023_202308251939.csv", header=0, sep=',', encoding='gbk')
    df22_4 = pd.read_csv("./data/0825_train/occur/2022_10_12_202308250913.csv", header=0, sep=',', encoding='gbk')
    df22_3 = pd.read_csv("./data/0825_train/occur/2022_7_10_202308250922.csv", header=0, sep=',', encoding='gbk')
    df22_2 = pd.read_csv("./data/0825_train/occur/2022_4_7_202308250927.csv", header=0, sep=',', encoding='gbk')
    df22_1 = pd.read_csv("./data/0825_train/occur/2022_1_4_202308250931.csv", header=0, sep=',', encoding='gbk')
    # print(df207.shape)
    df_all = pd.concat([df22_1,df22_2,df22_3,df22_4,df23])
    print(df_all.shape)
    del df22_1,df22_2,df22_3,df22_4,df23

    #col = df_all.columns.tolist().remove('CUSTOMER_ID').remove('RDATE').remove('Y')
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

    n_line_tail = 90  # (1~5) * 30
    n_line_head = 90  # = tail
    n_max_time = 7*4*6 + n_line_tail  # half_year + n_line_tail
    n_step_time = 7  # occur 7, treat 1
    type = 'occur'  # occur  treat
    date_str = datetime(2023, 9, 1).strftime("%Y%m%d")

    network = PaddleBaseClassifier.load('./model/'+date_str+'_'+type+'_50_20_16_244_fft_p_t_SS_t'+str(n_line_tail)+'_y16_m01_y23_m01_fl_35000_7.itc')

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230101)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('df_all.shape:', df_all.shape)

    ###################### for test 1:100
    df_all_0 = df_all[df_all['Y'] == 0]
    df_all_1 = df_all[df_all['Y'] == 1]  # 24
    print('df_all_1.shape:', df_all_1.shape)
    # 从 0 中 筛选出 2400 个
    selected_groups = df_all_0['CUSTOMER_ID'].drop_duplicates().sample(n=2400, random_state=2400)
    # 获取每个选中组的所有样本
    df_part1_0_selected = df_all_0.groupby('CUSTOMER_ID').apply(
        lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    df_part1_0_selected = df_part1_0_selected.dropna(subset=['Y'])
    df_all = pd.concat([df_part1_0_selected, df_all_1])
    print('df_all.shape: ', df_all.shape)
    del df_all_0, df_all_1, df_part1_0_selected

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

def get_cur_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

################################### for predict online start
def prepare_data(csv_file_path_base:str, csv_file_path_credit:str):
    '''
    load csv data and do prepare
    :param csv_file_path_base: CUSTOMER_ID,RDATE,Y,INV_AVG_60,INV_RATIO_90...
    :param csv_file_path_credit: CUSTOMER_ID,RDATE,ICA_30,PCA_30,ZCA_30
    :return: dataframe
    '''
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
    if not os.path.exists(csv_file_path_base) or not os.path.exists(csv_file_path_credit):
        print('%s or %s not exists, please check.' % (csv_file_path_base, csv_file_path_credit))
        return -1

    df_base = pd.read_csv(csv_file_path_base, header=0, usecols=usecols, sep=',', encoding='gbk')
    credit_usecols = ['CUSTOMER_ID', 'RDATE', 'ICA_30', ]  # ICA_30,PCA_30,ZCA_30  'PCA_30', 'ZCA_30'
    df_credit = pd.read_csv(csv_file_path_credit, header=0, usecols=credit_usecols, sep=',', encoding='gbk')

    print('df_base.shape:',df_base.shape)
    df_all = pd.merge(df_base, df_credit, on=['CUSTOMER_ID', 'RDATE'], how='left')
    print('after merge df_all.shape:', df_all.shape)
    del df_credit, df_base

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
           'LRR_AVG_90', 'LSR_91_AVG_30', 'ICA_30']  # 90 + 1

    df_all[col] = df_all[col].astype(float)

    n_line_tail = 30  # (1-5) * 30
    n_line_head = 30  # = tail

    step = 5
    ftr_num_str = '91'
    filter_num_ratio = 1 / 8  # 1/5

    # df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20230901)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('after filter length df_all.shape:', df_all.shape)
    df_all = df_all.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = size - i - batch_size
            if start_position < 0:
                break
            end_position = size - i
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_all = df_all.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_all.head:', df_all.head(32))
    print('after generate_new_groups df_all.shape:', df_all.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和; 3 -> watch out
    count_df = df_all.groupby('CUSTOMER_ID').apply(lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('threshold K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_all = df_all[~df_all['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0 or null df_all.shape:', df_all.shape)

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)
    df_all = df_all.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('at last df_all.shape: ', df_all.shape)

    print('1. prepare_data done at ', get_cur_time())
    return df_all

def transform_data(datasets: pd.DataFrame):
    '''
    get data from prepare_data for transformation
    :param datasets:
    :return: transform_data, label, customer_id
    '''
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
           'LRR_AVG_90', 'LSR_91_AVG_30', 'ICA_30']  # 90 + 1

    tsdatasets_all = TSDataset.load_from_dataframe(
        df=datasets,
        group_id='CUSTOMER_ID',
        target_cols=col,
        # known_cov_cols='CUSTOMER_ID',
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )

    fft = FFT(fs=1, half=False)  # _amplitude  half
    for data in tsdatasets_all:
        resfft = fft(data)
        for x in data.columns:
            # ----------------- fft
            resfft[x + "_amplitude"].index = data[x].index
            resfft[x + "_phase"].index = data[x].index
            data.set_column(column=x + "_amplitude", value=resfft[x + "_amplitude"], type='target')
            data.set_column(column=x + "_phase_fft", value=resfft[x + "_phase"], type='target')

    y_all = []  # fake
    y_all_customerid = []
    for dataset in tsdatasets_all:
        y_all.append(dataset.static_cov['Y'])
        y_all_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_all = np.array(y_all)
    y_all_customerid = np.array(y_all_customerid)

    from paddlets.transform import StandardScaler
    ss_scaler = StandardScaler()
    tsdatasets_all = ss_scaler.fit_transform(tsdatasets_all)

    print('2. transform_data done at :', get_cur_time())
    return tsdatasets_all,y_all,y_all_customerid

def group_data(tsdatasets: List[TSDataset], y_labels: np.ndarray, y_cutomersid: np.ndarray,
                cluster_model_path: str, cluster_model_file: str, del_num:int = 1):
    '''
    just call function of ts2vec_cluster_datagroup_model from tc_fft_train_occur.py
    :param tsdatasets:
    :param y_labels:
    :param y_cutomersid:
    :param cluster_model_path:
    :param cluster_model_file:
    :param del_num: 1 -> all test data is retained and none is deleted
    :return:
    '''
    tsdataset_list_all, label_list_all, customersid_list_all = ts2vec_cluster_datagroup_model(tsdatasets,
                                                                                              y_labels,
                                                                                              y_cutomersid,
                                                                                              cluster_model_path,
                                                                                              cluster_model_file,
                                                                                              del_num)
    print('3. group_data done at :', get_cur_time())
    return tsdataset_list_all, label_list_all, customersid_list_all

def dl_predict(tsdataset_list_all: List,label_list_all: List,customersid_list_all: List,):
    '''
    get data of group_data and do deeplearning predict
    :param tsdataset_list_all:
    :param label_list_all:
    :param customersid_list_all:
    :return:
    '''
    for i in range(len(label_list_all)):
        model_index = i
        dataset_group_index = i
        model_file_path = './model/20231024_occur_2017_addcredit_step5_reclass_less_800_200_100_' \
                          '20230101_2_1_16_ftr_91_t30_fl_aug_' + str(model_index) + '.itc'
        if not os.path.exists(model_file_path):  #  model 0 must exist
            model_file_path = './model/20231024_occur_2017_addcredit_step5_reclass_less_800_200_100_' \
                          '20230101_2_1_16_ftr_91_t30_fl_aug_0.itc'
            model_index = 0
        result_file_path = './result/' + 'dl_' +str(model_index) + '_' + str(dataset_group_index) + '.csv'
        dl_model_forward_ks_roc(model_file_path, result_file_path, tsdataset_list_all[i], label_list_all[i], customersid_list_all[i])

    print('4.a dl_predict done at :', get_cur_time())

def ml_predict(tsdataset_list_all: List,label_list_all: List,customersid_list_all: List,df_all: pd.DataFrame):
    '''
    get data of group_data and do machinelearning predict
    :param tsdataset_list_all:
    :param label_list_all:
    :param customersid_list_all:
    :param df_all:  from  prepare_data()
    :return:
    '''
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
    fdr_level = 0.05
    for i in range(len(label_list_all)):
        model_index = i
        dataset_group_index = i
        df_all_part = df_all[df_all['CUSTOMER_ID'].isin(customersid_list_all[i])]
        model_file_path = './model/20231024_occur_2017_addcredit_augmentftr_step5_reclass_less_800_200_100_' \
                            '20230101_3_7_100_balanced_0.05_ftr_91_t30_ftr_select_' + str(model_index) + '.pkl'
        if not os.path.exists(model_file_path): #  model 0 must exist
            model_file_path = './model/20231024_occur_2017_addcredit_augmentftr_step5_reclass_less_800_200_100_' \
                            '20230101_3_7_100_balanced_0.05_ftr_91_t30_ftr_select_0.pkl'
            model_index = 0
        ftr_list_file_path = './model/20231024_occur_2017_addcredit_augmentftr_step5_reclass_less_800_200_100_' \
                                 '20230101_0.05_ftr_list_'+ str(model_index) + '.pkl'
        if not os.path.exists(ftr_list_file_path):
            print('{} not exists, please check.'.format(ftr_list_file_path))
            return -1
        with open(ftr_list_file_path, 'rb') as f:
            select_cols = pickle.load(f)
        print('length of select cols is:', len(select_cols))
        result_file_path = './result/' + 'ml_' +str(model_index) + '_' + str(dataset_group_index) + '.csv'
        df_test_ftr_select_notime = tsfresh_ftr_augment_select(df_all_part, usecols, select_cols, fdr_level)
        ml_model_forward_ks_roc(model_file_path, result_file_path, df_test_ftr_select_notime.loc[:,select_cols], np.array(df_test_ftr_select_notime.loc[:,'Y']),
                                 np.array(df_test_ftr_select_notime.loc[:,'CUSTOMER_ID']))

    print('4.b ml_predict done at :', get_cur_time())

def ensemble_predict(dl_result_file_path: str,ml_result_file_path: str,ensemble_model_file_path: str, ensemble_result_file_path: str):
    ensemble_dl_ml_base_score_test(dl_result_file_path,ml_result_file_path,ensemble_model_file_path,ensemble_result_file_path)
    print('5 ensemble_predict done at :', get_cur_time())

def predict_pipeline():
    base_path = './data/0720_2639/2023_8_202310241410.csv'
    credit_path =  './data/0720_2639/credit/202310241401.csv'
    df = prepare_data(base_path, credit_path)
    tsdatasets_all,y_all,y_all_customerid = transform_data(df)
    cluster_model_path = './model/cluster_step5_credit1_90_2017_20231024/'
    cluster_model_file = '20231024-repr-cluster-partial-train-6.pkl'
    tsdataset_list_all, label_list_all, customersid_list_all = group_data(tsdatasets_all,
                                                                          y_all,
                                                                          y_all_customerid,
                                                                          cluster_model_path,
                                                                          cluster_model_file)
    dl_predict(tsdataset_list_all, label_list_all, customersid_list_all)
    ml_predict(tsdataset_list_all, label_list_all, customersid_list_all, df)

################################### for predict online end
def ensemble_dl_ml_predict():
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
               'LRR_AVG_90', 'LSR_91_AVG_30']  # 90 cols  1/8  Y ->
    #df221 = pd.read_csv("./data/0720_2639/22_1_202307201615.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    #df227 = pd.read_csv("./data/0720_2639/22_7_202307201610.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    df23_1 = pd.read_csv("./data/0720_2639/2023_1_5_202308171425.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    # df23_2 = pd.read_csv("./data/0720_2639/2023_5_8_202308171416.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    df23_2 = pd.read_csv("./data/0720_2639/2023_5_6_202309081528.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    df23_3 = pd.read_csv("./data/0720_2639/2023_6_7_202309081530.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    df23_4 = pd.read_csv("./data/0720_2639/2023_7_8_202309081532.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    df23_5 = pd.read_csv("./data/0720_2639/2023_8_202310241410.csv", header=0, usecols=usecols,sep=',', encoding='gbk')
    credit_usecols = ['CUSTOMER_ID', 'RDATE', 'ICA_30',]  # ICA_30,PCA_30,ZCA_30  'PCA_30', 'ZCA_30' 2023_8_202310241410
    df_credit = pd.read_csv("./data/0720_2639/credit/202310241401.csv", header=0, usecols=credit_usecols, sep=',',encoding='gbk')

    df_all = pd.concat([df23_1, df23_2, df23_3, df23_4, df23_5])
    del  df23_1, df23_2, df23_3, df23_4, df23_5
    print(df_all.shape)
    df_all = pd.merge(df_all, df_credit, on=['CUSTOMER_ID', 'RDATE'], how='left')
    print('after merge df_all.shape:', df_all.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('1 read csv :', formatted_time)

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
           'LRR_AVG_90', 'LSR_91_AVG_30', 'ICA_30']  # 90 + 1

    df_all[col] = df_all[col].astype(float)

    n_line_tail = 30  # (1-5) * 30
    n_line_head = 30  # = tail

    step = 5
    date_str = datetime(2023, 10, 24).strftime("%Y%m%d")
    split_date_str = '20230101'
    ftr_num_str = '91'
    filter_num_ratio = 1 / 8  # 1/5
    ftr_good_year_split = 2017
    ########## model cnn dt
    epochs = 2
    patiences = 1  # 10
    kernelsize = 4
    max_depth = 2 # 2 3 4 5
    num_leaves = 3 # 3 7 15 31
    n_estimators = 50 # 50 100
    class_weight =  None # 'balanced'  None
    lc_c = [0.06, 0.03, 2.0,] #
    fdr_level = 0.001 # 0.05(default)  0.04 0.03 0.02 0.01
    cluster_model_path = './model/cluster_step' + str(step) + '_credit1_90_' + str(ftr_good_year_split) + '_' + date_str + '/'
    cluster_model_file = date_str + '-repr-cluster-partial-train-6.pkl'
    cluster_less_train_num = 800
    cluster_less_val_num = 200
    cluster_less_test_num = 100

    dl_type = 'occur_' + str(ftr_good_year_split) + '_addcredit_step' + str(step) + '_reclass_less_' + \
              str(cluster_less_train_num) + '_' + str(cluster_less_val_num)+ '_' + str(cluster_less_test_num)
    ml_type = 'occur_' + str(ftr_good_year_split) + '_addcredit_augmentftr_step' + str(step) + '_reclass_less_' + \
              str(cluster_less_train_num) + '_' + str(cluster_less_val_num) + '_' + str(cluster_less_test_num)
    ensemble_type = 'occur_ensemble'

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: max(x["RDATE"]) >= 20231001)
    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_tail)
    print('1 df_all.shape:', df_all.shape)

    df_all = df_all[df_all['CUSTOMER_ID'].isin(['SMCRWSQ2206', 'SMCRWSQ200U'])]
    print('2 df_all.shape:', df_all.shape)

    # selected_groups = df_all['CUSTOMER_ID'].drop_duplicates().sample(n=100)
    # 获取每个选中组的所有样本
    # df_all_selected = df_all.groupby('CUSTOMER_ID').apply(lambda x: x if x.name in selected_groups.values else None).reset_index(drop=True)
    # df_all = df_all_selected.dropna(subset=['Y'])
    # print('df_all_selected.shape:', df_all.shape)
    # 'SMCRWSQ2206' 'SMCRWSQ200U'
    # selected_rows = df_all[df_all['CUSTOMER_ID'].isin(['SMCRWSQ2206', 'SMCRWSQ200U'])]

    df_all = df_all.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)).reset_index(drop=True)

    # 定义每次读取的数量
    batch_size = n_line_head

    def generate_new_groups(group):
        new_groups = []
        size = len(group)
        # 循环切片生成新的组
        for i in range(0, size, step):  # range(0,size,2)
            start_position = size - i - batch_size
            if start_position < 0:
                break
            end_position = size - i
            # 获取当前组的一部分数据
            batch = group.iloc[start_position:end_position].copy()
            # 修改组名
            batch['CUSTOMER_ID'] = f'{group.iloc[i]["CUSTOMER_ID"]}_{i}'
            # 将切片后的数据添加到新的组列表中
            new_groups.append(batch)
        # 将新的组数据合并为一个 DataFrame
        new_df = pd.concat(new_groups)
        return new_df

    # 将数据按照 CUSTOMER_ID 列的值分组，并应用函数生成新的组
    df_all = df_all.groupby('CUSTOMER_ID').apply(generate_new_groups).reset_index(drop=True)
    # 输出结果
    print('df_all.head:', df_all.head(32))
    print('df_all.shape:', df_all.shape)

    # 按照 group 列进行分组，统计每个分组中所有列元素为 0 或 null 的个数的总和; 3 -> watch out
    count_df = df_all.groupby('CUSTOMER_ID').apply(lambda x: (x.iloc[:, 3:] == 0).sum() + x.iloc[:, 3:].isnull().sum()).sum(axis=1)
    # 设定阈值 K
    K = n_line_head * int(ftr_num_str) * filter_num_ratio
    print('K:', K)
    # 删除满足条件的组
    filtered_groups = count_df[count_df.gt(K)].index
    print(filtered_groups)
    df_all = df_all[~df_all['CUSTOMER_ID'].isin(filtered_groups)]
    print('after filter 0/null df_all.shape:', df_all.shape)

    df_all = df_all.groupby(['CUSTOMER_ID']).filter(lambda x: len(x) >= n_line_head)

    df_all = df_all.groupby(['CUSTOMER_ID']).apply(lambda x: x.sort_values(["RDATE"], ascending=True)). \
        reset_index(drop=True).groupby(['CUSTOMER_ID']).tail(n_line_head)
    print('df_all.shape: ', df_all.shape)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('2 load data:', formatted_time)

    from paddlets import TSDataset
    from paddlets.analysis import FFT
    tsdatasets_all = TSDataset.load_from_dataframe(
        df=df_all,
        group_id='CUSTOMER_ID',
        target_cols=col,
        # known_cov_cols='CUSTOMER_ID',
        fill_missing_dates=True,
        fillna_method="zero",
        static_cov_cols=['Y', 'CUSTOMER_ID'],
    )

    fft = FFT(fs=1, half=False)  # _amplitude  half
    # cwt = CWT(scales=n_line_tail/2)
    for data in tsdatasets_all:
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
    print('3 transform data:', formatted_time)

    y_all = [] # fake
    y_all_customerid = []
    for dataset in tsdatasets_all:
        y_all.append(dataset.static_cov['Y'])
        y_all_customerid.append(dataset.static_cov['CUSTOMER_ID'])
        dataset.static_cov = None
    y_all = np.array(y_all)
    y_all_customerid = np.array(y_all_customerid)

    from paddlets.transform import StandardScaler
    ss_scaler = StandardScaler()
    tsdatasets_all = ss_scaler.fit_transform(tsdatasets_all)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print('4 group data:', formatted_time)
    # cluster_less_train_num
    tsdataset_list_all, label_list_all, customersid_list_all = ts2vec_cluster_datagroup_model(tsdatasets_all,
                                                                                                    y_all,
                                                                                                    y_all_customerid,
                                                                                                    cluster_model_path,
                                                                                                    cluster_model_file,
                                                                                                    1)
    for i in range(len(label_list_all)):
        for j in range(len(lc_c)):
            model_file_path = './model/' + date_str + '_' + dl_type + '_' + split_date_str + '_' + str(epochs) + '_' + \
                              str(patiences) + '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                              '_fl_aug_' + str(j) + '.itc'
            result_file_path = './result/' + date_str + '_' + dl_type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) + \
                               '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_predict_aug_' + \
                               str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            if os.path.exists(result_file_path):
                print('{} already exists, so just remove it and still infer.'.format(result_file_path))
                os.remove(result_file_path)
                print(f" file '{result_file_path}' is removed.")
            dl_model_forward_ks_roc(model_file_path, result_file_path, tsdataset_list_all[i], label_list_all[i], customersid_list_all[i])

    for i in range(len(label_list_all)):
        df_all_part = df_all[df_all['CUSTOMER_ID'].isin(customersid_list_all[i])]

        for j in range(len(lc_c)):
            model_file_path = './model/' + date_str + '_' + ml_type + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                              str(num_leaves) + '_' + str(n_estimators)+'_' +str(class_weight)+ '_'+str(fdr_level) + '_ftr_' + ftr_num_str + \
                              '_t' + str(n_line_tail) + '_ftr_select_' + str(j) + '.pkl'
            ftr_list_file_path = './model/' + date_str + '_' + ml_type + '_' + split_date_str+ '_'+str(fdr_level) +'_ftr_list_' + str(j) + '.pkl'
            print(ftr_list_file_path)
            with open(ftr_list_file_path, 'rb') as f:
                select_cols = pickle.load(f)
            print('len select cols:', len(select_cols))
            result_file_path = './result/' + date_str + '_' + ml_type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                               '_' + str(n_estimators)+'_' +str(class_weight)+ '_'+str(fdr_level) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                               '_ftr_select_predict_' + str(j) + '_' + str(i) + '.csv'
            print(result_file_path)
            if os.path.exists(result_file_path):
                print('{} already exists, so just remove it and still infer.'.format(result_file_path))
                os.remove(result_file_path)
                print(f" file '{result_file_path}' is removed.")
            df_test_ftr_select_notime = tsfresh_ftr_augment_select(df_all_part, usecols, select_cols, fdr_level)
            ml_model_forward_ks_roc(model_file_path, result_file_path, df_test_ftr_select_notime.loc[:,select_cols], np.array(df_test_ftr_select_notime.loc[:,'Y']),
                                 np.array(df_test_ftr_select_notime.loc[:,'CUSTOMER_ID']))

    for i in range(len(label_list_all)):
        for j in range(len(lc_c)):
            dl_result_file_path = './result/' + date_str + '_' + dl_type + '_' + split_date_str + '_' + str(epochs) + '_' + str(patiences) + \
                                  '_' + str(kernelsize) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_fl_predict_aug_' + str(j) + '_' + str(i) + '.csv'
            ml_result_file_path = './result/' + date_str + '_' + ml_type + '_' + split_date_str + '_' + str(max_depth) + '_' + str(num_leaves) + \
                                  '_' + str(n_estimators) + '_' + str(class_weight) + '_' + str(fdr_level) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + \
                                  '_ftr_select_predict_' + str(j) + '_' + str(i) + '.csv'
            ensemble_model_file_path = './model/' + date_str + '_' + ensemble_type + '_' +str(lc_c[j]) + '_'+ split_date_str + '_' + str(epochs) + '_' + \
                                       str(patiences) + '_' + str(kernelsize) + '_' + str(max_depth) + '_' + str(num_leaves) + '_' + str(n_estimators) + \
                                       '_' + str(class_weight) +'_'+str(fdr_level) + '_ftr_' + ftr_num_str + '_t' + str(n_line_tail) + '_' + str(j) + '_lr.pkl'
            ensemble_result_file_path = './result/' + date_str + '_' + ensemble_type + '_'+str(lc_c[j]) + '_' + split_date_str + '_' + str(max_depth) + '_' + \
                                    str(num_leaves) + '_' + str(n_estimators) + '_' + str(class_weight) + '_'+str(fdr_level) +  '_ftr_' + ftr_num_str + '_t' + \
                                    str(n_line_tail) + '_predict_' + str(j) + '_' + str(i) + '.csv'
            #if os.path.exists(ensemble_result_file_path) or (i != j):
            #    print('{} already exists, but still infer.'.format(ensemble_result_file_path))
            #    continue
            print(ensemble_result_file_path)
            ensemble_dl_ml_base_score_test(dl_result_file_path,ml_result_file_path,ensemble_model_file_path,ensemble_result_file_path)

if __name__ == '__main__':
    # test_for_report()
    # predict_weekly()
    ensemble_dl_ml_predict()
