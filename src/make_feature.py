# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

def gen_pos_neg_aid_fea():
    train_data = pd.read_csv('input/train.csv')
    test2_data = pd.read_csv('input/test2.csv')

    train_user = train_data.uid.unique()
    
    # user-aid dict
    uid_dict = defaultdict(list)
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        uid_dict[row[2]].append([row[1], row[3]])

    # user convert
    uid_convert = {}
    for uid in tqdm(train_user):
        pos_aid, neg_aid = [], []
        for data in uid_dict[uid]:
            if data[1] > 0:
                pos_aid.append(data[0])
            else:
                neg_aid.append(data[0])
        uid_convert[uid] = [pos_aid, neg_aid]  

    test2_neg_pos_aid = {}
    for row in tqdm(test2_data.itertuples(), total=len(test2_data)):
        aid = row[1]
        uid = row[2]
        if uid_convert.get(uid, []) == []:
            test2_neg_pos_aid[row[0]] =  ['', '', -1] 
        else:
            pos_aid, neg_aid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
            convert = len(pos_aid)  / (len(pos_aid) + len(neg_aid)) if (len(pos_aid) + len(neg_aid)) > 0 else -1
            test2_neg_pos_aid[row[0]] =  [' '.join(map(str, pos_aid)), ' '.join(map(str, neg_aid)), convert]
    df_test2 =  pd.DataFrame.from_dict(data=test2_neg_pos_aid, orient='index')
    df_test2.columns = ['pos_aid', 'neg_aid', 'uid_convert']

    train_neg_pos_aid = {}
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        aid = row[1]
        uid = row[2]
        pos_aid, neg_aid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
        if aid in pos_aid:
            pos_aid.remove(aid)
        if aid in neg_aid:
            neg_aid.remove(aid)
        convert = len(pos_aid)  / (len(pos_aid) + len(neg_aid)) if (len(pos_aid) + len(neg_aid)) > 0 else -1
        train_neg_pos_aid[row[0]] = [' '.join(map(str, pos_aid)), ' '.join(map(str, neg_aid)), convert]

    df_train = pd.DataFrame.from_dict(data=train_neg_pos_aid, orient='index')
    df_train.columns = ['pos_aid', 'neg_aid', 'uid_convert']
     
    df_train.to_csv("dataset/train_neg_pos_aid.csv", index=False)
    df_test2.to_csv("dataset/test2_neg_pos_aid.csv", index=False)

def gen_uid_aid_fea():
    '''
    载入数据，　提取aid, uid的全局统计特征
    '''
    train_data = pd.read_csv('input/train.csv')
    test1_data = pd.read_csv('input/test1.csv')
    test2_data = pd.read_csv('input/test2.csv')

    ad_Feature = pd.read_csv('input/adFeature.csv')

    train_len = len(train_data)  # 45539700
    test1_len = len(test1_data)
    test2_len = len(test2_data)  # 11727304

    ad_Feature = pd.merge(ad_Feature, ad_Feature.groupby(['campaignId']).aid.nunique().reset_index(
    ).rename(columns={'aid': 'campaignId_aid_nunique'}), how='left', on='campaignId')

    df = pd.concat([train_data, test1_data, test2_data], axis=0)
    df = pd.merge(df, df.groupby(['uid'])['aid'].nunique().reset_index().rename(
        columns={'aid': 'uid_aid_nunique'}), how='left', on='uid')

    df = pd.merge(df, df.groupby(['aid'])['uid'].nunique().reset_index().rename(
        columns={'uid': 'aid_uid_nunique'}), how='left', on='aid')

    df['uid_count'] = df.groupby('uid')['aid'].transform('count')
    df = pd.merge(df, ad_Feature[['aid', 'campaignId_aid_nunique']], how='left', on='aid')

    fea_columns = ['campaignId_aid_nunique', 'uid_aid_nunique', 'aid_uid_nunique', 'uid_count', ]

    df[fea_columns].iloc[:train_len].to_csv('dataset/train_uid_aid.csv', index=False)
    df[fea_columns].iloc[train_len: train_len+test1_len].to_csv('dataset/test1_uid_aid.csv', index=False)
    df[fea_columns].iloc[-test2_len:].to_csv('dataset/test2_uid_aid.csv', index=False)

def digitize():
    uid_aid_train = pd.read_csv('dataset/train_uid_aid.csv')
    uid_aid_test1 = pd.read_csv('dataset/test1_uid_aid.csv')
    uid_aid_test2 = pd.read_csv('dataset/test2_uid_aid.csv')
    uid_aid_df = pd.concat([uid_aid_train, uid_aid_test1, uid_aid_test2], axis=0)
    for col in range(3):
        bins = []
        for percent in [0, 20, 35, 50, 65, 85, 100]:
            bins.append(np.percentile(uid_aid_df.iloc[:, col], percent))
        uid_aid_train.iloc[:, col] =  np.digitize(uid_aid_train.iloc[:, col], bins, right=True)
        uid_aid_test1.iloc[:, col] =  np.digitize(uid_aid_test1.iloc[:, col], bins, right=True)
        uid_aid_test2.iloc[:, col] =  np.digitize(uid_aid_test2.iloc[:, col], bins, right=True)
    
    count_bins = [1, 2, 4, 6, 8, 10, 16, 27, 50]
    uid_aid_train.iloc[:, 3] =  np.digitize(uid_aid_train.iloc[:, 3], count_bins, right=True)
    uid_aid_test1.iloc[:, 3] =  np.digitize(uid_aid_test1.iloc[:, 3], count_bins, right=True)
    uid_aid_test2.iloc[:, 3] =  np.digitize(uid_aid_test2.iloc[:, 3], count_bins, right=True)

    uid_convert_train = pd.read_csv("dataset/train_neg_pos_aid.csv", usecols=['uid_convert'])
    uid_convert_test2 = pd.read_csv("dataset/test2_neg_pos_aid.csv", usecols=['uid_convert'])

    convert_bins = [-1, 0, 0.1, 0.3, 0.5, 0.7, 1]
    uid_convert_train.iloc[:, 0] =  np.digitize(uid_convert_train.iloc[:, 0], convert_bins, right=True)
    uid_convert_test2.iloc[:, 0] =  np.digitize(uid_convert_test2.iloc[:, 0], convert_bins, right=True)

    
    uid_aid_train = pd.concat([uid_aid_train, uid_convert_train], axis=1)
    uid_aid_test2 = pd.concat([uid_aid_test2, uid_convert_test2], axis=1)

    uid_aid_train.to_csv('dataset/train_uid_aid_bin.csv', index=False)
    uid_aid_test2.to_csv('dataset/test2_uid_aid_bin.csv', index=False)

if __name__ == '__main__':
    print("Make Feature...")
    print("1. Generate pos_neg_aid_fea")
    gen_pos_neg_aid_fea()
    print("2. Generate uid_aid_fea")
    gen_uid_aid_fea()
    print("3. Digitize numerical feature")
    digitize()
    print("Make Feature Done")